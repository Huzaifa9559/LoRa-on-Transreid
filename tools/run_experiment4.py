#!/usr/bin/env python3
"""
tools/run_experiment4.py

Sequential orchestrator for Experiment 4 (BitFit, LN-tuning, Bottleneck Adapters)
on vast.ai. Runs the 9 configs one at a time as train.py subprocesses, streams each
run's output to logs/experiment4/run_XX_<method>_<window>.log, and records a
machine-readable summary in logs/experiment4/progress.json.

Resumable: completed runs are skipped on restart unless --force.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# (label, config_path) — also defines the log numbering order.
EXPERIMENT4_CONFIGS = [
    ("bitfit_0_11", "configs/Market/bitfit_blocks_0_11.yml"),
    ("bitfit_4_11", "configs/Market/bitfit_blocks_4_11.yml"),
    ("bitfit_6_11", "configs/Market/bitfit_blocks_6_11.yml"),
    ("lntune_0_11", "configs/Market/lntune_blocks_0_11.yml"),
    ("lntune_4_11", "configs/Market/lntune_blocks_4_11.yml"),
    ("lntune_6_11", "configs/Market/lntune_blocks_6_11.yml"),
    ("adapter_0_11", "configs/Market/adapter_blocks_0_11_r16.yml"),
    ("adapter_4_11", "configs/Market/adapter_blocks_4_11_r16.yml"),
    ("adapter_6_11", "configs/Market/adapter_blocks_6_11_r16.yml"),
]

LOG_DIR = REPO_ROOT / "logs" / "experiment4"
PROGRESS_FILE = LOG_DIR / "progress.json"

MAP_RE = re.compile(r"mAP:\s*([0-9.]+)%")
# \b after the rank digit prevents "Rank-1" from matching "Rank-10".
RANK_RE = re.compile(r"CMC curve, Rank-(\d+)\b[^:]*:\s*([0-9.]+)%")
TRAINABLE_RE = re.compile(r"Trainable params: ([\d,]+) / Total params: ([\d,]+) \(([0-9.]+)%\)")


def _load_progress() -> dict:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_progress(progress: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def _gpu_stats() -> tuple | None:
    """Return (util%, used_mib, total_mib) for the first GPU via nvidia-smi, or None."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return None
        line = out.stdout.strip().splitlines()[0]
        util, used, total = [int(x.strip()) for x in line.split(",")]
        return util, used, total
    except Exception:
        return None


class _GpuMonitor:
    """Polls nvidia-smi in a thread while a run executes; records peaks."""

    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self.peak_util = 0
        self.peak_vram_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def _loop(self):
        while not self._stop.is_set():
            stats = _gpu_stats()
            if stats is not None:
                util, used_mib, _ = stats
                self.peak_util = max(self.peak_util, util)
                self.peak_vram_mib = max(self.peak_vram_mib, used_mib)
            self._stop.wait(self.interval)


def _parse_metrics(log_path: Path) -> dict:
    text = log_path.read_text(errors="ignore")
    metrics = {}
    m = MAP_RE.findall(text)
    if m:
        metrics["mAP"] = float(m[-1])
    for rank, value in RANK_RE.findall(text):
        metrics[f"R{rank}"] = float(value)
    m = TRAINABLE_RE.findall(text)
    if m:
        metrics["trainable_params"] = int(m[-1][0].replace(",", ""))
        metrics["total_params"] = int(m[-1][1].replace(",", ""))
        metrics["param_ratio"] = float(m[-1][2])
    return metrics


def _train_running() -> bool:
    """True if a train.py (or another run_experiment4) is already running.

    Prevents two runners fighting over the same GPU — the cause of the
    "CUDA-capable device(s) is/are busy or unavailable" failures and the
    stuck-VRAM state during Experiment 4.
    """
    try:
        out = subprocess.run(["pgrep", "-f", "train.py"], capture_output=True, text=True)
        return out.returncode == 0
    except Exception:
        return False


def run_one(label: str, config_rel: str, seed: int, epochs: int | None, workers: int,
            force: bool, stop_on_error: bool, cpu_only: bool, progress: dict) -> int:
    if _train_running():
        print("ERROR: a train.py process is already running. Kill it first:")
        print("       pkill -f train.py   (or wait for it to finish)")
        sys.exit(1)
    key = f"{label}_s{seed}"
    if not force and key in progress and progress[key].get("returncode") == 0:
        print(f"[skip] {label} (seed {seed}) already completed — use --force to rerun")
        return 0

    log_path = LOG_DIR / f"run_{len(progress) + 1:02d}_{label}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(REPO_ROOT / "train.py"), "--config_file", config_rel,
           "SOLVER.SEED", str(seed)]
    if epochs is not None:
        cmd += ["SOLVER.MAX_EPOCHS", str(epochs)]
    cmd += ["DATALOADER.NUM_WORKERS", str(workers)]
    if cpu_only:
        cmd += ["MODEL.DEVICE", "cpu", "DATALOADER.NUM_WORKERS", "0"]

    env = dict(os.environ)
    # PyTorch 2.x: use expandable segments to reduce CUDA fragmentation (the
    # OOM we hit was fragmentation, not capacity — the model fits in ~7 GB).
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print(f"\n===> [{label}] running: {' '.join(cmd)}")
    print(f"     log: {log_path}")

    start = time.time()
    monitor = _GpuMonitor(interval=5.0)
    monitor.start()
    with open(log_path, "wb") as logf:
        # Tee: stream train.py output to the per-run log file AND the terminal
        # so progress is visible live (not just after the run finishes).
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT), env=env
        )
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, b""):
            logf.write(line)
            logf.flush()
            sys.stdout.buffer.write(line)
            sys.stdout.flush()
        proc.wait()
    monitor.stop()
    wall = time.time() - start

    entry = {
        "config": config_rel,
        "seed": seed,
        "returncode": proc.returncode,
        "wall_seconds": round(wall, 1),
        # MiB / 1024 => binary gigabytes (GiB). Maximum sampled total GPU memory.
        "peak_vram_gib": round(monitor.peak_vram_mib / 1024, 2),
        "peak_gpu_util": monitor.peak_util,
        **{"mAP": None, "R1": None, "param_ratio": None,
           "trainable_params": None, "total_params": None},
        **(_parse_metrics(log_path) if proc.returncode == 0 else {}),
    }
    progress[key] = entry
    _save_progress(progress)

    status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
    print(f"     [{label}] {status} in {wall:.0f}s | "
          f"peak VRAM {entry['peak_vram_gib']} GiB | peak GPU {entry['peak_gpu_util']}%")
    if proc.returncode != 0:
        print(f"     check log: {log_path}")
        if stop_on_error:
            print("     --stop-on-error set; aborting.")
            sys.exit(proc.returncode)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Experiment 4 sequentially (BitFit/LN/Adapter)")
    parser.add_argument("--seed", type=int, default=1234, help="SOLVER.SEED override (default 1234)")
    parser.add_argument("--epochs", type=int, default=None, help="SOLVER.MAX_EPOCHS override (debug)")
    parser.add_argument("--workers", type=int, default=4,
                        help="DATALOADER.NUM_WORKERS (default 4; lower reduces dataloader "
                             "fork memory pressure / segfaults on low-RAM hosts)")
    parser.add_argument("--only", type=str, default=None,
                        help="Filter: METHOD[,WINDOW] e.g. 'bitfit' or 'lntune,6_11'")
    parser.add_argument("--only-run", type=int, default=None, help="Run only the Nth config (1-based)")
    parser.add_argument("--force", action="store_true", help="Rerun even if recorded as completed")
    parser.add_argument("--stop-on-error", action="store_true", help="Abort on first failed run")
    parser.add_argument("--cpu-only", action="store_true", help="CPU run (local sanity only)")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    progress = _load_progress()

    print("=" * 70)
    print("  EXPERIMENT 4 — BitFit / LN-tuning / Bottleneck Adapters (sequential)")
    print("=" * 70)

    for idx, (label, config_rel) in enumerate(EXPERIMENT4_CONFIGS, start=1):
        if args.only_run is not None and idx != args.only_run:
            continue
        if args.only:
            method, _, window = label.partition("_")
            only_parts = args.only.replace(" ", "").split(",")
            if method not in only_parts:
                continue
            if len(only_parts) > 1 and window not in only_parts[1:]:
                continue
        run_one(label, config_rel, seed=args.seed, epochs=args.epochs, workers=args.workers,
                force=args.force, stop_on_error=args.stop_on_error, cpu_only=args.cpu_only,
                progress=progress)

    print("\n" + "=" * 70)
    print("  SUMMARY — logs/experiment4/progress.json")
    print("=" * 70)
    for key, entry in progress.items():
        print(f"  {key:<22} rc={entry['returncode']}  "
              f"params={entry.get('param_ratio')}%  mAP={entry.get('mAP')}  R1={entry.get('R1')}  "
              f"VRAM={entry.get('peak_vram_gib')}GiB  {entry.get('wall_seconds')}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

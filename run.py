#!/usr/bin/env python3
"""
One-shot bootstrap for vast.ai / Jupyter (zip upload workflow).

Upload TransReID-PEFT.zip to Jupyter, then run:

    python run.py

Market-1501 is downloaded automatically (~153 MB from Google Drive).
Optionally upload market1501.zip to skip the download.

Or with options:

    python run.py --config configs/Market/ssf_0_11_case1.yml
    python run.py --setup-only
    python run.py --skip-unzip --skip-setup

Jupyter notebook:

    !python run.py
    # or after unzipping into the repo folder:
    %run run.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR))

from datasets.download_market1501 import ensure_market1501
from datasets.download_duke import ensure_dukemtmcreid
from datasets.build_occ_duke import ensure_occ_duke
REPO_ZIP_NAMES = ("TransReID-PEFT.zip", "transreid-peft.zip")
WEIGHT_DIR = REPO_DIR / ".cache" / "torch" / "checkpoints"
WEIGHT_FILE = WEIGHT_DIR / "jx_vit_base_p16_224-80ecf9dd.pth"
VIT_URL = (
    "https://github.com/rwightman/pytorch-image-models/releases/download/"
    "v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth"
)
MARKET_DIR = REPO_DIR / "data" / "market1501"
DUKE_DIR = REPO_DIR / "data" / "dukemtmcreid"
OCC_DUKE_DIR = REPO_DIR / "data" / "Occluded_Duke"
DEFAULT_CONFIG = "configs/Market/lora_blocks_4_11_r32.yml"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd or REPO_DIR)


def _find_zip(search_dirs: list[Path], names: tuple[str, ...]) -> Path | None:
    for directory in search_dirs:
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def unzip_file(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Unzipping {zip_path.name} -> {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def resolve_repo_dir() -> Path:
    """Return repo root — handles running from parent dir after zip extract."""
    if (REPO_DIR / "train.py").is_file():
        return REPO_DIR
    nested = REPO_DIR / "TransReID-PEFT"
    if (nested / "train.py").is_file():
        return nested
    return REPO_DIR


def _try_dataset(label: str, ensure_fn, root: Path, search_dirs: list[Path] | None = None) -> bool:
    try:
        ensure_fn(root=root, search_dirs=search_dirs)
        print(f"    {label}: OK")
        return True
    except Exception as exc:
        print(f"    WARNING: {label} setup failed: {exc}")
        return False


def unzip_archives(skip_unzip: bool) -> Path:
    global REPO_DIR, WEIGHT_DIR, WEIGHT_FILE, MARKET_DIR, DUKE_DIR, OCC_DUKE_DIR

    if skip_unzip:
        REPO_DIR = resolve_repo_dir()
        WEIGHT_DIR = REPO_DIR / ".cache" / "torch" / "checkpoints"
        WEIGHT_FILE = WEIGHT_DIR / "jx_vit_base_p16_224-80ecf9dd.pth"
        MARKET_DIR = REPO_DIR / "data" / "market1501"
        DUKE_DIR = REPO_DIR / "data" / "dukemtmcreid"
        OCC_DUKE_DIR = REPO_DIR / "data" / "Occluded_Duke"
        return REPO_DIR

    cwd = Path.cwd()
    search_dirs = [cwd, REPO_DIR, REPO_DIR.parent, cwd.parent]

    repo_zip = _find_zip(search_dirs, REPO_ZIP_NAMES)
    if repo_zip and not (REPO_DIR / "train.py").is_file():
        # Extract next to the zip so TransReID-PEFT/ appears
        unzip_file(repo_zip, repo_zip.parent)

    REPO_DIR = resolve_repo_dir()
    WEIGHT_DIR = REPO_DIR / ".cache" / "torch" / "checkpoints"
    WEIGHT_FILE = WEIGHT_DIR / "jx_vit_base_p16_224-80ecf9dd.pth"
    MARKET_DIR = REPO_DIR / "data" / "market1501"
    DUKE_DIR = REPO_DIR / "data" / "dukemtmcreid"
    OCC_DUKE_DIR = REPO_DIR / "data" / "Occluded_Duke"

    data_root = REPO_DIR / "data"
    search_dirs = [Path.cwd(), REPO_DIR, REPO_DIR.parent, Path.cwd().parent]
    _try_dataset("Market-1501", ensure_market1501, data_root, search_dirs)

    return REPO_DIR


def setup(skip_smoke: bool = False, download_all_datasets: bool = False) -> None:
    print(f"\n==> TransReID-PEFT setup")
    print(f"    Repo: {REPO_DIR}")
    print(f"    Python: {sys.version.split()[0]}")

    try:
        import torch
        print(f"    PyTorch: {torch.__version__}")
        has_cuda = torch.cuda.is_available()
    except ImportError:
        print("    PyTorch not found — installing PyTorch and torchvision...")
        run([sys.executable, "-m", "pip", "install", "torch", "torchvision"])
        import torch
        print(f"    PyTorch installed: {torch.__version__}")
        has_cuda = torch.cuda.is_available()

    req_file = "requirements-vast.txt" if has_cuda else "requirements.txt"
    if has_cuda:
        print(f"    CUDA OK: {torch.cuda.get_device_name(0)}")
    else:
        print("    WARNING: CUDA not available — using full requirements.txt")

    run([sys.executable, "-m", "pip", "install", "-q", "-r", req_file])

    WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
    if WEIGHT_FILE.exists():
        print(f"    ViT weights present: {WEIGHT_FILE.name}")
    else:
        print("    Downloading ViT-Base pretrained weights...")
        urllib.request.urlretrieve(VIT_URL, WEIGHT_FILE)
        print(f"    Saved to {WEIGHT_FILE}")

    data_root = REPO_DIR / "data"
    search_dirs = [Path.cwd(), REPO_DIR, REPO_DIR.parent, Path.cwd().parent]

    if (MARKET_DIR / "bounding_box_train").is_dir():
        print("    Market-1501: OK")
    else:
        print("    Downloading Market-1501 dataset...")
        _try_dataset("Market-1501", ensure_market1501, data_root, search_dirs)

    if download_all_datasets:
        if (DUKE_DIR / "bounding_box_train").is_dir():
            print("    DukeMTMC-reID: OK")
        else:
            print("    Downloading DukeMTMC-reID dataset...")
            _try_dataset("DukeMTMC-reID", ensure_dukemtmcreid, data_root, search_dirs)

        if (OCC_DUKE_DIR / "bounding_box_train").is_dir():
            print("    Occluded-Duke: OK")
        else:
            print("    Building Occluded-Duke dataset...")
            _try_dataset("Occluded-Duke", ensure_occ_duke, data_root, search_dirs)

    if not skip_smoke:
        print("    Running smoke test...")
        run([
            sys.executable,
            "tools/check_peft.py",
            "--config_file",
            DEFAULT_CONFIG,
            "--cpu-only",
        ])


def train(config: str, extra_args: list[str]) -> None:
    if not (MARKET_DIR / "bounding_box_train").is_dir():
        raise SystemExit(
            "Cannot train: Market-1501 not found at data/market1501/.\n"
            "Re-run setup: python run.py --setup-only"
        )

    cmd = [
        sys.executable,
        "train.py",
        "--config_file",
        config,
        "MODEL.DEVICE_ID",
        "('0')",
        *extra_args,
    ]
    print(f"\n==> Starting training")
    print(f"    Config: {config}")
    run(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Unzip, setup, and train TransReID-PEFT")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Training config YAML (default: LoRA 4-11 r32)",
    )
    parser.add_argument("--skip-unzip", action="store_true", help="Skip zip extraction")
    parser.add_argument("--skip-setup", action="store_true", help="Skip dependency install / weights")
    parser.add_argument("--setup-only", action="store_true", help="Setup only, do not train")
    parser.add_argument(
        "--download-all-datasets",
        action="store_true",
        help="Also download DukeMTMC-reID and build Occluded-Duke (large; not needed for paper Market runs)",
    )
    parser.add_argument("--skip-smoke", action="store_true", help="Skip smoke test during setup")
    parser.add_argument("extra", nargs="*", help="Extra args passed to train.py")
    args = parser.parse_args()

    print("==> TransReID-PEFT run.py")
    repo = unzip_archives(skip_unzip=args.skip_unzip)
    os.chdir(repo)

    if not args.skip_setup:
        setup(skip_smoke=args.skip_smoke, download_all_datasets=args.download_all_datasets)

    if args.setup_only:
        print("\nSetup complete (--setup-only).")
        return 0

    train(args.config, args.extra)
    print("\nTraining finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

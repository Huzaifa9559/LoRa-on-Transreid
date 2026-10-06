#!/usr/bin/env python3
"""
tools/run_classification_control.py

Automated script to run Classification Control experiments on TransReID-PEFT.
Compares the 5 non-dominated frontier configurations under a pure classification head
(softmax loss only, no BNneck, no triplet loss, no JPM/SIE).

Proves that PEFT accuracy gaps are objective-driven (metric-learning ranking vs classification).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import cfg
from config.loader import merge_config_file
from datasets.make_dataloader import make_dataloader
from loss.make_loss import make_loss
from model.make_model import make_model
from processor.processor import do_train
from solver.make_optimizer import make_optimizer
from solver.scheduler_factory import create_scheduler

CLASSIFICATION_CONFIGS = [
    ("Full FT Baseline", "configs/exploratory/classification/vit_transreid_full_ft.yml"),
    ("LoRA 0–11 (r=8, α=16)", "configs/exploratory/classification/lora_blocks_0_11_r8.yml"),
    ("LoRA 4–11 (r=32, α=64)", "configs/exploratory/classification/lora_blocks_4_11_r32.yml"),
    ("LoRA 6–11 (r=16, α=32)", "configs/exploratory/classification/lora_blocks_6_11_r16.yml"),
    ("SSF 0–11 (Case 2)", "configs/exploratory/classification/ssf_0_11_case2.yml"),
]


def setup_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def run_experiment(config_path: str, cpu_only: bool = False, max_epochs: int | None = None) -> dict:
    cfg.defrost()
    merge_config_file(cfg, config_path)
    if cpu_only:
        cfg.MODEL.DEVICE = "cpu"
        cfg.DATALOADER.NUM_WORKERS = 0
    if max_epochs is not None:
        cfg.SOLVER.MAX_EPOCHS = max_epochs
    cfg.freeze()

    logger = setup_logger("transreid.train")
    logger.info(f"Loaded config: {config_path}")

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    start_time = time.time()
    if torch.cuda.is_available() and not cpu_only:
        torch.cuda.reset_peak_memory_stats()

    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query,
        local_rank=0,
    )

    elapsed_time = time.time() - start_time
    peak_vram = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
        if torch.cuda.is_available() and not cpu_only
        else 0.0
    )

    return {
        "trainable_params": trainable_params,
        "total_params": total_params,
        "param_ratio": 100.0 * trainable_params / total_params,
        "peak_vram": peak_vram,
        "time_seconds": elapsed_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Classification Control Experiments")
    parser.add_argument("--cpu-only", action="store_true", help="Run on CPU for testing")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count for quick runs")
    args = parser.parse_args()

    print("==========================================================================")
    print("      CLASSIFICATION CONTROL EXPERIMENT (TransReID-PEFT)")
    print("==========================================================================")

    results = []
    for name, config_rel in CLASSIFICATION_CONFIGS:
        config_path = str(REPO_ROOT / config_rel)
        print(f"\n---> Running: {name} [{config_rel}]")
        res = run_experiment(config_path, cpu_only=args.cpu_only, max_epochs=args.epochs)
        res["name"] = name
        results.append(res)

    print("\n" + "=" * 75)
    print("                      CLASSIFICATION CONTROL RESULTS                      ")
    print("=" * 75)
    print(f"{'Configuration':<25} | {'Params (%)':<10} | {'VRAM (GB)':<10} | {'Time (s)':<10}")
    print("-" * 75)
    for r in results:
        print(f"{r['name']:<25} | {r['param_ratio']:>9.2f}% | {r['peak_vram']:>9.2f} | {r['time_seconds']:>9.1f}")
    print("=" * 75)


if __name__ == "__main__":
    main()

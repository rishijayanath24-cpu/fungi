#!/usr/bin/env python3
"""Train all four baseline models sequentially with a single command.

Usage:
    python scripts/run_baselines.py --data-root /path/to/defungi

Trains: ResNet-50, EfficientNet-B3, Swin-Tiny, ViT-B/16 (224px)
All use identical seed, augmentation flags, and evaluation pipeline.
Results are written to outputs/results/ for collection by generate_paper.py.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_baselines")

BASELINE_CONFIGS = [
    "configs/resnet50.yaml",
    "configs/efficientnet_b3.yaml",
    "configs/swin_tiny.yaml",
    "configs/vit_224.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all baseline models")
    parser.add_argument("--data-root", required=True, type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only, don't run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results: dict[str, bool] = {}

    for config_path in BASELINE_CONFIGS:
        cmd = [
            sys.executable,
            "scripts/train.py",
            "--config", config_path,
            "--data-root", args.data_root,
            "--device", args.device,
        ]
        logger.info("Running: %s", " ".join(cmd))

        if args.dry_run:
            results[config_path] = True
            continue

        result = subprocess.run(cmd, capture_output=False)
        success = result.returncode == 0
        results[config_path] = success

        if not success:
            logger.error("FAILED: %s (returncode=%d)", config_path, result.returncode)
        else:
            logger.info("OK: %s", config_path)

    logger.info("\n=== Baseline Training Summary ===")
    for config, ok in results.items():
        status = "✓" if ok else "✗"
        logger.info("  %s  %s", status, config)

    failed = [c for c, ok in results.items() if not ok]
    if failed:
        logger.error("%d baseline(s) failed: %s", len(failed), failed)
        sys.exit(1)
    else:
        logger.info("All baselines complete. Run: python scripts/generate_paper.py")


if __name__ == "__main__":
    main()

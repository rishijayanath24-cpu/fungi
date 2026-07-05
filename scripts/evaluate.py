#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the DeFungi test set.

Usage:
    python scripts/evaluate.py \\
        --config configs/vit_224.yaml \\
        --checkpoint outputs/checkpoints/vit_b16_224_best.pt \\
        --data-root /path/to/defungi

Runs full evaluation: accuracy, F1, AUC, ROC/PR curves, confusion matrix,
inference time, GPU memory. Writes all outputs to outputs/results/ and paper/.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from datasets.defungi import build_loaders
from datasets.transforms import build_transforms_for_config
from models.builder import build_model
from trainer.checkpoint import CheckpointManager
from trainer.reproducibility import set_seed
from evaluation.evaluator import Evaluator
from evaluation.inference_bench import benchmark_inference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a DeFungi checkpoint")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Resolve base config if needed (same logic as train.py)
    if "defaults" in cfg:
        from scripts.train import load_config
        cfg = load_config(args.config)

    if args.data_root:
        cfg["dataset"]["root"] = args.data_root
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.no_figures:
        cfg.setdefault("evaluation", {})["generate_figures"] = False

    set_seed(cfg.get("seed", 42))

    eval_tf = build_transforms_for_config(cfg, is_training=False)
    # Build all loaders but only use test
    _, _, test_loader = build_loaders(
        cfg,
        train_transform=eval_tf,   # train slot unused
        eval_transform=eval_tf,
    )

    model = build_model(cfg)
    CheckpointManager.load(args.checkpoint, model, device=args.device)

    experiment_name = args.checkpoint.stem
    evaluator = Evaluator(
        model, cfg,
        experiment_name=experiment_name,
        device=args.device,
    )
    metrics = evaluator.evaluate(test_loader, checkpoint_path=args.checkpoint)

    logger.info("\nFinal test metrics:")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            logger.info("  %-30s %.4f", k, v)


if __name__ == "__main__":
    main()

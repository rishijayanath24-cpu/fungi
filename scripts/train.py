#!/usr/bin/env python3
"""Main training entry point.

Usage:
    python scripts/train.py --config configs/vit_224.yaml --data-root /path/to/defungi
    python scripts/train.py --config configs/vit_224.yaml --data-root /path/to/defungi --resume outputs/checkpoints/vit_b16_224_best.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from datasets.defungi import build_loaders
from datasets.transforms import build_transforms_for_config
from models.builder import build_model
from trainer.reproducibility import set_seed
from trainer.trainer import Trainer
from evaluation.evaluator import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DeFungi classifier")
    parser.add_argument("--config", required=True, type=Path, help="YAML config path")
    parser.add_argument("--data-root", type=str, default=None, help="Path to DeFungi dataset root")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--no-eval", action="store_true", help="Skip final test evaluation")
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    """Load YAML config; merge with base.yaml if 'defaults' key is present."""
    base_dir = config_path.parent
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if "defaults" in cfg:
        merged: dict = {}
        for default_name in cfg.pop("defaults"):
            base_path = base_dir / f"{default_name}.yaml"
            if base_path.exists():
                with open(base_path) as f:
                    base_cfg = yaml.safe_load(f) or {}
                _deep_update(merged, base_cfg)
        _deep_update(merged, cfg)
        return merged

    return cfg


def _deep_update(base: dict, override: dict) -> None:
    """Recursively update base dict with override values."""
    for k, v in override.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    experiment_name = cfg.get("experiment", {}).get("name", args.config.stem)

    # CLI overrides
    if args.data_root:
        cfg["dataset"]["root"] = args.data_root
    elif cfg["dataset"].get("root") is None:
        import os
        cfg["dataset"]["root"] = os.environ.get("FUNGI_DATA")
        if not cfg["dataset"]["root"]:
            logger.error(
                "Dataset root not specified. Use --data-root or set env var FUNGI_DATA."
            )
            sys.exit(1)

    # Reproducibility
    set_seed(cfg.get("seed", 42), deterministic=cfg.get("deterministic", True))

    # Data
    train_tf = build_transforms_for_config(cfg, is_training=True)
    eval_tf = build_transforms_for_config(cfg, is_training=False)
    train_loader, val_loader, test_loader = build_loaders(cfg, train_tf, eval_tf)

    logger.info(
        "Dataset: train=%d val=%d test=%d | batch=%d",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
        cfg["training"]["batch_size"],
    )

    # Model
    model = build_model(cfg)
    logger.info("Model: %s | params=%.2fM", experiment_name,
                sum(p.numel() for p in model.parameters()) / 1e6)

    # Train
    trainer = Trainer(model, cfg, experiment_name=experiment_name, device=args.device)
    best_metrics = trainer.fit(train_loader, val_loader, resume_from=args.resume)

    logger.info(
        "Training complete | best val_acc=%.4f val_f1=%.4f",
        best_metrics.get("val_accuracy", 0),
        best_metrics.get("val_f1_macro", 0),
    )

    # Evaluate on test set
    if not args.no_eval:
        ckpt_dir = Path(cfg["checkpoint"]["save_dir"])
        best_ckpt = ckpt_dir / f"{experiment_name}_best.pt"

        if best_ckpt.exists():
            from trainer.checkpoint import CheckpointManager
            CheckpointManager.load(best_ckpt, model, device=args.device)

        evaluator = Evaluator(model, cfg, experiment_name=experiment_name, device=args.device)
        test_metrics = evaluator.evaluate(test_loader, checkpoint_path=best_ckpt)

        logger.info(
            "Test results | acc=%.4f f1=%.4f auc=%.4f",
            test_metrics.get("accuracy", 0),
            test_metrics.get("f1_macro", 0),
            test_metrics.get("auc_macro", float("nan")),
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the full ablation study sequentially.

Trains 6 configurations of ViT-B/16 with progressively more components:
  A1: baseline
  A2: + MixUp
  A3: + CutMix
  A4: + Multi-scale (S1 equal weight)
  A5: + ACMF (S3 confidence-guided)
  A6: + XAI visualization

Usage:
    python scripts/run_ablation.py \\
        --base-config configs/vit_224.yaml \\
        --data-root /path/to/defungi
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
from evaluation.ablation import ABLATION_CONFIGS, apply_overrides, collect_ablation_results
from evaluation.ablation import ablation_to_latex_table
from models.builder import build_model
from trainer.reproducibility import set_seed
from trainer.trainer import Trainer
from evaluation.evaluator import Evaluator
from visualization.comparison_plot import plot_ablation_bars

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_ablation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeFungi ablation study")
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--configs", nargs="+", default=None,
        help="Run specific ablation names only (e.g. A1_vit_baseline A2_plus_mixup)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.base_config) as f:
        base_cfg = yaml.safe_load(f)

    # Resolve base config defaults
    if "defaults" in base_cfg:
        from scripts.train import load_config
        base_cfg = load_config(args.base_config)

    base_cfg["dataset"]["root"] = args.data_root

    configs_to_run = ABLATION_CONFIGS
    if args.configs:
        configs_to_run = [c for c in ABLATION_CONFIGS if c["name"] in args.configs]
        logger.info("Running %d selected ablation configs", len(configs_to_run))

    results_dir = Path(base_cfg["logging"].get("output_dir", "outputs")) / "results"

    for abl in configs_to_run:
        logger.info("\n=== Ablation: %s (%s) ===", abl["name"], abl["label"])

        cfg = apply_overrides(base_cfg, abl["overrides"])
        cfg.setdefault("experiment", {})["name"] = abl["name"]

        set_seed(cfg.get("seed", 42), deterministic=cfg.get("deterministic", True))

        train_tf = build_transforms_for_config(cfg, is_training=True)
        eval_tf = build_transforms_for_config(cfg, is_training=False)
        train_loader, val_loader, test_loader = build_loaders(cfg, train_tf, eval_tf)

        model = build_model(cfg)
        trainer = Trainer(model, cfg, experiment_name=abl["name"], device=args.device)
        trainer.fit(train_loader, val_loader)

        # Load best checkpoint for evaluation
        ckpt_path = Path(cfg["checkpoint"]["save_dir"]) / f"{abl['name']}_best.pt"
        if ckpt_path.exists():
            from trainer.checkpoint import CheckpointManager
            CheckpointManager.load(ckpt_path, model, device=args.device)

        evaluator = Evaluator(model, cfg, experiment_name=abl["name"], device=args.device)
        evaluator.evaluate(test_loader)

        logger.info("Ablation %s complete.", abl["name"])

    # Collect all results and generate table + figure
    rows = collect_ablation_results(results_dir)
    if rows:
        latex_table = ablation_to_latex_table(rows)
        table_dir = Path(base_cfg.get("evaluation", {}).get("table_dir", "paper/tables"))
        table_dir.mkdir(parents=True, exist_ok=True)
        (table_dir / "ablation_table.tex").write_text(latex_table)
        logger.info("Ablation LaTeX table written to paper/tables/ablation_table.tex")

        fig_dir = Path(base_cfg.get("evaluation", {}).get("figure_dir", "paper/figures"))
        plot_ablation_bars(
            rows,
            save_path=fig_dir / "ablation_f1.png",
            dpi=300,
            metric="f1_macro",
            title="Ablation Study — F1 Macro (DeFungi Test Set)",
        )
        logger.info("Ablation figure written to paper/figures/ablation_f1.png")
    else:
        logger.warning("No ablation results found to collect.")


if __name__ == "__main__":
    main()

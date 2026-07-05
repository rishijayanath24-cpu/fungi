"""Ablation study runner.

Trains and evaluates 6 configurations in sequence:
  1. ViT baseline (no augmentation beyond standard transforms)
  2. + MixUp
  3. + CutMix (on top of MixUp)
  4. + Multi-scale fusion (equal weight, S1)
  5. + ACMF (confidence-guided, S3)
  6. + Explainability visualization (same model, adds XAI output)

Each configuration inherits from the base ViT-224 config and overrides
specific augmentation flags. Results are collected into a unified table.

Usage:
    python scripts/run_ablation.py --base-config configs/vit_224.yaml
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ABLATION_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "A1_vit_baseline",
        "label": "ViT-B/16 (baseline)",
        "overrides": {
            "augmentation.use_mixup": False,
            "augmentation.use_cutmix": False,
            "augmentation.use_random_erasing": False,
            "training.label_smoothing": 0.0,
        },
    },
    {
        "name": "A2_plus_mixup",
        "label": "+ MixUp",
        "overrides": {
            "augmentation.use_mixup": True,
            "augmentation.use_cutmix": False,
            "augmentation.use_random_erasing": False,
            "training.label_smoothing": 0.1,
        },
    },
    {
        "name": "A3_plus_cutmix",
        "label": "+ CutMix",
        "overrides": {
            "augmentation.use_mixup": True,
            "augmentation.use_cutmix": True,
            "augmentation.use_random_erasing": False,
            "training.label_smoothing": 0.1,
        },
    },
    {
        "name": "A4_plus_multiscale",
        "label": "+ Multi-Scale (S1 equal weight)",
        "overrides": {
            "augmentation.use_mixup": True,
            "augmentation.use_cutmix": True,
            "augmentation.use_random_erasing": True,
            "training.label_smoothing": 0.1,
            "_fusion_strategy": "equal_weight",
        },
    },
    {
        "name": "A5_plus_acmf",
        "label": "+ ACMF (S3 confidence-guided)",
        "overrides": {
            "augmentation.use_mixup": True,
            "augmentation.use_cutmix": True,
            "augmentation.use_random_erasing": True,
            "training.label_smoothing": 0.1,
            "_fusion_strategy": "confidence_guided",
        },
    },
    {
        "name": "A6_plus_xai",
        "label": "+ Explainability",
        "overrides": {
            "augmentation.use_mixup": True,
            "augmentation.use_cutmix": True,
            "augmentation.use_random_erasing": True,
            "training.label_smoothing": 0.1,
            "_fusion_strategy": "confidence_guided",
            "_generate_xai": True,
        },
    },
]


def apply_overrides(cfg: dict, overrides: dict[str, Any]) -> dict:
    """Apply dot-notation key overrides to a nested config dict."""
    cfg = copy.deepcopy(cfg)
    for key, value in overrides.items():
        if key.startswith("_"):
            cfg[key] = value
            continue
        parts = key.split(".")
        node = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return cfg


def collect_ablation_results(results_dir: str | Path) -> list[dict]:
    """Scan results_dir for completed ablation JSON files and return a sorted list."""
    results_dir = Path(results_dir)
    rows = []
    for abl_cfg in ABLATION_CONFIGS:
        path = results_dir / f"{abl_cfg['name']}_metrics.json"
        if not path.exists():
            logger.warning("Ablation result not found: %s (not yet trained)", path)
            continue
        with open(path) as f:
            metrics = json.load(f)
        rows.append({
            "name": abl_cfg["name"],
            "label": abl_cfg["label"],
            **metrics,
        })
    return rows


def ablation_to_latex_table(rows: list[dict]) -> str:
    """Generate a complete LaTeX table from ablation results."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Ablation Study on DeFungi Test Set}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Configuration & Accuracy (\%) & F1 Macro (\%) & Precision (\%) & AUC Macro (\%) \\",
        r"\hline",
    ]
    for row in rows:
        acc = row.get("accuracy", 0) * 100
        f1 = row.get("f1_macro", 0) * 100
        prec = row.get("precision_macro", 0) * 100
        auc = row.get("auc_macro", float("nan")) * 100
        auc_str = f"{auc:.2f}" if auc == auc else "--"  # NaN check
        lines.append(
            f"{row['label']} & {acc:.2f} & {f1:.2f} & {prec:.2f} & {auc_str} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)

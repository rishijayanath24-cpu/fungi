#!/usr/bin/env python3
"""ACMF Fusion Evaluation — runs all 4 strategies on the DeFungi test set.

Requires trained checkpoints for both resolutions:
    outputs/checkpoints/vit_b16_224_best.pt
    outputs/checkpoints/vit_b16_384_best.pt

Produces four result files in outputs/results/:
    fusion_equal_weight_metrics.json      (S1)
    fusion_accuracy_weighted_metrics.json (S2)
    fusion_confidence_guided_metrics.json (S3)
    fusion_learnable_metrics.json         (S4)

Usage:
    python scripts/run_fusion.py --data-root "D:\\" --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm

from datasets.defungi import build_loaders
from datasets.transforms import build_transforms_for_config
from evaluation.metrics import compute_classification_metrics, format_classification_report
from fusion.factory import build_fusion_strategy, STRATEGY_LABELS
from models.builder import build_model
from trainer.checkpoint import CheckpointManager
from trainer.reproducibility import set_seed
from visualization.confusion_matrix import plot_confusion_matrix
from visualization.roc_pr import plot_roc_curves, plot_pr_curves

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_fusion")

ROOT = Path(__file__).parent.parent
CKPT_DIR  = ROOT / "outputs" / "checkpoints"
RESULTS_DIR = ROOT / "outputs" / "results"
FIGURE_DIR  = ROOT / "paper" / "figures"
TABLE_DIR   = ROOT / "paper" / "tables"

NUM_CLASSES  = 5
CLASS_NAMES  = ["H1", "H2", "H3", "H5", "H6"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(name: str) -> dict:
    """Load a model config, merging base.yaml."""
    from scripts.train import load_config as _load
    return _load(ROOT / "configs" / f"{name}.yaml")


def build_and_load(config_name: str, ckpt_name: str, device: str) -> nn.Module:
    cfg = load_config(config_name)
    model = build_model(cfg)
    ckpt_path = CKPT_DIR / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Make sure {config_name} training has completed."
        )
    CheckpointManager.load(ckpt_path, model, device=device)
    model.eval()
    return model.to(device)


@torch.no_grad()
def collect_probs(model: nn.Module, loader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, softmax_probs) arrays over the full loader."""
    all_labels, all_probs = [], []
    for images, labels in tqdm(loader, desc="  inference", leave=False):
        images = images.to(device, non_blocking=True)
        with autocast("cuda", enabled=(device == "cuda")):
            logits = model(images)
        probs = F.softmax(logits, dim=-1).cpu().float().numpy()
        all_probs.append(probs)
        all_labels.extend(labels.tolist())
    return np.array(all_labels), np.vstack(all_probs)


def read_val_accuracy(experiment: str) -> float:
    """Read val_accuracy from a saved results JSON. Returns 0.5 on failure."""
    for suffix in ("_metrics.json", "_results.json"):
        p = RESULTS_DIR / f"{experiment}{suffix}"
        if p.exists():
            try:
                data = json.loads(p.read_text())
                acc = data.get("val_accuracy") or data.get("accuracy")
                if acc is not None:
                    return float(acc)
            except Exception:
                pass
    log.warning("Could not read val_accuracy for %s; using 0.5", experiment)
    return 0.5


def save_results(experiment: str, metrics: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{experiment}_metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Saved metrics: %s", path)


def generate_figures(
    experiment: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    dpi: int = 300,
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    label = STRATEGY_LABELS.get(experiment.replace("fusion_", ""), experiment)

    plot_confusion_matrix(
        y_true, y_pred,
        class_names=CLASS_NAMES,
        save_path=FIGURE_DIR / f"{experiment}_confusion.png",
        dpi=dpi,
        title=f"Confusion Matrix — {label}",
    )
    plot_roc_curves(
        y_true, y_prob,
        class_names=CLASS_NAMES,
        save_path=FIGURE_DIR / f"{experiment}_roc.png",
        dpi=dpi,
        title=f"ROC Curves — {label}",
    )
    plot_pr_curves(
        y_true, y_prob,
        class_names=CLASS_NAMES,
        save_path=FIGURE_DIR / f"{experiment}_pr.png",
        dpi=dpi,
        title=f"PR Curves — {label}",
    )


def evaluate_fusion(
    experiment: str,
    y_true: np.ndarray,
    fused_probs: np.ndarray,
) -> dict:
    """Compute metrics for fused probabilities and persist outputs."""
    y_pred = fused_probs.argmax(axis=-1)
    metrics = compute_classification_metrics(
        y_true, y_pred,
        num_classes=NUM_CLASSES,
        y_prob=fused_probs,
    )
    metrics["experiment"] = experiment
    log.info(
        "  acc=%.4f  f1_macro=%.4f  auc_macro=%.4f",
        metrics.get("accuracy", 0),
        metrics.get("f1_macro", 0),
        metrics.get("auc_macro", float("nan")),
    )
    log.info("\n%s", format_classification_report(y_true, y_pred))
    save_results(experiment, metrics)
    generate_figures(experiment, y_true, y_pred, fused_probs)
    return metrics


# ── S4 learnable fusion training ──────────────────────────────────────────────

def train_learnable_head(
    fusion: nn.Module,
    train_probs_224: np.ndarray,
    train_probs_384: np.ndarray,
    train_labels: np.ndarray,
    device: str,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> nn.Module:
    """Train the MLP weight head on training-set probabilities."""
    fusion = fusion.to(device)
    fusion.train()

    p224 = torch.tensor(train_probs_224, dtype=torch.float32)
    p384 = torch.tensor(train_probs_384, dtype=torch.float32)
    labels = torch.tensor(train_labels, dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(p224, p384, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(fusion.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    log.info("  Training S4 learnable head for %d epochs ...", epochs)
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        for pb224, pb384, lbl in loader:
            pb224, pb384, lbl = pb224.to(device), pb384.to(device), lbl.to(device)
            fused = fusion(pb224, pb384)
            loss = F.cross_entropy(torch.log(fused + 1e-8), lbl)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (fused.argmax(-1) == lbl).sum().item()
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            acc = correct / len(dataset)
            log.info("  S4 epoch %02d/%d  loss=%.4f  train_acc=%.4f",
                     epoch + 1, epochs, total_loss / len(loader), acc)

    fusion.eval()
    return fusion


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--strategies", nargs="+",
        default=["equal_weight", "accuracy_weighted", "confidence_guided", "learnable"],
        help="Subset of strategies to run",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)

    device = args.device
    log.info("Loading ViT-B/16 224px model ...")
    cfg_224 = load_config("vit_224")
    cfg_224["dataset"]["root"] = args.data_root
    model_224 = build_and_load("vit_224", "vit_b16_224_best.pt", device)

    log.info("Loading ViT-B/16 384px model ...")
    cfg_384 = load_config("vit_384")
    cfg_384["dataset"]["root"] = args.data_root
    model_384 = build_and_load("vit_384", "vit_b16_384_best.pt", device)

    # Build loaders: test set uses model-specific transforms (different image sizes)
    log.info("Building data loaders ...")
    train_tf_224 = build_transforms_for_config(cfg_224, is_training=True)
    eval_tf_224  = build_transforms_for_config(cfg_224, is_training=False)
    eval_tf_384  = build_transforms_for_config(cfg_384, is_training=False)

    train_loader_224, _, test_loader_224 = build_loaders(cfg_224, train_tf_224, eval_tf_224)
    train_loader_384, _, test_loader_384 = build_loaders(cfg_384, eval_tf_384, eval_tf_384)

    # Collect test-set probabilities from each model
    log.info("Running test-set inference with ViT-224 ...")
    y_true_224, test_probs_224 = collect_probs(model_224, test_loader_224, device)

    log.info("Running test-set inference with ViT-384 ...")
    y_true_384, test_probs_384 = collect_probs(model_384, test_loader_384, device)

    assert np.array_equal(y_true_224, y_true_384), \
        "Label mismatch between 224 and 384 loaders — check shuffle/ordering"
    y_true = y_true_224

    log.info("Test set: %d samples, %d classes", len(y_true), NUM_CLASSES)

    # ── S1: Equal Weight ─────────────────────────────────────────────────────
    if "equal_weight" in args.strategies:
        log.info("\n=== S1: Equal Weight Fusion ===")
        fused = 0.5 * test_probs_224 + 0.5 * test_probs_384
        evaluate_fusion("fusion_equal_weight", y_true, fused)

    # ── S2: Accuracy Weighted ─────────────────────────────────────────────────
    if "accuracy_weighted" in args.strategies:
        log.info("\n=== S2: Accuracy-Weighted Fusion ===")
        acc_224 = read_val_accuracy("vit_b16_224")
        acc_384 = read_val_accuracy("vit_b16_384")
        log.info("  val_acc_224=%.4f  val_acc_384=%.4f", acc_224, acc_384)
        strategy = build_fusion_strategy("accuracy_weighted",
                                         num_classes=NUM_CLASSES,
                                         acc_224=acc_224, acc_384=acc_384)
        p224_t = torch.tensor(test_probs_224)
        p384_t = torch.tensor(test_probs_384)
        fused = strategy(p224_t, p384_t).numpy()
        evaluate_fusion("fusion_accuracy_weighted", y_true, fused)

    # ── S3: Confidence Guided ─────────────────────────────────────────────────
    if "confidence_guided" in args.strategies:
        log.info("\n=== S3: Confidence-Guided Fusion (ACMF) ===")
        strategy = build_fusion_strategy("confidence_guided", num_classes=NUM_CLASSES)
        p224_t = torch.tensor(test_probs_224)
        p384_t = torch.tensor(test_probs_384)
        fused = strategy(p224_t, p384_t).numpy()
        evaluate_fusion("fusion_confidence_guided", y_true, fused)

    # ── S4: Learnable Fusion ──────────────────────────────────────────────────
    if "learnable" in args.strategies:
        log.info("\n=== S4: Learnable Fusion ===")

        # Collect training-set probs for MLP head training
        log.info("  Collecting train-set probs for head training ...")
        train_labels_224, train_probs_224_np = collect_probs(model_224, train_loader_224, device)
        _, train_probs_384_np = collect_probs(model_384, train_loader_384, device)

        strategy = build_fusion_strategy("learnable", num_classes=NUM_CLASSES)
        strategy = train_learnable_head(
            strategy,
            train_probs_224_np, train_probs_384_np, train_labels_224,
            device=device, epochs=20, lr=1e-3,
        )

        with torch.no_grad():
            p224_t = torch.tensor(test_probs_224).to(device)
            p384_t = torch.tensor(test_probs_384).to(device)
            fused = strategy(p224_t, p384_t).cpu().numpy()
        evaluate_fusion("fusion_learnable", y_true, fused)

    log.info("\nAll fusion strategies complete. Results in: %s", RESULTS_DIR)


if __name__ == "__main__":
    main()

"""Classification metrics for the DeFungi evaluation suite.

All metrics are computed from numpy arrays of integer labels and predictions.
No projections or estimates — every value comes from a completed forward pass.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from datasets.defungi import CLASS_NAMES, SPECIES_NAMES


def compute_classification_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    num_classes: int = 5,
    y_prob: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Compute the full DeFungi evaluation metric suite.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        num_classes: Number of classes.
        y_prob: Softmax probabilities [N, C] for AUC computation (optional).

    Returns:
        Flat dict of metric_name → float value.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: dict[str, float] = {}

    # ── Overall metrics ───────────────────────────────────────────────────────
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # ── Per-class F1 ─────────────────────────────────────────────────────────
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, cls in enumerate(CLASS_NAMES[:num_classes]):
        if i < len(per_class_f1):
            metrics[f"f1_{cls}"] = float(per_class_f1[i])

    # ── AUC (macro OVR, requires probability scores) ─────────────────────────
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            metrics["auc_macro"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
            per_class_auc = _per_class_auc(y_true, y_prob, num_classes)
            for i, cls in enumerate(CLASS_NAMES[:num_classes]):
                if i < len(per_class_auc):
                    metrics[f"auc_{cls}"] = float(per_class_auc[i])
        except (ValueError, TypeError):
            pass  # Not enough classes represented in batch

    return metrics


def _per_class_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
) -> list[float]:
    """One-vs-Rest AUC for each class."""
    aucs = []
    for c in range(num_classes):
        binary_true = (y_true == c).astype(int)
        try:
            auc = float(roc_auc_score(binary_true, y_prob[:, c]))
        except ValueError:
            auc = float("nan")
        aucs.append(auc)
    return aucs


def format_classification_report(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    num_classes: int = 5,
) -> str:
    """Return a formatted sklearn classification report with species names."""
    target_names = [
        f"{cls} ({SPECIES_NAMES.get(cls, cls)})"
        for cls in CLASS_NAMES[:num_classes]
    ]
    return classification_report(
        y_true, y_pred,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )


def metrics_to_latex_row(metrics: dict, experiment_name: str) -> str:
    """Format metrics as a single LaTeX table row for the paper."""
    acc = metrics.get("accuracy", 0) * 100
    f1 = metrics.get("f1_macro", 0) * 100
    prec = metrics.get("precision_macro", 0) * 100
    rec = metrics.get("recall_macro", 0) * 100
    auc = metrics.get("auc_macro", float("nan")) * 100

    auc_str = f"{auc:.2f}" if not (auc != auc) else "--"  # nan check
    return (
        f"{experiment_name} & {acc:.2f} & {prec:.2f} & {rec:.2f} & {f1:.2f} & {auc_str} \\\\"
    )

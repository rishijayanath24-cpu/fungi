"""ROC and Precision-Recall curve visualization for multi-class classification.

Generates one-vs-rest curves for each fungal class, plus the macro-average.
All figures saved at publication DPI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from datasets.defungi import CLASS_NAMES, SPECIES_NAMES

_PALETTE = [
    "#2563EB",  # blue
    "#DC2626",  # red
    "#059669",  # green
    "#D97706",  # amber
    "#7C3AED",  # purple
]


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    save_path: str | Path,
    dpi: int = 300,
    title: Optional[str] = None,
) -> None:
    """One-vs-Rest ROC curves for all classes + macro average.

    Args:
        y_true: Integer class labels [N].
        y_prob: Softmax probability matrix [N, C].
        class_names: Class label strings.
        save_path: Output PNG path.
        dpi: Publication DPI.
    """
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    all_fpr = np.linspace(0, 1, 200)
    mean_tpr = np.zeros(200)

    for i, (cls, color) in enumerate(zip(class_names, _PALETTE)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
        species = SPECIES_NAMES.get(cls, cls)
        ax.plot(fpr, tpr, color=color, linewidth=1.4,
                label=f"{cls} — {species} (AUC={auc:.3f})")
        mean_tpr += np.interp(all_fpr, fpr, tpr)

    mean_tpr /= n_classes
    macro_auc = np.trapezoid(mean_tpr, all_fpr) if hasattr(np, "trapezoid") else np.trapz(mean_tpr, all_fpr)
    ax.plot(all_fpr, mean_tpr, color="black", linestyle="--", linewidth=1.8,
            label=f"Macro avg (AUC={macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k:", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate", fontsize=9)
    ax.set_title(title or "ROC Curves (One-vs-Rest)", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.02])

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    save_path: str | Path,
    dpi: int = 300,
    title: Optional[str] = None,
) -> None:
    """Precision-Recall curves for all classes + macro average.

    Args:
        y_true: Integer class labels [N].
        y_prob: Softmax probability matrix [N, C].
        class_names: Class label strings.
        save_path: Output PNG path.
        dpi: Publication DPI.
    """
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    for i, (cls, color) in enumerate(zip(class_names, _PALETTE)):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_bin[:, i], y_prob[:, i])
        species = SPECIES_NAMES.get(cls, cls)
        ax.plot(recall, precision, color=color, linewidth=1.4,
                label=f"{cls} — {species} (AP={ap:.3f})")

    macro_ap = average_precision_score(y_bin, y_prob, average="macro")
    ax.axhline(y=macro_ap, color="black", linestyle="--", linewidth=1.0,
               label=f"Macro AP = {macro_ap:.3f}")

    ax.set_xlabel("Recall", fontsize=9)
    ax.set_ylabel("Precision", fontsize=9)
    ax.set_title(title or "Precision-Recall Curves (One-vs-Rest)", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.02])

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

"""Publication-quality confusion matrix visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from datasets.defungi import SPECIES_NAMES


def plot_confusion_matrix(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    class_names: list[str],
    save_path: str | Path,
    dpi: int = 300,
    normalize: bool = True,
    cmap: str = "Blues",
    title: Optional[str] = None,
) -> None:
    """Plot and save a normalized confusion matrix.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        class_names: List of class label strings (e.g. ['H1', 'H2', ...]).
        save_path: Output PNG path.
        dpi: Publication DPI.
        normalize: If True, normalize rows to percentages.
        cmap: Matplotlib colormap.
        title: Optional figure title override.
    """
    from typing import Optional
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = cm.astype(float) / (row_sums + 1e-8)
    else:
        cm_display = cm

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(4.5, n * 1.2), max(4.0, n * 1.1)))

    im = ax.imshow(cm_display, interpolation="nearest", cmap=cmap, vmin=0, vmax=1 if normalize else None)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Tick labels with species names
    tick_labels = [
        f"{cls}\n({SPECIES_NAMES.get(cls, cls)})" for cls in class_names
    ]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(tick_labels, fontsize=7, rotation=30, ha="right")
    ax.set_yticklabels(tick_labels, fontsize=7)

    # Annotate cells
    thresh = cm_display.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = cm_display[i, j]
            text = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, text,
                    ha="center", va="center", fontsize=7,
                    color="white" if val > thresh else "black")

    ax.set_xlabel("Predicted Label", fontsize=9)
    ax.set_ylabel("True Label", fontsize=9)
    ax.set_title(title or "Confusion Matrix", fontsize=10)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

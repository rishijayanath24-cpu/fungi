"""Training curve visualization.

Produces publication-quality loss and accuracy curves from TensorBoard logs
or from a list of per-epoch metric dicts returned by the Trainer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# Publication style constants
_FONT_SIZE = 9
_TITLE_SIZE = 10
_LINE_WIDTH = 1.5
_PALETTE = {
    "train": "#2563EB",   # blue
    "val":   "#DC2626",   # red
}


def plot_training_curves(
    history: list[dict],
    experiment_name: str,
    save_path: str | Path,
    dpi: int = 300,
    metrics: tuple[str, str] = ("loss", "accuracy"),
) -> None:
    """Plot train vs. val curves for the given metrics.

    Args:
        history: List of per-epoch metric dicts (from Trainer.fit()).
        experiment_name: Used in the figure title and filename.
        save_path: Output PNG path.
        dpi: Publication DPI.
        metrics: Tuple of metric names to plot (without train_/val_ prefix).
    """
    epochs = [m.get("epoch", i) for i, m in enumerate(history)]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 3.2))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        train_key = f"train_{metric}"
        val_key = f"val_{metric}"

        train_vals = [m.get(train_key, float("nan")) for m in history]
        val_vals = [m.get(val_key, float("nan")) for m in history]

        ax.plot(
            epochs, train_vals,
            label="Train",
            color=_PALETTE["train"],
            linewidth=_LINE_WIDTH,
            linestyle="-",
        )
        ax.plot(
            epochs, val_vals,
            label="Val",
            color=_PALETTE["val"],
            linewidth=_LINE_WIDTH,
            linestyle="--",
        )

        # Mark best val epoch
        valid = [(i, v) for i, v in enumerate(val_vals) if v == v]  # filter nan
        if valid:
            best_i, best_v = max(valid, key=lambda x: x[1]) if "acc" in metric or "f1" in metric \
                else min(valid, key=lambda x: x[1])
            ax.axvline(x=epochs[best_i], color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
            ax.annotate(
                f"{best_v:.4f}",
                xy=(epochs[best_i], best_v),
                fontsize=6.5,
                ha="left",
                va="bottom",
                color="gray",
            )

        ax.set_xlabel("Epoch", fontsize=_FONT_SIZE)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=_FONT_SIZE)
        ax.set_title(f"{metric.replace('_', ' ').title()} — {experiment_name}", fontsize=_TITLE_SIZE)
        ax.legend(fontsize=_FONT_SIZE - 1)
        ax.tick_params(labelsize=_FONT_SIZE - 1)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        ax.grid(True, alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

"""Fusion strategy comparison bar charts.

Generates the main ACMF comparison figure for the paper:
a grouped bar chart showing Accuracy and F1 Macro for all four
fusion strategies (S1–S4) side by side.

Also generates ablation study bar charts from collected result dicts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from fusion.factory import STRATEGY_LABELS

_STRATEGY_COLORS = {
    "equal_weight":              "#E85D04",  # orange
    "accuracy_weighted":         "#2563EB",  # blue
    "confidence_guided":         "#059669",  # green
    "learnable":                 "#DC2626",  # red
    "fusion_equal_weight":       "#E85D04",  # orange
    "fusion_accuracy_weighted":  "#2563EB",  # blue
    "fusion_confidence_guided":  "#059669",  # green
    "fusion_learnable":          "#DC2626",  # red
}


def plot_fusion_comparison(
    results: dict[str, dict],
    save_path: str | Path,
    dpi: int = 300,
    metrics: tuple[str, ...] = ("accuracy", "f1_macro"),
    title: Optional[str] = None,
) -> None:
    """Grouped bar chart comparing all fusion strategies.

    Args:
        results: Dict mapping strategy_name → metrics_dict.
                 e.g. {"equal_weight": {"accuracy": 0.87, "f1_macro": 0.85}, ...}
        save_path: Output PNG path.
        dpi: Publication DPI.
        metrics: Which metrics to plot as bar groups.
        title: Optional figure title.
    """
    strategies = list(results.keys())
    n_strategies = len(strategies)
    n_metrics = len(metrics)

    x = np.arange(n_metrics)
    bar_width = 0.8 / n_strategies

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    for i, strategy in enumerate(strategies):
        values = [results[strategy].get(m, 0) * 100 for m in metrics]
        offset = (i - n_strategies / 2 + 0.5) * bar_width
        color = _STRATEGY_COLORS.get(strategy, "#94A3B8")
        label = STRATEGY_LABELS.get(strategy, strategy)
        bars = ax.bar(
            x + offset, values,
            width=bar_width * 0.92,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        # Value annotations on bars
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=6.5,
            )

    ax.set_xlabel("Metric", fontsize=9)
    ax.set_ylabel("Score (%)", fontsize=9)
    ax.set_title(title or "ACMF Fusion Strategy Comparison — DeFungi Test Set", fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics], fontsize=8)
    ax.legend(fontsize=6.5, loc="lower right")
    ax.tick_params(labelsize=8)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_bars(
    rows: list[dict],
    save_path: str | Path,
    dpi: int = 300,
    metric: str = "f1_macro",
    title: Optional[str] = None,
) -> None:
    """Horizontal bar chart for ablation study results.

    Args:
        rows: List of dicts with 'label' and metric values.
        save_path: Output PNG path.
        dpi: Publication DPI.
        metric: Metric to plot.
        title: Figure title.
    """
    labels = [r["label"] for r in rows]
    values = [r.get(metric, 0) * 100 for r in rows]
    colors = ["#64748B"] + ["#2563EB"] * (len(rows) - 1)
    colors[-1] = "#059669"  # Highlight the best configuration

    fig, ax = plt.subplots(figsize=(7.0, max(3.0, len(rows) * 0.55)))

    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}%", va="center", ha="left", fontsize=7.5,
        )

    ax.set_xlabel(f"{metric.replace('_', ' ').title()} (%)", fontsize=9)
    ax.set_title(title or "Ablation Study — DeFungi Test Set", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.set_xlim(0, max(values) + 5)
    ax.grid(True, axis="x", alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

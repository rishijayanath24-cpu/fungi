#!/usr/bin/env python3
"""Collect all experiment results and generate IEEE paper materials.

Scans outputs/results/ for completed experiment JSON files, then:
  - Builds main results table (LaTeX + CSV)
  - Builds baseline comparison table (LaTeX + CSV)
  - Builds ACMF fusion comparison table (LaTeX + CSV)
  - Generates fusion strategy comparison bar chart
  - Generates ablation study bar chart (if ablation results present)
  - Writes a Markdown summary of all results

Usage:
    python scripts/generate_paper.py
    python scripts/generate_paper.py --results-dir outputs/results --figure-dpi 600
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.ablation import collect_ablation_results, ablation_to_latex_table
from evaluation.metrics import metrics_to_latex_row
from visualization.comparison_plot import plot_fusion_comparison, plot_ablation_bars

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_paper")

BASELINE_NAMES = ["resnet50", "efficientnet_b3", "swin_tiny", "vit_b16_224"]
FUSION_NAMES = [
    "fusion_equal_weight",
    "fusion_accuracy_weighted",
    "fusion_confidence_guided",
    "fusion_learnable",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate IEEE paper materials")
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/results"))
    parser.add_argument("--figure-dir", type=Path, default=Path("paper/figures"))
    parser.add_argument("--table-dir", type=Path, default=Path("paper/tables"))
    parser.add_argument("--summary-dir", type=Path, default=Path("paper/summaries"))
    parser.add_argument("--figure-dpi", type=int, default=300)
    return parser.parse_args()


def load_results(results_dir: Path) -> dict[str, dict]:
    """Load all JSON metric files from the results directory."""
    all_results: dict[str, dict] = {}
    for json_path in sorted(results_dir.glob("*_metrics.json")):
        exp_name = json_path.stem.replace("_metrics", "")
        with open(json_path) as f:
            all_results[exp_name] = json.load(f)
    return all_results


def build_latex_table(
    results: dict[str, dict],
    exp_names: list[str],
    caption: str,
    label: str,
) -> str:
    """Build a complete LaTeX table for selected experiments."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"Model & Accuracy (\%) & Precision (\%) & Recall (\%) & F1 Macro (\%) & AUC Macro (\%) \\",
        r"\hline",
    ]
    for name in exp_names:
        if name in results:
            row = metrics_to_latex_row(results[name], name.replace("_", " ").title())
            lines.append(row)
        else:
            lines.append(f"{name.replace('_', ' ').title()} & -- & -- & -- & -- & -- \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def write_csv_table(
    results: dict[str, dict],
    exp_names: list[str],
    save_path: Path,
) -> None:
    fieldnames = ["experiment", "accuracy", "f1_macro", "precision_macro", "recall_macro", "auc_macro",
                  "latency_mean_ms", "params_millions"]
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for name in exp_names:
            if name in results:
                row = {"experiment": name, **results[name]}
                writer.writerow(row)


def write_markdown_summary(results: dict[str, dict], save_path: Path) -> None:
    lines = ["# DeFungi V2 — Experiment Results Summary\n"]
    lines.append("| Experiment | Accuracy | F1 Macro | AUC Macro | Latency (ms) | Params (M) |")
    lines.append("|:-----------|:--------:|:--------:|:---------:|:------------:|:----------:|")
    for name, m in sorted(results.items()):
        acc = m.get("accuracy", 0) * 100
        f1 = m.get("f1_macro", 0) * 100
        auc = m.get("auc_macro", float("nan")) * 100
        lat = m.get("latency_mean_ms", 0)
        params = m.get("params_millions", 0)
        auc_s = f"{auc:.2f}" if auc == auc else "--"
        lines.append(
            f"| {name} | {acc:.2f}% | {f1:.2f}% | {auc_s}% | {lat:.1f} | {params:.1f} |"
        )
    save_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()

    for d in [args.figure_dir, args.table_dir, args.summary_dir]:
        d.mkdir(parents=True, exist_ok=True)

    all_results = load_results(args.results_dir)
    if not all_results:
        logger.error("No results found in %s. Run training first.", args.results_dir)
        sys.exit(1)

    logger.info("Loaded %d experiment results: %s", len(all_results), sorted(all_results.keys()))

    # ── Baseline comparison table ─────────────────────────────────────────────
    baseline_tex = build_latex_table(
        all_results, BASELINE_NAMES,
        caption="Baseline Model Comparison on DeFungi Test Set",
        label="tab:baselines",
    )
    (args.table_dir / "baseline_comparison.tex").write_text(baseline_tex)
    write_csv_table(all_results, BASELINE_NAMES, args.table_dir / "baseline_comparison.csv")
    logger.info("Baseline table written.")

    # ── Fusion comparison table ───────────────────────────────────────────────
    fusion_tex = build_latex_table(
        all_results, FUSION_NAMES,
        caption="ACMF Fusion Strategy Comparison on DeFungi Test Set",
        label="tab:fusion",
    )
    (args.table_dir / "fusion_comparison.tex").write_text(fusion_tex)
    write_csv_table(all_results, FUSION_NAMES, args.table_dir / "fusion_comparison.csv")
    logger.info("Fusion comparison table written.")

    # ── Fusion bar chart ──────────────────────────────────────────────────────
    fusion_results = {
        name: all_results[name]
        for name in FUSION_NAMES
        if name in all_results
    }
    if fusion_results:
        plot_fusion_comparison(
            fusion_results,
            save_path=args.figure_dir / "fusion_comparison.png",
            dpi=args.figure_dpi,
        )
        logger.info("Fusion comparison figure written.")

    # ── Ablation table + figure ───────────────────────────────────────────────
    ablation_rows = collect_ablation_results(args.results_dir)
    if ablation_rows:
        abl_tex = ablation_to_latex_table(ablation_rows)
        (args.table_dir / "ablation_table.tex").write_text(abl_tex)
        plot_ablation_bars(
            ablation_rows,
            save_path=args.figure_dir / "ablation_f1.png",
            dpi=args.figure_dpi,
        )
        logger.info("Ablation table and figure written.")

    # ── Markdown summary ──────────────────────────────────────────────────────
    write_markdown_summary(all_results, args.summary_dir / "results_summary.md")
    logger.info("Markdown summary written.")

    logger.info("\nAll paper materials generated in paper/")


if __name__ == "__main__":
    main()

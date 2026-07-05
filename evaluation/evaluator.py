"""Full evaluation pipeline for a trained model on the DeFungi test set.

Runs all metrics, generates publication figures, writes CSV and LaTeX outputs.
Called automatically at the end of training and from scripts/evaluate.py.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.defungi import CLASS_NAMES
from evaluation.metrics import (
    compute_classification_metrics,
    format_classification_report,
    metrics_to_latex_row,
)
from evaluation.inference_bench import benchmark_inference

logger = logging.getLogger(__name__)


class Evaluator:
    """Runs the complete evaluation suite for a single model."""

    def __init__(
        self,
        model: nn.Module,
        cfg: dict,
        experiment_name: str,
        device: str = "cuda",
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.experiment_name = experiment_name
        self.device = torch.device(device)

        eval_cfg = cfg.get("evaluation", {})
        self.generate_figures = eval_cfg.get("generate_figures", True)
        self.figure_dpi = eval_cfg.get("figure_dpi", 300)
        self.figure_dir = Path(eval_cfg.get("figure_dir", "paper/figures"))
        self.table_dir = Path(eval_cfg.get("table_dir", "paper/tables"))
        self.results_dir = Path(cfg["logging"].get("output_dir", "outputs")) / "results"

        for d in [self.figure_dir, self.table_dir, self.results_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        test_loader: DataLoader,
        checkpoint_path: Optional[str] = None,
        run_benchmark: bool = True,
    ) -> dict:
        """Run full evaluation and persist all outputs.

        Returns:
            Dict of all computed metrics.
        """
        y_true, y_pred, y_prob = self._collect_predictions(test_loader)

        metrics = compute_classification_metrics(
            y_true, y_pred,
            num_classes=self.cfg["dataset"].get("num_classes", 5),
            y_prob=y_prob,
        )

        if run_benchmark:
            image_size = self.cfg["augmentation"].get("image_size", 224)
            bench = benchmark_inference(
                self.model,
                image_size=image_size,
                batch_size=1,
                device=str(self.device),
                checkpoint_path=checkpoint_path,
            )
            metrics.update(bench)

        # Print formatted report
        report = format_classification_report(y_true, y_pred)
        logger.info("\n%s", report)

        # Persist results
        self._save_json(metrics)
        self._save_csv(metrics)
        self._save_latex(metrics)

        if self.generate_figures:
            self._generate_figures(y_true, y_pred, y_prob)

        return metrics

    # ── Prediction collection ─────────────────────────────────────────────────

    def _collect_predictions(
        self,
        loader: DataLoader,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the model on all batches and collect labels, preds, probabilities."""
        self.model.eval()
        all_labels: list[int] = []
        all_preds: list[int] = []
        all_probs: list[np.ndarray] = []

        use_amp = self.cfg.get("training", {}).get("amp", True)

        with torch.no_grad():
            for images, labels in tqdm(loader, desc=f"Evaluating {self.experiment_name}"):
                images = images.to(self.device, non_blocking=True)

                with autocast(enabled=use_amp):
                    logits = self.model(images)

                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                preds = probs.argmax(axis=-1)

                all_probs.append(probs)
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())

        return (
            np.array(all_labels),
            np.array(all_preds),
            np.vstack(all_probs),
        )

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save_json(self, metrics: dict) -> None:
        path = self.results_dir / f"{self.experiment_name}_metrics.json"
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("JSON metrics: %s", path)

    def _save_csv(self, metrics: dict) -> None:
        path = self.results_dir / f"{self.experiment_name}_metrics.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in metrics.items():
                writer.writerow([k, v])
        logger.info("CSV metrics: %s", path)

    def _save_latex(self, metrics: dict) -> None:
        row = metrics_to_latex_row(metrics, self.experiment_name)
        path = self.table_dir / f"{self.experiment_name}_row.tex"
        path.write_text(row + "\n")
        logger.info("LaTeX row: %s", path)

    def _generate_figures(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
    ) -> None:
        from visualization.confusion_matrix import plot_confusion_matrix
        from visualization.roc_pr import plot_roc_curves, plot_pr_curves

        num_classes = self.cfg["dataset"].get("num_classes", 5)
        class_names = CLASS_NAMES[:num_classes]

        plot_confusion_matrix(
            y_true, y_pred,
            class_names=class_names,
            save_path=self.figure_dir / f"{self.experiment_name}_confusion.png",
            dpi=self.figure_dpi,
        )

        if y_prob is not None:
            plot_roc_curves(
                y_true, y_prob,
                class_names=class_names,
                save_path=self.figure_dir / f"{self.experiment_name}_roc.png",
                dpi=self.figure_dpi,
            )
            plot_pr_curves(
                y_true, y_prob,
                class_names=class_names,
                save_path=self.figure_dir / f"{self.experiment_name}_pr.png",
                dpi=self.figure_dpi,
            )

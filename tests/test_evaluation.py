"""Unit tests for the evaluation metrics module.

All tests use synthetic labels and probabilities — no dataset or GPU required.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.metrics import (
    compute_classification_metrics,
    format_classification_report,
    metrics_to_latex_row,
)

NUM_CLASSES = 5


def make_perfect_labels(n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Return labels and matching predictions (100% accuracy)."""
    labels = np.tile(np.arange(NUM_CLASSES), n // NUM_CLASSES + 1)[:n]
    return labels, labels.copy()


def make_random_probs(n: int = 100) -> np.ndarray:
    np.random.seed(42)
    raw = np.random.rand(n, NUM_CLASSES)
    return raw / raw.sum(axis=1, keepdims=True)


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y_true, y_pred = make_perfect_labels()
        metrics = compute_classification_metrics(y_true, y_pred, NUM_CLASSES)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["f1_macro"] == pytest.approx(1.0)
        assert metrics["precision_macro"] == pytest.approx(1.0)
        assert metrics["recall_macro"] == pytest.approx(1.0)

    def test_all_wrong_predictions(self):
        y_true = np.zeros(50, dtype=int)       # all class 0
        y_pred = np.ones(50, dtype=int)        # all class 1
        metrics = compute_classification_metrics(y_true, y_pred, NUM_CLASSES)
        assert metrics["accuracy"] == pytest.approx(0.0)

    def test_per_class_f1_present(self):
        y_true, y_pred = make_perfect_labels()
        metrics = compute_classification_metrics(y_true, y_pred, NUM_CLASSES)
        from datasets.defungi import CLASS_NAMES
        for cls in CLASS_NAMES[:NUM_CLASSES]:
            assert f"f1_{cls}" in metrics, f"Missing per-class F1 for {cls}"

    def test_auc_with_probs(self):
        y_true, _ = make_perfect_labels()
        y_prob = make_random_probs(len(y_true))
        metrics = compute_classification_metrics(y_true, y_true, NUM_CLASSES, y_prob=y_prob)
        assert "auc_macro" in metrics
        assert 0.0 <= metrics["auc_macro"] <= 1.0

    def test_returns_float_values(self):
        y_true, y_pred = make_perfect_labels()
        metrics = compute_classification_metrics(y_true, y_pred, NUM_CLASSES)
        for k, v in metrics.items():
            assert isinstance(v, float), f"Metric {k} is not float: {type(v)}"

    def test_list_inputs_accepted(self):
        y_true, y_pred = make_perfect_labels()
        metrics = compute_classification_metrics(y_true.tolist(), y_pred.tolist(), NUM_CLASSES)
        assert "accuracy" in metrics


class TestFormatReport:
    def test_report_contains_species_names(self):
        y_true, y_pred = make_perfect_labels()
        report = format_classification_report(y_true, y_pred)
        # Should contain at least one species abbreviation
        assert any(species in report for species in ["C. albicans", "A. niger", "T. rubrum"])

    def test_report_is_string(self):
        y_true, y_pred = make_perfect_labels()
        report = format_classification_report(y_true, y_pred)
        assert isinstance(report, str) and len(report) > 0


class TestLatexRow:
    def test_row_contains_experiment_name(self):
        y_true, y_pred = make_perfect_labels()
        metrics = compute_classification_metrics(y_true, y_pred, NUM_CLASSES)
        row = metrics_to_latex_row(metrics, "My Model")
        assert "My Model" in row

    def test_row_ends_with_latex_newline(self):
        y_true, y_pred = make_perfect_labels()
        metrics = compute_classification_metrics(y_true, y_pred, NUM_CLASSES)
        row = metrics_to_latex_row(metrics, "Test")
        assert row.endswith("\\\\")

    def test_row_has_five_columns(self):
        y_true, y_pred = make_perfect_labels()
        metrics = compute_classification_metrics(y_true, y_pred, NUM_CLASSES)
        row = metrics_to_latex_row(metrics, "Test")
        # Split by & gives: name + 4 metrics + trailing \\ → 5 fields
        fields = [f.strip() for f in row.split("&")]
        assert len(fields) == 6  # name + acc + prec + rec + f1 + auc

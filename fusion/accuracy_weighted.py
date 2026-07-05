"""Strategy S2 — Accuracy-Weighted Fusion.

Weights each resolution's prediction by its validation accuracy.
Weights are computed once after training and fixed at inference.

    w_224 = acc_224 / (acc_224 + acc_384)
    fused = w_224 * probs_224 + w_384 * probs_384
"""

from __future__ import annotations

import torch

from fusion.base import FusionStrategy


class AccuracyWeightedFusion(FusionStrategy):
    """S2: Fusion weights derived from per-model validation accuracy."""

    strategy_name = "accuracy_weighted"

    def __init__(
        self,
        num_classes: int = 5,
        acc_224: float = 0.5,
        acc_384: float = 0.5,
    ) -> None:
        super().__init__(num_classes)
        total = acc_224 + acc_384
        if total == 0.0:
            raise ValueError("acc_224 + acc_384 must be > 0")
        self.w_224 = acc_224 / total
        self.w_384 = acc_384 / total

    def fuse(
        self,
        probs_224: torch.Tensor,
        probs_384: torch.Tensor,
    ) -> torch.Tensor:
        return self.w_224 * probs_224 + self.w_384 * probs_384

    def update_weights(self, acc_224: float, acc_384: float) -> None:
        """Update weights after receiving actual validation accuracies."""
        total = acc_224 + acc_384
        self.w_224 = acc_224 / total
        self.w_384 = acc_384 / total

    def __repr__(self) -> str:
        return (
            f"AccuracyWeightedFusion("
            f"w_224={self.w_224:.4f}, w_384={self.w_384:.4f})"
        )

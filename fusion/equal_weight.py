"""Strategy S1 — Equal-Weight Fusion.

Replicates Version 1's approach: simple unweighted average of the two
softmax probability vectors. Serves as the baseline for comparing ACMF.

    fused = (probs_224 + probs_384) / 2
"""

from __future__ import annotations

import torch

from fusion.base import FusionStrategy


class EqualWeightFusion(FusionStrategy):
    """S1: Equal-weight average of 224px and 384px softmax predictions."""

    strategy_name = "equal_weight"

    def fuse(
        self,
        probs_224: torch.Tensor,
        probs_384: torch.Tensor,
    ) -> torch.Tensor:
        return (probs_224 + probs_384) * 0.5

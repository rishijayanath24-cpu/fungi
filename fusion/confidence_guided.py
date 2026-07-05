"""Strategy S3 — Confidence-Guided Fusion (ACMF core contribution).

Dynamically weights each resolution's prediction by the inverse of its
Shannon entropy. A high-confidence (low-entropy) prediction contributes
more to the final result; an uncertain prediction is down-weighted.

Per-image entropy:
    H_i = -sum_c p_ic * log(p_ic + eps)

Per-image fusion weight (unnormalized):
    w_i = 1 / (H_i + eps)

Normalized weight:
    w_i_norm = w_i / (w_224 + w_384)

Fused prediction:
    fused = w_224_norm * probs_224 + w_384_norm * probs_384

This requires NO training and NO validation accuracy — it operates
purely on the softmax distribution of each forward pass.
"""

from __future__ import annotations

import torch

from fusion.base import FusionStrategy


class ConfidenceGuidedFusion(FusionStrategy):
    """S3: Adaptive per-image entropy-based confidence weighting.

    This is the primary novel contribution of Version 2.
    Works at inference time with no additional training required.
    """

    strategy_name = "confidence_guided"

    def __init__(
        self,
        num_classes: int = 5,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(num_classes)
        self.eps = eps

    def fuse(
        self,
        probs_224: torch.Tensor,
        probs_384: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-image entropy and use inverse entropy as fusion weights."""
        h_224 = self._entropy(probs_224)   # [B]
        h_384 = self._entropy(probs_384)   # [B]

        # Inverse entropy: low entropy → high weight
        w_224 = 1.0 / (h_224 + self.eps)  # [B]
        w_384 = 1.0 / (h_384 + self.eps)  # [B]

        # Normalize weights to sum to 1 per image
        w_sum = w_224 + w_384 + self.eps
        w_224_norm = (w_224 / w_sum).unsqueeze(-1)  # [B, 1]
        w_384_norm = (w_384 / w_sum).unsqueeze(-1)  # [B, 1]

        fused = w_224_norm * probs_224 + w_384_norm * probs_384
        return fused

    def _entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Shannon entropy H = -sum p * log(p), shape [B]."""
        log_p = torch.log(probs + self.eps)
        return -(probs * log_p).sum(dim=-1)

    def get_weights(
        self,
        probs_224: torch.Tensor,
        probs_384: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the normalized weights for analysis/visualization."""
        h_224 = self._entropy(probs_224)
        h_384 = self._entropy(probs_384)
        w_224 = 1.0 / (h_224 + self.eps)
        w_384 = 1.0 / (h_384 + self.eps)
        w_sum = w_224 + w_384 + self.eps
        return w_224 / w_sum, w_384 / w_sum

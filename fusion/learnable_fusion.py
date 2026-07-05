"""Strategy S4 — Adaptive Learnable Fusion.

An MLP fusion head learns to weight each resolution's softmax predictions
conditioned on the concatenated probability vectors. The head is trained
end-to-end after the individual classifiers are frozen.

Architecture:
    input:  [probs_224 || probs_384]  → dim = 2 * num_classes
    hidden: Linear → LayerNorm → ReLU → Dropout
    output: 2 scalars (softmax → normalized weights for 224 and 384)
    fused:  w_224 * probs_224 + w_384 * probs_384

Unlike S3, S4 can learn asymmetric, class-conditioned weighting.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion.base import FusionStrategy


class LearnableFusion(FusionStrategy):
    """S4: Learnable MLP fusion head trained to weight per-resolution predictions."""

    strategy_name = "learnable"

    def __init__(
        self,
        num_classes: int = 5,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(num_classes)

        in_dim = 2 * num_classes  # concatenated probs from both resolutions

        self.weight_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # output: unnormalized weights for [224, 384]
        )

        # Initialize so weights start near equal (neutral initialization)
        nn.init.zeros_(self.weight_head[-1].weight)
        nn.init.zeros_(self.weight_head[-1].bias)

    def fuse(
        self,
        probs_224: torch.Tensor,
        probs_384: torch.Tensor,
    ) -> torch.Tensor:
        concat = torch.cat([probs_224, probs_384], dim=-1)   # [B, 2C]
        raw_weights = self.weight_head(concat)                # [B, 2]
        weights = F.softmax(raw_weights, dim=-1)              # [B, 2] summing to 1

        w_224 = weights[:, 0].unsqueeze(-1)   # [B, 1]
        w_384 = weights[:, 1].unsqueeze(-1)   # [B, 1]

        return w_224 * probs_224 + w_384 * probs_384

    def get_weights(
        self,
        probs_224: torch.Tensor,
        probs_384: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return learned weights per image for analysis."""
        concat = torch.cat([probs_224, probs_384], dim=-1)
        raw_weights = self.weight_head(concat)
        weights = F.softmax(raw_weights, dim=-1)
        return weights[:, 0], weights[:, 1]

    def freeze_head(self) -> None:
        for p in self.weight_head.parameters():
            p.requires_grad_(False)

    def unfreeze_head(self) -> None:
        for p in self.weight_head.parameters():
            p.requires_grad_(True)

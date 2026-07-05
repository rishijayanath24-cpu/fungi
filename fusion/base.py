"""Abstract base class for all ACMF fusion strategies.

All strategies receive two softmax probability vectors (one per resolution)
and return a single fused probability vector.

Strategy contract:
  - Input:  probs_224 [B, C], probs_384 [B, C]  (softmax probabilities)
  - Output: fused_probs [B, C]                   (normalized probabilities)

The output must always sum to 1.0 along dim=-1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class FusionStrategy(ABC, nn.Module):
    """Base class for multi-resolution fusion strategies.

    Subclasses implement fuse() which is called by forward().
    """

    strategy_name: str = "base"

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(
        self,
        probs_224: torch.Tensor,
        probs_384: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse two probability vectors. Validates shapes before delegating."""
        assert probs_224.shape == probs_384.shape, (
            f"Shape mismatch: {probs_224.shape} vs {probs_384.shape}"
        )
        assert probs_224.shape[-1] == self.num_classes, (
            f"Expected {self.num_classes} classes, got {probs_224.shape[-1]}"
        )
        return self.fuse(probs_224, probs_384)

    @abstractmethod
    def fuse(
        self,
        probs_224: torch.Tensor,
        probs_384: torch.Tensor,
    ) -> torch.Tensor:
        """Implement fusion logic. Must return a valid probability distribution."""
        ...

    def predict(
        self,
        probs_224: torch.Tensor,
        probs_384: torch.Tensor,
    ) -> torch.Tensor:
        """Return class indices from fused probabilities."""
        return self.forward(probs_224, probs_384).argmax(dim=-1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(strategy='{self.strategy_name}', num_classes={self.num_classes})"

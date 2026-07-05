"""Swin Transformer Tiny baseline classifier.

Swin-T operates on 224×224 px with 4×4 patch size and shifted window attention.
Included as the hierarchical transformer baseline between ViT and CNN models.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import timm

logger = logging.getLogger(__name__)


class SwinClassifier(nn.Module):
    """Swin-Tiny fine-tuned for fungal species classification."""

    def __init__(
        self,
        backbone: str = "swin_tiny_patch4_window7_224",
        num_classes: int = 5,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes

        self.model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=num_classes,
        )

        logger.info(
            "Swin loaded: %s | pretrained=%s | params=%.2fM",
            backbone,
            pretrained,
            sum(p.numel() for p in self.model.parameters()) / 1e6,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return globally averaged feature map from the final stage."""
        features = self.model.forward_features(x)
        # Swin returns (B, H, W, C) — average over spatial dims
        if features.dim() == 4:
            features = features.mean(dim=(1, 2))
        return features

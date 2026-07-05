"""ResNet-50 baseline classifier.

Used as a CNN baseline for comparison against ViT.
The final fully-connected layer is replaced to match num_classes.
Grad-CAM hooks target the final convolutional layer (layer4).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import timm

logger = logging.getLogger(__name__)


class ResNetClassifier(nn.Module):
    """ResNet-50 fine-tuned for fungal species classification."""

    def __init__(
        self,
        backbone: str = "resnet50",
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

        # Grad-CAM target layer (populated lazily)
        self._gradcam_layer: nn.Module | None = None
        self._gradients: torch.Tensor | None = None
        self._activations: torch.Tensor | None = None
        self._gradcam_hooks: list = []

        logger.info(
            "ResNet loaded: %s | pretrained=%s | params=%.2fM",
            backbone,
            pretrained,
            sum(p.numel() for p in self.model.parameters()) / 1e6,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return global average-pooled feature vector."""
        return self.model.forward_features(x)

    # ── Grad-CAM hook API ─────────────────────────────────────────────────────

    def register_gradcam_hooks(self) -> None:
        """Register hooks on layer4 for Grad-CAM computation."""
        target = self.model.layer4  # final residual block group

        def save_activation(module, input, output):
            self._activations = output

        def save_gradient(module, grad_input, grad_output):
            self._gradients = grad_output[0]

        self._gradcam_hooks = [
            target.register_forward_hook(save_activation),
            target.register_full_backward_hook(save_gradient),
        ]
        self._gradcam_layer = target

    def remove_gradcam_hooks(self) -> None:
        for h in self._gradcam_hooks:
            h.remove()
        self._gradcam_hooks = []

    def get_gradcam_tensors(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self._gradients, self._activations

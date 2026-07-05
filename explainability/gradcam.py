"""Grad-CAM for CNN baseline models (ResNet-50, EfficientNet-B3).

Reference: Selvaraju et al., 2017 — "Grad-CAM: Visual Explanations from Deep
           Networks via Gradient-based Localization"
           https://arxiv.org/abs/1610.02391

Algorithm:
    1. Forward pass + hook saves final conv feature map A [C, H, W]
    2. Backward pass on target class logit; hook saves gradients dScore/dA
    3. Importance weights: alpha_c = GAP(dScore/dA_c)
    4. Grad-CAM = ReLU(sum_c alpha_c * A_c)
    5. Bilinear upsample to input resolution
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Computes Grad-CAM heatmaps for a CNN model with registered hooks."""

    def __init__(
        self,
        model: nn.Module,
        target_class: Optional[int] = None,
    ) -> None:
        self.model = model
        self.target_class = target_class  # None → use predicted class

        # Models must expose register_gradcam_hooks() and get_gradcam_tensors()
        if not hasattr(model, "register_gradcam_hooks"):
            raise TypeError(
                f"{type(model).__name__} must implement register_gradcam_hooks(). "
                "Use ResNetClassifier or EfficientNetClassifier."
            )

    def __call__(
        self,
        image: torch.Tensor,
        image_size: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        """Compute Grad-CAM heatmap for a single image.

        Args:
            image: Preprocessed tensor [1, 3, H, W].
            image_size: Output heatmap spatial size. Defaults to input H×W.

        Returns:
            2D numpy array [H, W] in [0, 1], same spatial size as input.
        """
        self.model.register_gradcam_hooks()
        self.model.eval()

        # Forward pass with gradient tracking
        image = image.requires_grad_(False)
        logits = self.model(image)              # [1, C]

        target = (
            self.target_class
            if self.target_class is not None
            else int(logits.argmax(dim=-1).item())
        )

        # Backward pass on the target class score
        self.model.zero_grad()
        score = logits[0, target]
        score.backward()

        gradients, activations = self.model.get_gradcam_tensors()
        self.model.remove_gradcam_hooks()

        if gradients is None or activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture tensors. Check hook registration.")

        # Global average pool the gradients → importance weights [C]
        weights = gradients[0].mean(dim=(-2, -1))   # [C]

        # Weighted combination of activation maps
        cam = (weights.view(-1, 1, 1) * activations[0]).sum(dim=0)  # [H_feat, W_feat]
        cam = F.relu(cam)

        # Upsample to input resolution
        if image_size is None:
            image_size = (image.shape[2], image.shape[3])
        cam = cam.detach().unsqueeze(0).unsqueeze(0)   # [1, 1, H_feat, W_feat]
        cam = F.interpolate(cam, size=image_size, mode="bilinear", align_corners=False)
        cam = cam[0, 0].numpy()

        # Normalize to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

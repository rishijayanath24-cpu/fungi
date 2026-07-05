"""Attention Rollout for Vision Transformers.

Reference: Abnar & Zuidema, 2020 — "Quantifying Attention Flow in Transformers"
           https://arxiv.org/abs/2005.00928

Rollout propagates attention through all transformer layers by recursively
multiplying the attention matrices, accounting for skip connections via the
identity matrix. This gives a more complete view of which input patches
a model attends to for a given prediction.

Algorithm:
    A = identity
    for each layer:
        attn = (layer_attn + identity) / 2   # account for residual connection
        A = A @ attn                          # propagate attention
    return A[0, 1:]                           # CLS → patch attention
"""

from __future__ import annotations

import math

import numpy as np
import torch

from models.vit import ViTClassifier


class AttentionRollout:
    """Computes attention rollout for a ViTClassifier instance."""

    def __init__(
        self,
        model: ViTClassifier,
        head_fusion: str = "mean",
        discard_ratio: float = 0.9,
    ) -> None:
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio

    def __call__(self, image: torch.Tensor) -> np.ndarray:
        """Compute rollout attention map for a single image.

        Args:
            image: Preprocessed tensor [1, 3, H, W].

        Returns:
            2D numpy heatmap [patch_h, patch_w] in [0, 1].
        """
        self.model.register_attention_hooks()
        self.model.clear_attention_weights()

        self.model.eval()
        with torch.no_grad():
            _ = self.model(image)

        attn_list = self.model.get_attention_weights()
        if not attn_list:
            raise RuntimeError("No attention weights found. Register hooks first.")

        return self._rollout(attn_list)

    def _rollout(self, attn_list: list[torch.Tensor]) -> np.ndarray:
        """Implements the Abnar & Zuidema rollout algorithm."""
        # All tensors: [1, heads, tokens, tokens]
        # Start with identity matrix
        num_tokens = attn_list[0].shape[-1]
        result = torch.eye(num_tokens)

        for attn in attn_list:
            attn = attn[0]  # remove batch dim → [heads, tokens, tokens]

            if self.head_fusion == "mean":
                attn_fused = attn.mean(dim=0)
            elif self.head_fusion == "max":
                attn_fused = attn.max(dim=0).values
            else:
                attn_fused = attn.min(dim=0).values

            # Symmetrize and add residual
            attn_fused = attn_fused + torch.eye(num_tokens)
            attn_fused = attn_fused / attn_fused.sum(dim=-1, keepdim=True)

            result = attn_fused @ result

        # CLS token → all patch tokens
        mask = result[0, 1:]   # [num_patches]

        # Discard low-attention noise
        if self.discard_ratio > 0:
            threshold = torch.quantile(mask, self.discard_ratio)
            mask = mask * (mask >= threshold).float()

        num_patches = mask.shape[0]
        patch_h = patch_w = int(math.sqrt(num_patches))
        heatmap = mask.reshape(patch_h, patch_w).numpy()

        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap

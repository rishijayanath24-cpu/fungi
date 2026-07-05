"""Fast per-image attention map from the last ViT transformer block.

Extracts the CLS→patch attention weights from the final block and
reshapes them into a spatial heatmap matching the patch grid.

This is the fast method (single block, single forward pass).
For the full multi-layer rollout see attention_rollout.py.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from models.vit import ViTClassifier


def extract_last_block_attention(
    model: ViTClassifier,
    image: torch.Tensor,
    head_fusion: str = "mean",
    discard_ratio: float = 0.9,
) -> np.ndarray:
    """Extract and upsample the last-block attention map for a single image.

    Args:
        model: ViTClassifier with attention hooks registered.
        image: Preprocessed image tensor [1, 3, H, W].
        head_fusion: How to aggregate attention heads — 'mean', 'max', or 'min'.
        discard_ratio: Zero-out the lowest N% attention values (noise reduction).

    Returns:
        2D numpy array of shape [patch_h, patch_w], values in [0, 1].
    """
    model.register_attention_hooks()
    model.clear_attention_weights()

    model.eval()
    with torch.no_grad():
        _ = model(image)

    attn_weights = model.get_attention_weights()
    if not attn_weights:
        raise RuntimeError(
            "No attention weights collected. Ensure ViTClassifier._patch_attn_block "
            "is applied and register_attention_hooks() was called."
        )

    # Use only the last block's attention
    last_attn = attn_weights[-1]  # [1, heads, tokens, tokens]

    # Aggregate over heads
    if head_fusion == "mean":
        attn = last_attn[0].mean(dim=0)   # [tokens, tokens]
    elif head_fusion == "max":
        attn = last_attn[0].max(dim=0).values
    else:
        attn = last_attn[0].min(dim=0).values

    # CLS token row: attention from CLS to all patches
    cls_attn = attn[0, 1:]   # [num_patches]

    # Discard low-attention tokens
    if discard_ratio > 0:
        threshold = torch.quantile(cls_attn, discard_ratio)
        cls_attn = cls_attn * (cls_attn >= threshold).float()

    # Reshape to patch grid
    num_patches = cls_attn.shape[0]
    patch_h = patch_w = int(math.sqrt(num_patches))
    attn_map = cls_attn.reshape(patch_h, patch_w).numpy()

    # Normalize to [0, 1]
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    return attn_map

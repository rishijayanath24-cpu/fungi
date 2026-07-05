"""Vision Transformer (ViT-B/16) classifier wrapper.

Supports both 224px and 384px resolutions via timm.
Attention maps are accessible through registered forward hooks
(used by the explainability module).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import timm

logger = logging.getLogger(__name__)


class ViTClassifier(nn.Module):
    """ViT-B/16 fine-tuned for fungal species classification.

    The final classification head is replaced with a linear layer
    matching num_classes. Intermediate attention weights from all
    transformer blocks are optionally stored for explainability.
    """

    def __init__(
        self,
        backbone: str = "vit_base_patch16_224",
        num_classes: int = 5,
        pretrained: bool = True,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes

        self.model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
        )

        # Storage for attention weights (populated by hooks when enabled)
        self._attn_weights: list[torch.Tensor] = []
        self._hooks: list = []
        self._hooks_registered = False

        logger.info(
            "ViT loaded: %s | pretrained=%s | num_classes=%d | params=%.2fM",
            backbone,
            pretrained,
            num_classes,
            sum(p.numel() for p in self.model.parameters()) / 1e6,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ── Attention hook API ────────────────────────────────────────────────────

    def register_attention_hooks(self) -> None:
        """Register forward hooks on every transformer block's attention module."""
        if self._hooks_registered:
            return
        self._attn_weights = []

        def _make_hook(block_idx: int):
            def hook(module, inputs, output):
                # timm ViT attention returns (out, attn_weights) when
                # fused_attn is False, but may return only out otherwise.
                # We extract attn_weights from the module's stored state.
                if hasattr(module, "attn_weights"):
                    self._attn_weights.append(module.attn_weights.detach().cpu())

            return hook

        for i, block in enumerate(self.model.blocks):
            # Patch the attention forward to store weights
            _patch_attn_block(block.attn)
            h = block.attn.register_forward_hook(_make_hook(i))
            self._hooks.append(h)

        self._hooks_registered = True
        logger.debug("Attention hooks registered on %d blocks", len(self._hooks))

    def remove_attention_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._hooks_registered = False

    def get_attention_weights(self) -> list[torch.Tensor]:
        """Return stored per-block attention maps from the last forward pass."""
        return self._attn_weights

    def clear_attention_weights(self) -> None:
        self._attn_weights = []

    # ── Feature extraction ────────────────────────────────────────────────────

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the CLS token feature vector before the classification head."""
        return self.model.forward_features(x)[:, 0]  # CLS token

    def get_image_size(self) -> int:
        return self.model.patch_embed.img_size[0]


def _patch_attn_block(attn_module: nn.Module) -> None:
    """Monkey-patch a timm Attention module to cache attn_weights on forward."""
    original_forward = attn_module.forward

    def patched_forward(x):
        # Temporarily disable fused attention to get attention weights
        B, N, C = x.shape
        qkv = attn_module.qkv(x)
        qkv = qkv.reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scale = attn_module.scale
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        attn_module.attn_weights = attn  # store for hooks
        attn = attn_module.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attn_module.proj(x)
        x = attn_module.proj_drop(x)
        return x

    # Only patch if the module hasn't been patched already
    if not getattr(attn_module, "_patched", False):
        attn_module.forward = patched_forward
        attn_module._patched = True

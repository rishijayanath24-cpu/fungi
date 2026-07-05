"""Model factory: build any supported architecture from a config dict."""

from __future__ import annotations

import torch.nn as nn

from models.vit import ViTClassifier
from models.resnet import ResNetClassifier
from models.efficientnet import EfficientNetClassifier
from models.swin import SwinClassifier

_REGISTRY: dict[str, type] = {
    "vit": ViTClassifier,
    "resnet": ResNetClassifier,
    "efficientnet": EfficientNetClassifier,
    "swin": SwinClassifier,
}


def build_model(cfg: dict) -> nn.Module:
    """Instantiate a model from the 'model' section of a config dict.

    Example config section:
        model:
          architecture: "vit"
          backbone: "vit_base_patch16_224"
          pretrained: true
          num_classes: 5
          drop_rate: 0.1

    Raises:
        ValueError: If architecture is not registered.
    """
    model_cfg = cfg["model"]
    arch = model_cfg["architecture"].lower()

    if arch not in _REGISTRY:
        raise ValueError(
            f"Unknown architecture '{arch}'. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )

    cls = _REGISTRY[arch]
    num_classes = cfg["dataset"].get("num_classes", 5)

    kwargs: dict = {
        "backbone": model_cfg["backbone"],
        "num_classes": num_classes,
        "pretrained": model_cfg.get("pretrained", True),
    }

    # Architecture-specific optional kwargs
    if arch == "vit":
        kwargs["drop_rate"] = model_cfg.get("drop_rate", 0.1)
        kwargs["attn_drop_rate"] = model_cfg.get("attn_drop_rate", 0.0)

    return cls(**kwargs)


def list_architectures() -> list[str]:
    return sorted(_REGISTRY.keys())

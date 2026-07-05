"""Fusion strategy factory."""

from __future__ import annotations

import torch.nn as nn

from fusion.base import FusionStrategy
from fusion.equal_weight import EqualWeightFusion
from fusion.accuracy_weighted import AccuracyWeightedFusion
from fusion.confidence_guided import ConfidenceGuidedFusion
from fusion.learnable_fusion import LearnableFusion

_STRATEGY_REGISTRY: dict[str, type[FusionStrategy]] = {
    "equal_weight": EqualWeightFusion,
    "accuracy_weighted": AccuracyWeightedFusion,
    "confidence_guided": ConfidenceGuidedFusion,
    "learnable": LearnableFusion,
}

STRATEGY_LABELS: dict[str, str] = {
    "equal_weight": "S1 — Equal Weight (V1 baseline)",
    "accuracy_weighted": "S2 — Accuracy Weighted",
    "confidence_guided": "S3 — Confidence Guided (ACMF)",
    "learnable": "S4 — Adaptive Learnable Fusion",
}


def build_fusion_strategy(
    strategy: str,
    num_classes: int = 5,
    **kwargs,
) -> FusionStrategy:
    """Instantiate a fusion strategy by name.

    Args:
        strategy: One of 'equal_weight', 'accuracy_weighted',
                  'confidence_guided', 'learnable'.
        num_classes: Number of output classes.
        **kwargs: Strategy-specific keyword arguments (e.g. acc_224, eps).

    Raises:
        ValueError: If strategy name is unknown.
    """
    key = strategy.lower()
    if key not in _STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown fusion strategy '{strategy}'. "
            f"Available: {sorted(_STRATEGY_REGISTRY.keys())}"
        )
    return _STRATEGY_REGISTRY[key](num_classes=num_classes, **kwargs)


def list_strategies() -> list[str]:
    return sorted(_STRATEGY_REGISTRY.keys())

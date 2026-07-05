"""MixUp and CutMix data augmentation for batch-level mixing.

Both strategies operate on already-normalized image tensors in the collate
function, so they compose cleanly with any per-image augmentation pipeline.

References:
  MixUp:  Zhang et al., 2018 (https://arxiv.org/abs/1710.09412)
  CutMix: Yun et al., 2019  (https://arxiv.org/abs/1905.04899)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.8,
    num_classes: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MixUp to a batch.

    Returns:
        mixed_x: Mixed images.
        mixed_y: Soft one-hot targets (float, shape [B, C]).
    """
    if alpha <= 0:
        return x, F.one_hot(y, num_classes).float()

    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]

    y_onehot = F.one_hot(y, num_classes).float()
    y_onehot_b = F.one_hot(y[index], num_classes).float()
    mixed_y = lam * y_onehot + (1 - lam) * y_onehot_b

    return mixed_x, mixed_y


def cutmix_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
    num_classes: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply CutMix to a batch.

    Returns:
        mixed_x: CutMix images.
        mixed_y: Soft one-hot targets weighted by patch area ratio.
    """
    if alpha <= 0:
        return x, F.one_hot(y, num_classes).float()

    lam = float(np.random.beta(alpha, alpha))
    batch_size, _, H, W = x.shape
    index = torch.randperm(batch_size, device=x.device)

    cut_ratio = (1.0 - lam) ** 0.5
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, H)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    actual_lam = 1.0 - (x2 - x1) * (y2 - y1) / (H * W)

    y_onehot = F.one_hot(y, num_classes).float()
    y_onehot_b = F.one_hot(y[index], num_classes).float()
    mixed_y = actual_lam * y_onehot + (1 - actual_lam) * y_onehot_b

    return mixed_x, mixed_y


def apply_batch_augmentation(
    x: torch.Tensor,
    y: torch.Tensor,
    use_mixup: bool = True,
    use_cutmix: bool = True,
    mixup_alpha: float = 0.8,
    cutmix_alpha: float = 1.0,
    num_classes: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly apply either MixUp or CutMix to a batch.

    When both are enabled, one is chosen with equal probability.
    Returns soft one-hot targets in all cases.
    """
    if not use_mixup and not use_cutmix:
        return x, F.one_hot(y, num_classes).float()

    r = np.random.rand()
    if use_mixup and use_cutmix:
        if r < 0.5:
            return mixup_data(x, y, mixup_alpha, num_classes)
        else:
            return cutmix_data(x, y, cutmix_alpha, num_classes)
    elif use_mixup:
        return mixup_data(x, y, mixup_alpha, num_classes)
    else:
        return cutmix_data(x, y, cutmix_alpha, num_classes)


def soft_cross_entropy(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy loss compatible with soft (one-hot) targets from MixUp/CutMix.

    Also applies label smoothing if label_smoothing > 0.
    """
    num_classes = logits.size(-1)
    if label_smoothing > 0:
        smooth_targets = (
            soft_targets * (1.0 - label_smoothing)
            + label_smoothing / num_classes
        )
    else:
        smooth_targets = soft_targets

    log_prob = F.log_softmax(logits, dim=-1)
    loss = -(smooth_targets * log_prob).sum(dim=-1).mean()
    return loss

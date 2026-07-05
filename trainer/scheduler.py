"""Learning rate scheduler: linear warmup + cosine annealing decay.

This combination is standard for fine-tuning ViT and related models.
Warmup prevents large gradient updates in the early frozen-to-unfrozen
transition; cosine decay provides smooth convergence.
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(
    optimizer: Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    """Create a warmup + cosine annealing LR schedule.

    Args:
        optimizer: The optimizer to schedule.
        warmup_epochs: Number of linear warmup epochs.
        total_epochs: Total training epochs.
        steps_per_epoch: Number of optimizer steps per epoch.
        min_lr_ratio: Final LR as a fraction of base LR (default 1% of base).

    Returns:
        A LambdaLR scheduler that steps every batch.
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup from min_lr_ratio to 1.0
            return min_lr_ratio + (1.0 - min_lr_ratio) * current_step / max(warmup_steps, 1)
        else:
            # Cosine annealing from 1.0 to min_lr_ratio
            progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def get_current_lr(optimizer: Optimizer) -> float:
    """Read the current learning rate from an optimizer's first param group."""
    return optimizer.param_groups[0]["lr"]

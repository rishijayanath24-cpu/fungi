"""Global reproducibility utilities.

Call set_seed(seed) at the start of every training run before any
data loading or model instantiation. This ensures that results are
reproducible across machines with the same hardware configuration.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Fix all random sources for reproducible training.

    Args:
        seed: Integer seed value (e.g. 42).
        deterministic: Enable CUDA deterministic algorithms.
                       Slightly reduces throughput but ensures bit-exact
                       reproducibility on the same hardware.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU safety
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch 1.11+: deterministic algorithm selection
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            pass

    logger.info("Global seed set to %d | deterministic=%s", seed, deterministic)

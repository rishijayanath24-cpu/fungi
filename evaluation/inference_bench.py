"""Inference time and GPU memory benchmarking.

Reports:
  - Mean and std of per-image inference latency (ms)
  - Throughput (images/second)
  - GPU memory footprint (MB) at peak
  - Model parameter count (M)
  - Model size on disk (MB) if checkpoint path provided
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_WARMUP_ITERS = 10
_BENCH_ITERS = 100


def benchmark_inference(
    model: nn.Module,
    image_size: int = 224,
    batch_size: int = 1,
    device: str = "cuda",
    checkpoint_path: Optional[str | Path] = None,
    num_classes: int = 5,
) -> dict[str, float]:
    """Measure latency, throughput, memory, and model size.

    Args:
        model: Trained model in eval mode.
        image_size: Input resolution (square).
        batch_size: Batch size for throughput measurement.
        device: 'cuda' or 'cpu'.
        checkpoint_path: Optional path to .pt file for size measurement.
        num_classes: Unused here, kept for API consistency.

    Returns:
        Dict of benchmark metrics.
    """
    model = model.to(device)
    model.eval()

    dummy = torch.randn(batch_size, 3, image_size, image_size, device=device)

    # ── Warmup ───────────────────────────────────────────────────────────────
    with torch.no_grad():
        for _ in range(_WARMUP_ITERS):
            _ = model(dummy)

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    # ── Single-image latency (batch=1) ────────────────────────────────────────
    single = torch.randn(1, 3, image_size, image_size, device=device)
    latencies_ms: list[float] = []

    with torch.no_grad():
        for _ in range(_BENCH_ITERS):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(single)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000)

    import numpy as np
    lat_arr = np.array(latencies_ms)

    # ── Throughput (batch) ────────────────────────────────────────────────────
    if device == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(_BENCH_ITERS):
            _ = model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    throughput = (batch_size * _BENCH_ITERS) / (t_end - t_start)

    # ── GPU memory ───────────────────────────────────────────────────────────
    peak_memory_mb = 0.0
    if device == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1e6

    # ── Parameter count ───────────────────────────────────────────────────────
    num_params_m = sum(p.numel() for p in model.parameters()) / 1e6

    # ── Disk size ─────────────────────────────────────────────────────────────
    disk_mb = 0.0
    if checkpoint_path and Path(checkpoint_path).exists():
        disk_mb = os.path.getsize(checkpoint_path) / 1e6

    results = {
        "latency_mean_ms": float(lat_arr.mean()),
        "latency_std_ms": float(lat_arr.std()),
        "latency_p95_ms": float(np.percentile(lat_arr, 95)),
        "throughput_imgs_per_sec": float(throughput),
        "gpu_memory_peak_mb": float(peak_memory_mb),
        "params_millions": float(num_params_m),
        "disk_size_mb": float(disk_mb),
        "image_size": image_size,
        "batch_size": batch_size,
    }

    logger.info(
        "Benchmark: latency=%.2f±%.2f ms | throughput=%.1f img/s | "
        "GPU mem=%.1f MB | params=%.2fM",
        results["latency_mean_ms"],
        results["latency_std_ms"],
        results["throughput_imgs_per_sec"],
        results["gpu_memory_peak_mb"],
        results["params_millions"],
    )
    return results

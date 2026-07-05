"""Heatmap overlay and publication figure generation.

Produces:
  - Single-image heatmap overlay (PIL Image or saved PNG)
  - Side-by-side grid: original | rollout | Grad-CAM across models
  - Batch export of XAI figures for the paper

All figures saved at publication DPI (default 300).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image


def overlay_heatmap(
    image: np.ndarray | Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> Image.Image:
    """Overlay a normalized heatmap on an original image.

    Args:
        image: Original image as numpy [H, W, 3] uint8 or PIL Image.
        heatmap: Normalized heatmap [H, W] in [0, 1].
        alpha: Heatmap opacity (0=transparent, 1=opaque).
        colormap: Matplotlib colormap name.

    Returns:
        PIL Image with heatmap overlay.
    """
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    H, W = image.shape[:2]

    # Resize heatmap to match image
    if heatmap.shape != (H, W):
        heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_pil = heatmap_pil.resize((W, H), Image.BILINEAR)
        heatmap = np.array(heatmap_pil) / 255.0

    cmap = cm.get_cmap(colormap)
    colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)

    overlay = (image * (1 - alpha) + colored * alpha).astype(np.uint8)
    return Image.fromarray(overlay)


def save_xai_grid(
    images: list[np.ndarray],
    heatmaps: list[np.ndarray],
    labels: list[str],
    model_name: str,
    save_path: str | Path,
    dpi: int = 300,
    n_cols: int = 4,
) -> None:
    """Save a grid of original images + their XAI heatmap overlays.

    Args:
        images: List of [H, W, 3] uint8 numpy arrays.
        heatmaps: List of [H, W] normalized heatmaps.
        labels: List of class label strings per image.
        model_name: Used in the figure title.
        save_path: Output file path (.png).
        dpi: Publication DPI.
        n_cols: Number of image pairs per row.
    """
    assert len(images) == len(heatmaps) == len(labels)
    n = len(images)
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(
        n_rows * 2, n_cols,
        figsize=(n_cols * 3.0, n_rows * 6.0),
        squeeze=False,
    )
    fig.suptitle(f"XAI Attention Maps — {model_name}", fontsize=10, fontweight="bold")

    for i in range(n):
        row_orig = (i // n_cols) * 2
        row_heat = row_orig + 1
        col = i % n_cols

        axes[row_orig][col].imshow(images[i])
        axes[row_orig][col].set_title(labels[i], fontsize=7)
        axes[row_orig][col].axis("off")

        overlay = overlay_heatmap(images[i], heatmaps[i])
        axes[row_heat][col].imshow(overlay)
        axes[row_heat][col].axis("off")

    # Hide unused axes
    for idx in range(n, n_rows * n_cols):
        r_o = (idx // n_cols) * 2
        r_h = r_o + 1
        c = idx % n_cols
        axes[r_o][c].axis("off")
        axes[r_h][c].axis("off")

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_side_by_side_comparison(
    image: np.ndarray,
    heatmaps: dict[str, np.ndarray],
    label: str,
    save_path: str | Path,
    dpi: int = 300,
) -> None:
    """Save original image + N XAI heatmap overlays side by side.

    Args:
        image: Original image [H, W, 3] uint8.
        heatmaps: Dict of method_name → normalized heatmap.
        label: True class label string for the figure title.
        save_path: Output PNG path.
        dpi: Publication DPI.
    """
    n = 1 + len(heatmaps)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5))
    fig.suptitle(f"Class: {label}", fontsize=9)

    axes[0].imshow(image)
    axes[0].set_title("Original", fontsize=8)
    axes[0].axis("off")

    for idx, (method_name, heatmap) in enumerate(heatmaps.items(), start=1):
        overlay = overlay_heatmap(image, heatmap)
        axes[idx].imshow(overlay)
        axes[idx].set_title(method_name, fontsize=8)
        axes[idx].axis("off")

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def denormalize(
    tensor: "torch.Tensor",
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Convert a normalized image tensor back to uint8 numpy array [H, W, 3]."""
    import numpy as np
    img = tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    img = img * np.array(std) + np.array(mean)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img

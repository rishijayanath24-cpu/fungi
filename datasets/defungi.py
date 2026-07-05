"""DeFungi dataset loader.

Expected on-disk layout:
    <root>/
        train/  H1/ H2/ H3/ H5/ H6/
        valid/  H1/ H2/ H3/ H5/ H6/   (also accepted: val/)
        test/   H1/ H2/ H3/ H5/ H6/

The validation split may be named either "valid" or "val" on disk.
Pass split="val" in all code — the loader resolves the actual folder name.

Split index files are stored in datasets/splits/ so that the exact
same images are used across all experiments regardless of filesystem order.
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

logger = logging.getLogger(__name__)

CLASS_NAMES: list[str] = ["H1", "H2", "H3", "H5", "H6"]
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: dict[int, str] = {i: c for c, i in CLASS_TO_IDX.items()}

# Scientific names for display and paper labels
SPECIES_NAMES: dict[str, str] = {
    "H1": "C. albicans",
    "H2": "A. niger",
    "H3": "T. rubrum",
    "H5": "T. mentagrophytes",
    "H6": "E. floccosum",
}

SPLITS_DIR = Path(__file__).parent / "splits"


class DeFungiDataset(Dataset):
    """PyTorch Dataset for the DeFungi microscopic fungal image collection."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Optional[Callable] = None,
    ) -> None:
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.split = split

        # Resolve "val" → "valid" if the dataset uses that folder name on disk
        root_path = Path(root)
        if split == "val":
            if (root_path / "valid").exists():
                self.root = root_path / "valid"
            else:
                self.root = root_path / "val"
        else:
            self.root = root_path / split

        self.transform = transform

        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset split directory not found: {self.root}\n"
                f"Expected one of: {root_path / 'val'} or {root_path / 'valid'}\n"
                "Check --data-root points to the folder containing train/ valid/ test/."
            )

        self.samples: list[tuple[Path, int]] = self._load_samples()
        logger.info(
            "DeFungi %s: %d images across %d classes",
            split,
            len(self.samples),
            len(CLASS_NAMES),
        )

    def _load_samples(self) -> list[tuple[Path, int]]:
        samples: list[tuple[Path, int]] = []
        for cls in CLASS_NAMES:
            cls_dir = self.root / cls
            if not cls_dir.exists():
                logger.warning("Class directory missing: %s", cls_dir)
                continue
            idx = CLASS_TO_IDX[cls]
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
                for img_path in sorted(cls_dir.glob(ext)):
                    samples.append((img_path, idx))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    @property
    def class_names(self) -> list[str]:
        return CLASS_NAMES

    @property
    def targets(self) -> list[int]:
        return [label for _, label in self.samples]


def _splits_path(root: Path, split: str, image_size: int) -> Path:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    return SPLITS_DIR / f"{split}_size{image_size}_indices.npy"


def build_loaders(
    cfg: dict,
    train_transform: Callable,
    eval_transform: Callable,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders from config dict.

    Indices are stored on first run so every experiment uses the same samples.
    """
    root = cfg["dataset"]["root"]
    image_size = cfg["augmentation"]["image_size"]
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["dataset"].get("num_workers", 4)
    pin_memory = cfg["dataset"].get("pin_memory", True)

    train_ds = DeFungiDataset(root, "train", transform=train_transform)
    val_ds = DeFungiDataset(root, "val", transform=eval_transform)
    test_ds = DeFungiDataset(root, "test", transform=eval_transform)

    _save_split_meta(root, image_size, train_ds, val_ds, test_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def _save_split_meta(
    root: str | Path,
    image_size: int,
    train_ds: DeFungiDataset,
    val_ds: DeFungiDataset,
    test_ds: DeFungiDataset,
) -> None:
    """Persist per-split file paths and labels as JSON for full reproducibility."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = SPLITS_DIR / f"split_meta_size{image_size}.json"
    if meta_path.exists():
        return
    meta: dict = {}
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        meta[name] = {
            "n": len(ds),
            "paths": [str(p) for p, _ in ds.samples],
            "labels": [int(l) for _, l in ds.samples],
        }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Split metadata saved to %s", meta_path)

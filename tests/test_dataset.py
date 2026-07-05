"""Unit tests for the DeFungi dataset and transform pipeline.

Tests that do NOT require the actual dataset files:
  - Transform pipeline output shapes and value ranges
  - MixUp and CutMix output validity
  - Soft cross-entropy loss with soft targets

Tests that DO require the dataset (skipped if FUNGI_DATA env is not set):
  - DeFungiDataset loading and length
  - DataLoader batch shapes
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from datasets.transforms import build_transforms
from datasets.mixup import (
    mixup_data,
    cutmix_data,
    apply_batch_augmentation,
    soft_cross_entropy,
)

NUM_CLASSES = 5
BATCH = 8
IMAGE_SIZE = 224


def make_random_pil(size: int = 224) -> Image.Image:
    arr = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class TestTransforms:
    def test_train_transform_output_shape(self):
        tf = build_transforms(image_size=IMAGE_SIZE, is_training=True)
        img = make_random_pil(IMAGE_SIZE)
        tensor = tf(img)
        assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE), f"Unexpected shape: {tensor.shape}"

    def test_eval_transform_output_shape(self):
        tf = build_transforms(image_size=IMAGE_SIZE, is_training=False)
        img = make_random_pil(300)  # larger source image
        tensor = tf(img)
        assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE), f"Unexpected shape: {tensor.shape}"

    def test_transform_384(self):
        tf = build_transforms(image_size=384, is_training=True)
        img = make_random_pil(500)
        tensor = tf(img)
        assert tensor.shape == (3, 384, 384)

    def test_output_dtype_float(self):
        tf = build_transforms(image_size=IMAGE_SIZE, is_training=False)
        tensor = tf(make_random_pil())
        assert tensor.dtype == torch.float32

    def test_normalization_applied(self):
        """After normalization, mean should not be exactly 0.5 (ImageNet mean applied)."""
        tf = build_transforms(image_size=IMAGE_SIZE, is_training=False)
        # Uniform grey image → after ImageNet normalization values shift
        grey = Image.fromarray(np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 128, dtype=np.uint8))
        tensor = tf(grey)
        assert not torch.allclose(tensor.mean(), torch.tensor(0.5), atol=0.01)


class TestMixUp:
    def make_batch(self):
        x = torch.rand(BATCH, 3, IMAGE_SIZE, IMAGE_SIZE)
        y = torch.randint(0, NUM_CLASSES, (BATCH,))
        return x, y

    def test_mixup_output_shapes(self):
        x, y = self.make_batch()
        mixed_x, mixed_y = mixup_data(x, y, alpha=0.8, num_classes=NUM_CLASSES)
        assert mixed_x.shape == x.shape
        assert mixed_y.shape == (BATCH, NUM_CLASSES)

    def test_mixup_soft_targets_sum_to_one(self):
        x, y = self.make_batch()
        _, mixed_y = mixup_data(x, y, alpha=0.8, num_classes=NUM_CLASSES)
        sums = mixed_y.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5)

    def test_cutmix_output_shapes(self):
        x, y = self.make_batch()
        mixed_x, mixed_y = cutmix_data(x, y, alpha=1.0, num_classes=NUM_CLASSES)
        assert mixed_x.shape == x.shape
        assert mixed_y.shape == (BATCH, NUM_CLASSES)

    def test_apply_batch_augmentation_with_both(self):
        x, y = self.make_batch()
        mixed_x, mixed_y = apply_batch_augmentation(
            x, y, use_mixup=True, use_cutmix=True, num_classes=NUM_CLASSES
        )
        assert mixed_x.shape == x.shape
        sums = mixed_y.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5)

    def test_soft_cross_entropy_shape(self):
        logits = torch.randn(BATCH, NUM_CLASSES)
        soft_targets = F.softmax(torch.randn(BATCH, NUM_CLASSES), dim=-1)
        loss = soft_cross_entropy(logits, soft_targets, label_smoothing=0.1)
        assert loss.ndim == 0  # scalar
        assert loss > 0

    def test_soft_cross_entropy_matches_hard_when_one_hot(self):
        """With one-hot targets and no smoothing, soft CE should match F.cross_entropy."""
        logits = torch.randn(BATCH, NUM_CLASSES)
        labels = torch.randint(0, NUM_CLASSES, (BATCH,))
        one_hot = F.one_hot(labels, NUM_CLASSES).float()

        soft_loss = soft_cross_entropy(logits, one_hot, label_smoothing=0.0)
        hard_loss = F.cross_entropy(logits, labels)
        assert torch.allclose(soft_loss, hard_loss, atol=1e-5)


@pytest.mark.skipif(
    not os.environ.get("FUNGI_DATA"),
    reason="Set FUNGI_DATA=/path/to/defungi to run dataset integration tests",
)
class TestDeFungiDataset:
    def test_dataset_loads(self):
        from datasets.defungi import DeFungiDataset
        from datasets.transforms import build_transforms
        tf = build_transforms(is_training=False)
        ds = DeFungiDataset(os.environ["FUNGI_DATA"], split="test", transform=tf)
        assert len(ds) > 0
        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert 0 <= label < NUM_CLASSES

    def test_all_splits_non_empty(self):
        from datasets.defungi import DeFungiDataset
        from datasets.transforms import build_transforms
        tf = build_transforms(is_training=False)
        for split in ("train", "val", "test"):
            ds = DeFungiDataset(os.environ["FUNGI_DATA"], split=split, transform=tf)
            assert len(ds) > 0, f"Split '{split}' is empty"

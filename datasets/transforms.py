"""Image augmentation pipelines.

Training augmentation follows DeiT (Touvron et al. 2021):
  RandomResizedCrop → RandomHorizontalFlip → ColorJitter → ToTensor → Normalize
  + optional Random Erasing (applied after normalize)

MixUp and CutMix operate on batches in the collate function (see mixup.py).
"""

from __future__ import annotations

from typing import Optional

import torchvision.transforms as T

# ImageNet statistics — standard for pretrained ViT/ResNet/EfficientNet/Swin
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(
    image_size: int = 224,
    use_random_erasing: bool = True,
    random_erasing_prob: float = 0.25,
    is_training: bool = True,
) -> T.Compose:
    """Return a torchvision transform pipeline.

    Args:
        image_size: Target square crop size in pixels.
        use_random_erasing: Apply Random Erasing during training.
        random_erasing_prob: Probability of applying Random Erasing.
        is_training: Build training or evaluation transforms.
    """
    if is_training:
        ops: list = [
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        if use_random_erasing:
            ops.append(
                T.RandomErasing(
                    p=random_erasing_prob,
                    scale=(0.02, 0.2),
                    ratio=(0.3, 3.3),
                    value="random",
                )
            )
        return T.Compose(ops)
    else:
        eval_size = int(image_size * 1.143)  # resize then centre crop (standard)
        return T.Compose([
            T.Resize(eval_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def build_transforms_for_config(cfg: dict, is_training: bool) -> T.Compose:
    """Convenience wrapper: build transforms directly from a config dict."""
    aug = cfg.get("augmentation", {})
    return build_transforms(
        image_size=aug.get("image_size", 224),
        use_random_erasing=aug.get("use_random_erasing", True) and is_training,
        random_erasing_prob=aug.get("random_erasing_prob", 0.25),
        is_training=is_training,
    )

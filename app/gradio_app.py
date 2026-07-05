#!/usr/bin/env python3
"""Gradio web application for DeFungi Version 2.

Features:
  - Model selector: ViT-B/16 (224px), ViT-B/16 (384px), ResNet-50,
                    EfficientNet-B3, Swin-Tiny
  - Fusion strategy selector (when two ViT checkpoints are available):
    S1 Equal Weight, S2 Accuracy-Weighted, S3 Confidence-Guided, S4 Learnable
  - Attention Rollout heatmap (ViT models)
  - Grad-CAM heatmap (CNN models)
  - Inference time display
  - Top-5 prediction confidence bar chart

Usage:
    python app/gradio_app.py --ckpt-224 outputs/checkpoints/vit_b16_224_best.pt

Optional additional checkpoints:
    --ckpt-384 outputs/checkpoints/vit_b16_384_best.pt
    --ckpt-resnet outputs/checkpoints/resnet50_best.pt
    etc.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
import gradio as gr
from PIL import Image

from datasets.defungi import CLASS_NAMES, SPECIES_NAMES
from datasets.transforms import build_transforms
from trainer.checkpoint import CheckpointManager
from explainability.visualizer import overlay_heatmap, denormalize
from fusion.factory import build_fusion_strategy, STRATEGY_LABELS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Global model registry ─────────────────────────────────────────────────────
_MODELS: dict[str, torch.nn.Module] = {}
_CHECKPOINTS: dict[str, Path] = {}


def load_model_if_needed(model_key: str, checkpoint_path: Path) -> torch.nn.Module:
    if model_key in _MODELS:
        return _MODELS[model_key]

    from models.vit import ViTClassifier
    from models.resnet import ResNetClassifier
    from models.efficientnet import EfficientNetClassifier
    from models.swin import SwinClassifier

    arch_map = {
        "vit_224": (ViTClassifier, {"backbone": "vit_base_patch16_224", "num_classes": 5}),
        "vit_384": (ViTClassifier, {"backbone": "vit_base_patch16_384", "num_classes": 5}),
        "resnet":  (ResNetClassifier, {"backbone": "resnet50", "num_classes": 5}),
        "efficientnet": (EfficientNetClassifier, {"backbone": "efficientnet_b3", "num_classes": 5}),
        "swin":    (SwinClassifier, {"backbone": "swin_tiny_patch4_window7_224", "num_classes": 5}),
    }

    cls, kwargs = arch_map[model_key]
    model = cls(**kwargs)
    CheckpointManager.load(checkpoint_path, model, device=DEVICE)
    model.eval()
    _MODELS[model_key] = model.to(DEVICE)
    return _MODELS[model_key]


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_single(
    image: Image.Image,
    model_key: str,
    image_size: int,
    generate_xai: bool = True,
) -> tuple[dict[str, float], np.ndarray, float]:
    """Run inference on a single image.

    Returns:
        probs: Dict of class_name → probability.
        heatmap: XAI heatmap [H, W] in [0, 1].
        latency_ms: Inference time in milliseconds.
    """
    tf = build_transforms(image_size=image_size, is_training=False)
    tensor = tf(image).unsqueeze(0).to(DEVICE)

    model = _MODELS[model_key]
    is_vit = "vit" in model_key

    t0 = time.perf_counter()

    if generate_xai and is_vit:
        from explainability.attention_rollout import AttentionRollout
        rollout = AttentionRollout(model)
        with torch.no_grad():
            logits = model(tensor)
        heatmap = rollout(tensor)
    elif generate_xai and not is_vit:
        from explainability.gradcam import GradCAM
        gcam = GradCAM(model)
        logits = model(tensor)   # needs grad
        heatmap = gcam(tensor, image_size=(image_size, image_size))
    else:
        with torch.no_grad():
            logits = model(tensor)
        heatmap = np.zeros((14, 14))

    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000

    probs = F.softmax(logits.detach(), dim=-1)[0].cpu().numpy()
    prob_dict = {
        f"{cls} — {SPECIES_NAMES.get(cls, cls)}": float(probs[i])
        for i, cls in enumerate(CLASS_NAMES)
    }
    return prob_dict, heatmap, latency_ms


def run_fusion(
    image: Image.Image,
    strategy: str,
    acc_224: float = 0.85,
    acc_384: float = 0.85,
) -> tuple[dict[str, float], float]:
    """Run ACMF fusion with specified strategy.

    Returns:
        fused_probs: Dict of class_name → fused probability.
        latency_ms: Total fusion inference time.
    """
    tf_224 = build_transforms(image_size=224, is_training=False)
    tf_384 = build_transforms(image_size=384, is_training=False)

    t224 = tf_224(image).unsqueeze(0).to(DEVICE)
    t384 = tf_384(image).unsqueeze(0).to(DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits_224 = _MODELS["vit_224"](t224)
        logits_384 = _MODELS["vit_384"](t384)

    probs_224 = F.softmax(logits_224, dim=-1)
    probs_384 = F.softmax(logits_384, dim=-1)

    kwargs = {}
    if strategy == "accuracy_weighted":
        kwargs = {"acc_224": acc_224, "acc_384": acc_384}

    fusion = build_fusion_strategy(strategy, num_classes=5, **kwargs)
    fused = fusion(probs_224, probs_384)[0].cpu().numpy()
    t1 = time.perf_counter()

    prob_dict = {
        f"{cls} — {SPECIES_NAMES.get(cls, cls)}": float(fused[i])
        for i, cls in enumerate(CLASS_NAMES)
    }
    return prob_dict, (t1 - t0) * 1000


# ── Gradio interface ──────────────────────────────────────────────────────────

def build_interface(available_models: list[str]) -> gr.Blocks:
    has_fusion = "vit_224" in available_models and "vit_384" in available_models

    with gr.Blocks(title="DeFungi V2 — Fungal Species Classifier", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# DeFungi V2 — Explainable Fungal Species Classification\n"
            "Upload a microscopic fungal image to classify it into one of 5 species. "
            "Optionally enable explainability (Attention Rollout / Grad-CAM) and multi-scale fusion (ACMF)."
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Microscopic Image")
                model_choice = gr.Dropdown(
                    choices=available_models,
                    value=available_models[0],
                    label="Model",
                )
                show_xai = gr.Checkbox(value=True, label="Generate XAI Heatmap")

                if has_fusion:
                    with gr.Accordion("Multi-Scale Fusion (ACMF)", open=False):
                        use_fusion = gr.Checkbox(value=False, label="Enable ACMF Fusion")
                        fusion_strategy = gr.Dropdown(
                            choices=list(STRATEGY_LABELS.keys()),
                            value="confidence_guided",
                            label="Fusion Strategy",
                        )
                        gr.Markdown(
                            "**S1** Equal weight (V1 baseline) | "
                            "**S2** Accuracy-weighted | "
                            "**S3** Confidence-guided (ACMF) | "
                            "**S4** Adaptive learnable"
                        )

                run_btn = gr.Button("Classify", variant="primary")

            with gr.Column(scale=1):
                output_probs = gr.Label(num_top_classes=5, label="Prediction Probabilities")
                output_heatmap = gr.Image(label="XAI Heatmap Overlay", type="pil")
                output_latency = gr.Textbox(label="Inference Time")

        def classify(img, model_key, xai, *fusion_args):
            if img is None:
                return {}, None, "No image provided"

            # Determine image size from model key
            size = 384 if "384" in model_key else 300 if "efficient" in model_key else 224

            probs, heatmap, lat = predict_single(img, model_key, size, generate_xai=xai)

            # Apply fusion if requested
            if has_fusion and len(fusion_args) == 2:
                use_f, strategy = fusion_args
                if use_f and "vit_224" in available_models and "vit_384" in available_models:
                    probs, lat = run_fusion(img, strategy)

            # Overlay heatmap
            img_np = np.array(img.convert("RGB").resize((size, size)))
            import cv2
            hmap_resized = np.array(
                Image.fromarray((heatmap * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
            ) / 255.0
            overlay = overlay_heatmap(img_np, hmap_resized, alpha=0.5)

            return probs, overlay, f"{lat:.1f} ms"

        inputs = [input_image, model_choice, show_xai]
        if has_fusion:
            inputs += [use_fusion, fusion_strategy]

        run_btn.click(fn=classify, inputs=inputs, outputs=[output_probs, output_heatmap, output_latency])

        gr.Examples(
            examples=[],
            inputs=[input_image],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch DeFungi V2 Gradio app")
    parser.add_argument("--ckpt-224", type=Path, default=None, help="ViT-B/16 224px checkpoint")
    parser.add_argument("--ckpt-384", type=Path, default=None, help="ViT-B/16 384px checkpoint")
    parser.add_argument("--ckpt-resnet", type=Path, default=None, help="ResNet-50 checkpoint")
    parser.add_argument("--ckpt-efficientnet", type=Path, default=None, help="EfficientNet-B3 checkpoint")
    parser.add_argument("--ckpt-swin", type=Path, default=None, help="Swin-Tiny checkpoint")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public Gradio share link")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ckpt_map = {
        "vit_224": args.ckpt_224,
        "vit_384": args.ckpt_384,
        "resnet": args.ckpt_resnet,
        "efficientnet": args.ckpt_efficientnet,
        "swin": args.ckpt_swin,
    }

    available: list[str] = []
    for key, path in ckpt_map.items():
        if path and path.exists():
            load_model_if_needed(key, path)
            _CHECKPOINTS[key] = path
            available.append(key)

    if not available:
        print("No checkpoints provided. Train a model first:")
        print("  python scripts/train.py --config configs/vit_224.yaml --data-root /path/to/defungi")
        sys.exit(1)

    print(f"Loaded models: {available}")
    demo = build_interface(available)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

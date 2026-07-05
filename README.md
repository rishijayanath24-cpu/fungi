# Adaptive Confidence-Guided Multi-Scale Vision Transformer Framework for Explainable Microscopic Fungal Species Classification

**Version 2 — IEEE Conference Submission**  
Rishi Jayanath A · URK23CS1105 · Karunya Institute of Technology and Sciences

---

## Overview

This repository implements an IEEE-quality research framework for automated microscopic fungal species classification on the [DeFungi](https://www.kaggle.com/datasets/defungi/defungi) dataset (5 classes, 6,801 images).

The primary research contribution is **Adaptive Confidence-Guided Multi-Scale Fusion (ACMF)** — a novel framework that dynamically weights predictions from a 224px and 384px ViT-B/16 model based on per-image prediction confidence (entropy), with a learnable variant trained end-to-end.

## Dataset

**DeFungi** — Microscopic Fungal Image Dataset

| Class | Species | Train | Val | Test |
|:-----:|:--------|------:|----:|-----:|
| H1 | *Candida albicans* | — | — | — |
| H2 | *Aspergillus niger* | — | — | — |
| H3 | *Trichophyton rubrum* | — | — | — |
| H5 | *Trichophyton mentagrophytes* | — | — | — |
| H6 | *Epidermophyton floccosum* | — | — | — |
| **Total** | | **5,000** | **899** | **902** |

Download: `import kagglehub; kagglehub.dataset_download("defungi/defungi")`

## Models

| Model | Backbone | Image Size | Params |
|:------|:---------|:----------:|-------:|
| ViT-B/16 (primary) | `vit_base_patch16_224` | 224 px | 86.6 M |
| ViT-B/16 (high-res) | `vit_base_patch16_384` | 384 px | 86.9 M |
| ResNet-50 | `resnet50` | 224 px | 25.6 M |
| EfficientNet-B3 | `efficientnet_b3` | 300 px | 12.2 M |
| Swin-Tiny | `swin_tiny_patch4_window7_224` | 224 px | 28.3 M |

## ACMF Fusion Strategies

| ID | Strategy | Description |
|:--:|:---------|:------------|
| S1 | Equal Weight | Simple average (V1 baseline) |
| S2 | Accuracy Weighted | Weights proportional to validation accuracy |
| **S3** | **Confidence Guided** | **Inverse entropy per image — no training required** |
| **S4** | **Adaptive Learnable** | **MLP head trained end-to-end** |

S3 and S4 are the novel contributions introduced in Version 2.

## Setup

```bash
# Clone and install
cd D:\Fungi\fungi-ai
pip install -r requirements.txt

# Download dataset
python -c "import kagglehub; path = kagglehub.dataset_download('defungi/defungi'); print(path)"
# Set the printed path as your --data-root argument
```

## Training

```bash
# Train primary ViT-B/16 model
python scripts/train.py --config configs/vit_224.yaml --data-root /path/to/defungi

# Train all 4 baselines with a single command
python scripts/run_baselines.py --data-root /path/to/defungi

# Run full ablation study (6 configurations)
python scripts/run_ablation.py --base-config configs/vit_224.yaml --data-root /path/to/defungi
```

## Evaluation

```bash
# Evaluate a checkpoint
python scripts/evaluate.py \
  --config configs/vit_224.yaml \
  --checkpoint outputs/checkpoints/vit_b16_224_best.pt \
  --data-root /path/to/defungi
```

## Generate Paper Materials

After training all models and the ablation study:

```bash
python scripts/generate_paper.py
```

Outputs written to `paper/`:
- `paper/tables/baseline_comparison.tex` — LaTeX baseline table
- `paper/tables/fusion_comparison.tex` — LaTeX ACMF table
- `paper/tables/ablation_table.tex` — LaTeX ablation table
- `paper/figures/` — 300 DPI PNG figures (ROC, PR, confusion matrices, comparison bars)
- `paper/summaries/results_summary.md` — Markdown results summary

## Gradio Web App

```bash
python app/gradio_app.py \
  --ckpt-224 outputs/checkpoints/vit_b16_224_best.pt \
  --ckpt-384 outputs/checkpoints/vit_b16_384_best.pt
```

## Tests

```bash
pytest tests/ -v
```

Dataset integration tests are skipped unless `FUNGI_DATA=/path/to/defungi` is set.

## Repository Structure

```
fungi-ai/
├── configs/        YAML experiment configurations
├── datasets/       DeFungi dataset class, transforms, MixUp/CutMix
├── models/         ViT, ResNet, EfficientNet, Swin, builder factory
├── fusion/         ACMF: 4 fusion strategies (S1–S4)
├── trainer/        Training loop, AMP, scheduler, checkpointing, seed control
├── evaluation/     Metrics, evaluator, inference benchmarking, ablation runner
├── explainability/ Attention Rollout (ViT), Grad-CAM (CNN), visualizer
├── visualization/  Publication figures (curves, confusion matrix, ROC, PR, comparison)
├── scripts/        train.py, evaluate.py, run_ablation.py, run_baselines.py, generate_paper.py
├── app/            Gradio web interface
├── tests/          Unit tests (fusion, transforms, metrics)
├── paper/          Auto-generated paper materials (LaTeX, 300 DPI figures)
└── outputs/        Training outputs (checkpoints, logs, results) — git-ignored
```

## Key Design Principles

- **No fabricated metrics** — every number in a results table comes from a completed training run and evaluation pipeline.
- **Reproducibility** — single `seed` value in `configs/base.yaml` fixes all random sources.
- **Publication-ready outputs** — every experiment writes LaTeX tables, 300 DPI figures, and CSV files automatically.
- **No hardcoded paths** — all paths and hyperparameters live in YAML configs.

## Citation

> Rishi Jayanath A. "Adaptive Confidence-Guided Multi-Scale Vision Transformer Framework for Explainable Microscopic Fungal Species Classification." *Proceedings of [venue], 2025.*

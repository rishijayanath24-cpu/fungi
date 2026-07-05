"""
Train all models sequentially, then generate paper outputs.

Run once and leave overnight:
    python scripts/train_all.py --data-root "D:\\" --device cuda

Order:
    1. ViT-B/16 @ 224px   (resume from _best.pt if it exists)
    2. ViT-B/16 @ 384px
    3. ResNet-50
    4. EfficientNet-B3
    5. Swin-Tiny
    6. generate_paper.py   (collects all results, writes LaTeX + figures)
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
CKPT_DIR = ROOT / "outputs" / "checkpoints"
RESULTS_DIR = ROOT / "outputs" / "results"


def run(cmd: list[str], label: str) -> bool:
    log.info("=" * 60)
    log.info("START: %s", label)
    log.info("CMD  : %s", " ".join(cmd))
    log.info("=" * 60)
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)
    if result.returncode == 0:
        log.info("DONE : %s  (elapsed %dm %ds)", label, mins, secs)
        return True
    else:
        log.error("FAIL : %s  exit_code=%d", label, result.returncode)
        return False


def already_done(experiment: str) -> bool:
    """Return True if a results JSON exists for this experiment."""
    result_path = RESULTS_DIR / f"{experiment}_results.json"
    if result_path.exists():
        try:
            data = json.loads(result_path.read_text())
            acc = data.get("val_accuracy", 0)
            log.info("SKIP : %s — results already exist (val_acc=%.4f)", experiment, acc)
            return True
        except Exception:
            pass
    return False


def find_best_checkpoint(experiment: str) -> Path | None:
    best = CKPT_DIR / f"{experiment}_best.pt"
    return best if best.exists() else None


def train(config: str, experiment: str, data_root: str, device: str, resume: bool = True) -> bool:
    if already_done(experiment):
        return True

    cmd = [
        sys.executable, str(ROOT / "scripts" / "train.py"),
        "--config", str(ROOT / "configs" / config),
        "--data-root", data_root,
        "--device", device,
    ]

    if resume:
        ckpt = find_best_checkpoint(experiment)
        if ckpt:
            log.info("Resuming %s from %s", experiment, ckpt.name)
            cmd += ["--resume", str(ckpt)]

    return run(cmd, experiment)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-paper", action="store_true",
                    help="Skip generate_paper.py at the end")
    args = ap.parse_args()

    steps = [
        ("vit_224.yaml",         "vit_b16_224"),
        ("vit_384.yaml",         "vit_b16_384"),
        ("resnet50.yaml",        "resnet50"),
        ("efficientnet_b3.yaml", "efficientnet_b3"),
        ("swin_tiny.yaml",       "swin_tiny"),
    ]

    failed = []
    for config, experiment in steps:
        ok = train(config, experiment, args.data_root, args.device, resume=True)
        if not ok:
            failed.append(experiment)
            log.warning("Continuing with remaining models despite failure in %s", experiment)

    log.info("=" * 60)
    log.info("Training complete.  Failed: %s", failed if failed else "none")
    log.info("=" * 60)

    if not args.skip_paper:
        # ACMF fusion evaluation (all 4 strategies: S1-S4)
        # Requires vit_b16_224 and vit_b16_384 checkpoints to exist.
        vit224_ckpt = ROOT / "outputs" / "checkpoints" / "vit_b16_224_best.pt"
        vit384_ckpt = ROOT / "outputs" / "checkpoints" / "vit_b16_384_best.pt"
        if vit224_ckpt.exists() and vit384_ckpt.exists():
            log.info("Running ACMF fusion evaluation (S1–S4) ...")
            ok = run(
                [sys.executable, str(ROOT / "scripts" / "run_fusion.py"),
                 "--data-root", args.data_root,
                 "--device", args.device],
                "run_fusion"
            )
            if not ok:
                log.error("Fusion evaluation failed — check logs above.")
        else:
            missing = []
            if not vit224_ckpt.exists(): missing.append("vit_b16_224_best.pt")
            if not vit384_ckpt.exists(): missing.append("vit_b16_384_best.pt")
            log.warning("Skipping fusion: missing checkpoints: %s", missing)

        # Compile all results into paper tables and figures
        log.info("Generating paper outputs ...")
        ok = run(
            [sys.executable, str(ROOT / "scripts" / "generate_paper.py")],
            "generate_paper"
        )
        if not ok:
            log.error("Paper generation failed — check outputs/results/ manually.")

    log.info("All done. Results in: %s", RESULTS_DIR)


if __name__ == "__main__":
    main()

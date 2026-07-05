"""Main training loop for all models.

Supports:
  - AMP (automatic mixed precision)
  - MixUp + CutMix batch augmentation
  - Gradient clipping
  - Label smoothing (via soft cross-entropy)
  - Warmup + cosine LR scheduling
  - Early stopping
  - TensorBoard logging
  - Checkpoint save/resume
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.mixup import apply_batch_augmentation, soft_cross_entropy
from trainer.checkpoint import CheckpointManager
from trainer.scheduler import build_scheduler, get_current_lr

logger = logging.getLogger(__name__)


class Trainer:
    """Encapsulates the complete training and validation loop."""

    def __init__(
        self,
        model: nn.Module,
        cfg: dict,
        experiment_name: str,
        device: str | torch.device = "cuda",
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.experiment_name = experiment_name
        self.device = torch.device(device)

        tr = cfg["training"]
        aug = cfg.get("augmentation", {})

        self.epochs = tr["epochs"]
        self.grad_clip = tr.get("grad_clip", 1.0)
        self.label_smoothing = tr.get("label_smoothing", 0.1)
        self.use_amp = tr.get("amp", True)
        self.use_mixup = aug.get("use_mixup", True)
        self.use_cutmix = aug.get("use_cutmix", True)
        self.mixup_alpha = aug.get("mixup_alpha", 0.8)
        self.cutmix_alpha = aug.get("cutmix_alpha", 1.0)
        self.num_classes = cfg["dataset"].get("num_classes", 5)
        self.early_stopping_patience = tr.get("early_stopping_patience", 10)

        self.optimizer = AdamW(
            model.parameters(),
            lr=tr.get("lr", 1e-4),
            weight_decay=tr.get("weight_decay", 1e-2),
        )
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        # Scheduler is built once DataLoaders are known (need steps_per_epoch)
        self._scheduler = None

        # Output directories
        output_dir = Path(cfg["logging"].get("output_dir", "outputs"))
        log_dir = output_dir / "logs" / experiment_name
        ckpt_dir = cfg["checkpoint"].get("save_dir", str(output_dir / "checkpoints"))

        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.ckpt_manager = CheckpointManager(
            save_dir=ckpt_dir,
            experiment_name=experiment_name,
            monitor=cfg["checkpoint"].get("monitor", "val_f1"),
            mode=cfg["checkpoint"].get("mode", "max"),
            save_top_k=cfg["checkpoint"].get("save_top_k", 3),
        )

        self._best_val_metric: Optional[float] = None
        self._epochs_no_improve = 0
        self._global_step = 0

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_from: Optional[str] = None,
    ) -> dict:
        """Train for self.epochs epochs, returning the best metrics dict."""
        start_epoch = 0

        self._scheduler = build_scheduler(
            optimizer=self.optimizer,
            warmup_epochs=self.cfg["training"].get("warmup_epochs", 5),
            total_epochs=self.epochs,
            steps_per_epoch=len(train_loader),
        )

        if resume_from:
            state = CheckpointManager.load(
                resume_from,
                self.model,
                self.optimizer,
                self._scheduler,
                self.scaler,
                device=str(self.device),
            )
            start_epoch = state["epoch"] + 1
            self._global_step = start_epoch * len(train_loader)
            logger.info("Resumed from epoch %d", state["epoch"])

        history: list[dict] = []

        for epoch in range(start_epoch, self.epochs):
            train_metrics = self._train_epoch(train_loader, epoch)
            val_metrics = self._val_epoch(val_loader, epoch)

            metrics = {**{f"train_{k}": v for k, v in train_metrics.items()},
                       **{f"val_{k}": v for k, v in val_metrics.items()},
                       "epoch": epoch}
            history.append(metrics)

            self._log_metrics(metrics, epoch)
            self.ckpt_manager.save(
                self.model, self.optimizer, self._scheduler, epoch, metrics, self.scaler
            )

            if self._should_stop_early(metrics):
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

        self.writer.close()
        best = self._select_best(history)
        self.ckpt_manager.save_metadata(best)
        return best

    # ── Training epoch ────────────────────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.epochs} [train]", leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            images, soft_labels = apply_batch_augmentation(
                images, labels,
                use_mixup=self.use_mixup,
                use_cutmix=self.use_cutmix,
                mixup_alpha=self.mixup_alpha,
                cutmix_alpha=self.cutmix_alpha,
                num_classes=self.num_classes,
            )

            self.optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=self.use_amp):
                logits = self.model(images)
                loss = soft_cross_entropy(logits, soft_labels, self.label_smoothing)

            self.scaler.scale(loss).backward()

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self._scheduler.step()

            # Track accuracy on hard labels (for monitoring)
            preds = logits.detach().argmax(dim=-1)
            hard_labels = labels  # original (not mixed)
            correct += (preds == hard_labels).sum().item()
            total += hard_labels.size(0)
            total_loss += loss.item()

            self._global_step += 1
            if batch_idx % self.cfg["logging"].get("log_interval", 10) == 0:
                self.writer.add_scalar("train/loss_step", loss.item(), self._global_step)
                self.writer.add_scalar("train/lr", get_current_lr(self.optimizer), self._global_step)

            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{get_current_lr(self.optimizer):.2e}")

        n_batches = len(loader)
        return {
            "loss": total_loss / n_batches,
            "accuracy": correct / total,
        }

    # ── Validation epoch ──────────────────────────────────────────────────────

    def _val_epoch(self, loader: DataLoader, epoch: int) -> dict:
        from evaluation.metrics import compute_classification_metrics

        self.model.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []
        total_loss = 0.0

        with torch.no_grad():
            for images, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{self.epochs} [val]", leave=False):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with autocast("cuda", enabled=self.use_amp):
                    logits = self.model(images)
                    soft_labels = torch.nn.functional.one_hot(labels, self.num_classes).float()
                    loss = soft_cross_entropy(logits, soft_labels, label_smoothing=0.0)

                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                total_loss += loss.item()

        metrics = compute_classification_metrics(all_labels, all_preds, self.num_classes)
        metrics["loss"] = total_loss / len(loader)
        return metrics

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log_metrics(self, metrics: dict, epoch: int) -> None:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, epoch)

        logger.info(
            "Epoch %03d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f val_f1=%.4f",
            epoch,
            metrics.get("train_loss", 0),
            metrics.get("train_accuracy", 0),
            metrics.get("val_loss", 0),
            metrics.get("val_accuracy", 0),
            metrics.get("val_f1_macro", 0),
        )

    def _should_stop_early(self, metrics: dict) -> bool:
        monitor = self.cfg["checkpoint"].get("monitor", "val_f1_macro")
        mode = self.cfg["checkpoint"].get("mode", "max")
        score = metrics.get(monitor)
        if score is None:
            return False

        improved = (
            self._best_val_metric is None
            or (mode == "max" and score > self._best_val_metric)
            or (mode == "min" and score < self._best_val_metric)
        )
        if improved:
            self._best_val_metric = score
            self._epochs_no_improve = 0
        else:
            self._epochs_no_improve += 1

        return self._epochs_no_improve >= self.early_stopping_patience

    @staticmethod
    def _select_best(history: list[dict]) -> dict:
        """Return the epoch with the highest val_f1_macro."""
        return max(history, key=lambda m: m.get("val_f1_macro", 0.0))

"""Checkpoint save/load/resume utilities.

Saves the top-K checkpoints ranked by a monitored metric.
Checkpoint files contain the full training state so training can
resume exactly from any saved step.
"""

from __future__ import annotations

import heapq
import json
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages top-K checkpoint saving with resume support.

    Args:
        save_dir: Directory to write checkpoint files.
        experiment_name: Used as filename prefix.
        monitor: Metric name to rank checkpoints by (e.g. 'val_f1').
        mode: 'max' or 'min' depending on the metric.
        save_top_k: Maximum number of checkpoints to retain.
    """

    def __init__(
        self,
        save_dir: str | Path,
        experiment_name: str,
        monitor: str = "val_f1",
        mode: str = "max",
        save_top_k: int = 3,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k

        # Min-heap: (score, epoch, path) — flip sign for max-mode
        self._heap: list[tuple[float, int, Path]] = []
        self.best_score: Optional[float] = None
        self.best_path: Optional[Path] = None

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler,
        epoch: int,
        metrics: dict[str, float],
        scaler=None,
    ) -> Optional[Path]:
        """Save a checkpoint if the monitored metric improved.

        Returns the path written, or None if the checkpoint was not saved.
        """
        score = metrics.get(self.monitor)
        if score is None:
            logger.warning("Monitor metric '%s' not found in metrics dict.", self.monitor)
            return None

        # For a min-heap we store the raw score in max-mode so that heappop()
        # removes the lowest score (= the worst checkpoint to prune).
        # In min-mode we negate so the highest actual score (worst) is popped.
        heap_score = score if self.mode == "max" else -score
        path = self.save_dir / f"{self.experiment_name}_epoch{epoch:03d}.pt"

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "metrics": metrics,
            "monitor": self.monitor,
        }
        torch.save(state, path)

        heapq.heappush(self._heap, (heap_score, epoch, path))

        # Track best
        improved = (
            self.best_score is None
            or (self.mode == "max" and score > self.best_score)
            or (self.mode == "min" and score < self.best_score)
        )
        if improved:
            self.best_score = score
            self.best_path = path
            logger.info(
                "New best checkpoint: epoch=%d  %s=%.4f  -> %s",
                epoch, self.monitor, score, path.name
            )

        # Prune worst checkpoints over the top-K limit
        while len(self._heap) > self.save_top_k:
            _, _, old_path = heapq.heappop(self._heap)
            if old_path.exists():
                old_path.unlink()
                logger.debug("Removed old checkpoint: %s", old_path.name)

        # Update _best.pt only when this epoch is the new best.
        # Write to a .tmp file first then os.replace() to avoid the Windows
        # file-lock PermissionError that hits unlink() on a just-written file.
        if improved:
            import os as _os
            best_final = self.save_dir / f"{self.experiment_name}_best.pt"
            best_tmp   = self.save_dir / f"{self.experiment_name}_best.tmp"
            torch.save(state, best_tmp)
            try:
                _os.replace(str(best_tmp), str(best_final))
            except PermissionError:
                best_tmp.unlink(missing_ok=True)
                logger.warning("Could not update _best.pt (file locked); numbered checkpoint still saved.")

        return path

    @staticmethod
    def load(
        checkpoint_path: str | Path,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler=None,
        scaler=None,
        device: str = "cuda",
    ) -> dict:
        """Load a checkpoint and restore state into model (and optionally optimizer).

        Returns:
            The full checkpoint dict (includes 'epoch' and 'metrics').
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])

        if optimizer and state.get("optimizer_state_dict"):
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler and state.get("scheduler_state_dict"):
            scheduler.load_state_dict(state["scheduler_state_dict"])
        if scaler and state.get("scaler_state_dict"):
            scaler.load_state_dict(state["scaler_state_dict"])

        logger.info(
            "Checkpoint loaded: epoch=%d  metrics=%s",
            state["epoch"],
            {k: f"{v:.4f}" for k, v in state.get("metrics", {}).items()},
        )
        return state

    def save_metadata(self, metrics: dict) -> None:
        """Write a JSON summary of all tracked metrics to the checkpoint directory."""
        meta_path = self.save_dir / f"{self.experiment_name}_results.json"
        with open(meta_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Results metadata saved: %s", meta_path)

"""Lightweight checkpoint manager for coursework experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch


class Checkpointer:
    """Simple checkpoint saver that keeps only the most recent files."""

    def __init__(self, directory: Path, keep_last: int = 3) -> None:
        self.directory = Path(directory)
        self.keep_last = keep_last
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        checkpoint_path = self.directory / f"checkpoint_{epoch:04d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "metrics": metrics or {},
            },
            checkpoint_path,
        )

        if metrics:
            metrics_path = checkpoint_path.with_suffix(".json")
            metrics_path.write_text(json.dumps({"epoch": epoch, **metrics}, indent=2))

        self._prune_old_checkpoints()
        return checkpoint_path

    def _prune_old_checkpoints(self) -> None:
        checkpoints = sorted(self.directory.glob("checkpoint_*.pt"))
        for checkpoint in checkpoints[:-self.keep_last]:
            checkpoint.unlink(missing_ok=True)
            metrics_file = checkpoint.with_suffix(".json")
            metrics_file.unlink(missing_ok=True)


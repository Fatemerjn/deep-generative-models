"""Structured logging utilities."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict


class MetricLogger:
    """Minimal metric logger that aggregates running statistics."""

    def __init__(self) -> None:
        self.buffer: Dict[str, float] = defaultdict(float)
        self.counts: Dict[str, int] = defaultdict(int)

    def update(self, **metrics: float) -> None:
        for key, value in metrics.items():
            self.buffer[key] += float(value)
            self.counts[key] += 1

    def flush(self, epoch: int, step: int) -> None:
        averaged = self.averages()
        formatted = ", ".join(f"{key}: {value:.4f}" for key, value in averaged.items())
        print(f"[Epoch {epoch:03d} | Step {step:05d}] {formatted}")

    def reset(self) -> None:
        self.buffer.clear()
        self.counts.clear()

    def averages(self) -> Dict[str, float]:
        return {key: self.buffer[key] / max(self.counts[key], 1) for key in self.buffer}

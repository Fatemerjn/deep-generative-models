"""Base classes and helpers for generative models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
from torch import nn

from ..config import ModelConfig


class GenerativeModel(ABC, nn.Module):
    """Abstract base class for all generative models used in the course."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass required by ``nn.Module``."""

    @abstractmethod
    def loss(self, batch: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the training loss and any additional logging metrics."""

    def sample(self, num_samples: int) -> torch.Tensor:
        """Sample from the model's prior; override if the model uses a custom sampler."""

        z = torch.randn(num_samples, self.config.latent_dim, device=self.device)
        return self.decode(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Optionally implemented by subclasses."""

        raise NotImplementedError("encode is not implemented for this model.")

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Optionally implemented by subclasses."""

        raise NotImplementedError("decode is not implemented for this model.")

    @property
    def device(self) -> torch.device:
        """Return the current device of the module."""

        return next(self.parameters()).device


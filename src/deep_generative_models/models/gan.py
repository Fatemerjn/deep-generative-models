"""Simple GAN implementation for coursework experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn

from ..config import ModelConfig
from .base import GenerativeModel


def _make_mlp(in_dim: int, hidden_dims: Tuple[int, ...], out_dim: int, activation: str = "relu") -> nn.Sequential:
    """Utility to build an MLP with the requested activation."""

    activations = {
        "relu": nn.ReLU,
        "leaky_relu": lambda: nn.LeakyReLU(0.2),
    }
    act = activations.get(activation.lower(), nn.ReLU)

    layers = []
    prev_dim = in_dim
    for hidden_dim in hidden_dims:
        layers.extend([nn.Linear(prev_dim, hidden_dim), act()])
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, out_dim))
    return nn.Sequential(*layers)


@dataclass
class GANComponents:
    generator: nn.Module
    discriminator: nn.Module


class GenerativeAdversarialNetwork(GenerativeModel):
    """Lightweight GAN based on fully-connected networks."""

    def __init__(self, config: ModelConfig, image_shape: Tuple[int, int, int] = (1, 28, 28)) -> None:
        super().__init__(config)
        self.image_shape = image_shape
        self.generator, self.discriminator = self._build_components()
        self._bce = nn.BCEWithLogitsLoss()

    def _build_components(self) -> GANComponents:
        channels, height, width = self.image_shape
        flatten_dim = channels * height * width
        generator = _make_mlp(
            in_dim=self.config.latent_dim,
            hidden_dims=self.config.hidden_dims,
            out_dim=flatten_dim,
            activation=self.config.activation,
        )
        discriminator = _make_mlp(
            in_dim=flatten_dim,
            hidden_dims=tuple(reversed(self.config.hidden_dims)),
            out_dim=1,
            activation=self.config.activation,
        )
        return GANComponents(generator, discriminator)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.generator(z)
        return torch.tanh(logits).view(z.size(0), *self.image_shape)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.forward(z)

    def loss(self, batch: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        real_images = batch.view(batch.size(0), -1)
        z = torch.randn(batch.size(0), self.config.latent_dim, device=batch.device)
        fake_images = self.generator(z)

        real_logits = self.discriminator(real_images)
        fake_logits = self.discriminator(fake_images.detach())

        real_labels = torch.ones_like(real_logits)
        fake_labels = torch.zeros_like(fake_logits)

        d_loss_real = self._bce(real_logits, real_labels)
        d_loss_fake = self._bce(fake_logits, fake_labels)
        d_loss = d_loss_real + d_loss_fake

        g_logits = self.discriminator(fake_images)
        g_loss = self._bce(g_logits, real_labels)

        total_loss = g_loss + d_loss
        metrics = {
            "loss/generator": g_loss.detach(),
            "loss/discriminator": d_loss.detach(),
            "logits/real_mean": real_logits.mean().detach(),
            "logits/fake_mean": fake_logits.mean().detach(),
        }
        return total_loss, metrics


"""Variational Autoencoder model used in coursework."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from ..config import ModelConfig
from .base import GenerativeModel


class VariationalAutoEncoder(GenerativeModel):
    """A compact VAE with configurable hidden dimensions."""

    def __init__(self, config: ModelConfig, image_shape: Tuple[int, int, int] = (1, 28, 28)) -> None:
        super().__init__(config)
        channels, height, width = image_shape
        self._flatten_dim = channels * height * width

        hidden_dims = config.hidden_dims
        self.encoder = self._build_encoder(self._flatten_dim, hidden_dims)
        self.fc_mu = nn.Linear(hidden_dims[-1], config.latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], config.latent_dim)

        decoder_dims = list(hidden_dims[::-1]) + [self._flatten_dim]
        self.decoder = self._build_decoder(config.latent_dim, decoder_dims)

    @staticmethod
    def _build_encoder(input_dim: int, hidden_dims: Tuple[int, ...]) -> nn.Sequential:
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        return nn.Sequential(*layers)

    @staticmethod
    def _build_decoder(latent_dim: int, hidden_dims: list[int]) -> nn.Sequential:
        layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims[:-1]:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, hidden_dims[-1]))
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.view(x.size(0), -1)
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        reconstruction = self.decoder(z)
        return torch.sigmoid(reconstruction).view(z.size(0), -1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    def loss(self, batch: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        reconstruction, mu, logvar = self.forward(batch)
        batch_flat = batch.view(batch.size(0), -1)

        reconstruction_loss = F.binary_cross_entropy(reconstruction, batch_flat, reduction="sum") / batch.size(0)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch.size(0)
        total_loss = reconstruction_loss + kld

        metrics = {
            "loss/reconstruction": reconstruction_loss.detach(),
            "loss/kl_divergence": kld.detach(),
        }
        return total_loss, metrics


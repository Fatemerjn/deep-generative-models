"""Normalizing flow model used for density estimation tasks."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from ..config import ModelConfig
from .base import GenerativeModel


class AffineCoupling(nn.Module):
    """Simple affine coupling layer inspired by RealNVP."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.scale_net = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim // 2),
            nn.Tanh(),
        )
        self.translate_net = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim // 2),
        )

    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        x_a, x_b = x.chunk(2, dim=1)
        scale = self.scale_net(x_a)
        translate = self.translate_net(x_a)
        if reverse:
            y_b = (x_b - translate) * torch.exp(-scale)
            log_det = -scale.sum(dim=1)
        else:
            y_b = x_b * torch.exp(scale) + translate
            log_det = scale.sum(dim=1)
        y = torch.cat([x_a, y_b], dim=1)
        return y, log_det


class NormalizingFlowModel(GenerativeModel):
    """Stack of affine coupling layers with simple permutations."""

    def __init__(self, config: ModelConfig, input_dim: int) -> None:
        super().__init__(config)
        self.prior = torch.distributions.MultivariateNormal(
            torch.zeros(config.latent_dim), torch.eye(config.latent_dim)
        )
        hidden_dim = config.extra_kwargs.get("hidden_dim", config.hidden_dims[0])
        num_layers = config.extra_kwargs.get("num_layers", 4)
        self.transforms = nn.ModuleList([AffineCoupling(input_dim, hidden_dim) for _ in range(num_layers)])

    def _permute(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=[1])

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        z = x
        log_det_total = torch.zeros(x.size(0), device=x.device)
        for i, transform in enumerate(self.transforms):
            z, log_det = transform(z, reverse=False)
            log_det_total += log_det
            if i % 2 == 0:
                z = self._permute(z)
        log_prob = self.prior.log_prob(z) + log_det_total
        return z, log_prob

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = z
        for i, transform in reversed(list(enumerate(self.transforms))):
            if i % 2 == 0:
                x = self._permute(x)
            x, _ = transform(x, reverse=True)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z, log_prob = self.encode(x)
        return z, log_prob

    def loss(self, batch: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z, log_prob = self.forward(batch.view(batch.size(0), -1))
        loss = -log_prob.mean()
        return loss, {"loss/nll": loss.detach(), "latent/mean": z.mean().detach()}


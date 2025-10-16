"""Common plotting utilities for notebooks and scripts."""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch


def plot_samples(images: torch.Tensor, nrow: int = 8, title: str | None = None) -> None:
    """Render a grid of images using matplotlib."""

    images = images.detach().cpu()
    if images.dim() == 4 and images.size(1) == 1:
        images = images.squeeze(1)

    grid = make_grid(images, nrow=nrow)
    plt.figure(figsize=(nrow, nrow))
    plt.imshow(grid, cmap="gray")
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()


def make_grid(images: torch.Tensor, nrow: int) -> torch.Tensor:
    """Custom grid builder to avoid pulling in torchvision just for visuals."""

    if images.dim() == 3:
        images = images.unsqueeze(1)
    num_images, channels, height, width = images.shape
    nrow = min(nrow, num_images)
    ncol = (num_images + nrow - 1) // nrow

    grid = images.new_zeros((channels, ncol * height, nrow * width))
    for idx, img in enumerate(images):
        row = idx // nrow
        col = idx % nrow
        grid[:, row * height : (row + 1) * height, col * width : (col + 1) * width] = img

    if channels == 1:
        grid = grid.squeeze(0)
    return (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)

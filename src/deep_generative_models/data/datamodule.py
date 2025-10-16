"""Data loading utilities for Deep Generative Models experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from ..config import DataConfig


TransformFactory = Callable[[DataConfig], transforms.Compose]
DatasetFactory = Callable[[DataConfig, transforms.Compose], Dataset]


def default_transform_factory(config: DataConfig) -> transforms.Compose:
    """Build a canonical set of transforms for image datasets."""

    transform_list = [transforms.ToTensor()]
    if config.normalize:
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))
    return transforms.Compose(transform_list)


def mnist_dataset_factory(config: DataConfig, transform: transforms.Compose) -> Dataset:
    """Return the MNIST dataset as a default example."""

    return datasets.MNIST(
        root=str(config.root),
        train=True,
        transform=transform,
        download=config.extra_kwargs.get("download", True),
    )


DATASET_REGISTRY = {
    "mnist": mnist_dataset_factory,
}


@dataclass
class DataModule:
    """Thin wrapper that standardises dataset/dataloader creation."""

    config: DataConfig
    transform_factory: TransformFactory = default_transform_factory
    dataset_factory: Optional[DatasetFactory] = None
    val_split: float = 0.1

    def __post_init__(self) -> None:
        if self.dataset_factory is None:
            try:
                self.dataset_factory = DATASET_REGISTRY[self.config.name.lower()]
            except KeyError as exc:
                raise ValueError(
                    f"Unknown dataset '{self.config.name}'. "
                    f"Register a factory in DATASET_REGISTRY or pass one explicitly."
                ) from exc

    def setup(self) -> Tuple[Dataset, Dataset]:
        """Prepare train/validation datasets."""

        transform = self.transform_factory(self.config)
        dataset = self.dataset_factory(self.config, transform)  # type: ignore[arg-type]

        if self.val_split <= 0.0:
            return dataset, dataset

        val_size = int(len(dataset) * self.val_split)
        train_size = len(dataset) - val_size
        generator = torch.Generator().manual_seed(self.config.extra_kwargs.get("split_seed", 0))
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        return train_dataset, val_dataset

    def dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation data loaders."""

        train_dataset, val_dataset = self.setup()
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        return train_loader, val_loader


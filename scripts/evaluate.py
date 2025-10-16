#!/usr/bin/env python3
"""Evaluate a trained model checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from deep_generative_models.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)
from deep_generative_models.data import DataModule
from deep_generative_models.training import build_model, evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_experiment(path: Path) -> ExperimentConfig:
    config = yaml.safe_load(path.read_text())
    data = DataConfig(root=Path(config["data"]["root"]), **{k: v for k, v in config["data"].items() if k != "root"})
    model = ModelConfig(**config["model"])
    optimizer = OptimizerConfig(**config.get("optimizer", {}))
    training = TrainingConfig(**config.get("training", {}))
    return ExperimentConfig(
        data=data,
        model=model,
        optimizer=optimizer,
        training=training,
        output_dir=Path(config.get("output_dir", "results/experiments")),
        notes=config.get("notes"),
    )


def main() -> None:
    args = parse_args()
    experiment = load_experiment(args.config)
    device = torch.device(args.device)

    data_module = DataModule(experiment.data)
    _, val_loader = data_module.dataloaders()

    model = build_model(experiment).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    metrics = evaluate(model, val_loader, device)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()


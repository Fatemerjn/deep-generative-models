#!/usr/bin/env python3
"""Command-line training entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

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
from deep_generative_models.training import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML configuration file.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--override", type=str, default=None, help="JSON string to override config values.")
    return parser.parse_args()


def load_config(path: Path, override: str | None = None) -> ExperimentConfig:
    data = yaml.safe_load(path.read_text())
    overrides: Dict[str, Any] = json.loads(override) if override else {}
    merged = _deep_update(data, overrides)

    experiment = ExperimentConfig(
        data=_build_data_config(merged["data"]),
        model=_build_model_config(merged["model"]),
        optimizer=OptimizerConfig(**merged.get("optimizer", {})),
        training=TrainingConfig(**merged.get("training", {})),
        output_dir=Path(merged.get("output_dir", "results/experiments")),
        notes=merged.get("notes"),
    )
    return experiment


def _build_data_config(config: Dict[str, Any]) -> DataConfig:
    config = dict(config)
    config["root"] = Path(config["root"])
    return DataConfig(**config)


def _build_model_config(config: Dict[str, Any]) -> ModelConfig:
    config = dict(config)
    extra = config.pop("extra_kwargs", {})
    model_config = ModelConfig(**config)
    model_config.extra_kwargs.update(extra)
    return model_config


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def main() -> None:
    args = parse_args()
    experiment = load_config(args.config, args.override)
    data_module = DataModule(experiment.data)
    train_loader, val_loader = data_module.dataloaders()

    device = torch.device(args.device)
    experiment.output_dir.mkdir(parents=True, exist_ok=True)
    run_training(experiment, train_loader, val_loader, device)


if __name__ == "__main__":
    main()


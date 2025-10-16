"""Top-level package for the Deep Generative Models coursework codebase."""

from .config import DataConfig, ModelConfig, OptimizerConfig, TrainingConfig, ExperimentConfig

# Re-export key namespaces for convenience
from . import data, models, training, utils

__all__ = [
    "data",
    "models",
    "training",
    "utils",
    "DataConfig",
    "ModelConfig",
    "OptimizerConfig",
    "TrainingConfig",
    "ExperimentConfig",
]


"""Configuration dataclasses used across the project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class DataConfig:
    """Parameters that control dataset preparation and batching."""

    name: str
    root: Path
    batch_size: int = 64
    num_workers: int = 4
    shuffle: bool = True
    normalize: bool = True
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Parameters shared by model definitions."""

    name: str
    latent_dim: int
    hidden_dims: Tuple[int, ...] = (256, 512)
    activation: str = "relu"
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizerConfig:
    """Optimizer and scheduler hyper-parameters."""

    optimizer: str = "adam"
    learning_rate: float = 2e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    scheduler: Optional[str] = None
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Controls training loop settings."""

    epochs: int = 100
    gradient_clip: Optional[float] = None
    log_every: int = 50
    eval_every: int = 1
    precision: str = "fp32"
    seed: int = 42


@dataclass
class ExperimentConfig:
    """Bundle all sections into a single object for scripting convenience."""

    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: Path = Path("results") / "experiments"
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return the configuration as a dictionary with Path objects stringified."""

        def _convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, (DataConfig, ModelConfig, OptimizerConfig, TrainingConfig, ExperimentConfig)):
                return {k: _convert(v) for k, v in value.__dict__.items()}
            if isinstance(value, dict):
                return {k: _convert(v) for k, v in value.items()}
            if isinstance(value, tuple):
                return tuple(_convert(v) for v in value)
            return value

        return _convert(self)


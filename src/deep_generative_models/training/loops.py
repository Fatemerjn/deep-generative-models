"""Training and evaluation loops."""

from __future__ import annotations

import time
from typing import Callable, Dict, Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..models import MODEL_REGISTRY, GenerativeModel
from ..utils.checkpointing import Checkpointer
from ..utils.logging import MetricLogger


Batch = torch.Tensor
StepOutput = Tuple[torch.Tensor, Dict[str, torch.Tensor]]


def build_model(config: ExperimentConfig) -> GenerativeModel:
    """Instantiate a model from the registry based on the configuration."""

    try:
        model_cls = MODEL_REGISTRY[config.model.name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown model '{config.model.name}'.") from exc
    model = model_cls(config.model, **config.model.extra_kwargs)
    return model


def build_optimizer(model: nn.Module, config: ExperimentConfig) -> torch.optim.Optimizer:
    """Construct an optimizer according to the provided configuration."""

    if config.optimizer.optimizer.lower() == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.optimizer.learning_rate,
            betas=config.optimizer.betas,
            weight_decay=config.optimizer.weight_decay,
        )
    if config.optimizer.optimizer.lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.optimizer.learning_rate,
            betas=config.optimizer.betas,
            weight_decay=config.optimizer.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer '{config.optimizer.optimizer}'.")


def run_training(
    experiment: ExperimentConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    hooks: Iterable[Callable[[int, Dict[str, float]], None]] = (),
) -> GenerativeModel:
    """Generic training loop shared across models."""

    model = build_model(experiment).to(device)
    optimizer = build_optimizer(model, experiment)
    scaler = torch.cuda.amp.GradScaler() if experiment.training.precision == "amp" else None
    logger = MetricLogger()
    checkpointer = Checkpointer(experiment.output_dir, keep_last=3)

    for epoch in range(experiment.training.epochs):
        start_time = time.time()
        model.train()
        for step, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss, metrics = _step(model, batch)
                scaler.scale(loss).backward()
                if experiment.training.gradient_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), experiment.training.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, metrics = _step(model, batch)
                loss.backward()
                if experiment.training.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), experiment.training.gradient_clip)
                optimizer.step()

            logger.update(loss=loss.detach(), **metrics)
            if (step + 1) % experiment.training.log_every == 0:
                logger.flush(epoch, step + 1)

        val_metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - start_time
        summary = {"epoch": epoch + 1, "elapsed": elapsed, **logger.buffer, **val_metrics}
        for hook in hooks:
            hook(epoch + 1, summary)

        logger.reset()
        checkpointer.save(model, optimizer, epoch + 1, metrics=val_metrics)

    return model


def evaluate(model: GenerativeModel, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Evaluate the model on the validation loader."""

    model.eval()
    metrics_accumulator: Dict[str, float] = {}
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            loss, metrics = _step(model, batch)
            metrics["loss/total"] = loss.detach()
            for key, value in metrics.items():
                metrics_accumulator.setdefault(key, 0.0)
                metrics_accumulator[key] += float(value)

    num_batches = len(dataloader)
    for key in metrics_accumulator:
        metrics_accumulator[key] /= num_batches
    return metrics_accumulator


def _step(model: GenerativeModel, batch: Batch) -> StepOutput:
    """Run a training step. Kept separate for readability and testing."""

    return model.loss(batch)


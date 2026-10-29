"""Reusable training, validation, checkpoint, and evaluation utilities."""

from __future__ import annotations

import csv
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from model import MLP


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    batch_size: int = 64
    max_epochs: int = 500
    learning_rate: float = 1e-3
    patience: int = 40
    min_delta: float = 1e-8
    weight_decay: float = 0.0
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        if self.min_delta < 0:
            raise ValueError("min_delta cannot be negative")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")


@dataclass
class TrainingResult:
    model: MLP
    history: list[dict[str, float | int]]
    best_epoch: int
    best_validation_loss: float
    training_seconds: float
    device: str


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_csv_dataset(path: str | Path, limit: int | None = None) -> TensorDataset:
    """Load an x1,x2,y CSV as float32 tensors; optionally take a stable prefix."""
    csv_path = Path(path)
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader, None)
        if header != ["x1", "x2", "y"]:
            raise ValueError(f"{csv_path} must have header x1,x2,y; got {header}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"{csv_path} contains no data rows")
    try:
        values = np.asarray(rows, dtype=np.float64)
    except ValueError as error:
        raise ValueError(f"{csv_path} contains a non-numeric or malformed row") from error
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"{csv_path} must contain exactly three columns")
    if not np.isfinite(values).all():
        raise ValueError(f"{csv_path} contains NaN or infinite values")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        if limit > len(values):
            raise ValueError(f"requested {limit} rows from {csv_path}, which has {len(values)}")
        values = values[:limit]

    inputs = torch.from_numpy(values[:, :2].astype(np.float32))
    targets = torch.from_numpy(values[:, 2:3].astype(np.float32))
    return TensorDataset(inputs, targets)


def _mean_loss(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            loss = loss_function(model(inputs), targets)
            batch_size = inputs.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    if total_samples == 0:
        raise ValueError("cannot evaluate an empty dataset")
    return total_loss / total_samples


def save_checkpoint(
    path: str | Path,
    model: MLP,
    config: TrainingConfig,
    best_epoch: int,
    best_validation_loss: float,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            },
            "hidden_sizes": list(model.hidden_sizes),
            "activation": model.activation_name,
            "input_scale": model.input_scale.detach().cpu().tolist(),
            "training_config": asdict(config),
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
        },
        checkpoint_path,
    )


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[MLP, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    model = MLP(
        hidden_sizes=checkpoint["hidden_sizes"],
        activation=checkpoint["activation"],
        input_scale=tuple(checkpoint["input_scale"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(map_location)
    model.eval()
    return model, checkpoint


def train_model(
    training_data: Dataset,
    validation_data: Dataset,
    hidden_sizes: tuple[int, ...] = (10,),
    activation: str = "tanh",
    config: TrainingConfig | None = None,
    checkpoint_path: str | Path | None = None,
) -> TrainingResult:
    """Fit an MLP, select its checkpoint using validation loss, and restore it."""
    config = config or TrainingConfig()
    if len(training_data) == 0 or len(validation_data) == 0:
        raise ValueError("training and validation datasets must not be empty")

    set_seed(config.seed)
    device = resolve_device(config.device)
    model = MLP(hidden_sizes=hidden_sizes, activation=activation).to(device)
    training_generator = torch.Generator().manual_seed(config.seed)
    training_loader = DataLoader(
        training_data,
        batch_size=config.batch_size,
        shuffle=True,
        generator=training_generator,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=config.batch_size,
        shuffle=False,
    )

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    start_time = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        sample_count = 0
        for inputs, targets in training_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = loss_function(predictions, targets)
            loss.backward()
            optimizer.step()

            batch_size = inputs.shape[0]
            train_loss_sum += loss.item() * batch_size
            sample_count += batch_size

        if sample_count == 0:
            raise ValueError("cannot train on an empty dataset")
        training_loss = train_loss_sum / sample_count
        validation_loss = _mean_loss(model, validation_loader, loss_function, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": training_loss,
                "validation_loss": validation_loss,
            }
        )

        if validation_loss < best_loss - config.min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    training_seconds = time.perf_counter() - start_time
    if best_state is None:
        raise RuntimeError("training did not produce a validation checkpoint")
    model.load_state_dict(best_state)

    if checkpoint_path is not None:
        save_checkpoint(checkpoint_path, model, config, best_epoch, best_loss)

    return TrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        training_seconds=training_seconds,
        device=str(device),
    )


def evaluate_model(
    model: nn.Module,
    dataset: Dataset,
    device: str | torch.device = "cpu",
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Compute regression metrics and predictions for a held-out dataset."""
    if len(dataset) == 0:
        raise ValueError("cannot evaluate an empty dataset")
    target_device = torch.device(device)
    model = model.to(target_device)
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for inputs, batch_targets in DataLoader(dataset, batch_size=256, shuffle=False):
            predictions.append(model(inputs.to(target_device)).cpu())
            targets.append(batch_targets.cpu())

    predicted_values = torch.cat(predictions).numpy().reshape(-1)
    target_values = torch.cat(targets).numpy().reshape(-1)
    errors = predicted_values - target_values
    metrics = {
        "mse": float(np.mean(np.square(errors))),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
    }
    return metrics, target_values, predicted_values


def save_history_csv(path: str | Path, history: list[dict[str, float | int]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "validation_loss"])
        writer.writeheader()
        writer.writerows(history)

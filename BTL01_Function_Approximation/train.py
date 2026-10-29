"""Train and evaluate the reproducible baseline MLP."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from training import (
    TrainingConfig,
    evaluate_model,
    load_checkpoint,
    load_csv_dataset,
    save_history_csv,
    train_model,
)


PROJECT_DIR = Path(__file__).parent


def _save_training_plot(history: list[dict[str, float | int]], path: Path) -> None:
    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    validation_loss = [float(row["validation_loss"]) for row in history]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, train_loss, label="Train")
    axis.plot(epochs, validation_loss, label="Validation")
    axis.set(xlabel="Epoch", ylabel="MSE loss", title="Baseline training and validation loss")
    axis.set_yscale("log")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _save_prediction_plot(
    targets: np.ndarray,
    predictions: np.ndarray,
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(targets, predictions, alpha=0.8)
    lower = float(min(targets.min(), predictions.min()))
    upper = float(max(targets.max(), predictions.max()))
    axes[0].plot([lower, upper], [lower, upper], "k--", label="Ideal prediction")
    axes[0].set(
        xlabel="True y",
        ylabel="Predicted y",
        title="Test predictions vs. truth",
    )
    axes[0].legend()
    axes[1].scatter(targets, predictions - targets, alpha=0.8)
    axes[1].axhline(0.0, color="black", linestyle="--")
    axes[1].set(xlabel="True y", ylabel="Prediction error", title="Test residuals")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _save_surface_plot(model: torch.nn.Module, path: Path, device: str) -> None:
    x1 = np.linspace(-2.0, 2.0, 100, dtype=np.float32)
    x2 = np.linspace(-4.0, 4.0, 100, dtype=np.float32)
    grid_x1, grid_x2 = np.meshgrid(x1, x2)
    inputs = torch.from_numpy(np.column_stack((grid_x1.ravel(), grid_x2.ravel())))
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(inputs.to(device)).cpu().numpy().reshape(grid_x1.shape)

    figure = plt.figure(figsize=(9, 6))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot_surface(grid_x1, grid_x2, predictions, cmap="viridis", linewidth=0)
    axis.set(
        xlabel="x1",
        ylabel="x2",
        zlabel="Predicted y",
        title="MLP approximation surface",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--hidden-width", type=int, default=10)
    parser.add_argument("--hidden-layers", type=int, default=1)
    parser.add_argument("--activation", choices=("relu", "tanh", "sigmoid"), default="tanh")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "results" / "baseline",
    )
    args = parser.parse_args()

    if args.hidden_width <= 0 or args.hidden_layers <= 0:
        parser.error("--hidden-width and --hidden-layers must be positive")

    data_dir = PROJECT_DIR / "data"
    training_data = load_csv_dataset(data_dir / "train.csv")
    validation_data = load_csv_dataset(data_dir / "validation.csv")
    config = TrainingConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        device=args.device,
    )
    hidden_sizes = (args.hidden_width,) * args.hidden_layers
    checkpoint_path = args.output_dir / "best_model.pt"
    result = train_model(
        training_data,
        validation_data,
        hidden_sizes=hidden_sizes,
        activation=args.activation,
        config=config,
        checkpoint_path=checkpoint_path,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_history_csv(args.output_dir / "training_history.csv", result.history)
    _save_training_plot(result.history, args.output_dir / "learning_curves.png")

    # Test data is loaded only after validation-based checkpoint selection is complete.
    model, checkpoint = load_checkpoint(checkpoint_path, map_location=result.device)
    test_data = load_csv_dataset(data_dir / "test.csv")
    test_metrics, test_targets, test_predictions = evaluate_model(
        model,
        test_data,
        device=result.device,
    )
    _save_prediction_plot(
        test_targets,
        test_predictions,
        args.output_dir / "test_predictions.png",
    )
    _save_surface_plot(
        model,
        args.output_dir / "predicted_surface.png",
        result.device,
    )

    run_summary = {
        "model": {
            "hidden_sizes": list(hidden_sizes),
            "activation": args.activation,
            "input_scale": [2.0, 4.0],
        },
        "training_config": asdict(config),
        "device_used": result.device,
        "epochs_completed": len(result.history),
        "best_epoch": checkpoint["best_epoch"],
        "best_validation_mse": checkpoint["best_validation_loss"],
        "training_seconds": result.training_seconds,
        "test_metrics": test_metrics,
        "checkpoint": checkpoint_path.name,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()

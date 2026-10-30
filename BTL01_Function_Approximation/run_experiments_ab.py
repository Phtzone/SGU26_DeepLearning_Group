"""Run controlled training-size (A) and hidden-width (B) experiments."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from training import (
    TrainingConfig,
    evaluate_model,
    load_csv_dataset,
    save_history_csv,
    train_model,
)


PROJECT_DIR = Path(__file__).parent
TRAINING_SIZES = tuple(range(200, 2001, 200))
HIDDEN_WIDTHS = tuple(range(1, 11))
DEFAULT_SEEDS = (42, 43, 44)
RUN_FIELDS = (
    "experiment",
    "value",
    "seed",
    "training_samples",
    "hidden_width",
    "hidden_layers",
    "activation",
    "batch_size",
    "learning_rate",
    "max_epochs",
    "patience",
    "epochs_completed",
    "best_epoch",
    "best_validation_mse",
    "train_mse",
    "validation_mse",
    "test_mse",
    "test_mae",
    "test_rmse",
    "training_seconds",
    "device",
    "run_dir",
)
METRIC_FIELDS = (
    "best_validation_mse",
    "train_mse",
    "validation_mse",
    "test_mse",
    "test_mae",
    "test_rmse",
    "training_seconds",
)


def _write_rows(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_run_plot(
    axis: str,
    value: int,
    history: list[dict[str, float | int]],
    output_path: Path,
) -> None:
    figure, plot_axis = plt.subplots(figsize=(8, 5))
    epochs = [int(row["epoch"]) for row in history]
    plot_axis.plot(epochs, [float(row["train_loss"]) for row in history], label="Train")
    plot_axis.plot(
        epochs,
        [float(row["validation_loss"]) for row in history],
        label="Validation",
    )
    plot_axis.set(
        xlabel="Epoch",
        ylabel="MSE loss",
        title=f"Experiment {axis}, swept value {value}",
    )
    plot_axis.set_yscale("log")
    plot_axis.grid(True, alpha=0.3)
    plot_axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _aggregate(
    experiment: str,
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["value"])].append(row)

    summary: list[dict[str, object]] = []
    for value in sorted(grouped):
        group = grouped[value]
        result: dict[str, object] = {
            "experiment": experiment,
            "value": value,
            "runs": len(group),
        }
        for field in METRIC_FIELDS:
            values = [float(row[field]) for row in group]
            result[f"{field}_mean"] = mean(values)
            result[f"{field}_std"] = stdev(values) if len(values) > 1 else 0.0
        summary.append(result)
    return summary


def _plot_experiment_a(summary: list[dict[str, object]], output_path: Path) -> None:
    sizes = [int(row["value"]) for row in summary]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, metric, label in (
        (axes[0], "test_mse", "Test MSE"),
        (axes[1], "test_mae", "Test MAE"),
    ):
        means = [float(row[f"{metric}_mean"]) for row in summary]
        deviations = [float(row[f"{metric}_std"]) for row in summary]
        axis.errorbar(sizes, means, yerr=deviations, marker="o", capsize=3)
        axis.set(xlabel="Training samples", ylabel=label, title=f"{label} vs. training-set size")
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_experiment_b(summary: list[dict[str, object]], output_dir: Path) -> None:
    widths = [int(row["value"]) for row in summary]
    figure, axis = plt.subplots(figsize=(8, 5))
    for split, label in (
        ("train_mse", "Train"),
        ("validation_mse", "Validation"),
        ("test_mse", "Test"),
    ):
        means = [float(row[f"{split}_mean"]) for row in summary]
        deviations = [float(row[f"{split}_std"]) for row in summary]
        axis.errorbar(widths, means, yerr=deviations, marker="o", capsize=3, label=label)
    axis.set(
        xlabel="Hidden units (B)",
        ylabel="MSE",
        title="Experiment B: error vs. hidden-layer width",
    )
    axis.set_xticks(widths)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "experiment_b_split_mse.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    means = [float(row["test_mae_mean"]) for row in summary]
    deviations = [float(row["test_mae_std"]) for row in summary]
    axis.errorbar(widths, means, yerr=deviations, marker="o", capsize=3)
    axis.set(
        xlabel="Hidden units (B)",
        ylabel="Test MAE",
        title="Experiment B: test MAE vs. hidden-layer width",
    )
    axis.set_xticks(widths)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "experiment_b_test_mae.png", dpi=180)
    plt.close(figure)


def _run_configuration(
    experiment: str,
    value: int,
    seed: int,
    training_data,
    validation_data,
    test_csv: Path,
    output_dir: Path,
    config: TrainingConfig,
    activation: str,
) -> dict[str, object]:
    if experiment == "A":
        hidden_width = 10
        training_samples = value
        run_training_data = load_csv_dataset(
            PROJECT_DIR / "data" / "train.csv",
            limit=training_samples,
        )
    else:
        hidden_width = value
        training_samples = len(training_data)
        run_training_data = training_data

    hidden_sizes = (hidden_width,)
    run_name = f"{experiment}_{value:04d}_seed{seed}"
    run_dir = output_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = TrainingConfig(
        seed=seed,
        batch_size=config.batch_size,
        max_epochs=config.max_epochs,
        learning_rate=config.learning_rate,
        patience=config.patience,
        min_delta=config.min_delta,
        weight_decay=config.weight_decay,
        device=config.device,
    )

    result = train_model(
        run_training_data,
        validation_data,
        hidden_sizes=hidden_sizes,
        activation=activation,
        config=run_config,
        checkpoint_path=run_dir / "best_model.pt",
    )
    train_metrics, _, _ = evaluate_model(
        result.model,
        run_training_data,
        device=result.device,
    )
    validation_metrics, _, _ = evaluate_model(
        result.model,
        validation_data,
        device=result.device,
    )

    # Load and evaluate test data only after training and validation checkpoint selection.
    test_data = load_csv_dataset(test_csv)
    test_metrics, _, _ = evaluate_model(
        result.model,
        test_data,
        device=result.device,
    )

    save_history_csv(run_dir / "training_history.csv", result.history)
    _save_run_plot(experiment, value, result.history, run_dir / "learning_curves.png")
    try:
        recorded_run_dir = run_dir.resolve().relative_to(PROJECT_DIR.resolve())
    except ValueError:
        recorded_run_dir = run_dir.resolve()
    row: dict[str, object] = {
        "experiment": experiment,
        "value": value,
        "seed": seed,
        "training_samples": training_samples,
        "hidden_width": hidden_width,
        "hidden_layers": 1,
        "activation": activation,
        "batch_size": run_config.batch_size,
        "learning_rate": run_config.learning_rate,
        "max_epochs": run_config.max_epochs,
        "patience": run_config.patience,
        "epochs_completed": len(result.history),
        "best_epoch": result.best_epoch,
        "best_validation_mse": result.best_validation_loss,
        "train_mse": train_metrics["mse"],
        "validation_mse": validation_metrics["mse"],
        "test_mse": test_metrics["mse"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "training_seconds": result.training_seconds,
        "device": result.device,
        "run_dir": str(recorded_run_dir),
    }
    (run_dir / "run_config_and_metrics.json").write_text(
        json.dumps(
            {
                "experiment": experiment,
                "swept_value": value,
                "seed": seed,
                "model": {
                    "hidden_sizes": list(hidden_sizes),
                    "activation": activation,
                    "input_scale": [2.0, 4.0],
                },
                "training_config": asdict(run_config),
                "epochs_completed": len(result.history),
                "best_epoch": result.best_epoch,
                "best_validation_mse": result.best_validation_loss,
                "train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
                "training_seconds": result.training_seconds,
                "device_used": result.device,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"{experiment}={value}, seed={seed}: "
        f"val_mse={row['validation_mse']:.6g}, "
        f"test_mse={row['test_mse']:.6g}, "
        f"test_mae={row['test_mae']:.6g}, "
        f"best_epoch={result.best_epoch}, "
        f"seconds={result.training_seconds:.2f}"
    )
    return row


def _run_experiment(
    experiment: str,
    values: tuple[int, ...],
    seeds: tuple[int, ...],
    training_data,
    validation_data,
    test_csv: Path,
    output_dir: Path,
    config: TrainingConfig,
    activation: str,
) -> None:
    experiment_dir = output_dir / experiment
    experiment_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for value in values:
        for seed in seeds:
            rows.append(
                _run_configuration(
                    experiment,
                    value,
                    seed,
                    training_data,
                    validation_data,
                    test_csv,
                    experiment_dir,
                    config,
                    activation,
                )
            )
            _write_rows(experiment_dir / "runs.csv", RUN_FIELDS, rows)

    summary = _aggregate(experiment, rows)
    summary_fields = ("experiment", "value", "runs") + tuple(
        f"{metric}_{statistic}"
        for metric in METRIC_FIELDS
        for statistic in ("mean", "std")
    )
    _write_rows(experiment_dir / "summary.csv", summary_fields, summary)
    if experiment == "A":
        _plot_experiment_a(summary, experiment_dir / "experiment_a_test_errors.png")
    else:
        _plot_experiment_b(summary, experiment_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=("A", "B"),
        default=("A", "B"),
        help="Experiment groups to run (default: both A and B)",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--sizes", nargs="+", type=int, default=TRAINING_SIZES)
    parser.add_argument("--widths", nargs="+", type=int, default=HIDDEN_WIDTHS)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--activation", choices=("relu", "tanh", "sigmoid"), default="tanh")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "results" / "experiments",
    )
    args = parser.parse_args()

    if not args.seeds or any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must contain one or more non-negative integers")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")
    if not args.sizes or any(size not in TRAINING_SIZES for size in args.sizes):
        parser.error("--sizes must be selected from 200, 400, ..., 2000")
    if not args.widths or any(width <= 0 for width in args.widths):
        parser.error("--widths must contain one or more positive integers")

    data_dir = PROJECT_DIR / "data"
    training_data = load_csv_dataset(data_dir / "train.csv")
    validation_data = load_csv_dataset(data_dir / "validation.csv")
    config = TrainingConfig(
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        device=args.device,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if "A" in args.experiments:
        _run_experiment(
            "A",
            tuple(args.sizes),
            tuple(args.seeds),
            training_data,
            validation_data,
            data_dir / "test.csv",
            output_dir,
            config,
            args.activation,
        )
    if "B" in args.experiments:
        _run_experiment(
            "B",
            tuple(args.widths),
            tuple(args.seeds),
            training_data,
            validation_data,
            data_dir / "test.csv",
            output_dir,
            config,
            args.activation,
        )

    print(f"Experiment results written to {output_dir}")


if __name__ == "__main__":
    main()

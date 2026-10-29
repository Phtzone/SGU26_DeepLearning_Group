"reproducible data and plots for the MLP function-approximation"

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


MEAN = np.array([0.0, 0.0], dtype=np.float64)
COVARIANCE = np.array([[0.3, -0.5], [-0.5, 2.0]], dtype=np.float64)
X1_RANGE = (-2.0, 2.0)
X2_RANGE = (-4.0, 4.0)
DEFAULT_SEED = 42
DEFAULT_TRAIN_SIZE = 2000
DEFAULT_VALIDATION_SIZE = 500
TEST_SIZE = 100


def gaussian_density(inputs: np.ndarray) -> np.ndarray:
    """Evaluate the specified bivariate Gaussian density at rows of 2D inputs."""
    inputs = np.asarray(inputs, dtype=np.float64)
    if inputs.ndim != 2 or inputs.shape[1] != 2:
        raise ValueError(f"inputs must have shape (n_samples, 2); got {inputs.shape}")

    centered = inputs - MEAN
    quadratic_form = np.sum(centered * np.linalg.solve(COVARIANCE, centered.T).T, axis=1)
    normalizer = 1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(COVARIANCE)))
    return normalizer * np.exp(-0.5 * quadratic_form)


def generate_dataset(size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw inputs uniformly from the required rectangle and compute exact labels."""
    if size <= 0:
        raise ValueError(f"dataset size must be positive; got {size}")

    inputs = np.column_stack(
        (
            rng.uniform(*X1_RANGE, size=size),
            rng.uniform(*X2_RANGE, size=size),
        )
    )
    return inputs, gaussian_density(inputs)


def save_dataset(path: Path, inputs: np.ndarray, targets: np.ndarray) -> None:
    np.savetxt(
        path,
        np.column_stack((inputs, targets)),
        delimiter=",",
        header="x1,x2,y",
        comments="",
        fmt="%.10g",
    )


def summarize_dataset(inputs: np.ndarray, targets: np.ndarray) -> dict[str, object]:
    columns = np.column_stack((inputs, targets))
    return {
        "n_samples": int(len(targets)),
        "columns": ["x1", "x2", "y"],
        "mean": dict(zip(("x1", "x2", "y"), columns.mean(axis=0).tolist())),
        "std": dict(zip(("x1", "x2", "y"), columns.std(axis=0).tolist())),
        "min": dict(zip(("x1", "x2", "y"), columns.min(axis=0).tolist())),
        "max": dict(zip(("x1", "x2", "y"), columns.max(axis=0).tolist())),
    }


def save_plots(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(9, 6))
    axis = figure.add_subplot(111, projection="3d")
    scatter = axis.scatter(
        train_inputs[:, 0],
        train_inputs[:, 1],
        train_targets,
        c=train_targets,
        cmap="viridis",
        s=14,
        alpha=0.8,
    )
    axis.set(xlabel="x1", ylabel="x2", zlabel="y = f(x1, x2)", title="Training samples")
    figure.colorbar(scatter, ax=axis, label="Gaussian density", shrink=0.7)
    figure.tight_layout()
    figure.savefig(figure_dir / "training_scatter_3d.png", dpi=180)
    plt.close(figure)

    x1 = np.linspace(*X1_RANGE, 100)
    x2 = np.linspace(*X2_RANGE, 100)
    grid_x1, grid_x2 = np.meshgrid(x1, x2)
    grid_inputs = np.column_stack((grid_x1.ravel(), grid_x2.ravel()))
    surface = gaussian_density(grid_inputs).reshape(grid_x1.shape)

    figure = plt.figure(figsize=(9, 6))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot_surface(grid_x1, grid_x2, surface, cmap="viridis", linewidth=0, antialiased=True)
    axis.set(xlabel="x1", ylabel="x2", zlabel="f(x1, x2)", title="True Gaussian surface")
    figure.tight_layout()
    figure.savefig(figure_dir / "true_gaussian_surface.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for axis, values, label in zip(
        axes,
        (train_inputs[:, 0], train_inputs[:, 1], train_targets),
        ("x1", "x2", "y"),
    ):
        axis.hist(values, bins=30, color="steelblue", edgecolor="white")
        axis.set(xlabel=label, ylabel="Count", title=f"Training {label} distribution")
    figure.tight_layout()
    figure.savefig(figure_dir / "training_distributions.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--validation-size", type=int, default=DEFAULT_VALIDATION_SIZE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Directory for CSV files and dataset_metadata.json",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path(__file__).parent / "figures",
        help="Directory for generated PNG figures",
    )
    args = parser.parse_args()

    if args.train_size < 2000:
        parser.error("--train-size must be at least 2000 for the required size sweep")
    if args.validation_size <= 0:
        parser.error("--validation-size must be positive")

    train_seed, validation_seed, test_seed = np.random.SeedSequence(args.seed).spawn(3)
    datasets = {
        "train": generate_dataset(args.train_size, np.random.default_rng(train_seed)),
        "validation": generate_dataset(
            args.validation_size, np.random.default_rng(validation_seed)
        ),
        "test": generate_dataset(TEST_SIZE, np.random.default_rng(test_seed)),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, (inputs, targets) in datasets.items():
        save_dataset(args.output_dir / f"{name}.csv", inputs, targets)

    metadata = {
        "seed": args.seed,
        "rng": "NumPy default_rng with independent SeedSequence child streams",
        "sampling": {
            "distribution": "uniform independently within the rectangular input domain",
            "x1_range": list(X1_RANGE),
            "x2_range": list(X2_RANGE),
        },
        "target": {
            "distribution": "bivariate Gaussian density",
            "mean": MEAN.tolist(),
            "covariance": COVARIANCE.tolist(),
        },
        "datasets": {
            name: summarize_dataset(inputs, targets)
            for name, (inputs, targets) in datasets.items()
        },
    }
    (args.output_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    save_plots(*datasets["train"], args.figure_dir)

    for name, (inputs, targets) in datasets.items():
        print(
            f"{name}: {len(targets)} samples; "
            f"x1 in [{inputs[:, 0].min():.4f}, {inputs[:, 0].max():.4f}], "
            f"x2 in [{inputs[:, 1].min():.4f}, {inputs[:, 1].max():.4f}], "
            f"y in [{targets.min():.6f}, {targets.max():.6f}]"
        )
    print(f"CSV and metadata files: {args.output_dir}")
    print(f"Figures: {args.figure_dir}")


if __name__ == "__main__":
    main()

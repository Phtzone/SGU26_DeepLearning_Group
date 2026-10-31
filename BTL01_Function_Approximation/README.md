# BTL01: Function Approximation with an MLP

## Environment setup

Python 3.10 or newer is recommended. From this folder, create and activate a
virtual environment, then install the project dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In VS Code, select `.venv\Scripts\python.exe` as the Python interpreter. The
dependencies include PyTorch for the required MLP implementation, NumPy for
data generation, and Matplotlib for plots. CPU training is supported; a GPU is
optional.

## Generate the datasets

```powershell
python generate_data.py
```

This writes the fixed train, validation, and test CSVs and metadata to `data\`,
and the dataset plots to `figures\`. See [data/README.md](data/README.md) for
the data definition, split sizes, and experiment reuse guidance.

## Train the PyTorch baseline

With the project environment activated, run:

```powershell
python train.py
```

The default baseline is a 2-10-1 MLP with a tanh hidden activation, Adam
(`learning_rate=0.001`), batch size 64, at most 500 epochs, and early stopping
after 40 epochs without validation improvement. Inputs are divided by their
fixed domain half-ranges (`x1 / 2`, `x2 / 4`). Override settings with
`python train.py --help`.

The runner selects and saves the best checkpoint using **validation MSE only**.
It loads the test set afterward for final MSE, MAE, and RMSE evaluation. Results
are written to `results\baseline\` (checkpoint, epoch-level CSV, run summary,
loss curves, test prediction/residual plot, and approximation surface).

For A/B/C experiment scripts, `training.py` exposes `load_csv_dataset`,
`TrainingConfig`, `train_model`, `evaluate_model`, and `load_checkpoint`.
Experiment A can use `load_csv_dataset("data/train.csv", limit=N)` to train on
the first `N` rows while keeping the validation and test splits fixed.

## Experiment C: hidden-layer depth

From `BTL01_Function_Approximation`, run:

```powershell
python run_experiments_c.py
```

This runs C = 1, 2, 3, 4 with seeds 42, 43, 44 (12 runs). Every hidden layer
has 10 units. The fixed data splits, tanh activation, linear output, input
scaling, MSE loss, Adam optimizer, learning rate 0.001, batch size 64,
500-epoch limit, and early stopping (patience 40, min_delta 1e-8) match the
baseline. Only depth changes within the experiment. CPU execution uses one
PyTorch thread by default; `--device cuda` is optional. Times measure the
training/validation loop and exclude plotting and final evaluation.

Checkpoints are selected using validation MSE. Test metrics are final
evaluations, never used to tune depth. The analysis recommends depth by mean
validation MSE across seeds. More layers also increase the parameter count;
this is a fixed-width comparison, not a fixed-parameter-budget comparison.

Outputs under `results/experiments/C/`:

- `runs.csv`: settings, epochs, parameter counts, train/validation/test metrics,
  and training time per run.
- `summary.csv`: metric means and sample standard deviations across seeds.
- `experiment_config.json`: setup, library versions, thread count, data hashes.
- `experiment_c_errors_and_time.png`: train/validation/test MSE, test MAE,
  and training time versus depth, with standard deviation bars.
- `experiment_c_convergence.png`: train and validation loss for each depth/seed.
- `analysis.md`: results table and Vietnamese discussion of depth and limitations.
- `runs/C_0001_seed42/`, etc.: `best_model.pt`, `training_history.csv`,
  `run_config_and_metrics.json`, `learning_curves.png`, `test_predictions.csv`,
  `test_predictions.png`, and `predicted_surface.png`.

Use `python run_experiments_c.py --help` for overrides. For a short execution
check, use a separate output folder so full results remain available:

```powershell
python run_experiments_c.py --depths 1 4 --seeds 42 --epochs 2 --output-dir results/smoke_c
```

The short check does not replace the complete 12-run experiment.

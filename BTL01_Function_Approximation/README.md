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

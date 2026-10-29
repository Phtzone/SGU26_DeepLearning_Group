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

# Dataset generation

Run `python generate_data.py` from this folder's parent directory to create the
datasets, summary statistics, and figures. The default seed is `42`; the default
split is 2,000 training samples, 500 validation samples, and 100 test samples.
Each split uses an independent random-number stream derived from the seed.

Inputs are sampled uniformly over `x1 ∈ [-2, 2]` and `x2 ∈ [-4, 4]`. The target
is the bivariate Gaussian density

`f(x) = exp(-0.5 * (x - μ)^T C^-1 (x - μ)) / (2π sqrt(det(C)))`

with `μ = [0, 0]` and `C = [[0.3, -0.5], [-0.5, 2.0]]` (`det(C) = 0.35`).
CSV files have the columns `x1,x2,y`. For Experiment A, use the first 200, 400,
..., 2,000 rows of the fixed training CSV; keep validation and test data unchanged.
Use validation for model selection and reserve test data for final evaluation.

Generated files:

- `train.csv`, `validation.csv`, `test.csv`: labeled datasets.
- `dataset_metadata.json`: seed, sampling setup, and per-split descriptive statistics.
- `../figures/training_scatter_3d.png`: 3D scatter of training samples.
- `../figures/true_gaussian_surface.png`: exact target surface over the input domain.
- `../figures/training_distributions.png`: histograms of `x1`, `x2`, and `y`.

To regenerate with another seed or validation size, pass `--seed` or
`--validation-size`; the test set always contains 100 samples.

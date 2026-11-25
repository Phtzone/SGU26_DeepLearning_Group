# GRU preprocessing: full-budget Train/Validation comparison

Maximum 100 epochs per fold, patience 15, min_delta 0.001. Same subjects, seed, model and optimizer. No Test evaluation.

| Fold | Control | Window z-score | Difference | Epochs control/new |
|---|---:|---:|---:|---:|
| 1 | 0.3548 | 0.3397 | -0.0151 | 16/16 |
| 2 | 0.4733 | 0.4298 | -0.0435 | 30/18 |
| 3 | 0.4771 | 0.5960 | +0.1189 | 16/21 |
| 4 | 0.4794 | 0.4305 | -0.0490 | 16/18 |
| 5 | 0.4308 | 0.7177 | +0.2869 | 20/48 |

Mean per-person Validation Macro-F1: 0.4431 -> 0.5027.

Exploratory comparison with one training seed. Validation was used for selection; this is not independent Test performance. Source data and prior CNN/GRU runs are preserved.

# CNN preprocessing: full-budget Train/Validation comparison

Maximum 100 epochs per fold, patience 15, min_delta 0.001. Same subjects, seed, model and optimizer. No Test evaluation.

| Fold | Control | Window z-score | Difference | Epochs control/new |
|---|---:|---:|---:|---:|
| 1 | 0.3115 | 0.2950 | -0.0166 | 21/17 |
| 2 | 0.3822 | 0.5074 | +0.1251 | 29/17 |
| 3 | 0.5537 | 0.6569 | +0.1032 | 40/23 |
| 4 | 0.6132 | 0.5435 | -0.0697 | 23/16 |
| 5 | 0.3894 | 0.8244 | +0.4350 | 47/31 |

Mean per-person Validation Macro-F1: 0.4500 -> 0.5654.

Exploratory comparison with one training seed. Validation was used for selection; this is not independent Test performance. Source data and prior CNN/GRU runs are preserved.

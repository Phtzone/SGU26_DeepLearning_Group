# Frozen CNN/GRU Test evaluation after window normalization

Metric: Test Macro-F1 computed per person then averaged over 15 people. No training, tuning, or checkpoint selection on Test.

| Model | Historical Test | Current control Test | Window z-score Test | Change vs current control |
|---|---:|---:|---:|---:|
| CNN | 0.2962 | 0.3402 | 0.3964 | +0.0562 |
| GRU | 0.2864 | 0.2864 | 0.3567 | +0.0702 |

The current control and window-z-score checkpoints were trained under the same local budget and split. Historical CNN training may differ numerically; both comparisons are shown.
All 20 Validation checkpoint scores and probabilities replayed before Test scoring. Each variant covers 15 people and 2,134 Test windows exactly once. Input checkpoints and scalers are unchanged.
Historical Test results were already inspected before this preprocessing exploration. Treat this as exploratory same-dataset evidence, not a fresh independent confirmation. Do not keep adjusting to these Test scores.

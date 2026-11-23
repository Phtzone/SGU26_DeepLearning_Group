Trạng thái: **completed**. Hoàn tất 5/5 folds. Group: `20261123T072126Z_2c81daac`.

Metric chính: Macro-F1 tính riêng từng người rồi trung bình đều trên 15 người, cho mỗi seed. Độ lệch chuẩn giữa người không phải độ lệch chuẩn giữa seeds.

| Seed | Fold | Epochs | Best epoch | Test Macro-F1 | Balanced Accuracy | Duration (s) |
|---|---|---|---|---|---|---|
| 42 | 1 | 28 | 13 | 0.2735 | 0.3824 | 863.5 |
| 42 | 2 | 29 | 14 | 0.3567 | 0.3542 | 849.4 |
| 42 | 3 | 40 | 25 | 0.3207 | 0.3372 | 915.6 |
| 42 | 4 | 23 | 8 | 0.3777 | 0.4053 | 478.9 |
| 42 | 5 | 32 | 17 | 0.4532 | 0.4527 | 660.2 |

Seed 42; sample standard deviation (`ddof=1`):

| Aggregation | Macro-F1 | Balanced Accuracy |
|---|---|---|
| fold_level | 0.3564 ± 0.0670 | 0.3864 ± 0.0453 |
| subject_level | 0.2962 ± 0.1467 | 0.3858 ± 0.1365 |
| pooled_oof | 0.3866 | 0.3847 |

Macro-F1 giữa folds: 0.2735–0.4532; giữa subjects: S16 (0.0924) đến S9 (0.5624). Pooled OOF có 2134 windows; tổng thời lượng folds 3767.7 giây.


| Seed | Subject | Fold | Windows | Macro-F1 | Balanced Accuracy |
|---|---|---|---|---|---|
| 42 | S10 | 1 | 147 | 0.1266 | 0.1472 |
| 42 | S16 | 1 | 143 | 0.0924 | 0.3333 |
| 42 | S9 | 1 | 141 | 0.5624 | 0.6667 |
| 42 | S8 | 2 | 142 | 0.2971 | 0.3803 |
| 42 | S17 | 2 | 147 | 0.2606 | 0.3333 |
| 42 | S15 | 2 | 144 | 0.3842 | 0.3604 |
| 42 | S7 | 3 | 142 | 0.1494 | 0.3333 |
| 42 | S4 | 3 | 140 | 0.3021 | 0.3415 |
| 42 | S11 | 3 | 144 | 0.2529 | 0.3333 |
| 42 | S5 | 4 | 142 | 0.2963 | 0.3394 |
| 42 | S6 | 4 | 142 | 0.4518 | 0.6276 |
| 42 | S14 | 4 | 143 | 0.0976 | 0.2474 |
| 42 | S2 | 5 | 137 | 0.2358 | 0.3333 |
| 42 | S3 | 5 | 136 | 0.4991 | 0.4902 |
| 42 | S13 | 5 | 144 | 0.4346 | 0.5195 |

Các metrics được tính lại từ predictions và kiểm tra với manifest. Một training seed chưa đánh giá được độ ổn định theo khởi tạo. Cần đối chiếu các lượt chạy có cùng giao thức để so sánh kiến trúc.

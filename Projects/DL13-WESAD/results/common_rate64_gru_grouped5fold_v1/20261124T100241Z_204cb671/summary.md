Trạng thái: **completed**. Hoàn tất 5/5 folds. Group: `20261124T100241Z_204cb671`.

Metric chính: Macro-F1 tính riêng từng người rồi trung bình đều trên 15 người, cho mỗi seed. Độ lệch chuẩn giữa người không phải độ lệch chuẩn giữa seeds.

| Seed | Fold | Epochs | Best epoch | Test Macro-F1 | Balanced Accuracy | Duration (s) |
|---|---|---|---|---|---|---|
| 42 | 1 | 16 | 1 | 0.1110 | 0.2468 | 529.2 |
| 42 | 2 | 30 | 15 | 0.3260 | 0.3192 | 3385.3 |
| 42 | 3 | 16 | 1 | 0.2774 | 0.3398 | 522.8 |
| 42 | 4 | 16 | 15 | 0.4514 | 0.4461 | 484.1 |
| 42 | 5 | 20 | 5 | 0.4719 | 0.4613 | 640.2 |

Seed 42; sample standard deviation (`ddof=1`):

| Aggregation | Macro-F1 | Balanced Accuracy |
|---|---|---|
| fold_level | 0.3275 ± 0.1463 | 0.3626 ± 0.0902 |
| subject_level | 0.2864 ± 0.1913 | 0.3609 ± 0.1890 |
| pooled_oof | 0.3614 | 0.3603 |

Macro-F1 giữa folds: 0.1110–0.4719; giữa subjects: S10 (0.0000) đến S6 (0.7423). Pooled OOF có 2134 windows; tổng thời lượng folds 5561.5 giây.


| Seed | Subject | Fold | Windows | Macro-F1 | Balanced Accuracy |
|---|---|---|---|---|---|
| 42 | S10 | 1 | 147.0 | 0.0000 | 0.0000 |
| 42 | S16 | 1 | 143 | 0.0924 | 0.3333 |
| 42 | S9 | 1 | 141 | 0.2233 | 0.4069 |
| 42 | S8 | 2 | 142 | 0.1632 | 0.1193 |
| 42 | S17 | 2 | 147 | 0.3806 | 0.4113 |
| 42 | S15 | 2 | 144 | 0.3287 | 0.4258 |
| 42 | S7 | 3 | 142 | 0.1494 | 0.3333 |
| 42 | S4 | 3 | 140 | 0.2678 | 0.3496 |
| 42 | S11 | 3 | 144 | 0.1560 | 0.3333 |
| 42 | S5 | 4 | 142 | 0.2169 | 0.2906 |
| 42 | S6 | 4 | 142 | 0.7423 | 0.8064 |
| 42 | S14 | 4 | 143 | 0.2821 | 0.2386 |
| 42 | S2 | 5 | 137 | 0.2914 | 0.3623 |
| 42 | S3 | 5 | 136 | 0.3981 | 0.3563 |
| 42 | S13 | 5 | 144 | 0.6046 | 0.6468 |

Các metrics được tính lại từ predictions và kiểm tra với manifest. Một training seed chưa đánh giá được độ ổn định theo khởi tạo. Cần đối chiếu các lượt chạy có cùng giao thức để so sánh kiến trúc.

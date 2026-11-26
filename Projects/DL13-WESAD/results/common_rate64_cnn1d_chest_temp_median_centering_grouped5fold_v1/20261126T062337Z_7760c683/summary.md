Trạng thái: **completed**. Hoàn tất 5/5 folds. Group: `20261126T062337Z_7760c683`.

Metric chính: Macro-F1 tính riêng từng người rồi trung bình đều trên 15 người, cho mỗi seed. Độ lệch chuẩn giữa người không phải độ lệch chuẩn giữa seeds.

| Seed | Fold | Epochs | Best epoch | Test Macro-F1 | Balanced Accuracy | Duration (s) |
|---|---|---|---|---|---|---|
| 42 | 1 | 24 | 9 | 0.3294 | 0.4329 | 570.6 |
| 42 | 2 | 18 | 3 | 0.5848 | 0.6067 | 516.7 |
| 42 | 3 | 26 | 11 | 0.4398 | 0.4901 | 497.0 |
| 42 | 4 | 23 | 8 | 0.3753 | 0.3702 | 463.0 |
| 42 | 5 | 52 | 37 | 0.4719 | 0.4826 | 1007.9 |

Seed 42; sample standard deviation (`ddof=1`):

| Aggregation | Macro-F1 | Balanced Accuracy |
|---|---|---|
| fold_level | 0.4402 ± 0.0980 | 0.4765 ± 0.0871 |
| subject_level | 0.3918 ± 0.1783 | 0.4765 ± 0.1770 |
| pooled_oof | 0.4697 | 0.4744 |

Macro-F1 giữa folds: 0.3294–0.5848; giữa subjects: S16 (0.0924) đến S6 (0.6386). Pooled OOF có 2134 windows; tổng thời lượng folds 3055.1 giây.


| Seed | Subject | Fold | Windows | Macro-F1 | Balanced Accuracy |
|---|---|---|---|---|---|
| 42 | S10 | 1 | 147 | 0.2171 | 0.3074 |
| 42 | S16 | 1 | 143 | 0.0924 | 0.3333 |
| 42 | S9 | 1 | 141 | 0.5712 | 0.6580 |
| 42 | S8 | 2 | 142 | 0.4963 | 0.5251 |
| 42 | S17 | 2 | 147 | 0.5301 | 0.6667 |
| 42 | S15 | 2 | 144 | 0.5170 | 0.6407 |
| 42 | S7 | 3 | 142 | 0.1586 | 0.3376 |
| 42 | S4 | 3 | 140 | 0.5006 | 0.4878 |
| 42 | S11 | 3 | 144 | 0.5779 | 0.6450 |
| 42 | S5 | 4 | 142 | 0.2490 | 0.2804 |
| 42 | S6 | 4 | 142 | 0.6386 | 0.7106 |
| 42 | S14 | 4 | 143 | 0.1562 | 0.1163 |
| 42 | S2 | 5 | 137 | 0.3139 | 0.3716 |
| 42 | S3 | 5 | 136 | 0.4434 | 0.5697 |
| 42 | S13 | 5 | 144 | 0.4141 | 0.4978 |

Các metrics được tính lại từ predictions và kiểm tra với manifest. Một training seed chưa đánh giá được độ ổn định theo khởi tạo. Cần đối chiếu các lượt chạy có cùng giao thức để so sánh kiến trúc.

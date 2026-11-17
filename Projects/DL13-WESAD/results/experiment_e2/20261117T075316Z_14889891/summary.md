Trạng thái: **completed**. Hoàn tất 5/5 folds. Group: `20261117T075316Z_14889891`.

Metric chính: Macro-F1 tính riêng từng người rồi trung bình đều trên 15 người, cho mỗi seed. Độ lệch chuẩn giữa người không phải độ lệch chuẩn giữa seeds.

| Seed | Fold | Epochs | Best epoch | Test Macro-F1 | Balanced Accuracy | Duration (s) |
|---|---|---|---|---|---|---|
| 42 | 1 | 17 | 2 | 0.0382 | 0.1111 | 300.6 |
| 42 | 2 | 41 | 26 | 0.3777 | 0.3879 | 756.3 |
| 42 | 3 | 17 | 2 | 0.2840 | 0.3689 | 305.6 |
| 42 | 4 | 27 | 12 | 0.4673 | 0.4702 | 457.5 |
| 42 | 5 | 49 | 34 | 0.5182 | 0.5271 | 784.5 |

Seed 42; sample standard deviation (`ddof=1`):

| Aggregation | Macro-F1 | Balanced Accuracy |
|---|---|---|
| fold_level | 0.3371 ± 0.1894 | 0.3731 ± 0.1597 |
| subject_level | 0.3102 ± 0.2322 | 0.3719 ± 0.2392 |
| pooled_oof | 0.3603 | 0.3712 |

Macro-F1 giữa folds: 0.0382–0.5182; giữa subjects: S10 (0.0000) đến S6 (0.8156). Pooled OOF có 2136 windows; tổng thời lượng folds 2604.4 giây.


| Seed | Subject | Fold | Windows | Macro-F1 | Balanced Accuracy |
|---|---|---|---|---|---|
| 42 | S10 | 1 | 147 | 0.0000 | 0.0000 |
| 42 | S16 | 1 | 143 | 0.0000 | 0.0000 |
| 42 | S9 | 1 | 141 | 0.0941 | 0.3333 |
| 42 | S8 | 2 | 142 | 0.0852 | 0.0826 |
| 42 | S17 | 2 | 147 | 0.4914 | 0.5087 |
| 42 | S15 | 2 | 144 | 0.4471 | 0.5628 |
| 42 | S7 | 3 | 142 | 0.1494 | 0.3333 |
| 42 | S4 | 3 | 140 | 0.2192 | 0.2591 |
| 42 | S11 | 3 | 144 | 0.4327 | 0.5119 |
| 42 | S5 | 4 | 142 | 0.1991 | 0.1869 |
| 42 | S6 | 4 | 142 | 0.8156 | 0.8579 |
| 42 | S14 | 4 | 143 | 0.3041 | 0.3701 |
| 42 | S2 | 5 | 137 | 0.4695 | 0.5096 |
| 42 | S3 | 5 | 138 | 0.5675 | 0.6016 |
| 42 | S13 | 5 | 144 | 0.3779 | 0.4607 |

Các metrics được tính lại từ predictions và kiểm tra với manifest. Một training seed chưa đánh giá được độ ổn định theo khởi tạo. Cần đối chiếu các lượt chạy có cùng giao thức để so sánh kiến trúc.

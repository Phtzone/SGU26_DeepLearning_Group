Trạng thái: **completed**. Hoàn tất 5/5 folds. Group: `20261108T154303Z_474b38f9`.

| Seed | Fold | Epochs | Best epoch | Test Macro-F1 | Balanced Accuracy | Duration (s) |
|---|---|---|---|---|---|---|
| 42 | 1 | 46 | 31 | 0.5052 | 0.4927 | 752.4 |
| 42 | 2 | 17 | 2 | 0.2897 | 0.3041 | 255.8 |
| 42 | 3 | 27 | 12 | 0.5049 | 0.5145 | 404.9 |
| 42 | 4 | 23 | 8 | 0.4423 | 0.4565 | 409.9 |
| 42 | 5 | 24 | 9 | 0.5045 | 0.5155 | 383.2 |

Seed 42; sample standard deviation (`ddof=1`):

| Aggregation | Macro-F1 | Balanced Accuracy |
|---|---|---|
| fold_level | 0.4493 ± 0.0933 | 0.4567 ± 0.0886 |
| subject_level | 0.4110 ± 0.1929 | 0.4554 ± 0.1911 |
| pooled_oof | 0.4500 | 0.4547 |

Macro-F1 giữa folds: 0.2897–0.5052; giữa subjects: S8 (0.0303) đến S11 (0.7298). Pooled OOF có 2136 windows; tổng thời lượng folds 2206.3 giây.


| Seed | Subject | Fold | Windows | Macro-F1 | Balanced Accuracy |
|---|---|---|---|---|---|
| 42 | S10 | 1 | 147 | 0.6170 | 0.6230 |
| 42 | S16 | 1 | 143 | 0.5201 | 0.5447 |
| 42 | S9 | 1 | 141 | 0.2682 | 0.3030 |
| 42 | S8 | 2 | 142 | 0.0303 | 0.0263 |
| 42 | S17 | 2 | 147 | 0.3543 | 0.3783 |
| 42 | S15 | 2 | 144 | 0.3740 | 0.5043 |
| 42 | S7 | 3 | 142 | 0.1407 | 0.1624 |
| 42 | S4 | 3 | 140 | 0.5901 | 0.6701 |
| 42 | S11 | 3 | 144 | 0.7298 | 0.7040 |
| 42 | S5 | 4 | 142 | 0.3133 | 0.4231 |
| 42 | S6 | 4 | 142 | 0.5325 | 0.6206 |
| 42 | S14 | 4 | 143 | 0.2557 | 0.3348 |
| 42 | S2 | 5 | 137 | 0.6014 | 0.5850 |
| 42 | S3 | 5 | 138 | 0.3677 | 0.4251 |
| 42 | S13 | 5 | 144 | 0.4699 | 0.5265 |

Các metrics được tính lại từ predictions và kiểm tra với manifest. Hiệu suất biến thiên theo subject trong WESAD; các kết quả này chưa xác định được nguyên nhân của khác biệt. Một training seed chưa đánh giá được độ ổn định theo khởi tạo. Chưa có kết quả E2/E3 để so sánh kiến trúc.

Trạng thái: **completed**. Hoàn tất 5/5 folds. Group: `20261115T101928Z_0ce52f72`.

| Seed | Fold | Epochs | Best epoch | Test Macro-F1 | Balanced Accuracy | Duration (s) |
|---|---|---|---|---|---|---|
| 42 | 1 | 57 | 42 | 0.4860 | 0.4740 | 243.8 |
| 42 | 2 | 17 | 2 | 0.2897 | 0.3041 | 74.8 |
| 42 | 3 | 27 | 12 | 0.5602 | 0.5535 | 103.6 |
| 42 | 4 | 23 | 8 | 0.4498 | 0.4653 | 61.4 |
| 42 | 5 | 24 | 9 | 0.5324 | 0.5396 | 64.3 |

Seed 42; sample standard deviation (`ddof=1`):

| Aggregation | Macro-F1 | Balanced Accuracy |
|---|---|---|
| fold_level | 0.4636 ± 0.1061 | 0.4673 ± 0.0992 |
| subject_level | 0.4242 ± 0.2106 | 0.4664 ± 0.2037 |
| pooled_oof | 0.4623 | 0.4654 |

Macro-F1 giữa folds: 0.2897–0.5602; giữa subjects: S8 (0.0303) đến S4 (0.7996). Pooled OOF có 2136 windows; tổng thời lượng folds 547.9 giây.


| Seed | Subject | Fold | Windows | Macro-F1 | Balanced Accuracy |
|---|---|---|---|---|---|
| 42 | S10 | 1 | 147 | 0.5724 | 0.5727 |
| 42 | S16 | 1 | 143 | 0.5149 | 0.5404 |
| 42 | S9 | 1 | 141 | 0.2652 | 0.3030 |
| 42 | S8 | 2 | 142 | 0.0303 | 0.0263 |
| 42 | S17 | 2 | 147 | 0.3543 | 0.3783 |
| 42 | S15 | 2 | 144 | 0.3740 | 0.5043 |
| 42 | S7 | 3 | 142 | 0.1522 | 0.1795 |
| 42 | S4 | 3 | 140 | 0.7996 | 0.8192 |
| 42 | S11 | 3 | 144 | 0.6931 | 0.6586 |
| 42 | S5 | 4 | 142 | 0.2969 | 0.4103 |
| 42 | S6 | 4 | 142 | 0.5526 | 0.6598 |
| 42 | S14 | 4 | 143 | 0.2557 | 0.3348 |
| 42 | S2 | 5 | 137 | 0.6461 | 0.6329 |
| 42 | S3 | 5 | 138 | 0.3751 | 0.4413 |
| 42 | S13 | 5 | 144 | 0.4807 | 0.5352 |

Các metrics được tính lại từ predictions và kiểm tra với manifest. Hiệu suất biến thiên theo subject trong WESAD; các kết quả này chưa xác định được nguyên nhân của khác biệt. Một training seed chưa đánh giá được độ ổn định theo khởi tạo. Chưa có kết quả E2/E3 để so sánh kiến trúc.

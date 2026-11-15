Trạng thái: **completed**. Hoàn tất 5/5 folds. Group: `20261115T101531Z_c2d7bd4b`.

| Seed | Fold | Epochs | Best epoch | Test Macro-F1 | Balanced Accuracy | Duration (s) |
|---|---|---|---|---|---|---|
| 42 | 1 | 39 | 24 | 0.5352 | 0.5199 | 106.4 |
| 42 | 2 | 17 | 2 | 0.2880 | 0.3027 | 43.5 |
| 42 | 3 | 27 | 12 | 0.5764 | 0.5646 | 74.9 |
| 42 | 4 | 30 | 15 | 0.4137 | 0.4311 | 111.3 |
| 42 | 5 | 24 | 9 | 0.5571 | 0.5651 | 109.0 |

Seed 42; sample standard deviation (`ddof=1`):

| Aggregation | Macro-F1 | Balanced Accuracy |
|---|---|---|
| fold_level | 0.4741 ± 0.1218 | 0.4767 ± 0.1115 |
| subject_level | 0.4385 ± 0.2423 | 0.4758 ± 0.2275 |
| pooled_oof | 0.4725 | 0.4750 |

Macro-F1 giữa folds: 0.2880–0.5764; giữa subjects: S8 (0.0303) đến S4 (0.9743). Pooled OOF có 2136 windows; tổng thời lượng folds 445.0 giây.


| Seed | Subject | Fold | Windows | Macro-F1 | Balanced Accuracy |
|---|---|---|---|---|---|
| 42 | S10 | 1 | 147 | 0.7063 | 0.7062 |
| 42 | S16 | 1 | 143 | 0.5172 | 0.5395 |
| 42 | S9 | 1 | 141 | 0.2690 | 0.2987 |
| 42 | S8 | 2 | 142 | 0.0303 | 0.0263 |
| 42 | S17 | 2 | 147 | 0.3543 | 0.3783 |
| 42 | S15 | 2 | 144 | 0.3673 | 0.5000 |
| 42 | S7 | 3 | 142 | 0.1442 | 0.1581 |
| 42 | S4 | 3 | 140 | 0.9743 | 0.9666 |
| 42 | S11 | 3 | 144 | 0.6131 | 0.5720 |
| 42 | S5 | 4 | 142 | 0.2712 | 0.3932 |
| 42 | S6 | 4 | 142 | 0.4628 | 0.5614 |
| 42 | S14 | 4 | 143 | 0.2621 | 0.3478 |
| 42 | S2 | 5 | 137 | 0.7066 | 0.6756 |
| 42 | S3 | 5 | 138 | 0.4342 | 0.4910 |
| 42 | S13 | 5 | 144 | 0.4642 | 0.5222 |

Các metrics được tính lại từ predictions và kiểm tra với manifest. Hiệu suất biến thiên theo subject trong WESAD; các kết quả này chưa xác định được nguyên nhân của khác biệt. Một training seed chưa đánh giá được độ ổn định theo khởi tạo. Chưa có kết quả E2/E3 để so sánh kiến trúc.

# Test CNN và GRU sau khi đổi tiền xử lý

Đánh giá các checkpoint đã chọn bằng Validation, không huấn luyện lại. Phần này chỉ gồm CNN và GRU, không có lượt E3. Nguồn dữ liệu, 14 kênh, 64 Hz, window 30 giây/stride 15 giây và manifest 5 folds 9/3/3 giữ nguyên.

| Mô hình | Test lịch sử | Đối chứng chạy lại | Window z-score | Chênh lệch so với đối chứng chạy lại |
|---|---:|---:|---:|---:|
| CNN | 0.2962 | 0.3402 | 0.3964 | +0.0562 |
| GRU | 0.2864 | 0.2864 | 0.3567 | +0.0702 |

Metric chính: Macro-F1 tính riêng từng người rồi trung bình đều trên 15 người. Không nhầm với Accuracy hoặc Macro-F1 gộp cửa sổ. CNN đối chứng chạy lại khác CNN lịch sử; dùng đối chứng cùng đợt để so ảnh hưởng của bước chuẩn hóa. Mới có một seed, không khẳng định cải thiện ổn định cho mọi người. Test lịch sử đã được xem trước đợt khám phá này nên đây là bằng chứng thăm dò trên cùng bộ dữ liệu.

## Minh chứng để trình bày

- `test_comparison.png`: so sánh điểm trung bình của CNN/GRU.
- `figures/test_by_subject.png`: xem người nào tăng, giảm; không chỉ nhìn trung bình.
- `figures/CNN_training_curves.png` và `figures/GRU_training_curves.png`: loss và Macro-F1 của Train/Validation trên đủ 5 folds. Train/Val pooled là gộp cửa sổ; Val mean/person là tiêu chí chọn checkpoint.
- `figures/CNN_test_confusion.png` và `figures/GRU_test_confusion.png`: nhầm lẫn giữa ba lớp, gộp dự đoán Test của 15 người. Đây là hình mô tả phân loại, không thay tiêu chí trung bình theo người.

## Số liệu gốc và truy vết

- `summary.json`: Macro-F1, Balanced Accuracy, Accuracy trung bình theo người.
- `CNN|GRU/control|window_zscore/subject_metrics.csv`: điểm từng người.
- `CNN|GRU/control|window_zscore/oof_predictions.csv`: 2.134 dự đoán Test, mỗi cửa sổ đúng một lần.
- Trong mỗi `fold_XX/`: predictions.csv, subject_metrics.csv, test.json, classification_report.json, confusion_matrix.csv và ảnh confusion matrix.
- `run.log`: tổng kết đánh giá theo fold; không phải log huấn luyện.
- `verification.json`: 20 lượt checkpoint tái tạo chính xác xác suất Validation trước đánh giá Test; scaler chỉ từ Train, không sửa kết quả cũ.
- `independent_metrics_check.json`: tính lại Test Macro-F1 từ bốn file dự đoán OOF, khớp tổng hợp.
- `frozen_inputs.json`, `source_checksums.json`, `window_index/`: đầu vào đã khóa, hash và chỉ mục cửa sổ.

Checkpoint, scaler và log huấn luyện gốc ở hai thư mục đã lưu trước đó:

- `results/preprocessing_window_zscore_cnn_full/20261124_full_seed42/`
- `results/preprocessing_window_zscore_gru_full/20261124_full_seed42/`

## Lệnh tái lập

Từ thư mục `Projects/DL13-WESAD` (đã có dữ liệu cục bộ đúng hash và hai lượt huấn luyện trên):

```powershell
python -u -B -X utf8 -m scripts.evaluate_window_zscore_test --output results/window_zscore_test/NEW_RUN_ID --disable-optional-pyarrow
python -B -m scripts.plot_window_zscore_test --result results/window_zscore_test/NEW_RUN_ID
```

Không ghi đè thư mục kết quả hiện tại. Script vẽ chỉ đọc các CSV đã có, không huấn luyện hay đánh giá thêm trên Test.

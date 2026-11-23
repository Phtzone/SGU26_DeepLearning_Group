# CNN-1D trên WESAD 14 kênh, 64 Hz

Thí nghiệm `common_rate64_cnn1d_grouped5fold_v1` dùng các tín hiệu ngực và cổ tay đã đồng bộ bởi [notebook 02](../../notebooks/02_preprocessing_common_rate.ipynb). [Cấu hình](config.yaml) đặt cả dữ liệu lẫn kết quả dưới mã thí nghiệm riêng. E1/E2 lịch sử vẫn dùng cấu hình và thư mục của từng thí nghiệm.

## Dữ liệu và giao thức

- Đặt nguyên các tệp xuất của notebook tại `data/processed/common_rate_64hz/`. Runner đối chiếu 15 subject, thứ tự 14 cột, kiểu `float32`, chiều dài, thời gian kết thúc nhãn, mask Boolean và SHA256. Các mảng được mở read-only bằng NumPy memory mapping.
- Với mỗi đoạn nhãn `[start, end)` ở 700 Hz, chỉ lấy mẫu 64 Hz có cả khoảng mẫu nằm trong đoạn. Chỉ giữ nhãn gốc 1, 2, 3 và ánh xạ lần lượt thành Baseline, Stress, Amusement. Cửa sổ 1.920 mẫu, bước 960 mẫu, bắt đầu từ biên 64 Hz bên trong từng đoạn. Cả 1.920 mẫu phải có mask `True`.
- `window_index/windows.csv` trong group là manifest đầy đủ. Các CSV theo người phục vụ Dataset và đối chiếu OOF. Dữ liệu mẫu được cắt khi DataLoader yêu cầu, không có bản sao mảng cửa sổ trên đĩa.
- Dùng [split seed 42](../../splits/grouped_5fold_seed42.json), 9 Train / 3 Validation / 3 Test trong mỗi fold. Scaler theo kênh và trọng số lớp lấy từ Train. Scaler tính population std trên các timestep trong cửa sổ Train, kể cả overlap, và đặt std về 1 khi nhỏ hơn `1e-8`.
- CNN-1D có ba khối convolution 32, 64, 128 kênh, BatchNorm và ReLU. Hai khối đầu có MaxPool stride 2. Global average pooling theo thời gian, Linear 128 → 64, dropout 0,3 và Linear 64 → 3 tạo logits. Đầu vào `[B, 14, 1920]`, đầu ra `[B, 3]`, 47.075 tham số trainable.
- Mỗi fold khởi tạo model và Adam từ seed 42. Dùng learning rate 0,001, weight decay 0,0001, batch 32, weighted cross-entropy, gradient clipping 1,0 và learning rate cố định. Tối đa 100 epochs, patience 15, min_delta 0,001. Chọn checkpoint theo Macro-F1 tính riêng từng người Validation rồi trung bình đều. Khi hòa trong `1e-12`, lấy Validation loss thấp hơn. Nạp checkpoint đó trước khi đánh giá Test.

## Chạy và tiếp tục

Chạy từ thư mục dự án với Python có các gói trong [requirements.txt](../../requirements.txt) và [requirements-tracking.txt](../../requirements-tracking.txt). Lệnh tạo run mới sẽ sinh GROUP_ID mới.

```powershell
# Kiểm tra toàn bộ dữ liệu và kế hoạch ở chế độ chỉ đọc
python -m scripts.run_grouped_cv --experiment experiments/common_rate64_cnn1d_grouped5fold_v1/config.yaml --dry-run

# Chạy tuần tự đủ năm fold
python -m scripts.run_grouped_cv --experiment experiments/common_rate64_cnn1d_grouped5fold_v1/config.yaml

# Chạy một fold trong một group riêng nếu cần khảo sát sau này
python -m scripts.run_grouped_cv --experiment experiments/common_rate64_cnn1d_grouped5fold_v1/config.yaml --folds 1

# Tiếp tục đúng group và fold bị gián đoạn
python -m scripts.run_grouped_cv --experiment experiments/common_rate64_cnn1d_grouped5fold_v1/config.yaml --resume --group-id GROUP_ID
```

Resume xác minh effective config, split, dữ liệu và window index. Các fold đã hoàn tất được bỏ qua sau khi kiểm tra `completion.json`. `last.pt` lưu trạng thái optimizer, RNG, lịch sử epoch đã commit và mô hình tốt nhất của chính fold đó.

## Kết quả và TensorBoard

Lượt chạy chính thức **`20261123T072126Z_2c81daac`** đã hoàn tất và được kiểm chứng đủ 5/5 fold. Macro-F1 trung bình của 15 người Test đạt **0,2962 ± 0,1467**, với độ lệch chuẩn mẫu. Xem [báo cáo](../../results/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac/report.md), [hậu kiểm](../../results/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac/verification.json) và [Research Journal](Research_journal_common_rate64_cnn1d_grouped5fold_v1.xlsx).

[Thư mục runs](../../runs/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac/) lưu split, window index, effective config, môi trường, source snapshot, logs và checkpoints. Mỗi `seed_42/fold_XX` có `normalization.json`, `logs/`, `tensorboard/`, `checkpoints/`, `metrics/` và `plots/`.

```powershell
python -m tensorboard.main --logdir runs/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac --port 6006
```

Mở `http://localhost:6006`. Stream `epochs` ghi Train/Validation và lựa chọn model, `batches` ghi loss, learning rate và gradient norm theo global step, `evaluation` ghi Test và hình đánh giá. `val/subject_macro_f1` ghi chỉ số lựa chọn theo người. `val/macro_f1` ghi chỉ số gộp các cửa sổ Validation như trong các run trước.

Sau khi đủ năm fold, chạy kiểm chứng artifacts bằng `python -m scripts.verify_common_rate --group-id GROUP_ID`. Kết quả chính là trung bình Macro-F1 của 15 người Test với trọng số bằng nhau. Báo cáo lưu riêng trung bình giữa fold, pooled OOF và độ lệch chuẩn mẫu. Macro-F1 và Balanced Accuracy dùng nhãn cố định `[0, 1, 2]`, gán 0 cho lớp không có mẫu thật khi tính trung bình theo lớp.

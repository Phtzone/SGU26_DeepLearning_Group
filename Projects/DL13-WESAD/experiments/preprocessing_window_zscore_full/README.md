# CNN: kiểm tra tiền xử lý bằng ngân sách huấn luyện đầy đủ

Lượt thử 5 epochs/fold cho thấy Validation tăng. Lượt này kiểm tra lại bằng cùng ngân sách với CNN gốc: tối đa 100 epochs/fold, patience 15, min_delta 0.001. Chạy lại cả control và window_zscore trên cùng máy, seed 42, đủ 5 folds 9 Train / 3 Validation / 3 Test. Chỉ tạo Dataset Train/Validation, không đánh giá Test.

Giữ nguyên 14 kênh, 64 Hz, window 30 giây/stride 15 giây, manifest, CNN, loss, optimizer và class weights từ Train. Nhánh mới chuẩn hóa mean/std riêng mỗi cửa sổ trên 10 kênh dao động; giữ EDA và nhiệt độ trước scaler theo kênh fit chỉ trên Train. Không sửa dữ liệu gốc hay kết quả CNN/GRU trước đó.

## Chạy

Từ thư mục `Projects/DL13-WESAD`:

```powershell
python -u -B -X utf8 -m scripts.probe_window_preprocessing --config experiments/preprocessing_window_zscore_full/config.yaml --output results/preprocessing_window_zscore_cnn_full/20261124_full_seed42 --disable-optional-pyarrow
```

Lượt này đã được khởi chạy. Không chạy thêm lệnh trên khi lượt hiện tại đang chạy; chương trình từ chối ghi đè thư mục đã có.

## Theo dõi

Kết quả: `results/preprocessing_window_zscore_cnn_full/20261124_full_seed42/`.

- `live.json`: fold, nhánh, epoch và điểm Validation tốt nhất hiện tại; trạng thái running/completed/failed.
- `run.log`: nhật ký huấn luyện.
- `progress.json`: các nhánh đã hoàn thành.
- `fold_XX/control/` và `fold_XX/window_zscore/`: batches.csv, epochs.csv, normalization.json, best.pt, TensorBoard và dự đoán Validation sau khi xong nhánh.
- `summary.json`, `verification.json`, `report.md`: chỉ có khi hoàn thành toàn bộ. Checkpoint tốt nhất được nạp lại và kiểm tra khớp điểm Validation trước khi tổng hợp.

```powershell
Get-Content results/preprocessing_window_zscore_cnn_full/20261124_full_seed42/live.json
python -m tensorboard.main --logdir results/preprocessing_window_zscore_cnn_full/20261124_full_seed42 --port 6007
```

Sau khi mở TensorBoard, xem `http://127.0.0.1:6007`. Train/Validation loss và Macro-F1 nằm ở Scalars. `val/subject_macro_f1` là tiêu chí chọn checkpoint; `val/macro_f1` gộp cửa sổ là số liệu phụ.

## Đọc kết quả

So sánh điểm tốt nhất trên Validation của từng fold rồi lấy trung bình theo người. Hai nhánh có cùng mức trần và quy tắc dừng sớm, số epochs thực tế có thể khác. Đây vẫn là lựa chọn phương pháp bằng Validation trên một seed, chưa phải kết quả Test độc lập và chưa chứng minh hiệu quả cho GRU. Chưa tự thay quy trình chính của nhóm.

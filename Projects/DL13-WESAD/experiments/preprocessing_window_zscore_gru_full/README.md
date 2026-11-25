# GRU: đối chiếu hai cách tiền xử lý trên Train/Validation

Chạy GRU đối chứng và GRU dùng window z-score với cùng dữ liệu, manifest 5 folds (9 Train / 3 Validation / 3 Test), seed 42, tối đa 100 epochs/fold và dừng sớm patience 15, min_delta 0.001. Chỉ tạo Dataset Train/Validation, không đánh giá Test.

Giữ nguyên mô hình GRU 19.715 tham số, 14 kênh, 64 Hz, window 30 giây/stride 15 giây, Adam, weighted cross entropy và class weights fit Train từ cấu hình GRU gốc. Hai nhánh chỉ khác phép chuẩn hóa riêng cửa sổ trên 10 kênh. EDA và nhiệt độ giữ nguyên trước scaler theo kênh fit Train. Dùng cùng phép biến đổi đã thử trên CNN; không đổi dữ liệu gốc hay kết quả CNN/GRU cũ.

Từ thư mục `Projects/DL13-WESAD`:

```powershell
python -u -B -X utf8 -m scripts.probe_window_preprocessing --config experiments/preprocessing_window_zscore_gru_full/config.yaml --output results/preprocessing_window_zscore_gru_full/20261124_full_seed42 --disable-optional-pyarrow
```

Không chạy thêm lệnh này nếu tiến trình đang chạy. Thư mục có sẵn không được ghi đè.

Theo dõi `results/preprocessing_window_zscore_gru_full/20261124_full_seed42/live.json` và `run.log`. Mỗi fold/nhánh có batches.csv, epochs.csv, normalization.json, best.pt và TensorBoard. Khi xong nhánh, checkpoint tốt nhất được nạp lại để kiểm tra điểm, rồi xuất validation_predictions.csv. Chỉ khi xong toàn bộ mới có summary.json, verification.json và report.md.

```powershell
python -m tensorboard.main --logdir results/preprocessing_window_zscore_gru_full/20261124_full_seed42 --port 6008
```

Mở `http://127.0.0.1:6008`, xem train/loss, train/macro_f1, val/loss và val/subject_macro_f1. Tiêu chí chọn checkpoint là Macro-F1 trung bình theo người Validation; không dùng điểm Test để lựa chọn.

Sau khi hoàn thành, đối chiếu với CNN ở `results/preprocessing_window_zscore_cnn_full/20261124_full_seed42/summary.json`. Cần xét cả trung bình và từng fold; đây là kết quả chọn phương pháp trên Validation với một seed, chưa phải kết luận về Test hay CNN-GRU.

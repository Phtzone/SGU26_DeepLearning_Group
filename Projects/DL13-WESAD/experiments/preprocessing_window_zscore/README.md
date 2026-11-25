# Thử tiền xử lý trên CNN — Train/Validation

Nhánh Random Forest đã xóa theo phạm vi nhóm thống nhất: ba mô hình chính vẫn là CNN-1D, GRU và CNN-GRU. Đây chỉ là thử nghiệm thay tiền xử lý của CNN, không thêm kiến trúc mới.

## Giả thuyết

Chênh lệch mức nền/biên độ giữa người có thể làm mô hình học các dấu hiệu khó tổng quát. Thử chuẩn hóa mean/std riêng từng cửa sổ cho 10 kênh dao động: ACC ngực/cổ tay, ECG, EMG, Resp và BVP. Giữ nguyên giá trị EDA ngực/cổ tay và nhiệt độ ngực/cổ tay trước khi fit scaler Train.

Thứ tự xử lý: lấy cửa sổ → biến đổi cửa sổ (nếu là nhánh mới) → fit mean/std theo kênh chỉ bằng Train → áp dụng scaler đó cho Train/Validation. Không dùng thống kê cả bản ghi người Validation/Test, không dùng nhãn để tính biến đổi. Kênh hằng được đưa về 0 sau trừ mean, tránh chia 0.

Chuẩn hóa theo cửa sổ có thể loại bỏ thông tin biên độ hữu ích; vì vậy đây là giả thuyết cần thử, không bảo đảm tăng điểm.

## Đối chứng

- Cùng dữ liệu đã khớp hash CNN, 14 kênh, 64 Hz, 2.134 cửa sổ, window 30s/stride 15s.
- Cùng manifest 5 folds 9 Train / 3 Validation / 3 Test và seed 42.
- Hai nhánh `control` và `window_zscore` đều dùng CNN hiện có, Adam/loss/dropout như cấu hình CNN gốc.
- Mỗi nhánh chạy 5 epochs/fold, chọn checkpoint bằng Validation Macro-F1 trung bình theo người; tổng 10 lượt ngắn. Đây là ngân sách khảo sát, không so trực tiếp với điểm Test hoặc full-run CNN đã huấn luyện tới 100 epochs.
- Không tạo Test Dataset, không đánh giá Test. Giữ nguyên dữ liệu gốc, các cấu hình và kết quả CNN/GRU cũ.

## Chạy và theo dõi

```powershell
python -m scripts.probe_window_preprocessing --disable-optional-pyarrow
```

Kết quả lần này: `results/preprocessing_window_zscore/20261124T140004Z_b893f0f6/`. Xem `run.log`, `progress.json`; chỉ khi có `summary.json` trạng thái completed và `verification.json` đạt mới coi là đã hoàn thành khảo sát. Mỗi fold/nhánh có epochs.csv, scaler, best checkpoint và dự đoán Validation.

7 kiểm thử Dataset/scaler/chuẩn hóa đạt; chi tiết `checks/pytest.xml`. Đã kiểm tra kênh EDA/TEMP giữ nguyên, kênh hằng không tạo NaN, đầu vào không bị sửa và scaler chỉ fit trên cửa sổ Train đã biến đổi.

## Kết quả khảo sát

Đã hoàn thành cả 5 folds, mỗi nhánh 5 epochs/fold. Validation Macro-F1 trung bình theo người tăng từ 0.3566 (control) lên 0.5222 (window_zscore), chênh lệch +0.1656; cả 5 folds tăng. Đây là khảo sát ngắn trên Validation, chưa phải kết quả Test hay kết luận cho huấn luyện đầy đủ.

Xem `report.md`, `validation_comparison.png` và `independent_verification.json` trong thư mục kết quả trên. Kiểm tra lại dự đoán/nhãn/ID cửa sổ và scaler Train đều đạt. Kết quả CNN/GRU trước đó giữ nguyên. So với log CNN lịch sử, control có 3/25 điểm epoch lệch nhẹ (tối đa khoảng 0.0070), ghi trong `historical_control_comparison.json`; bảng so sánh chính dùng hai nhánh cùng chạy lần này.

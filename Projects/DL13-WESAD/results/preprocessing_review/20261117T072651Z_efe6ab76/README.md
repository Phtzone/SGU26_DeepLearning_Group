# Kiểm tra dữ liệu trước khi làm E2/E3

Có 15 người và 2136 cửa sổ. Mỗi cửa sổ dài 30 giây, bước 15 giây.

| Lớp | Số cửa sổ | Tỷ lệ |
|---|---:|---:|
| baseline | 1150 | 53.84% |
| stress | 640 | 29.96% |
| amusement | 346 | 16.20% |

Đã kiểm tra hash dữ liệu gốc/cache, cấu hình cache, nhãn, kích thước và NaN/Inf cho đủ 15 người.
Có 27 cặp cửa sổ–kênh có giá trị không đổi hoàn toàn; xem channel_checks.csv. Đây là thống kê, không tự động coi là lỗi hay loại dữ liệu.

Các trường hợp phẳng thuộc các kênh: ACC-x, ACC-y. Một cửa sổ có thể được đếm ở nhiều kênh.

Ba hình tín hiệu lấy từ S2, thuộc Train của Fold 1. Mỗi lớp lấy cửa sổ đầu tiên, hiển thị 10 giây đầu.
Ba cột lần lượt là tín hiệu gốc, sau resampling về 32 Hz và sau chuẩn hóa bằng scaler Train đã lưu.
Resampling được tính lại từ dữ liệu gốc và đối chiếu khớp cache cho cả ba cửa sổ.

- [Bảng số cửa sổ từng người](windows_by_subject.csv)
- [Phân bố ba lớp](class_distribution.png)
- [Kiểm tra từng kênh](channel_checks.csv)
- [baseline — S2](S2_baseline_preprocessing.png)
- [stress — S2](S2_stress_preprocessing.png)
- [amusement — S2](S2_amusement_preprocessing.png)

Chưa đổi tiền xử lý hay chạy mô hình. Ảnh chỉ minh họa các đoạn đã chọn, chưa đủ để kết luận toàn bộ dữ liệu không có nhiễu.
Các cửa sổ chồng lấp nên số cửa sổ không phải số quan sát độc lập. Khi báo cáo metric chính vẫn tính theo người.
Nếu chưa thấy vấn đề cụ thể cần xử lý, giữ nguyên pipeline này khi so sánh CNN, GRU và CNN-GRU.

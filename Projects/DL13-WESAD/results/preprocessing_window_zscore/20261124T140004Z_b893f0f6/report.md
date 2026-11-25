# CNN — thăm dò chuẩn hóa theo cửa sổ

Cùng 14 kênh, nguồn, split 9/3/3, window 30s/stride 15s và seed 42. Hai nhánh có cùng ngân sách 5 epochs mỗi fold. Nhánh mới chuẩn hóa riêng từng cửa sổ cho 10 kênh dao động; giữ EDA và nhiệt độ ở thang gốc trước scaler Train-only. Không đánh giá Test.

| Nhánh | Validation Macro-F1 trung bình theo người |
|---|---:|
| control | 0.3566 |
| window_zscore | 0.5222 |

Thử ngắn 5 epochs chưa đủ kết luận hiệu quả khi huấn luyện đầy đủ hoặc trên Test. Không so điểm này trực tiếp với Test của CNN/GRU đã chạy trước.

| Fold | Control | Chuẩn hóa cửa sổ | Chênh lệch |
|---|---:|---:|---:|
| 1 | 0.2699 | 0.2948 | +0.0249 |
| 2 | 0.2668 | 0.5074 | +0.2406 |
| 3 | 0.4040 | 0.5504 | +0.1464 |
| 4 | 0.5033 | 0.5435 | +0.0402 |
| 5 | 0.3392 | 0.7149 | +0.3757 |

Trung bình tăng 0.1656; cả 5 folds tăng trong phép thử này. Điểm lấy ở epoch có Validation tốt nhất trong 5 epochs của từng nhánh. Cần thử huấn luyện đầy đủ trước khi chốt tiền xử lý.

Kiểm tra độc lập (`independent_verification.json`): tính lại điểm từ dự đoán Validation, kiểm tra nhãn/ID cửa sổ, scaler chỉ fit Train và giữ nguyên kết quả CNN/GRU cũ đều đạt. Mỗi nhánh có tổng 2.134 cửa sổ Validation qua 5 folds. Không đánh giá Test.

Đối chiếu control với 5 epochs đầu của log CNN lịch sử: 22/25 điểm trùng, 3 điểm lệch, lớn nhất khoảng 0.0070; chưa xác định nguyên nhân. Chi tiết trong `historical_control_comparison.json`. So sánh chính dùng hai nhánh vừa chạy trên cùng máy, không giả định lượt control tái lập tuyệt đối log lịch sử.

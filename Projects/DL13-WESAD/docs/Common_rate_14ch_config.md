# Cấu hình chính: Common-rate 14 kênh

**Cập nhật 24/11/2026:** tài liệu bên dưới ghi lại bản đề xuất cục bộ trước khi pull runner mới. Cấu hình CNN hoạt động hiện nằm tại `experiments/common_rate64_cnn1d_grouped5fold_v1/config.yaml`; cấu hình GRU đối chứng tại `experiments/common_rate64_gru_grouped5fold_v1/config.yaml`. Không dùng bản trong `configs/` để huấn luyện. GRU còn chờ đối chiếu dữ liệu nguồn với CNN; xem README trong thư mục GRU.

File: `configs/common_rate_14ch.yaml`. Đây là cấu hình riêng, không kế thừa ngầm danh sách cảm biến cổ tay của `configs/default.yaml`.

- Đọc cửa sổ đã tạo từ `data/processed/common_rate_windows_64hz_30s_15s/`.
- Đầu vào: 14 kênh ngực/cổ tay, 64 Hz, 30 giây, stride 15 giây; mỗi mẫu `[14, 1920]`.
- Thứ tự kênh phải khớp cả dữ liệu liên tục, cache cửa sổ và metadata của từng người.
- Nhãn: 1 Baseline → 0, 2 Stress → 1, 3 Amusement → 2.
- Dùng manifest 5 folds theo người đang có; mỗi fold 9 Train / 3 Validation / 3 Test.
- Dự kiến chuẩn hóa Z-score bằng Train của từng fold, weighted cross-entropy, Adam, seed 42. Chưa fit scaler hay huấn luyện ở bước này. Thông số huấn luyện là điểm bắt đầu, không phải thông số đã được chứng minh tối ưu.
- Chỉ số Test chính: Macro-F1 tính riêng từng người rồi lấy trung bình đều. Checkpoint chọn bằng Validation Macro-F1.
- `status: awaiting_runner_integration`: cấu hình đã chuẩn bị, runner chưa nối cache mới. Output namespace dự kiến riêng cho bài 14 kênh; runner cũ chưa dùng trường này.

Kiểm tra cấu hình và dữ liệu hiện có, không huấn luyện:

```powershell
python -m scripts.check_common_rate_config
```

Nếu đưa cấu hình này vào `scripts.run_grouped_cv` cũ, lệnh sẽ dừng ngay, tránh vô tình chạy lại preprocessing 5 kênh. Bước tiếp theo là tích hợp đọc cache mới, chuẩn hóa Train và namespace đầu ra vào runner.

## Vì sao hiện chưa chạy LOSO?

**LOSO làm được trên WESAD.** Giới hạn hiện tại là runner và manifest của dự án đang được xây cho 5 folds cố định, không phải do dữ liệu có 14 kênh.

| Giao thức | Một lượt chia | Số lượt ngoài |
|---|---|---:|
| Grouped 5-fold hiện tại | 9 Train / 3 Validation / 3 Test | 5 |
| LOSO | 1 người Test; chia Train/Validation từ 14 người còn lại | 15 |

Muốn đổi sang LOSO cần tạo manifest mới, quy định Validation rõ ràng (ví dụ 11 người Train / 3 Validation / 1 Test), hỗ trợ manifest đó trong loader/runner và kiểm tra lại tổng hợp kết quả. Không dùng người Test để early stopping hoặc tuning. Cả CNN, GRU, CNN-GRU và các ablation phải dùng cùng giao thức mới.

LOSO có 15 lượt ngoài thay vì 5 nên cần nhiều lượt huấn luyện hơn; thời gian thực tế còn phụ thuộc số người Train, mô hình và early stopping. Cả hai cách đều có thể đánh giá trên người chưa thấy nếu giữ đúng tách người và chuẩn hóa Train-only. Không trộn kết quả hai giao thức thành cùng một phép so sánh.

Nhóm có thể chọn LOSO trước khi bắt đầu thí nghiệm 14 kênh chính; kết quả cũ không bắt buộc nhóm giữ 5 folds. Cấu hình hiện tại tiếp tục dùng 5 folds theo quyết định đang áp dụng, chưa tự đổi thiết kế nghiên cứu.

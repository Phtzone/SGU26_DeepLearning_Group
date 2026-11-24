# Debug hậu kiểm — đã khắc phục

Huấn luyện hoàn tất đủ 5 folds. Hậu kiểm đầu tiên dừng khi đọc `subject_metrics.csv`: cột số cửa sổ có giá trị `147.0`, trong khi bộ đọc số nguyên dùng `int('147.0')`.

Đã sửa `src/evaluation/history.py` để chấp nhận biểu diễn số nguyên chính xác như `147`, `147.0` hoặc `1.47e2` bằng Decimal. Giá trị phân số, NaN/Inf hoặc không hợp lệ vẫn bị từ chối; không âm thầm làm tròn. Không sửa dữ liệu, cấu hình, weights hoặc CSV kết quả để vượt kiểm tra; không huấn luyện lại.

20 kiểm thử bộ đọc số nguyên, so sánh và tổng hợp đạt; chi tiết `integer_fix_tests.xml`. Hậu kiểm chạy lại đạt 5 folds, 10 checkpoints, 15 event streams và 134 hình. Đã phát lại 2.134 dự đoán Test từ best checkpoint: chênh lệch xác suất lớn nhất ở cả 5 folds bằng 0.

Trạng thái cuối trong `status.json` là `completed`; báo cáo so sánh nằm tại `results/common_rate64_gru_grouped5fold_v1/20261124T100241Z_204cb671/cnn_gru_comparison.md`. File `review.log` giữ cả lỗi ban đầu và lần chạy thành công để truy lại quá trình.

# Theo dõi hậu kiểm và so sánh CNN–GRU

Lượt GRU `20261124T100241Z_204cb671` chạy với dữ liệu 14 kênh, 64 Hz, cửa sổ 30 giây/stride 15 giây và manifest 5 folds 9/3/3 của nhóm. Đối chứng CNN: `20261123T072126Z_2c81daac`.

Tác vụ `scripts.finish_common_rate_gru --wait` đang đợi huấn luyện hoàn tất. Xem `status.json` và `review.log`; không coi việc khởi động tác vụ là đã hậu kiểm đạt.

Khi hoàn tất, tác vụ thực hiện:

1. Kiểm tra đủ 5 folds, nguồn dữ liệu, scaler Train, trọng số lớp, checkpoint, lịch sử chọn checkpoint, log và TensorBoard bằng verifier Common-rate.
2. Phát lại dự đoán GRU từ best checkpoint của từng fold, đối chiếu xác suất và lớp dự đoán với CSV đã lưu. Không tuning hay cập nhật trọng số trong bước này.
3. Kiểm tra CNN/GRU dùng cùng nguồn, split, cấu hình huấn luyện/selection và từng ID cửa sổ/nhãn/thời gian.
4. Tạo bảng so sánh Test Macro-F1 theo người, Balanced Accuracy, Accuracy; CSV và biểu đồ từng người.

Đầu ra dự kiến dưới `results/common_rate64_gru_grouped5fold_v1/20261124T100241Z_204cb671/`:

- `verification.json`
- `checkpoint_prediction_replay.json`
- `cnn_gru_comparison.md`, `cnn_gru_comparison.json`
- `cnn_gru_subject_comparison.csv`
- `cnn_gru_subject_macro_f1.png`, `.svg`

Các file dự kiến chỉ được tạo khi đủ điều kiện. Nếu huấn luyện hoặc hậu kiểm lỗi, tác vụ lưu trạng thái failed, không coi kết quả chưa kiểm chứng là kết quả cuối cùng. Không sửa Excel nhật ký tự động.

Mã so sánh/metrics/aggregation đã qua 12 kiểm thử; chi tiết ở `review_tests.xml`. Một seed chưa đủ để kết luận độ ổn định giữa các lần khởi tạo hoặc mô hình nào luôn tốt hơn.

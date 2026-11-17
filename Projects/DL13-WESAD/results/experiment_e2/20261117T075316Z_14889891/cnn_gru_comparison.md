# CNN và GRU — lượt đầu, seed 42

Hai lượt dùng cùng máy, thư viện, dữ liệu, split và thông số huấn luyện. GRU dùng một lớp 64 hidden units; CNN giữ kiến trúc gốc. Chưa thử thêm cấu hình GRU.

| Model | Macro-F1 trung bình 15 người | Accuracy trung bình 15 người |
|---|---:|---:|
| CNN | 0.4242 | 0.5221 |
| GRU | 0.3102 | 0.3893 |

Đây là điểm Test sau khi chọn checkpoint bằng Validation. Không dùng Test để sửa cấu hình giữa các folds. Số trung bình theo người khác với số trung bình 5 folds đã dùng trong các ghi chép trước.

Một cấu hình và một seed chưa đủ kết luận mô hình nào tốt hơn một cách ổn định. CNN-GRU chưa được triển khai nên chưa trả lời trọn RQ1. Các lượt này là thử nghiệm triển khai, chưa phải kết quả xác nhận cuối cùng.

- [So sánh từng người](cnn_gru_subject_comparison.csv)
- [Kết quả và cách tổng hợp](summary.md)
- [Kiểm chứng checkpoint, dữ liệu và dự đoán](verification.json)

CNN group: 20261115T101928Z_0ce52f72
GRU group: 20261117T075316Z_14889891

[Biểu đồ theo người](cnn_gru_subject_macro_f1.png)

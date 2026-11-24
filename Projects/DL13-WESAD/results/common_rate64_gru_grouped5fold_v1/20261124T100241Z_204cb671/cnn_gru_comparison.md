# CNN và GRU trên 14 kênh — seed 42

Cùng 2.134 cửa sổ, split 9/3/3, chuẩn hóa Train-only và tiêu chí chọn checkpoint bằng Validation Macro-F1 trung bình theo người. Đã hậu kiểm đủ 5 folds và phát lại dự đoán từ checkpoint GRU.

| Mô hình | Test Macro-F1 trung bình 15 người | Balanced Accuracy theo người | Accuracy theo người |
|---|---:|---:|---:|
| CNN | 0.2962 | 0.3858 | 0.4325 |
| GRU | 0.2864 | 0.3609 | 0.4031 |

Đây là kết quả một seed; chưa kết luận mô hình nào luôn tốt hơn. CNN-GRU và ablation chưa có kết quả. Không dùng điểm Test để chọn lại cấu hình đã chạy.

[Từng người](cnn_gru_subject_comparison.csv) · [Biểu đồ](cnn_gru_subject_macro_f1.png) · [Hậu kiểm](verification.json) · [Phát lại dự đoán](checkpoint_prediction_replay.json)

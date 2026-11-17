# E2 — GRU

Đã chạy xong 5 folds ngày 17/11/2026. GROUP_ID: `20261117T075316Z_14889891`. Tổng 151 epochs, 6.133 batches và 2.136 dự đoán Test, đủ 15 người.

| Model, seed 42 | Macro-F1 trung bình 15 người |
|---|---:|
| CNN chạy lại ngày 15/11 | 0.4242 |
| GRU | 0.3102 |

Ở cấu hình này, GRU có điểm thấp hơn CNN. Chưa kết luận cho mọi cấu hình hay seeds. Không đổi thông số trong lượt chạy để cải thiện điểm Test.

- [So sánh và biểu đồ theo người](../../results/experiment_e2/20261117T075316Z_14889891/cnn_gru_comparison.md)
- [Kết quả đầy đủ](../../results/experiment_e2/20261117T075316Z_14889891/summary.md)
- [Log, checkpoint và code lúc chạy](../../runs/experiment_e2/20261117T075316Z_14889891/)
- [Hình của đủ 5 folds](../../results/experiment_e2/20261117T075316Z_14889891/enhanced_visualizations/scientific-v2_5743853baeca1007/)
- [Kiểm chứng lại kết quả](../../results/experiment_e2/20261117T075316Z_14889891/verification.json)
- [Kiểm thử: 58 đạt, 2 bỏ qua do thiếu artifacts lịch sử](checks/pytest.log)

Các số Test trung bình 5 folds để ghi chú bổ sung: Accuracy 38.98%, Macro-F1 0.3371. Không nhầm chúng với Macro-F1 trung bình theo người ở bảng trên. Thư mục hình của lượt chạy thử một fold vẫn được giữ lại; liên kết hình trên là bản đủ 5 folds.

GRU một chiều, một lớp, 64 hidden units. Mỗi cửa sổ có 5 kênh và 960 mẫu thời gian. Model lấy trạng thái cuối của GRU, qua Linear 64→64, ReLU, Dropout 0.3 và Linear 64→3. Đầu ra là logits cho baseline, stress và amusement. Có 17.987 tham số trainable.

Đợt này chạy một cấu hình, seed 42; không dò thêm thông số. Learning rate 0.001, weight decay 0.0001, batch size 32, tối đa 100 epochs và patience 15, giống CNN gốc. Giữ nguyên cache, split, chuẩn hóa Train-only, weighted cross-entropy và cách chọn checkpoint bằng Validation Macro-F1. Không sửa cấu hình giữa lượt chạy.

Đây là lượt đầu triển khai GRU. Kết quả một seed chưa đủ để kết luận RQ1 hay lập báo cáo độ ổn định giữa seeds. CNN-GRU chưa được triển khai. Các lượt tuning CNN trước đó vẫn giữ trong lịch sử; không xem chúng như ngân sách tuning đã cân bằng cho mọi kiến trúc.

Metric chính để báo cáo là `subject_level.macro_f1.mean` trong `summary.json`: tính Macro-F1 riêng từng người, rồi lấy trung bình đều 15 người. Đây là điểm Test sau chọn checkpoint. `fold_level` và `pooled_oof` là các cách tổng hợp bổ sung. Độ lệch chuẩn giữa 15 người không phải độ lệch chuẩn giữa seeds.

Lệnh chạy từ thư mục DL13-WESAD:

```powershell
python -m scripts.run_grouped_cv --experiment experiments/e2_gru/config.yaml --dry-run
python -m scripts.run_grouped_cv --experiment experiments/e2_gru/config.yaml --folds 1 --non-interactive
# Sau khi kiểm tra Fold 1, tiếp tục cùng lượt để đủ 5 folds:
python -m scripts.run_grouped_cv --experiment experiments/e2_gru/config.yaml --resume --group-id GROUP_ID --folds 1 2 3 4 5 --non-interactive
```

Log, checkpoint, code đã dùng: `runs/experiment_e2/<GROUP_ID>/`.

Kết quả và hình: `results/experiment_e2/<GROUP_ID>/`.

Không chỉnh file trong source snapshot để làm thí nghiệm mới. Muốn đổi thông số thì tạo cấu hình và GROUP_ID mới. Test chỉ dùng báo cáo, không dùng chọn hướng tuning tiếp.

Tham khảo API: [PyTorch GRU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html). Dropout được đặt ở classifier vì GRU một lớp không có dropout giữa các lớp recurrent.

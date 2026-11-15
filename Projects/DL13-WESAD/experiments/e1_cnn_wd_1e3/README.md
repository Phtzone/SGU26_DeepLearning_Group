# CNN thử weight decay 0.001

Lần này tăng weight decay từ 0.0001 lên 0.001 để xem điểm Validation có tốt hơn không. Các thông số huấn luyện còn lại giữ nguyên.

Đã chạy xong ngày 15/11/2026, đủ 5 folds, seed 42. Có chạy lại cấu hình gốc trên cùng máy để đối chiếu vì môi trường hiện tại khác máy chạy E1 trước đây.

| Cấu hình | Validation Macro-F1 trung bình |
|---|---:|
| Weight decay 0.0001 | 0.6000 |
| Weight decay 0.001 | 0.6033 |

Mức tăng là 0.0032, tương đương khoảng 0.32 điểm phần trăm. Hai folds tăng điểm, ba folds giảm. Kết quả một seed này chưa cho thấy cải thiện rõ ràng, nên chưa thay baseline.

- [Bảng so sánh từng fold](../../results/experiment_e1/20261115T101531Z_c2d7bd4b/comparison.md)
- [Kết quả lượt thử](../../results/experiment_e1/20261115T101531Z_c2d7bd4b/summary.json)
- [Log lượt thử](../../runs/experiment_e1/20261115T101531Z_c2d7bd4b/)
- [Log lượt đối chứng](../../runs/experiment_e1/20261115T101928Z_0ce52f72/)

Chạy lại từ thư mục DL13-WESAD:

```powershell
python -m scripts.run_grouped_cv --experiment experiments/e1_cnn_wd_1e3/config.yaml
```

Lệnh trên tạo lượt chạy mới. Hồ sơ E1 gốc ngày 08/11 vẫn giữ nguyên.

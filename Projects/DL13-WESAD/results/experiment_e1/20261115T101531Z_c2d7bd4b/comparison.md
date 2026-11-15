# Thử tăng weight decay cho CNN

Đối chứng: 20261115T101928Z_0ce52f72 (weight decay 0.0001).
Cấu hình thử: 20261115T101531Z_c2d7bd4b (weight decay 0.001).

Hai lượt đều chạy đủ 5 folds trên cùng máy, cùng phiên bản thư viện, dữ liệu, split và seed. Chỉ thay weight decay trong cấu hình huấn luyện. Mỗi fold lấy checkpoint có Validation Macro-F1 tốt nhất.

So sánh Validation Macro-F1:

| Fold | WD 0.0001 | WD 0.001 | Chênh lệch |
|---|---:|---:|---:|
| 1 | 0.5037 | 0.4870 | -0.0167 |
| 2 | 0.6854 | 0.6807 | -0.0047 |
| 3 | 0.5055 | 0.4545 | -0.0510 |
| 4 | 0.6366 | 0.6665 | +0.0299 |
| 5 | 0.6690 | 0.7276 | +0.0586 |
| Trung bình | 0.6000 | 0.6033 | +0.0032 |

Cấu hình thử cao hơn ở 2/5 folds. Đây là kết quả của một seed; chưa đánh giá độ ổn định qua nhiều seeds. Chênh lệch nhỏ không đủ để khẳng định cải thiện ổn định.

Test scores được lưu trong summary.json của mỗi lượt để báo cáo; không dùng chúng để chọn hướng tuning tiếp. Hồ sơ E1 gốc ngày 08/11/2026 giữ nguyên, đã kiểm tra hash các fold artifacts.

Chi tiết: validation_comparison.csv. Log/checkpoint nằm trong runs/experiment_e1/ theo hai GROUP_ID trên.

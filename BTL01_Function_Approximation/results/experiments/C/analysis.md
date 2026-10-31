# Thực nghiệm C – Ảnh hưởng của số tầng ẩn

Chỉ thay đổi số tầng ẩn C; mọi tầng ẩn có cùng độ rộng. Dùng lại nguyên vẹn các CSV train/validation/test của A và B.

Cấu hình: seeds [42, 43, 44]; 2000 mẫu train, 500 validation, 100 test; 10 nơ-ron/tầng; activation tanh; Adam; batch size 64; learning rate 0.001; tối đa 500 epochs; early stopping patience 40, min_delta 1e-08; weight decay 0. Chuẩn hóa đầu vào x1/2, x2/4; đầu ra tuyến tính; loss MSE. Checkpoint mỗi run được chọn bằng validation MSE, không dùng test.

## Bảng kết quả

Trung bình ± độ lệch chuẩn mẫu giữa các seeds (không phải khoảng tin cậy).

| C | Tham số | Train MSE | Validation MSE | Test MSE | Test MAE | Test RMSE | Train (s) |
|---|---|---|---|---|---|---|---|
| 1 | 41 | 0.000383543 ± 0.000101 | 0.000394494 ± 0.000114 | 0.000468638 ± 0.000101 | 0.0158412 ± 0.00187 | 0.0215677 ± 0.00228 | 16.5221 ± 0.501 |
| 2 | 151 | 6.6689e-06 ± 3.84e-06 | 6.72057e-06 ± 4.67e-06 | 6.71109e-06 ± 2.26e-06 | 0.00207418 ± 0.000424 | 0.0025648 ± 0.000446 | 18.0592 ± 0.177 |
| 3 | 261 | 4.24215e-06 ± 3.46e-06 | 3.87778e-06 ± 2.97e-06 | 5.03872e-06 ± 4.71e-06 | 0.00167013 ± 0.000799 | 0.00209346 ± 0.000992 | 20.0176 ± 0.0589 |
| 4 | 371 | 6.54072e-07 ± 8.46e-08 | 6.51142e-07 ± 1.07e-07 | 7.94925e-07 ± 5.09e-08 | 0.00072102 ± 1.92e-05 | 0.000891281 ± 2.85e-05 | 23.9818 ± 1.51 |

## Nhận xét

- Chọn C = 4 theo validation MSE trung bình nhỏ nhất (6.51142e-07); test chỉ dùng mô tả chất lượng cuối cùng.
- Từ C = 1 đến C = 4, train MSE đổi từ 0.000383543 thành 6.54072e-07; validation MSE từ 0.000394494 thành 6.51142e-07; test MSE từ 0.000468638 thành 7.94925e-07.
- Nhiều tầng tạo thêm các phép biến đổi phi tuyến, giúp biểu diễn đỉnh Gaussian và vùng biến thiên hẹp. Tuy nhiên độ sâu cũng tăng số tham số và độ khó tối ưu; không thể quy toàn bộ chênh lệch cho độ sâu độc lập với dung lượng mô hình.
- Thời gian train trung bình ở hai đầu dải C là 16.522s và 23.982s. Đây là thời gian vòng train/validation, không gồm xuất hình hay đánh giá test. Thời gian còn phụ thuộc số epoch dừng và môi trường máy; không so trực tiếp với thời gian A/B từ một phiên chạy khác.
- Trong dải đã khảo sát, validation MSE trung bình giảm khi tăng C. Khởi tạo ngẫu nhiên và quỹ đạo tối ưu có thể tạo khác biệt giữa seeds; mạng sâu hơn không bảo đảm tốt hơn ở mọi lần chạy. Cần đọc cả độ lệch chuẩn và đường hội tụ thay vì chỉ một giá trị trung bình.
- C = 1: validation/train MSE = 1.029. Train và validation cùng bậc sai số; chưa thấy khoảng cách lớn chứng minh quá khớp.
  Số epoch thực chạy: 500–500; epoch checkpoint tốt nhất: 496–499. Có run chạm trần epoch, nên chưa thể khẳng định đã hội tụ hoàn toàn.
- C = 2: validation/train MSE = 1.008. Train và validation cùng bậc sai số; chưa thấy khoảng cách lớn chứng minh quá khớp.
  Số epoch thực chạy: 500–500; epoch checkpoint tốt nhất: 495–500. Có run chạm trần epoch, nên chưa thể khẳng định đã hội tụ hoàn toàn.
- C = 3: validation/train MSE = 0.914. Train và validation cùng bậc sai số; chưa thấy khoảng cách lớn chứng minh quá khớp.
  Số epoch thực chạy: 500–500; epoch checkpoint tốt nhất: 476–490. Có run chạm trần epoch, nên chưa thể khẳng định đã hội tụ hoàn toàn.
- C = 4: validation/train MSE = 0.996. Train và validation cùng bậc sai số; chưa thấy khoảng cách lớn chứng minh quá khớp.
  Số epoch thực chạy: 455–500; epoch checkpoint tốt nhất: 415–500. Có run chạm trần epoch, nên chưa thể khẳng định đã hội tụ hoàn toàn.
- Xem `experiment_c_convergence.png`: nét liền là train, nét đứt là validation; từng seed được vẽ riêng và dừng đúng epoch thực tế. Loss train trong log là trung bình các minibatch khi cập nhật trọng số; train MSE trong bảng được tính lại tại checkpoint tốt nhất nên có thể khác điểm cuối đường loss.
- Giới hạn: 3 seeds mặc định, test chỉ 100 mẫu và một bộ dữ liệu cố định; không khẳng định ý nghĩa thống kê hay độ sâu tối ưu cho mọi bài toán. Early stopping và trần epoch cố định có thể khiến cấu hình dừng trước khi hội tụ hoàn toàn.

## Minh chứng

- `runs.csv`, `summary.csv`: số liệu từng run và thống kê tổng hợp.
- `experiment_config.json`: thiết lập, phiên bản môi trường và SHA-256 dữ liệu.
- `runs/C_*/`: checkpoint, cấu hình/metrics, loss CSV, learning curves, test predictions CSV/hình và bề mặt xấp xỉ cho từng run.
- `experiment_c_errors_and_time.png`: sai số và thời gian theo C.
- `experiment_c_convergence.png`: so sánh hội tụ train/validation.

# Chia cửa sổ Common-rate 14 kênh

Đầu vào là dữ liệu liên tục do `notebooks/02_preprocessing_common_rate.ipynb` tạo ra: 15 người, 14 kênh ngực và cổ tay, 64 Hz. Bước này chưa chuẩn hóa, chia Train/Validation/Test hay huấn luyện.

## Chạy và xem kết quả

Mở `notebooks/03_windowing_common_rate.ipynb`, hoặc chạy từ thư mục `Projects/DL13-WESAD`:

```powershell
python -m scripts.prepare_common_rate_windows
```

- Dữ liệu cửa sổ: `data/processed/common_rate_windows_64hz_30s_15s/<subject>/`.
- Mỗi lần chạy có báo cáo riêng tại `results/common_rate_windowing/<run_id>/`.
- Chạy lại chỉ dùng cache nếu hash đầu vào, cấu hình và file đầu ra khớp; vẫn kiểm tra lại nội dung từng cửa sổ. Nếu cache khác, lệnh dừng và yêu cầu dùng thư mục `--output` mới.
- Các file dữ liệu trong `data/processed/` được Git bỏ qua. Báo cáo nhỏ, log, CSV và code có thể commit để nhóm kiểm tra.

## Quy tắc tạo mẫu

1. Ánh xạ nhãn gốc: 1 Baseline → 0, 2 Stress → 1, 3 Amusement → 2. Loại các trạng thái khác.
2. Từ ranh giới nhãn 700 Hz, lấy đầu đoạn bằng `ceil(start*64/700)` và cuối đoạn bằng `floor(end*64/700)`. Không gán nhãn cho một khoảng mẫu 64 Hz đi qua ranh giới gốc.
3. Tạo cửa sổ 30 giây = 1.920 mốc, stride 15 giây = 960 mốc; overlap 50%. Bắt đầu lại lưới stride ở đầu mỗi vùng nhãn hợp lệ liên tục, giống cách tạo cửa sổ trước đây. Chỉ nhận cửa sổ đủ độ dài, nằm trọn một vùng nhãn; không lấy đa số nhãn.
4. Loại cửa sổ nếu bất kỳ mốc nào có `quality_mask=False`. Không di chuyển lưới stride sau khi loại; không tự điền hay sửa dữ liệu tín hiệu.
5. Lưu `windows.npy` dạng `[số cửa sổ, 14, 1920]`, kiểu `float32`, chưa chuẩn hóa.

Quality mask của notebook 02 đã bao gồm vùng đệm 2 giây. Bước này dùng nguyên mask, không thêm một lần đệm nữa. Đây vẫn là quy tắc chất lượng ban đầu, không chứng minh tín hiệu đã sạch mọi nhiễu.

## File cần xem

| File | Nội dung |
|---|---|
| `<subject>/windows.npy` | Giá trị 14 kênh của từng cửa sổ |
| `<subject>/windows.csv` | ID người/cửa sổ, thời gian, chỉ số đầu/cuối và nhãn |
| `<subject>/rejected_quality.csv` | Những cửa sổ bị loại do mask và số mốc bị đánh dấu |
| `<subject>/metadata.json` | Cấu hình, thứ tự kênh, thống kê, hash đầu vào/đầu ra |
| Báo cáo `windowing.log` | Log theo từng người và trạng thái hoàn thành/lỗi |
| Báo cáo `windows_by_subject.csv` | Số cửa sổ từng lớp, giữ lại và bị loại theo người |
| Báo cáo `verification.json` | Đối chiếu từng cửa sổ với tín hiệu, mask và nhãn 700 Hz gốc |
| Báo cáo `summary.json` | Tổng hợp toàn bộ lần chạy |

`label_safe_candidates` là số cửa sổ nằm trọn trong nhãn hợp lệ **trước** lọc chất lượng; `rejected_quality` là phần bị loại do mask. Cửa sổ qua chuyển trạng thái bị ngăn từ lúc tạo nên không nằm trong hai số này. `label_audit.global_grid_rejections` chỉ là kiểm tra phụ trên lưới bắt đầu từ 0; không cộng số đó vào thống kê lưới theo vùng nhãn.

## Kết quả ngày 23/11/2026

Lần chạy `20261123T083922Z_f1ddf9de` có 2.136 cửa sổ nhãn hợp lệ; loại 2 cửa sổ Stress của S3 do quality mask, còn **2.134 cửa sổ**:

| Lớp | Số cửa sổ giữ lại |
|---|---:|
| Baseline | 1.150 |
| Stress | 638 |
| Amusement | 346 |

Cả 15 người đều đạt đối chiếu. Các cửa sổ trùng khớp dữ liệu nguồn, đúng nhãn, không qua ranh giới nhãn và không chạm mask không hợp lệ. Đây là kiểm tra bước tạo mẫu, chưa phải kết quả mô hình.

Bước tiếp: nối cache này vào runner 14 kênh, dùng manifest 5 folds theo người; fit chuẩn hóa bằng Train của từng fold. Lệnh huấn luyện cũ vẫn đọc dữ liệu cổ tay cũ, chưa tự chuyển sang bộ này.

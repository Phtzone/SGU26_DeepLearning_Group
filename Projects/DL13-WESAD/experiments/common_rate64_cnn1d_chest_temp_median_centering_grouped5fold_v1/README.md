# CNN-1D + Chest Temperature Median Centering + Z-score

**Preprocessing verified — Training not started**

Experiment ID `common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1`. Câu hỏi nghiên cứu là liệu việc bỏ mức nhiệt độ ngực nền trong từng cửa sổ có thu hẹp khác biệt phân phối giữa người. Trong baseline `20261123T072126Z_2c81daac`, mean của `chest_Temp` ở Validation Fold 4 thấp hơn Train khoảng 3,35 độ lệch chuẩn. [Kiểm toán baseline](../../docs/audits/common_rate64_cnn1d_grouped5fold_v1/AUDIT_REPORT.md) · [Cấu hình thí nghiệm](config.yaml).

## Dữ liệu và phép biến đổi

- Common-rate 64 Hz với 15 subject, 14 kênh, 2.134 cửa sổ hợp lệ dài 1.920 timestep và stride 960 timestep. Mỗi fold dùng đúng 9 subject Train, 3 Validation và 3 Test từ baseline, seed 42. [Window Manifest](../../runs/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac/window_index/windows.csv) · [Split Manifest](../../runs/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac/split_manifest.json).
- `chest_Temp` là cột thứ 7 (index 6 tính từ 0) trong thứ tự kênh thực tế. Dataset tra tên từ metadata, lấy một cửa sổ `[14,1920]` và trừ riêng trên kênh này median của chính 1.920 mẫu trong cửa sổ. Cả ba tập dùng phép trừ giống nhau. [Metadata 64 Hz](../../data/processed/common_rate_64hz/config.json) · [Dataset](../../src/data/dataset.py).
- Sau centering, `ChannelScaler` fit lại 14 cặp mean/std trên cửa sổ **Train** của từng fold, tính cả các lượt timestep chồng lấp. Validation và Test dùng scaler đó. Quá trình tạo tensor diễn ra khi đọc cửa sổ, giữ layout `[B,14,1920]` và dtype `float32`. [ChannelScaler](../../src/data/normalization.py).

Bước fit lại std của nhiệt độ sau centering làm thay đổi cả offset lẫn thang đo. Trong phân tích phân phối, cần theo dõi độ lớn Z-score và các tỷ lệ ở đuôi vì std Train thấp có thể làm biến thiên nhỏ chiếm nhiều đơn vị Z-score. Train và Validation cung cấp số liệu so sánh phương pháp. Test dùng để xác minh tính đúng đắn của transform.

## Kiểm chứng và hiện vật

Từ thư mục `Projects/DL13-WESAD`, dùng Python có NumPy, PyTorch và Matplotlib để chạy phép kiểm chứng. Script tạo group kết quả tiền xử lý riêng và đọc Window Manifest của baseline.

```powershell
python -m scripts.verify_chest_temp_centering
```

Group kiểm chứng được bàn giao là `20261126T025058Z_fa1c92ea`. [Thư mục kết quả](../../results/common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1/20261126T025058Z_fa1c92ea/) · [Báo cáo](../../results/common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1/20261126T025058Z_fa1c92ea/preprocessing_report.md) · [Verification JSON](../../results/common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1/20261126T025058Z_fa1c92ea/preprocessing_verification.json).

| Hiện vật | Nội dung |
| --- | --- |
| `preprocessing_report.md` và `preprocessing_verification.json` | Kết quả kiểm chứng trên 5 fold và trạng thái thực thi |
| `seed_42/fold_XX/normalization.json` | Mean/std của scaler mới fit từ Train, riêng từng fold |
| `normalization_statistics.csv` | Thống kê 14 kênh ở từng fold và tập |
| `temperature_stage_statistics.csv` | Thống kê nhiệt độ trước/sau centering và sau Z-score theo tập, người và lớp |
| `temperature_distribution_comparison.csv` | Dịch chuyển phân phối và đuôi Z-score so với baseline |
| `figures/` và `execution.log` | Biểu đồ PNG/SVG và nhật ký kiểm chứng |

## Research Journal

CSV là định dạng chính thức từ group `20261126T025058Z_fa1c92ea`. Sáu bảng giữ thứ tự cột và nội dung từ workbook. Ô có đích liên kết khác chữ hiển thị giữ cả hai trên hai dòng trong cùng trường. Đọc đường dẫn tương đối từ thư mục thí nghiệm chứa README này.

[Thí nghiệm](research_journal/experiments.csv) · [Hình](research_journal/figures.csv) · [Nhật ký](research_journal/activity_log.csv) · [Debug](research_journal/debug.csv) · [Tham khảo](research_journal/references.csv) · [Việc cần làm](research_journal/todo.csv).

Script đọc bằng chứng kiểm chứng và bổ sung các dòng CSV mới. Chạy lại với cùng dữ liệu sẽ giữ nguyên các bảng.

```powershell
python -m scripts.report_chest_temp_centering --group-id 20261126T025058Z_fa1c92ea
```

[Workbook XLSX lưu trữ](Research_journal_common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1.xlsx) · [Quy ước CSV cho thí nghiệm mới](../journal/README.md#research-journal-cho-thí-nghiệm-mới)

Giai đoạn này ghi nhận tính đúng đắn của tiền xử lý và dịch chuyển phân phối. Các chỉ số nhận diện Baseline, Stress và Amusement của cấu hình mới chỉ có thể đánh giá sau khi huấn luyện và đánh giá mô hình.

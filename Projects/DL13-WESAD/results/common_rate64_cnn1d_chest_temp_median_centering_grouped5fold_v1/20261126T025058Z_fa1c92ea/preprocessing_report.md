# Kiểm chứng Median Centering và Z-score cho CNN-1D

**Preprocessing verified — Training not started**

Thí nghiệm `common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1`, group `20261126T025058Z_fa1c92ea`, đối chiếu baseline `20261123T072126Z_2c81daac`. Dùng nguyên 15 subject, 14 kênh 64 Hz, cửa sổ 1.920 mẫu, stride 960, split 9/3/3 và seed 42. Hai manifest, mask và 33 tệp nguồn được đối chiếu fingerprint. [Định danh và bằng chứng](preprocessing_verification.json) · [Cấu hình](../../../experiments/common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1/config.yaml).

## Phép biến đổi và kiểm chứng

`chest_Temp` là kênh thứ **7**, index **6** tính từ 0. Dataset tra tên kênh từ metadata, lấy một bản sao cửa sổ và chỉ trừ median của 1.920 mẫu nhiệt độ của cửa sổ đó. Tính median và trừ ở float64, rồi lưu kết quả vào tensor float32. `ChannelScaler` fit mean/std population trên chính các cửa sổ **Train đã center** của fold, gồm các timestep lặp do overlap. Cùng scaler được dùng để transform cả ba tập. [Dataset](../../../src/data/dataset.py) · [Scaler](../../../src/data/normalization.py).

- Kiểm trực tiếp **2,134 cửa sổ duy nhất** theo manifest. Median sau centering bằng 0 trong cả bộ. Chênh lệch tối đa về sai phân hai timestep là 0, peak-to-peak là 0 và std mỗi cửa sổ là 1.1e-16. 13 kênh còn lại khớp từng phần tử. Nhãn, ID, shape và mask khớp baseline.
- Đã kiểm **343 batch**, **10,670 lượt cửa sổ** ở cả 5 fold. Tensor `[B,14,1920]` float32 khớp phép biến đổi độc lập cho mọi split. Cả 5 scaler khớp phép tính độc lập theo Train. Trên Train, sai số lớn nhất của mean so với 0 là 1.42e-06, của std so với 1 là 5.20e-08. Không phát hiện NaN/Inf hay kênh phải dùng fallback std. [Thống kê 14 kênh](normalization_statistics.csv) · [Scaler Fold 4](seed_42/fold_04/normalization.json).
- Nhánh tắt centering cho kết quả scaler và tensor bằng baseline đã lưu. Ở nhánh bật centering, cả mean/std scaler lẫn tensor sau Z-score của 13 kênh còn lại khớp baseline. Checksum trước/sau của 583 tệp gốc khớp, gồm data, masks, index, split, logs, checkpoints và notebook 02. Test chỉ dùng xác minh transform.

## So sánh phân phối nhiệt độ ngực

| Fold | Std Train gốc (°C) | Std Train đã center (°C) | Tỷ số hai std | Mean Z Validation, cũ → mới | Max trị tuyệt đối Z Validation, cũ → mới |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,5569 | 0,0755 | 20,62× | 0,4867 → -0,0094 | 1,22 → 3,88 |
| 2 | 1,5637 | 0,0755 | 20,72× | 0,3453 → -0,0041 | 1,01 → 4,03 |
| 3 | 1,5842 | 0,0753 | 21,03× | 0,3800 → 0,0044 | 1,13 → 3,71 |
| 4 | 0,6514 | 0,0215 | 30,26× | -3,3484 → 0,0323 | 9,30 → 133,69 |
| 5 | 0,6471 | 0,0235 | 27,49× | 0,3687 → 0,0003 | 2,37 → 14,73 |

Ở **Validation Fold 4**, mean nhiệt độ gốc là **32,024 °C**, so với **34,206 °C** ở Train. Chênh lệch mean Validation–Train giảm từ -2,1812 °C còn 0,0007 °C sau centering. Mean Train sau centering là -0,0004 °C, mean Validation là 0,0003 °C. Median bằng 0 trong từng cửa sổ **trước** Z-score. Z-score sau đó trừ mean Train chung, nên median mỗi cửa sổ có thể lệch 0.

| Subject Validation Fold 4 | Mean gốc (°C) | Mean sau center (°C) | Mean Z baseline | Mean Z mới | Std Z mới | Max trị tuyệt đối Z mới |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S2 | 30,1158 | 0,0014 | -6,2785 | 0,0806 | 9,6576 | 133,69 |
| S3 | 32,0009 | 0,0002 | -3,3846 | 0,0264 | 3,5843 | 37,23 |
| S13 | 33,8627 | -0,0005 | -0,5266 | -0,0081 | 1,1306 | 8,31 |

Scale mới làm mean Validation Fold 4 gần 0 hơn, đồng thời tăng std Z từ **2,919 lên 5,939**. Tỷ lệ mẫu có trị tuyệt đối Z trên 5 giảm từ 20,84% xuống 5,07%. Tỷ lệ trên 10 tăng từ 0,00% lên 2,00%. P99 độ lớn sai phân hai timestep tăng từ 0,0449 lên 1,3579 trong đơn vị Z-score.

Cực trị Train/Validation của toàn bộ kiểm chứng nằm ở `S2_84900_86820`, fold 4 (validation), tại giây 1326,921875, Z = **-133,69**. Tất cả mẫu của cửa sổ này có mask True. Khi std Train nhỏ, cùng một mức dao động tính bằng °C chiếm nhiều đơn vị Z-score. Điều này có thể khuếch đại cả biến thiên sinh lý và nhiễu. Phân phối và mask hiện có chưa xác định được nguyên nhân của cực trị.

Các bảng phân tầng theo tập, từng người và từng lớp có đủ mean, std, median, IQR, min–max, phân vị và tỷ lệ ở đuôi. Báo cáo mô tả sự thay đổi phân phối ở các cửa sổ, gồm các lượt mẫu chồng lấp. Hiệu quả nhận diện Stress của phép biến đổi vẫn chưa được đo. [Bảng bốn trạng thái](temperature_stage_statistics.csv) · [So sánh theo fold](temperature_distribution_comparison.csv) · [Cửa sổ mẫu](figures/temperature_windows.png) · [Boxplot theo subject](figures/temperature_subject_boxplots.png) · [Density Fold 4](figures/temperature_density_fold_04.png) · [So sánh 5 fold](figures/temperature_shift_by_fold.png).

[Nhật ký kiểm chứng](execution.log) · [Research Journal](../../../experiments/common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1/Research_journal_common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1.xlsx).

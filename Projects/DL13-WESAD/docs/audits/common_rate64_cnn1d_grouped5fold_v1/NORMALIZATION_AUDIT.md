# Kiểm toán chuẩn hóa và dữ liệu đầu vào CNN-1D

**Run** `common_rate64_cnn1d_grouped5fold_v1` · `20261123T072126Z_2c81daac` · seed 42.

## Kết luận

**Verified correct.** CNN-1D nhận Z-score theo từng kênh, với một scaler fit từ 9 subject Train của chính fold. Cả 70 cặp mean/std độc lập đều khớp bit sau khi làm tròn về float32. Trên toàn bộ cửa sổ Train đã transform, giá trị lớn nhất của \(|\mathrm{mean}|\) qua 5 × 14 kênh là \(2,70 \times 10^{-6}\), sai lệch lớn nhất của std so với 1 là \(5,21 \times 10^{-8}\). Cả 14 kênh có std hợp lệ ở mọi fold. [Thống kê trước và sau chuẩn hóa](normalization_statistics.csv) · [Bằng chứng dữ liệu](data_evidence.json).

## Dữ liệu và windowing

Notebook 02 xuất mỗi subject thành một chuỗi float32 \([N,14]\) ở 64 Hz. Nó xử lý từng người, resample theo kênh bằng `resample_poly`, ghi mask và các đoạn nhãn gốc 700 Hz. `config.json` khai báo `normalized: false`, `windowed: false`. Đã kiểm SHA-256 của 15 mảng tín hiệu, 15 mask và 3 tệp metadata theo fingerprint lúc chạy. [Notebook 02](../../../notebooks/02_preprocessing_common_rate.ipynb) · [Cấu hình dữ liệu](../../../data/processed/common_rate_64hz/config.json) · [Fingerprint](data_evidence.json).

`build_index()` chuyển mỗi đoạn nhãn \([a,b)\) ở 700 Hz về \([\lceil64a/700\rceil,\lfloor64b/700\rfloor)\) trên lưới 64 Hz, neo lưới cửa sổ tại đầu từng đoạn. Mỗi cửa sổ có đúng 1.920 timestep, bước 960 timestep và toàn bộ mask `True`. Tái dựng độc lập từng cửa sổ từ các đoạn nhãn và mask thu được đúng **2.134 ID duy nhất**, khớp index lưu, với 1.150 Baseline, 638 Stress và 346 Amusement. Mapping nhãn gốc là \(1\to0\), \(2\to1\), \(3\to2\). Nhãn 0 gốc (Undefined) và nhãn ngoài ba lớp đã bị bỏ. Ranh giới ở đây là **chỉ số đầu/cuối của cửa sổ trên chuỗi đã resample**. Bộ lọc trên chuỗi cùng subject có thể nhận ảnh hưởng từ mẫu lân cận ranh giới trước resampling. Khi áp dụng trực tuyến cần đánh giá riêng hiệu ứng đó và cách phát cửa sổ khi chưa biết nhãn. [common_rate.py, `build_index()` dòng 28–106](../../../src/data/common_rate.py#L28) · [Window manifest](../../../runs/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac/window_index/windows.csv) · [SciPy, `resample_poly`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html).

**Confirmed issue ở tín hiệu nguồn, đã được loại khỏi đầu vào.** EDA ghi nhận đoạn nhiệt độ ngực và gia tốc ngực bất thường ở S3 trong khoảng 2531,014–2531,131 s của bản ghi gốc. Ở 64 Hz, mask đánh dấu đúng 265 timestep trong \([161856,162121)\), tức từ 2529,000 đến 2533,140625 s (cận cuối loại trừ). Hai cửa sổ Stress \([160887,162807)\) và \([161847,163767)\) bị loại. `chest_Temp` S3 có min/max −297,37/57,91 trên chuỗi 64 Hz đầy đủ và 30,77/34,43 trên các cửa sổ hợp lệ. Con số −297,37 là mẫu sau resampling. Quan sát của EDA trên tín hiệu gốc có min xấp xỉ −273,15. Không có NaN/Inf trong 15 chuỗi 64 Hz. CSV tách `continuous_all`, `continuous_mask_true` và `valid_windows`, gồm min/max, mean/std, median, P01/P05/P25/P75/P95/P99, IQR và tỷ lệ ngoài ngưỡng Tukey \(3\times\mathrm{IQR}\). Các đuôi thống kê dùng để khoanh vùng xem lại tín hiệu. [EDA](../../../notebooks/preprocessing.ipynb) · [Thống kê chuỗi và cửa sổ](pre_normalization_statistics.csv) · [Vùng mask](quality_mask_intervals.csv) · [Cửa sổ bị loại](rejected_windows.csv).

## Luồng từ file đến CNN1D.forward()

| Bước | Cơ chế đã đối chiếu | Điểm kiểm tra |
| --- | --- | --- |
| Index và split | Runner dùng nguồn `common_rate`, dựng index của group và gọi trainer với fold từ manifest. Ba tập rời nhau theo subject trong từng fold, 9/3/3. Mỗi subject xuất hiện 1 lần ở Test qua 5 folds. | [Runner, `run()` dòng 67 và 147](../../../scripts/run_grouped_cv.py#L67) · [Split của run](../../../runs/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac/split_manifest.json) |
| Đọc cửa sổ | `WindowDataset` mở `Sx_X.npy` read-only, lấy thứ tự cột từ notebook config và policy. 14 kênh khớp thứ tự `config.json` và effective YAML. `raw_window()` cắt `[1920,14]`, rồi `.T` thành `[14,1920]`. | [dataset.py, dòng 12–25 và 50–54](../../../src/data/dataset.py#L12) |
| Fit | `train_fold()` tạo ba dataset chưa transform, gọi `ChannelScaler().fit(datasets['train'])`, lưu `normalization.json` và gán cùng scaler cho cả ba dataset. `fit()` gộp mean và phương sai từng block ở float64, `ddof=0`. Lưu về float32, thay std dưới 1e−8 bằng 1. | [trainer.py, dòng 70–78](../../../src/training/trainer.py#L70) · [normalization.py, `fit()` dòng 7–30](../../../src/data/normalization.py#L7) |
| Transform | `__getitem__()` tạo bản sao C-contiguous, gọi `transform()` một lần. Công thức \(z_{c,t}=(x_{c,t}-\mu_c)/\sigma_c\), broadcasting `[14,1]` dọc trục thời gian. | [dataset.py, `__getitem__()` dòng 64–69](../../../src/data/dataset.py#L64) · [normalization.py, `transform()` dòng 32–33](../../../src/data/normalization.py#L32) |
| Forward | DataLoader batch tối đa 32, float32 `[B,14,1920]`. `forward_pre_hook` xác nhận chính tensor DataLoader ở đầu vào CNN. So độc lập từng phần tử với lát cắt timeline + phép transform. Nhãn và index cũng được kiểm. | [cnn.py, `forward()` dòng 14](../../../src/models/cnn.py#L14) · [Mã hậu kiểm tensor](audit_replay.py) · [Bằng chứng inference](replay_evidence.json) |

Thứ tự cột là `chest_ACC_X`, `chest_ACC_Y`, `chest_ACC_Z`, `chest_ECG`, `chest_EMG`, `chest_EDA`, `chest_Temp`, `chest_Resp`, `wrist_ACC_X`, `wrist_ACC_Y`, `wrist_ACC_Z`, `wrist_BVP`, `wrist_EDA`, `wrist_TEMP`. Cả năm scaler được tính lại bằng cách lấy trực tiếp các lát cắt Train rồi tính mean/std trên trục window và timestep. Số lượt timestep mỗi kênh là số cửa sổ Train nhân 1.920. Khác biệt float64 nhỏ giữa phép fit và hiện vật nằm trong phép làm tròn khi lưu float32. [Mã tính độc lập](audit_data.py) · [Normalization metadata Fold 1](../../../runs/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac/seed_42/fold_01/normalization.json).

| Fold | Cửa sổ Train | Lượt timestep/kênh khi fit | Max trị tuyệt đối mean(z) | Max sai lệch std(z) so với 1 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.270 | 2.438.400 | 1,09e−6 | 4,15e−8 |
| 2 | 1.275 | 2.448.000 | 1,01e−6 | 5,19e−8 |
| 3 | 1.281 | 2.459.520 | 1,10e−6 | 4,70e−8 |
| 4 | 1.290 | 2.476.800 | 2,70e−6 | 4,95e−8 |
| 5 | 1.286 | 2.469.120 | 2,41e−6 | 5,20e−8 |

Trong phạm vi đã kiểm, cả năm fold đọc đúng dữ liệu 64 Hz. Validation/Test dùng mean/std Train đã lưu của fold. Index và tensor thực tế khớp phép transform duy nhất. Tệp thống kê lưu 210 hàng ở mức fold × tập × kênh, gồm số mẫu, min/max, phân vị, NaN/Inf, tỷ lệ ngoài P01–P99 của Train và tỷ lệ trị tuyệt đối của \(z\) vượt 3, 5 và 10. Với người mới, mean/std Train được giữ cố định. Vì vậy mean/std của Validation và Test có thể khác 0/1. [Bảng trước và sau chuẩn hóa](normalization_statistics.csv).

## Overlap và trọng số khi fit

Có **2.088** cặp cửa sổ liền kề chồng đúng 960 timestep. Trên mỗi kênh, 2.134 cửa sổ chứa 4.097.280 lượt timestep và bao phủ 2.092.800 vị trí riêng, gồm 2.004.480 lượt lặp (48,92%). Scaler cho mỗi lượt xuất hiện trọng số bằng nhau. Phần giữa các đoạn đủ dài thường xuất hiện hai lần, mép đoạn hoặc bên cạnh cửa sổ bị loại có thể chỉ xuất hiện một lần. Vì vậy cửa sổ của cùng subject có phụ thuộc lẫn nhau.

Khi so hai cửa sổ kề nhau tại cùng vị trí tương đối (lag 15 s), median Pearson có thể cao ở kênh biến thiên chậm, chẳng hạn S3 `chest_Temp` 0,745 và S3 `wrist_EDA` 0,688. Đoạn 960 mẫu trùng khớp tuyệt đối khi căn theo timestamp. Các hệ số Pearson ở lag 15 s đo sự giống nhau tại vị trí tương đối. Overlap nằm trong từng subject và trong đúng một tập của từng fold. [Thống kê overlap theo subject và kênh](window_overlap.csv).

## Dịch chuyển phân phối

Đặt \(p_T,p_E\) là tỷ lệ lớp tính theo lượt timestep của Train và tập đánh giá, \(m_T,m_E\) là mean từng lớp. Chênh lệch tổng, tính theo đơn vị std Train, bằng tổng của thành phần tỷ lệ lớp \(\sum_c(p_E-p_T)m_T/\sigma_T\) và thành phần trong lớp \(\sum_c p_E(m_E-m_T)/\sigma_T\). Đây là phân rã mô tả tại mean. Bảng subject × lớp có thêm median, IQR, độ phân tán, min/max và đuôi. [140 hàng phân rã](distribution_shift_decomposition.csv) · [4.200 hàng subject × lớp × kênh × fold](subject_distribution_statistics.csv).

| Fold, tập, kênh | Mean shift (std Train) | Do tỷ lệ lớp | Trong cùng lớp |
| --- | ---: | ---: | ---: |
| 4 Validation `chest_Temp` | −3,348 | −0,009 | −3,340 |
| 5 Test `chest_Temp` | −3,255 | −0,004 | −3,251 |
| 3 Test `chest_EDA` | +1,815 | −0,004 | +1,818 |
| 2 Validation `chest_EDA` | +1,295 | −0,001 | +1,295 |
| 1 Validation `wrist_TEMP` | −1,227 | −0,008 | −1,218 |

Ở Validation Fold 4, `chest_Temp` có mean 32,02 so với 34,21 của Train, std \(z\) 2,92 và 55,51% lượt timestep nằm ngoài P01–P99 Train. Chênh lệch cũng có riêng trong Baseline (−4,15 std Train), Stress (−2,79) và Amusement (−1,67). Validation Fold 4 và Test Fold 5 chia sẻ nhóm subject, nên hai hàng `chest_Temp` ở trên là hai đối chiếu với Train khác nhau trên cùng nhóm người. Fold 4 Test `wrist_BVP` có cực trị \(z=33,69\), với 0,37% lượt timestep có trị tuyệt đối của \(z\) lớn hơn 10. Đuôi lớn là dấu hiệu để khoanh vùng thời điểm cần xem. Nguồn audit chưa đủ để gán đuôi đó cho chuyển động, tiếp xúc cảm biến hay biến thiên sinh lý. [Bảng thống kê](normalization_statistics.csv) · [Heatmap từng subject Test](figures/test_subject_channel_heatmap.png) · [Boxplot Fold 1 trước/sau](figures/fold_01_subject_boxplots.png) · [Histogram Fold 4 trước/sau](figures/distribution_density_example.png).

Ba người Validation Fold 4 có độ lệch mean `chest_Temp` khác nhau dưới cùng scaler. S2 có \(z=-6,28\) và 96,6% lượt timestep ngoài P01–P99, S3 có \(z=-3,38\) và 72,9%, S13 có \(z=-0,53\) và 0%. Điều này giúp định vị phần dịch chuyển theo người. [Bảng từng subject](subject_distribution_statistics.csv).

**Mức độ bằng chứng.** Subject shift theo vị trí, độ phân tán và đuôi đã được đo. Trong nhiều ví dụ lớn, thành phần trong cùng lớp chiếm phần chủ đạo. Kết quả mô tả 15 subject và cửa sổ có overlap. Audit chưa xác định kênh nào gây giảm Macro-F1 hay mức ảnh hưởng nhân quả của từng kênh.

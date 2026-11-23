# BÁO CÁO KIỂM CHỨNG NGUỒN VÀ ĐỐI CHIẾU THỐNG NHẤT — CHƯƠNG 3

**Tài liệu bàn giao đi kèm:** `D:\Deep_learning_group\Projects\DL13-WESAD\docs\Chuong_3_Bo_du_lieu_va_tien_xu_ly.docx`  
**Tài liệu gốc đối chiếu (chỉ đọc, không sửa đổi):** `D:\Deep_learning_group\Projects\project_DL.docx`  
**Repository đối chiếu:** `D:\Deep_learning_group\Projects\DL13-WESAD`

---

## 1. Hệ thống nguồn bằng chứng đã sử dụng (theo thứ tự ưu tiên)

1. **Dữ liệu và artifacts thực tế (ưu tiên cao nhất):**
   - `data/processed/common_rate_64hz/config.json`: xác nhận chiến lược `common_rate`, `target_fs = 64`, `label_source_fs = 700`, danh sách và thứ tự chính xác 14 kênh đầu ra, danh sách 15 subject (`S2`–`S11`, `S13`–`S17`), `quality_guard_seconds = 2.0`, `resampling_method = scipy.signal.resample_poly`, `normalized = false`, `windowed = false`.
   - `data/processed/common_rate_64hz/subjects_summary.csv`: thời lượng bản ghi (giây), số mẫu đầu ra ở 64 Hz, số kênh (14), số mẫu bất thường ở 700 Hz (`S3 = 83`, 14 subject còn lại `= 0`), số mẫu đánh dấu không hợp lệ ở 64 Hz (`S3 = 265`, 14 subject còn lại `= 0`).
   - `data/processed/common_rate_64hz/label_segments.csv`: 253 đoạn nhãn liên tiếp ở độ phân giải gốc 700 Hz trên toàn bộ 15 subject (`S2` có 15 đoạn; `S3`–`S17` mỗi subject có 17 đoạn).
   - 15 cặp tệp `SX_X.npy` (tổng 296,91 MiB, kiểu `float32`, tổng 5.559.552 mốc thời gian × 14 kênh) và `SX_quality_mask.npy` (tổng 5,30 MiB, kiểu `bool`) — được đọc và hậu kiểm trực tiếp qua `np.load(..., mmap_mode="r")`.
   - Dữ liệu gốc `data/raw/WESAD/wesad_readme.pdf`, 15 tệp `SX_readme.txt` và kiểm chứng cục bộ đoạn bất thường `S3.pkl` (`[1771710:1771793]`) cùng cực trị `S5.pkl` (`wrist.BVP`).
2. **Notebook đã thực thi và có đầu ra lưu trữ:**
   - `notebooks/preprocessing.ipynb` (45 cells, exec 2–21): kiểm tra tải dữ liệu 15/15 subject, kiểm tra tính nhất quán cấu trúc (165 mảng, 0 thiếu tín hiệu, 0 sai shape), thống kê mô tả 120×14, kiểm tra NaN/`+Inf`/`-Inf` (0/60.807.600 ở 700 Hz và toàn bộ các kênh cổ tay), khảo sát giá trị đáng nghi (`chest_Temp ≈ -273,15`, `chest_ACC ≈ -6,6`, `|wrist_BVP| > 500`), trực quan hóa S3 và S5, phân phối nhãn 8 mã và 3 lớp mục tiêu, dòng thời gian nhãn 15 subject, histogram/boxplot, kiểm tra vùng đáng nghi S3 thuộc hoàn toàn nhãn 2 (Stress), và kiểm tra căn chỉnh cửa sổ 30 giây theo mốc thời gian.
   - `notebooks/02_preprocessing_common_rate.ipynb` (7 cells, exec 1–3): triển khai `convert_sampling_rate` bằng `scipy.signal.resample_poly` dựa trên `gcd(original_fs, 64)`, ghi trực tiếp `SX_X.npy` bằng `open_memmap`, bảo toàn `label_segments.csv` ở 700 Hz, tạo `SX_quality_mask.npy` với khoảng đệm `±2,0` giây, và chạy bước hậu kiểm đạt `15 / 15`.
3. **Mã nguồn, cấu hình và tài liệu nội bộ hiện hành (để đối chiếu trạng thái triển khai):**
   - `configs/default.yaml`, `src/data/preprocessing.py`, `src/data/resampling.py`, `src/data/windowing.py`, `src/data/normalization.py`, `src/data/dataset.py`, `src/data/splits.py`, `src/models/cnn.py`, `src/models/gru.py`, `src/models/cnn_gru.py`.
   - `docs/S2_Complete_Data_Understanding.md`, `README.md`, `Projects/README.md`, `data/processed/c369cd27d24abb09/S2/metadata.json`.
4. **Báo cáo gốc đang biên soạn:**
   - `D:\Deep_learning_group\Projects\project_DL.docx` (dùng để tham chiếu cấu trúc, định dạng, thuật ngữ, ký hiệu toán học và đối chiếu nhất quán; tuyệt đối không chỉnh sửa).
5. **Tài liệu khoa học và kỹ thuật gốc:**
   - Schmidt và cộng sự (2018), *Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection*, ACM ICMI 2018, DOI: `10.1145/3242969.3242985`.
   - Tài liệu kỹ thuật `wesad_readme.pdf` (kèm bộ dữ liệu WESAD) và đặc tả bộ lọc đa pha `scipy.signal.resample_poly` / lý thuyết lấy mẫu đa tốc độ (Oppenheim & Schafer; Crochiere & Rabiner).

---

## 2. Truy xuất nguồn của từng bảng thống kê và hình minh họa trong Chương 3

| Thành phần trong Chương 3 | Nội dung chính | Nguồn truy xuất trực tiếp | Loại bằng chứng |
|---|---|---|---|
| **Bảng 3.1** (Mục 3.1) | Thành phần 5 tệp trong mỗi thư mục `SX` và cấu trúc khóa trong `SX.pkl` | `data/raw/WESAD/wesad_readme.pdf` (mục I.1, II.3), `docs/S2_Complete_Data_Understanding.md`, `preprocessing.ipynb` (Cell 9–11) | Dữ liệu thực tế + tài liệu WESAD gốc |
| **Bảng 3.2** (Mục 3.2) | Tổng hợp 14 kênh thuộc 10 tín hiệu (8 ngực + 6 cổ tay), thứ tự cột, tần số gốc, dtype gốc, đơn vị vật lý và vai trò | `common_rate_64hz/config.json`, `preprocessing.ipynb` (Cell 11), `wesad_readme.pdf` (mục II.1–II.2) | Artifact + notebook + tài liệu gốc |
| **Bảng 3.3** (Mục 3.2) | Hệ thống 8 mã nhãn gốc WESAD (0–7), ánh xạ 3 lớp mục tiêu (`1→0, 2→1, 3→2`), số đoạn, số mẫu 700 Hz, thời lượng và tỷ lệ % | `common_rate_64hz/label_segments.csv` đối chiếu khớp 100% với `preprocessing.ipynb` (Cell 24–25) | Artifact thực tế + notebook EDA |
| **Bảng 3.4** (Mục 3.3) | Đối chiếu thông số chuyển đổi tần số lấy mẫu (tần số gốc, tần số đích 64 Hz, `gcd`, hệ số `L`/`M`, loại thao tác và tác động băng thông) | `02_preprocessing_common_rate.ipynb` (Cell 2, Cell 4) | Mã nguồn notebook đã thực thi |
| **Bảng 3.5** (Mục 3.4) | Cấu hình chia cửa sổ thời gian dự kiến ở 64 Hz (30 s / 15 s / 50% overlap / 1.920 mẫu) và quy tắc chấp nhận/loại bỏ cửa sổ | `project_DL.docx` (Mục 2.2.3, 2.3), `02_preprocessing_common_rate.ipynb` (Cell 0–1), `config.json` (`windowed: false`) | Thiết kế phương pháp (chưa thực thi) |
| **Bảng 3.6** (Mục 3.5) | Quy cách kích thước tensor đầu vào và luồng biến đổi cho CNN-1D, GRU và CNN-GRU | `project_DL.docx` (Mục 2.3.1–2.3.3), `common_rate_64hz/config.json` | Thiết kế phương pháp + artifact |
| **Bảng 3.7** (Mục 3.6) | Thống kê chi tiết 15 subject sau đồng bộ 64 Hz (thời lượng, số mẫu 700 Hz, số mẫu 64 Hz, số kênh, dtype, dung lượng MiB, trạng thái kiểm tra) | `common_rate_64hz/subjects_summary.csv`, hậu kiểm trực tiếp 15 tệp `SX_X.npy` và `02_preprocessing_common_rate.ipynb` (Cell 6) | Artifact thực tế đã hậu kiểm |
| **Bảng 3.8** (Mục 3.6) | Thời lượng (giây) và tỷ lệ 3 lớp mục tiêu theo từng subject | Tổng hợp trực tiếp từ `common_rate_64hz/label_segments.csv` và `subjects_summary.csv` | Artifact thực tế |
| **Bảng 3.9** (Mục 3.6) | Tổng hợp kiểm soát chất lượng dữ liệu sau tiền xử lý (NaN/Inf, bất thường S3 83→265 mẫu, hiệu ứng biên `resample_poly`, biến động BVP S5, ghi chú cảm biến `SX_readme.txt`) | `S3_quality_mask.npy`, `SX_X.npy`, `S3.pkl`, `preprocessing.ipynb` (Cell 16–23, 39–41), `SX_readme.txt` | Artifact thực tế + dữ liệu gốc |
| **Bảng 3.10** (Mục 3.6) | Thống kê mô tả toàn cục 14 kênh trên 5.559.552 mốc thời gian sau đồng bộ 64 Hz (Mean, Std, Min, Max) | Tính trực tiếp theo block trên 15 tệp `SX_X.npy` (`mmap_mode="r"`) | Artifact thực tế |
| **Hình 3.1** | Sơ đồ cấu trúc thư mục WESAD và tổ chức tệp `SX.pkl` | `wesad_readme.pdf`, `preprocessing.ipynb` | **Sơ đồ minh họa phương pháp/cấu trúc** |
| **Hình 3.2** | Phân phối 8 nhãn gốc, mất cân bằng 3 lớp mục tiêu và dòng thời gian 15 subject | `common_rate_64hz/label_segments.csv` | **Biểu đồ từ dữ liệu thật** |
| **Hình 3.3** | Dạng sóng bất thường S3 (`chest_Temp`, `chest_ACC_X`: gốc 700 Hz vs sau 64 Hz + vùng đệm `quality_mask`) và cực trị `wrist_BVP` ở S5 | `S3.pkl`, `S3_X.npy`, `S3_quality_mask.npy`, `S5.pkl` | **Biểu đồ từ dữ liệu thật** |
| **Hình 3.4** | Sơ đồ tổng thể pipeline tiền xử lý Common-rate 64 Hz đã triển khai | `02_preprocessing_common_rate.ipynb` | **Sơ đồ minh họa phương pháp** |
| **Hình 3.5** | Minh họa cơ chế trượt cửa sổ (Sliding Window) 30 s / bước trượt 15 s gắn vùng nhãn 700 Hz và `quality_mask` | Thiết kế mục 3.4 | **Sơ đồ minh họa phương pháp** |
| **Hình 3.6** | Luồng biến đổi tensor từ chuỗi 14 kênh 64 Hz qua Z-score (Fit-on-Train) đến CNN-1D, GRU và CNN-GRU | Thiết kế mục 3.5 & Chương 2 `project_DL.docx` | **Sơ đồ minh họa phương pháp** |
| **Hình 3.7** | Thời lượng bản ghi, số mốc 64 Hz và phân bổ 3 trạng thái mục tiêu theo từng subject | `subjects_summary.csv`, `label_segments.csv` | **Biểu đồ từ dữ liệu thật** |
| **Hình 3.8** | Boxplot phân phối sau đồng bộ 64 Hz của 15 subject trên 6 kênh đại diện | 15 tệp `SX_X.npy` và `SX_quality_mask.npy` (lấy mẫu trên phần hợp lệ) | **Biểu đồ từ dữ liệu thật** |

---

## 3. Phân biệt rõ trạng thái triển khai thực tế

### 3.1. Các bước đã xác minh hoàn thành và có bằng chứng thực tế
- **EDA trên toàn bộ 15 subject (`preprocessing.ipynb`):**
  - Đọc tuần tự thành công 15/15 tệp `SX.pkl` (`S2`–`S11`, `S13`–`S17`), tổng thời lượng **86.868,0 giây** (1.447,80 phút ≈ 24,13 giờ), tổng **60.807.600** mẫu ở tần số 700 Hz.
  - Kiểm tra tính nhất quán cấu trúc 165 mảng (`15 × 11`): 0 tín hiệu thiếu, 0 lỗi shape, dtype đồng nhất (`float64` cho 9 tín hiệu, `float32` cho `chest_Temp`, `int32` cho `label`).
  - Kiểm tra giá trị không hữu hạn trên toàn bộ 14 kênh × 15 subject: **0 NaN, 0 +Inf, 0 -Inf** (tỷ lệ 0%).
  - Phát hiện đoạn bất thường duy nhất về giá trị vật lý tại **S3**: 83 mẫu liên tiếp `[1771710, 1771793)` (`2531,0143`–`2531,1329` giây, kéo dài `0,1186` giây), thuộc hoàn toàn **nhãn 2 (Stress)**, trong đó cả 8 kênh RespiBAN đồng loạt rơi về giá trị quy đổi tương ứng mã ADC bằng 0 (`chest_Temp = -273,15 °C`, `chest_ACC = -6,6 g` cả 3 trục, `ECG = EMG = -1,5 mV`, `EDA = 0,0 µS`, `Resp = -50,0%`).
  - Khảo sát biên độ `wrist_BVP` với ngưỡng thăm dò `|BVP| > 500`: xuất hiện ở cả 15 subject, nhiều nhất tại `S5` (4.457 mẫu, 1,1128%, 555 đoạn, cực trị `1789,0` tại `4122,84375 s`, đoạn vượt ngưỡng dài nhất 37 mẫu = `0,578125 s`) và ít nhất tại `S16` (3 mẫu, 0,0008%).
  - Thống kê đầy đủ phân phối 8 nhãn gốc, 3 lớp mục tiêu (`Baseline = 53,12%`, `Stress = 30,06%`, `Amusement = 16,82%`, tỷ lệ Baseline/Amusement ≈ 3,16 lần) và dòng thời gian 253 đoạn liên tiếp.
- **Tiền xử lý Common-rate 64 Hz (`02_preprocessing_common_rate.ipynb` → `data/processed/common_rate_64hz/`):**
  - Đồng bộ toàn bộ 14 kênh của 15 subject về 64 Hz bằng `scipy.signal.resample_poly` với bộ lọc đa pha FIR (`up = 64 / gcd`, `down = f_source / gcd`; riêng `wrist_BVP` 64 Hz giữ nguyên).
  - Xuất 15 tệp `SX_X.npy` dạng memory-mapped `float32` (tổng 5.559.552 mốc thời gian × 14 kênh, 296,91 MiB), bảo toàn thứ tự 14 kênh trong `config.json`.
  - Xuất `label_segments.csv` giữ nguyên ranh giới nhãn ở 700 Hz (253 đoạn).
  - Xuất 15 tệp `SX_quality_mask.npy` kiểu `bool`, giữ nguyên chuỗi liên tục nhưng đánh dấu `False` vùng bất thường S3 mở rộng ±2,0 giây mỗi bên (`[161856, 162121)` → **265 mẫu ở 64 Hz**, tức `2529,0000`–`2533,1406 s`, chiếm 0,0638% của S3 và 0,0048% toàn tập).
  - Hậu kiểm tự động đạt **15/15 subject** (`ndim == 2`, `shape[1] == 14`, `dtype == float32`, `len(quality_mask) == len(X)`, `isfinite == True`).

### 3.2. Các chức năng có trong source code cũ nhưng thuộc nhánh 5 kênh 32 Hz (KHÔNG thuộc dữ liệu 14 kênh 64 Hz)
- Các module `src/data/preprocessing.py`, `src/data/resampling.py`, `src/data/windowing.py`, `src/data/normalization.py`, `src/data/splits.py`, `configs/default.yaml` và thư mục cache `data/processed/c369cd27d24abb09/` được xây dựng và chạy cho nhánh cũ: **5 kênh cổ tay (`ACC_X/Y/Z, BVP, EDA`), 32 Hz, 960 mẫu/cửa sổ, 2.136 cửa sổ hợp lệ, Grouped 5-Fold (`splits/grouped_5fold_seed42.json`)**, nơi E1 (CNN-1D) và E2 (GRU) đã có kết quả chạy.
- Các số liệu cửa sổ (2.136 cửa sổ) và kết quả E1/E2 của nhánh 32 Hz **không được trộn lẫn hoặc thay thế** cho nhánh 14 kênh 64 Hz trong Chương 3.

### 3.3. Các bước mới ở mức thiết kế / dự kiến cho giai đoạn thực nghiệm (chưa thực thi trên dữ liệu 14 kênh 64 Hz)
- **Chia cửa sổ thời gian (Sliding Window)** cấu hình `64 Hz`, độ dài `30 giây` (`T = 1.920`), bước trượt `15 giây` (`overlap = 50%`), kết hợp lọc vùng nhãn mục tiêu từ `label_segments.csv` và lọc mẫu không hợp lệ từ `SX_quality_mask.npy` (`config.json` xác nhận `"windowed": false`).
- **Chuẩn hóa Z-score theo từng kênh** theo nguyên tắc chỉ ước lượng $(\mu_c, \sigma_c)$ trên tập Train của từng lượt chia (`config.json` xác nhận `"normalized": false`).
- **Giao thức chia tập theo người tham gia định hướng LOSO** (Leave-One-Subject-Out, 15 lượt đánh giá) và huấn luyện/đánh giá ba kiến trúc CNN-1D, GRU, CNN-GRU trên tensor 14 kênh 64 Hz (`src/models/cnn_gru.py` hiện mới là file placeholder).

---

## 4. Các mâu thuẫn phát hiện giữa `project_DL.docx` và tình trạng nghiên cứu hiện tại

> **Nguyên tắc tuân thủ:** Không chỉnh sửa `project_DL.docx`, Chương 1 hay Chương 2. Các mâu thuẫn dưới đây được ghi nhận để nhóm thống nhất khi ghép toàn bộ báo cáo.

| STT | Vị trí trong `project_DL.docx` / Repo | Nội dung hiện có | Thực tế đã xác minh cho hướng 14 kênh 64 Hz | Khuyến nghị đồng bộ cho nhóm |
|---|---|---|---|---|
| 1 | Chương 1 (Mục 1.1, 1.2, 1.3, 1.4) | Mô tả chỉ dùng tín hiệu **cổ tay**, đồng bộ về **32 Hz**, cửa sổ `960` mẫu (`5 × 960`), đánh giá **5-fold theo người** (9 Train / 3 Val / 3 Test). | Chương 2 và nhánh `common_rate_64hz` dùng **14 kênh (8 ngực + 6 cổ tay)** ở **64 Hz**, cửa sổ `1.920` mẫu, định hướng đánh giá theo người (LOSO). | Nhóm cần cập nhật lại Chương 1 cho thống nhất với Chương 2 và Chương 3 (hoặc giải thích rõ quá trình mở rộng phạm vi từ cổ tay sang đa thiết bị). |
| 2 | Chương 1 (Mục 1.2, đoạn P025 & Mục 1.4, đoạn P037) | Mục 1.2 viết *"Đầu vào gồm sáu kênh tín hiệu từ cổ tay: ba trục ACC, một BVP và một EDA"* (liệt kê 5 kênh) và công thức ghi `5 × 960`; Mục 1.4 lại liệt kê 4 nhóm `ACC, BVP, EDA, TEMP` (= 6 kênh). | Nhánh cũ `c369cd27d24abb09` dùng đúng **5 kênh** (không có TEMP); nhánh mới `common_rate_64hz` dùng đủ **14 kênh** (có cả `chest_Temp` và `wrist_TEMP`). | Sửa mâu thuẫn nội bộ 5 kênh vs 6 kênh trong Mục 1.2 và Mục 1.4 của Chương 1. |
| 3 | Chương 1 (Mục 1.2, đoạn P022) | Viết *"một trong ba trạng thái được gán nhãn trong WESAD là baseline hoặc stress"* (thiếu *amusement*). | Bài toán phân loại 3 lớp: `Baseline`, `Stress`, `Amusement`. | Bổ sung *amusement* trong câu phát biểu bài toán ở Mục 1.2. |
| 4 | Chương 1 (Mục 1.3, đoạn P033) | Ký hiệu câu hỏi thứ ba ghi `R3:` thay vì `RQ3:` (trong khi câu 1 và 2 là `RQ1:`, `RQ2:`). | Quy ước nhóm là `RQ1`, `RQ2`, `RQ3`. | Chỉnh `R3:` thành `RQ3:`. |
| 5 | Chương 1 vs Chương 2 (`project_DL.docx`) | Định dạng cỡ chữ thân bài không đồng nhất: Chương 1 dùng **Times New Roman 13 pt**, Chương 2 dùng **Times New Roman 14 pt**; cả hai đều dùng style `normal` định dạng thủ công thay vì Heading Styles chuẩn. | Chương 3 được chuẩn hóa dùng **Times New Roman 13 pt** cho thân bài (theo đúng yêu cầu kế thừa định dạng chuẩn của tài liệu), khổ A4, lề `2,54 cm`, giãn dòng `1,3`, thụt đầu dòng `1,27 cm` và áp dụng Heading Styles (`Heading 1/2/3`) để hỗ trợ Mục lục tự động. | Khi ghép báo cáo hoàn chỉnh, cần thống nhất lại cỡ chữ thân bài (13 pt) và gắn Heading Styles cho Chương 1 và Chương 2. |
| 6 | Chương 2 (Mục 2.3.3.3, đoạn P248) | Viết đầu ra CNN là $X \in \mathbb{R}^{B \times D \times T'}$ (trùng ký hiệu đầu vào $X$, trong khi ngay trước đó định nghĩa $F = \mathrm{CNN}(X)$). | Trong Chương 3 sử dụng ký hiệu nhất quán $F \in \mathbb{R}^{B \times D \times T'}$ và $F' = \mathrm{Transpose}(F) \in \mathbb{R}^{B \times T' \times D}$. | Sửa ký hiệu $X$ thành $F$ tại đoạn P248 của Chương 2 khi biên tập chung. |
| 7 | Chương 2 (`project_DL.docx`) | Có trích dẫn dạng `[3]` (Ordóñez và Roggen, 2016) lẫn trích dẫn tác giả–năm, chưa có mục Tài liệu tham khảo cuối báo cáo; Bảng 1 mục 2.1 có lỗi chính tả `Accelorometer`. | Chương 3 dùng hệ thống trích dẫn nhất quán, đúng chính tả (`Accelerometer`), kèm danh sách tài liệu tham khảo xác minh đầy đủ ở cuối chương để dễ gom vào cuối báo cáo. | Nhóm cần thống nhất 1 chuẩn trích dẫn (IEEE `[1]` hoặc APA tác giả–năm) cho toàn báo cáo và sửa lỗi chính tả `Accelorometer`. |
| 8 | `README.md` và `src/` so với `notebooks/` | `README.md` mô tả `notebooks/` chỉ có file giữ chỗ và pipeline chỉ dùng 5 kênh cổ tay 32 Hz + Grouped 5-Fold. | Thực tế `notebooks/` đã chứa `preprocessing.ipynb` và `02_preprocessing_common_rate.ipynb` đã thực thi xong dữ liệu `data/processed/common_rate_64hz/`. | Cập nhật `README.md` khi nhóm hoàn tất chuyển đổi pipeline sang 14 kênh 64 Hz. |

---

## 5. Các phát hiện kỹ thuật mới và thông tin cần nhóm xác nhận trước giai đoạn thực nghiệm

1. **Hiệu ứng biên của `scipy.signal.resample_poly` với cấu hình đệm mặc định (`padtype='constant'`, đệm 0):**
   - Kiểm tra trực tiếp trên 15 tệp `SX_X.npy` cho thấy các tín hiệu có độ lệch DC lớn như `chest_Temp` (~34 °C) và `wrist_TEMP` (~32,6 °C) bị kéo giảm mạnh ở một vài mẫu đầu/cuối bản ghi (ví dụ `chest_Temp[0]` rơi xuống `16,45 °C` ở S2, `17,20 °C` ở S3; cuối bản ghi `wrist_TEMP` giảm về `1,34–1,58 °C`).
   - Do toàn bộ 15 subject đều mở đầu và kết thúc bằng vùng nhãn `0 (Undefined)` dài (ít nhất `26,20 s` ở đầu S14 và `135,75 s` ở cuối S9), hiệu ứng biên này **không lọt vào các vùng nhãn mục tiêu `{1, 2, 3}`**. Tuy nhiên, nhóm nên cân nhắc dùng `padtype='line'` hoặc `padtype='edge'` nếu chạy lại bước resampling trong tương lai.
2. **Dao động Gibbs của bộ lọc FIR quanh đoạn mất mẫu 83 mẫu ở S3:**
   - Vì notebook 02 giữ nguyên chuỗi gốc trước khi resampling để bảo toàn cấu trúc thời gian, xung nhảy bậc của S3 (`-273,15 °C`, `-6,6 g` cùng các kênh ngực khác tại `2531,0143`–`2531,1329 s`) tạo dao động quá độ (`chest_Temp` trong vùng mask đạt `[-297,37; +57,91] °C`).
   - Khoảng đệm `QUALITY_GUARD_SECONDS = 2,0` giây (265 mẫu 64 Hz, từ `2529,0000` đến `2533,1406 s`) đã bao trọn phần dao động quá độ này; cần xác nhận giữ quy tắc loại bỏ mọi cửa sổ giao với `quality_mask == False` khi chạy windowing (và có thể đánh giá độ nhạy của ngưỡng đệm 2,0 giây nếu cần).
3. **Thông tin cần bổ sung vào Chương 3 sau khi bước thực nghiệm bắt đầu:**
   - **Bảng thống kê cửa sổ hợp lệ** cho Mục 3.6 (số cửa sổ theo từng subject, theo từng lớp 0/1/2, số cửa sổ bị loại do giao vùng `quality_mask == False` tại S3, và tỷ lệ mất cân bằng ở cấp cửa sổ) — hiện chưa đưa vào vì `windowed = false`.
   - **Quy tắc xử lý `wrist_BVP` biên độ lớn (ví dụ S5) và ghi chú `chest_Temp` chưa gắn chắc (S2, S17):** nhóm cần xác nhận có giữ nguyên để đánh giá khả năng chống nhiễu của mô hình hay áp dụng lọc/cắt ngưỡng bổ sung.
   - **Giao thức đánh giá chính thức (LOSO vs Grouped 5-Fold):** cần chốt thống nhất giữa Chương 1, Chương 3 và Chương 4 để quy trình fit Z-score theo từng fold/lượt chia được trình bày đồng bộ xuyên suốt.

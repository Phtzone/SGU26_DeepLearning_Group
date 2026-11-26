# Điều tra đuôi Z-score của nhiệt độ ngực sau Median Centering

**Thí nghiệm** `common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1` · **Group** `20261126T025058Z_fa1c92ea`.

Điểm **Z = −133,69** xuất hiện ở đầu một đoạn tăng nhiệt đã có trong tín hiệu gốc của S2. Mẫu 64 Hz tái hiện đúng từ bản ghi 700 Hz. Trong cửa sổ đang xét, nhiệt độ thấp hơn median của chính cửa sổ **2,8784 °C**, còn std Train Fold 4 sau centering chỉ **0,02153 °C**. Sau biến đổi, cả 5 tập Train có mẫu vượt \(|Z|=10\). Trên Validation, các giá trị vượt \(|Z|=100\) tập trung ở hai cửa sổ của S2 thuộc lớp Baseline. Ghi chú đầu đo của S2 khiến thay đổi tiếp xúc là một giả thuyết có căn cứ. Dữ liệu chưa cho phép xác định nguyên nhân của đoạn tăng. [Bằng chứng kiểm chứng](audit_verification.json).

## 1. Truy vết cửa sổ S2_84900_86820

| Nguồn hoặc bước xử lý | Kết quả đối chiếu |
| --- | --- |
| Cửa sổ 64 Hz | 1.920 mẫu, chỉ số `[84900,86820)`, từ **1326,5625 đến 1356,5625 s** tính từ đầu PKL. Mẫu có trị tuyệt đối Z lớn nhất ở chỉ số **84923**, **1326,921875 s**, tức 0,359375 s sau đầu cửa sổ. |
| RespiBAN gốc | Header OpenSignals xác nhận TEMP ở **CH4**, 700 Hz và 16 bit. Đã đối chiếu đủ **21.000 mẫu** trong cửa sổ, từ nSeq 1.020.611 đến 1.041.610, với đoạn `[928594,949594)` của PKL. Theo công thức điện trở nhiệt trong tài liệu WESAD, chênh lệch lớn nhất so với PKL là **0,000047 °C**. Mốc offset 92.017 hàng được kiểm lại trên toàn đoạn đích. |
| PKL 700 Hz → Common-rate | Nhiệt độ gốc trong cửa sổ có min/median/max **28,5532 / 31,4607 / 32,8428 °C**. Dùng lại `resample_poly(up=16, down=175)` trên toàn chuỗi S2, cast float32, thu được **389.056/389.056 mẫu** khớp từng phần tử với `S2_X.npy`. Quanh điểm đích ±0,25 s, PKL dao động **28,5532–28,7336 °C**. Mẫu 64 Hz **28,5822 °C** nằm trong khoảng quan sát này. |
| Nhãn | Cả 21.000 nhãn 700 Hz là **1 (Baseline)**, ánh xạ **0** trong window index. Đoạn nhãn bao phủ `[214583,1015383)` ở 700 Hz, từ 306,547 đến 1450,547 s. Cửa sổ nằm trọn trong đoạn này. |
| Quality mask | **1.920/1.920 mẫu True** trong cửa sổ, cũng như toàn bộ 389.056 mẫu 64 Hz của S2. Kiểm lại hai quy tắc đã dùng trong notebook, gần −273,15 °C ở Temp hoặc đồng thời ba trục ACC gần −6,6, không có mẫu nào của S2 khớp. Quy tắc hiện hành không đo mức tiếp xúc đầu đo hoặc phát hiện một đoạn nhiệt thay đổi từ từ. |
| Trừ median → Z-score | Median 64 Hz **31,4606 °C**. Mẫu đích sau centering là **−2,878355 °C**. Với mean Train của residual **−0,0003534 °C** và std **0,0215272 °C**, mẫu này thành **−133,6912**. Z-score baseline tại cùng mẫu là **−8,6326**. |

[Bảng tín hiệu 64 Hz](s2_target_trace_64hz.csv) · [Zoom quanh điểm đích ở 700 Hz](s2_native_peak_zoom.csv) · [Dữ liệu RespiBAN](../../../../data/raw/WESAD/S2/S2_respiban.txt) · [WESAD README, mục II.1–II.3](../../../../data/raw/WESAD/wesad_readme.pdf) · [Notebook tiền xử lý](../../../../notebooks/02_preprocessing_common_rate.ipynb) · [SciPy resample_poly](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html).

Trong 30 s, biên độ 64 Hz là **4,1493 °C**. Median 5 s đầu và 5 s cuối lần lượt là **29,1453** và **32,5751 °C**. Sai phân lớn nhất giữa hai timestep 64 Hz là **0,03244 °C**. Đồ thị cho thấy phần tăng kéo dài, bắt đầu gần mẫu đích và tiếp tục qua nhiều cửa sổ. Cả ba đồ thị giữ nguyên lưới thời gian. Bước trừ median dịch trục tung, bước Z-score tiếp tục dịch và đổi thang đo. [Tín hiệu trước và sau mỗi bước, PNG](figures/s2_window_three_stages.png) · [SVG](figures/s2_window_three_stages.svg) · [Bối cảnh và zoom 700/64 Hz](figures/s2_native_context_and_neighbors.png).

### Cửa sổ lân cận trên Validation Fold 4

Các hàng dưới đều thuộc S2, Baseline và có toàn bộ quality mask True. Bảng CSV đầy đủ có ba cửa sổ mỗi phía, kèm các fold mà S2 thuộc Train hoặc Validation.

| Cửa sổ | Khoảng nhiệt 64 Hz (°C) | Median (°C) | Z có trị tuyệt đối lớn nhất | Số mẫu vượt trị tuyệt đối Z = 100 | Z tại timestamp 84923 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `S2_82980_84900` | 28,4451–28,6140 | 28,5240 | +4,20 | 0 | — |
| `S2_83940_85860` | 28,5141–31,4627 | 28,6012 | +132,94 | 304 | −0,86 |
| `S2_84900_86820` | 28,5822–32,7316 | 31,4606 | −133,69 | 201 | −133,69 |
| `S2_85860_87780` | 31,4585–33,5591 | 32,7301 | −59,05 | 0 | — |
| `S2_86820_88740` | 32,7268–34,0560 | 33,5503 | −38,24 | 0 | — |

Hai cửa sổ liền nhau chồng **960 mẫu**. Ví dụ, timestamp 84923 có cùng nhiệt độ 64 Hz nhưng cho Z = −0,86 trong cửa sổ trước và Z = −133,69 trong cửa sổ đích, vì median hai cửa sổ chênh khoảng **2,86 °C**. Điểm dương +132,94 của cửa sổ trước nằm ở mẫu 85859, tại đầu còn lại của phần chồng lấp. Đây là hệ quả có thể tái tạo của centering theo cửa sổ. [Bảng các cửa sổ lân cận](s2_neighbor_windows.csv).

## 2. Đuôi phân phối trên cả 5 fold

Phân tích chỉ dùng Train và Validation, với đúng mean/std Train đã lưu của từng fold và từng phương pháp. Tổng Validation có **2.134 cửa sổ**, mỗi subject xuất hiện trong đúng một tập Validation. Mỗi cửa sổ vẫn gồm 1.920 mẫu và các cửa sổ có overlap. Bảng dùng **lượt timestep** để mô tả phân phối mà pipeline nhìn thấy, và thêm **timestamp riêng** để kiểm mức tập trung. Một timestamp được tính là vượt ngưỡng nếu có ít nhất một cửa sổ chứa nó vượt ngưỡng. Số đoạn liền kề vượt ngưỡng chỉ mô tả vị trí thời gian, chưa xác định số sự cố cảm biến. Kết quả giữa các fold Train có thể lặp cùng một subject và cùng một cửa sổ.

| Fold | Std Train cũ → mới (°C) | Tỷ số std | Max trị tuyệt đối Z Train, cũ → mới | Max trị tuyệt đối Z Validation, cũ → mới | Lượt Train / Validation vượt trị tuyệt đối Z = 10 sau biến đổi |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,55689 → 0,07549 | 20,62× | 3,39 → 38,13 | 1,22 → 3,88 | 4.529 / 0 |
| 2 | 1,56372 → 0,07545 | 20,72× | 3,44 → 38,15 | 1,01 → 4,03 | 4.531 / 0 |
| 3 | 1,58417 → 0,07535 | 21,03× | 3,41 → 38,20 | 1,13 → 3,71 | 4.542 / 0 |
| 4 | 0,65142 → 0,02153 | 30,26× | 2,55 → 16,11 | 9,30 → 133,69 | 1.881 / 15.975 |
| 5 | 0,64710 → 0,02354 | 27,49× | 2,45 → 12,92 | 2,37 → 14,73 | 971 / 622 |

Baseline có **0 lượt trên ngưỡng 10** trong cả Train và Validation. Sau centering + Z-score, mức vượt 10 xuất hiện ở **6–9 cửa sổ Train mỗi fold**. Bốn cửa sổ S2 thuộc Baseline có \(|Z|>20\) lặp lại trong Train của Fold 1, 2 và 3. Trên Validation, Fold 4 có **15.975/16.597** lượt vượt 10, khoảng 96,3% số lượt của cả 5 fold. Fold 5 có 622 lượt thuộc hai cửa sổ S9. Validation Fold 1–3 không vượt 5. Vì vậy hệ số đổi thang lớn ở mọi fold, còn cực trị nghiêm trọng tập trung ở vài chủ thể và fold. [Thống kê theo fold, subject, class và subject × class](zscore_statistics_by_group.csv) · [Thống kê từng cửa sổ](zscore_statistics_by_window.csv) · [So sánh đuôi phân phối theo fold](figures/absolute_z_survival_by_fold.png).

### Phân vị và ngưỡng vượt

Phân vị nội suy tuyến tính, tính riêng cho Z có dấu và trị tuyệt đối Z. Bảng dưới minh họa Z có dấu trên Validation. CSV chứa đủ **min, P01, P05, P25, P50, P75, P95, P99, P99,9 và max** cho mọi fold, vai trò, subject, lớp, subject × lớp và cửa sổ.

| Chỉ số | Fold 4 baseline | Fold 4 sau biến đổi | Gộp Validation baseline | Gộp Validation sau biến đổi |
| --- | ---: | ---: | ---: | ---: |
| Min | −9,303 | −133,691 | −9,303 | −133,691 |
| P01 | −9,126 | −10,015 | −8,942 | −3,135 |
| P05 | −8,947 | −3,012 | −4,750 | −1,025 |
| P25 | −4,775 | −0,599 | −0,285 | −0,180 |
| P50 | −3,197 | +0,016 | +0,231 | +0,002 |
| P75 | −0,988 | +0,648 | +0,650 | +0,183 |
| P95 | +0,155 | +3,101 | +1,043 | +1,032 |
| P99 | +0,596 | +9,946 | +2,248 | +3,181 |
| P99,9 | +0,821 | +66,300 | +2,322 | +21,653 |
| Max | +0,855 | +132,940 | +2,370 | +132,940 |

Trên Validation gộp, **P50 trị tuyệt đối Z giảm 0,537 → 0,182**, trong khi **P99,9 tăng 9,219 → 37,090**. Riêng Fold 4, P99,9 trị tuyệt đối Z tăng **9,265 → 84,922**. Phần giữa thu hẹp nhờ centering, đồng thời phần đuôi xa mở rộng. [Histogram trị tuyệt đối Z](absolute_z_histogram.csv) · [Biểu đồ tỷ lệ vượt ngưỡng](figures/threshold_rates_by_fold.png).

| Ngưỡng nghiêm ngặt | Baseline, số lượt (%) | Sau biến đổi, số lượt (%) | Số cửa sổ sau biến đổi | Số subject sau biến đổi | Timestamp riêng sau biến đổi |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 417.914 (10,1998%) | 88.408 (2,1577%) | 264 | 9 | 73.662 |
| 5 | 166.828 (4,0717%) | 42.074 (1,0269%) | 92 | 6 | 34.466 |
| 10 | 0 (0%) | 16.597 (0,4051%) | 25 | 3 | 12.423 |
| 20 | 0 (0%) | 9.040 (0,2206%) | 11 | 2 | 7.224 |
| 50 | 0 (0%) | 2.848 (0,0695%) | 6 | 1 | 2.626 |
| 100 | 0 (0%) | 505 (0,0123%) | 2 | 1 | 505 |

Mẫu số của tỷ lệ lượt trên Validation gộp là **4.097.280**. Mẫu số timestamp riêng là **2.092.800**. Các ngưỡng trong bảng đều dùng \(|Z|>\) ngưỡng. Với ngưỡng 100, **304** lượt ở cửa sổ trước và **201** lượt ở cửa sổ đích. Tổng 505 cũng là số timestamp riêng, thuộc hai đoạn vượt ngưỡng liên tục tách nhau. Có **25/2.134 cửa sổ (1,17%)** chứa mẫu vượt \(|Z|=10\), ứng với S2, S3 và S9.

### Subject và lớp

| Lớp Validation | Cửa sổ | Lượt vượt 10 (% trong lớp) | Lượt vượt 20 | Lượt vượt 50 | Lượt vượt 100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 1.150 | 11.224 (0,5083%) | 7.351 | 2.848 | 505 |
| Stress | 638 | 5.371 (0,4385%) | 1.689 | 0 | 0 |
| Amusement | 346 | 2 (0,0003%) | 0 | 0 | 0 |

Tất cả các lượt Validation vượt 50 đến từ **S2, Baseline**, ở 6 cửa sổ. Ở ngưỡng 10, S2 đóng góp 9.966 lượt trên 9 cửa sổ, S3 có 6.009 lượt trên 14 cửa sổ và S9 có 622 lượt trên 2 cửa sổ. Ba cửa sổ Stress S3 vượt 20. S2 và S3 cũng có mẫu vượt 10 trong Train Fold 1–3. Train Fold 4 có mẫu vượt 10 ở S9, S15, S4, còn Train Fold 5 có S15, S4, S5. Chênh lệch theo lớp ở đây chịu ảnh hưởng của subject, giai đoạn đo và hệ số từng fold. Chưa có cơ sở coi các cực trị là dấu hiệu phân biệt lớp. [CSV phân tầng](zscore_statistics_by_group.csv).

## 3. Biến thiên nhiệt độ sau khi bỏ mức nền

Trên 2.134 cửa sổ duy nhất, xét độ lệch chuẩn trong cửa sổ, biên độ đỉnh–đỉnh, dải P95–P05, và P99 độ lớn sai phân kề nhau. Trừ một hằng số trong cửa sổ giữ nguyên tất cả các độ đo trên, đến sai số số thực. Kết quả dưới tính bằng °C, trước khi chia std Train.

| Subject | Median std trong cửa sổ | P95 std trong cửa sổ | Median biên độ đỉnh–đỉnh | P95 biên độ đỉnh–đỉnh | Max biên độ đỉnh–đỉnh |
| --- | ---: | ---: | ---: | ---: | ---: |
| S2 | 0,0251 | 0,2340 | 0,1381 | 0,8480 | 4,1493 |
| S3 | 0,0202 | 0,1454 | 0,0921 | 0,5039 | 1,5454 |
| S4 | 0,0078 | 0,0328 | 0,0615 | 0,1514 | 0,3826 |
| S9 | 0,0139 | 0,0305 | 0,0984 | 0,1488 | 0,4175 |
| S14 | 0,0098 | 0,0192 | 0,0728 | 0,1076 | 0,1455 |

S2 và S3 có đuôi biên độ dài hơn, dù median của nhiều người nằm trong khoảng hẹp. Trong S2, P95 biên độ ở Baseline là **2,1934 °C**, so với **0,2385 °C** ở Stress. Trong S3, P95 biên độ cao hơn ở Stress (**1,0075 °C**) so với Baseline (**0,3256 °C**). Hai hình thái này hỗ trợ việc kiểm theo từng subject và từng đoạn liên tục. [Bảng đầy đủ theo subject và lớp](centered_window_variation_by_subject.csv) · [Boxplot của 15 subject](figures/centered_window_variation_by_subject.png).

**Giải thích được tới đâu.** Scale thấp khuếch đại mọi chênh lệch sau centering, dù là sai số đo, nhiễu ngắn hạn hay biến thiên nhiệt thực. Với S2, hướng tăng kéo dài đã xuất hiện từ kênh RespiBAN gốc và vẫn thấy trên 64 Hz. Biên độ cổ tay trong cùng cửa sổ chỉ khoảng **0,1194 °C**. Biên độ từng trục ACC ngực khoảng **0,044 / 0,108 / 0,078 g**. Đây là thông tin bối cảnh từ cảm biến và vị trí khác. Chúng chưa đủ xác nhận nhiệt độ da ngực thực sự thay đổi như vậy. Ghi chú S2 nói đầu đo ngực có lúc gắn không sát, nhưng không ghi thời điểm độ bám thay đổi. Giả thuyết thay đổi tiếp xúc phù hợp với ghi chú và dạng tín hiệu. Chưa thể kết luận đoạn này là nhiễu, tái tiếp xúc hay biến thiên sinh lý có ý nghĩa. Nghiên cứu thực nghiệm trên cảm biến tiếp xúc cũng cho thấy cách gắn và chênh lệch nhiệt với môi trường có thể ảnh hưởng số đo. [Ghi chú gốc S2](../../../../data/raw/WESAD/S2/S2_readme.txt) · [MacRae và cộng sự, thí nghiệm về sai số cảm biến tiếp xúc](https://pmc.ncbi.nlm.nih.gov/articles/PMC8309895/).

## 4. Đề xuất kiểm chứng tiếp theo

Đặt `r = x − median(window)`, `μr` là mean Train của residual, `σr` là std Train của residual hiện tại. Mỗi đề xuất giữ 13 kênh còn lại, các cửa sổ và split. Ước lượng tham số riêng từ Train từng fold, rồi đánh giá đối chứng trên đúng Validation của fold đó.

| Phương án | Ưu điểm | Hạn chế và nguy cơ mất thông tin | Phép kiểm trên Train/Validation |
| --- | --- | --- | --- |
| **Giữ Median Centering, dùng std Train trước centering** | Dùng `(r − μr) / σgốc`, giữ centering và chỉ thay mẫu số so với pipeline đang kiểm. Thang đo vẫn fit theo fold, giảm mức khuếch đại điểm lớn. Thứ tự và dạng sóng theo thời gian được giữ khi đổi thang. | Std gốc chứa khác biệt nhiệt nền giữa người, lớn hơn std residual 20,62–30,26 lần trong 5 fold này. Các biến thiên nhỏ có thể trở nên yếu khi đặt cạnh 13 kênh khác. | Đối chứng đầu tiên. So P99,9 và tỷ lệ vượt ngưỡng cùng median/IQR và biên độ thường gặp theo subject/lớp. Theo dõi cả độ giảm cực trị và mức nén tín hiệu điển hình. |
| **Robust Scaling** | Fit median và IQR trên các residual Train, để các cực trị Train ít chi phối trung tâm và mẫu số. Median theo cửa sổ vẫn bỏ được mức nền. | IQR residual Train thực tế chỉ **0,01685–0,02030 °C**, nhỏ hơn std residual ở **cả 5 fold**. Chia trực tiếp cho IQR có thể làm đuôi xa lớn hơn. Một lựa chọn nghiên cứu riêng là fit mẫu số robust trên dữ liệu Train trước centering, có thể nén biến thiên nhỏ tương tự std gốc. | Chốt trước sẽ dùng IQR của residual hay nhiệt độ gốc và quy tắc khi IQR rất nhỏ. So đồng thời độ lớn phần giữa và đuôi; đừng suy ra tail được khống chế chỉ từ tên Robust Scaling. [Tài liệu scikit-learn RobustScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html). |
| **Đặt cận dưới cho mẫu số** | Dùng `(r − μr) / max(σr, s_min)`. Hạn chế hệ số khuếch đại khi std Train thấp, vẫn giữ thứ tự và sự khác biệt giữa các mẫu. | Cận dưới quá cao làm giảm tín hiệu nhiệt độ nhỏ, quá thấp để lại đuôi dài. Cực trị đủ lớn vẫn có thể vượt ngưỡng. Giá trị của `s_min` cần căn cứ đo lường hoặc quy tắc đã định từ Train. | Chốt `s_min` trước dựa trên độ lặp lại của đầu đo hoặc một quy tắc theo thang nhiệt độ Train gốc. Đánh giá độ nhạy với cận dưới, kiểm phần giữa và các đoạn lớn như S2. |

Ưu tiên kiểm lại các đoạn lớn ở mức **đoạn liên tục**, đối chiếu PKL/RespiBAN và kênh bối cảnh với các đoạn ổn định cùng subject và cùng lớp. Nếu có điều kiện, phép đo lặp với nhật ký độ bám đầu đo và đầu đo tham chiếu cùng vị trí sẽ giúp phân biệt hiệu ứng tiếp xúc với biến thiên da thực. Khi so các mẫu số, sử dụng Validation mô tả hiệu ứng theo từng subject/lớp, ghi cả tỷ lệ cửa sổ và timestamp riêng. Giảm đuôi đơn thuần chưa đủ để chọn phương án nếu phần lớn biến thiên nhỏ cũng bị nén. [Std và IQR tính từ Train](temperature_train_scale_context.csv).

## 5. Tệp bàn giao và phạm vi

- [Kiểm chứng nguồn, nhãn, mask và SHA-256](audit_verification.json) và [mã phân tích chỉ đọc](analyze_extremes.py).
- [Phân vị và tỷ lệ theo nhóm](zscore_statistics_by_group.csv), [theo cửa sổ](zscore_statistics_by_window.csv), [histogram trị tuyệt đối](absolute_z_histogram.csv) và [biến thiên theo subject/lớp](centered_window_variation_by_subject.csv).
- [Tín hiệu cửa sổ 64 Hz](s2_target_trace_64hz.csv), [zoom PKL và RespiBAN 700 Hz](s2_native_peak_zoom.csv), [các cửa sổ lân cận](s2_neighbor_windows.csv) và năm biểu đồ ở cả PNG/SVG. Các hình được liệt kê trong verification JSON.

Đã đối chiếu SHA‑256 trước và sau của **576 tệp đầu vào**, gồm dữ liệu đã đọc, baseline, window index, split, scaler và hiện vật có sẵn của group. Chỉ ghi đầu ra phân tích riêng. Dùng nguyên các scaler đã lưu. Không loại mẫu, clipping, huấn luyện, forward, backpropagation hay optimizer step. Test không tham gia thống kê hoặc đề xuất.

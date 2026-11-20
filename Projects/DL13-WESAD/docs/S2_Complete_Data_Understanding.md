# Hiểu toàn bộ dữ liệu WESAD của người tham gia S2

Thư mục S2 chứa **năm tệp** phục vụ các mục đích khác nhau. Hai tệp ghi tín hiệu từ thiết bị, hai tệp mô tả người tham gia và buổi thí nghiệm, còn `S2.pkl` tập hợp **10 loại tín hiệu theo thiết bị, 14 kênh số và một dãy nhãn theo thời gian**. Với bài toán phân loại Baseline, Stress và Amusement của DL13-WESAD, PKL là nguồn đầu vào chính vì nhóm tác giả đã căn chỉnh dữ liệu hai thiết bị và cung cấp nhãn 700 Hz cùng mốc thời gian. Các tệp gốc cho phép kiểm tra phép căn chỉnh, đơn vị và chất lượng ghi.

Các ký hiệu chỉ nguồn bằng chứng trong tài liệu

| Ký hiệu | Căn cứ |
|---|---|
| **[PDF]** | Nội dung bốn trang của [wesad_readme.pdf](../data/raw/WESAD/wesad_readme.pdf). Số mục như II.3 tham chiếu trực tiếp bản PDF này. |
| **[S2]** | Nội dung được đọc hoặc kết quả được tính trực tiếp từ năm tệp trong [thư mục S2](../data/raw/WESAD/S2/). |
| **[Bổ sung]** | Bài báo gốc WESAD, tài liệu hãng, tài liệu kỹ thuật hoặc nghiên cứu sinh lý, có liên kết gần nội dung liên quan. Các giải thích sinh lý mô tả khả năng diễn giải, chưa phải kết quả phân tích của S2. |
| **[Dự án]** | Quy ước triển khai được lưu trong tài liệu và mã của DL13-WESAD. |
| **[Chưa xác minh]** | Thông tin chưa đủ căn cứ hoặc suy luận từ phép đối chiếu, có nêu giới hạn tương ứng. |

## 1. Giới thiệu WESAD và người tham gia S2

**[PDF] [Bổ sung]** WESAD, viết đầy đủ là *Wearable Stress and Affect Detection*, là bộ dữ liệu đa tín hiệu về chuyển động và sinh lý thu trong một thí nghiệm có kiểm soát. Nhóm tác giả công bố dữ liệu của 15 trên 17 người tham gia. Mỗi thư mục `SX` gom bản ghi và tài liệu của một người. `S2` là mã người tham gia, và trong PKL trường `subject` chứa chuỗi `'S2'`. WESAD có thể dùng cho phân loại ba trạng thái, phân loại stress theo hai lớp, nghiên cứu kết hợp nhiều cảm biến và đánh giá khả năng tổng quát hóa sang người mới. [Hồ sơ WESAD tại UCI](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection) · [Schmidt và cộng sự, 2018](https://doi.org/10.1145/3242969.3242985)

**[PDF] [Bổ sung]** Người tham gia đeo RespiBAN Professional quanh ngực và Empatica E4 ở cổ tay không thuận. Các giai đoạn chính gồm Baseline, khi người tham gia có thể đọc tài liệu trung tính, Stress, Amusement khi xem video hài, và hai lượt thiền có hướng dẫn sau các giai đoạn kích thích. Giai đoạn Stress dùng *Trier Social Stress Test* (TSST), bài thử gây căng thẳng tâm lý xã hội bằng thuyết trình và tính nhẩm trước hội đồng. Nhãn giúp ghép từng khoảng tín hiệu với điều kiện thí nghiệm để huấn luyện và đánh giá mô hình. Bảng hỏi tự báo cáo bổ sung góc nhìn của người tham gia. [Bài báo WESAD, mục 3.2–3.4](https://www.eti.uni-siegen.de/ubicomp/papers/ubi_icmi2018.pdf)

| Thuật ngữ | Nghĩa trong S2 |
|---|---|
| Người tham gia (*subject*) | Một người, với một phiên ghi gồm nhiều giai đoạn. |
| Cảm biến (*sensor*) | Bộ phận thực hiện phép đo, chẳng hạn gia tốc kế trên vòng E4. Một thiết bị đeo có thể chứa nhiều cảm biến. |
| Tín hiệu (*signal*) | Chuỗi giá trị biến đổi theo thời gian, chẳng hạn gia tốc cổ tay hoặc điện tâm đồ. |
| Kênh (*channel*) | Một thành phần số tại mỗi lần đo. ACC có ba kênh tương ứng ba trục. |
| Mẫu đo (*sample*) | Kết quả của một lần lấy mẫu. Một mẫu ACC có ba giá trị, một mẫu ECG có một giá trị. Mẫu đầu vào của mô hình có thể là cả một cửa sổ gồm nhiều lần đo. |
| Nhãn (*label*) | Mã của điều kiện thí nghiệm tại một vị trí trên lưới thời gian nhãn. |
| Thời lượng bản ghi (*recording duration*) | Độ dài thời gian mà chuỗi mẫu bao phủ theo tần số danh định. Một bản ghi của cùng một người có thể có nhiều phiên bản với điểm đầu và cuối khác nhau. |

## 2. Cấu trúc toàn bộ thư mục S2

**[S2]** Trong dự án, `data/` có các thư mục `raw/`, `interim/` và `processed/`. Phần triển khai ở `src/`, `scripts/`, `configs/` và `tests/`. Các tài liệu thực nghiệm nằm dưới `experiments/` và `results/`. Bản dữ liệu WESAD tại máy được tìm thấy ở `data/raw/WESAD/`, với PDF nằm cùng cấp thư mục con `S2/`. Kiểm kê đệ quy của S2 cho đúng năm tệp, không có thư mục con. [README của dự án](../README.md)

```text
data/raw/WESAD/
├── wesad_readme.pdf
└── S2/
    ├── S2_readme.txt
    ├── S2_quest.csv
    ├── S2_respiban.txt
    ├── S2_E4_Data.zip
    └── S2.pkl
```

| Tệp | Định dạng | Kích thước (byte) | Nguồn dữ liệu | Nội dung chính | Mục đích và vai trò nghiên cứu | Mức cần thiết cho pipeline hiện tại |
|---|---|---:|---|---|---|---|
| [S2_readme.txt](../data/raw/WESAD/S2/S2_readme.txt) | Văn bản | 481 | Hồ sơ người tham gia và ghi chú của nhóm thu thập | Đặc điểm S2, điều kiện trước phiên, ghi chú nhiệt độ ngực | Diễn giải điều kiện thu và vấn đề tiếp xúc của cảm biến | Tham chiếu, đặc biệt khi nghiên cứu `chest.Temp` |
| [S2_quest.csv](../data/raw/WESAD/S2/S2_quest.csv) | Bảng văn bản phân tách bằng `;` | 1.064 | Lịch của nhóm thu thập và câu trả lời của S2 | Thứ tự, mốc bắt đầu và kết thúc, điểm từng mục tự báo cáo | Kiểm chứng nhãn theo lịch, đọc trải nghiệm tự báo cáo | Tham chiếu cho bài toán hiện tại, có thể làm nguồn cho nghiên cứu tự báo cáo |
| [S2_respiban.txt](../data/raw/WESAD/S2/S2_respiban.txt) | Văn bản OpenSignals phân tách bằng tab | 261.272.944 | RespiBAN Professional | Header và tám kênh ghi ở 700 Hz | Kiểm tra cột, mã số hóa, đổi đơn vị và đối chiếu tín hiệu ngực | Tùy mục tiêu, hữu ích khi kiểm toán dữ liệu ngực |
| [S2_E4_Data.zip](../data/raw/WESAD/S2/S2_E4_Data.zip) | Tệp nén ZIP | 1.837.606 | Bản xuất Empatica E4 | Bốn tín hiệu gốc, HR, IBI, tệp sự kiện và mô tả | Kiểm tra tần số, đơn vị, gốc thời gian và các đoạn cắt | Tùy mục tiêu, hữu ích khi kiểm toán dữ liệu cổ tay |
| [S2.pkl](../data/raw/WESAD/S2/S2.pkl) | Pickle của Python | 975.117.737 | Bản phát hành đã đồng bộ của WESAD | Tín hiệu hai thiết bị, mã người tham gia và nhãn | Nguồn truy xuất mảng theo cảm biến và căn chỉnh nhãn | Nguồn chính |

Các tệp bổ sung cho nhau. PKL có phần nội dung trùng với hai bản ghi thiết bị, vì nó chứa dữ liệu trên khoảng đã căn chỉnh. Hai bản ghi gốc còn giữ dữ liệu ngoài khoảng đó và metadata xuất tệp. README và CSV lưu bối cảnh, lịch và tự báo cáo, những trường không xuất hiện trong dictionary PKL. Phép đối chiếu cụ thể nằm ở chương 5, 6 và 13. [wesad_readme.pdf, mục I.1 và II](../data/raw/WESAD/wesad_readme.pdf)

## 3. S2_readme.txt

**[S2]** Toàn bộ tệp gồm ba nhóm thông tin.

- **Thông tin cá nhân.** S2 là nam, 27 tuổi, cao 175 cm, nặng 80 kg và thuận tay phải. Theo quy tắc đeo E4 ở tay không thuận, vị trí E4 của S2 được suy ra là cổ tay trái. Tệp S2 không ghi trực tiếp phía đeo.
- **Điều kiện trước nghiên cứu.** Cả sáu câu trả lời là `NO`. S2 khai không uống cà phê trong ngày hoặc trong một giờ trước đó, không chơi thể thao trong ngày, không hút thuốc thường xuyên hoặc trong một giờ trước đó và không cảm thấy ốm.
- **Ghi chú thu thập.** Tác giả ghi cảm biến nhiệt độ RespiBAN không được gắn hoàn toàn chắc trong suốt giao thức. Tệp không cho biết thời điểm, mức độ hay bao nhiêu mẫu bị ảnh hưởng. Ghi chú chỉ đích danh nhiệt độ RespiBAN. [S2_readme.txt](../data/raw/WESAD/S2/S2_readme.txt)

Ghi chú nhiệt độ cung cấp một khả năng để giải thích biến động của `chest.Temp`, cần đối chiếu theo thời gian trước khi quyết định dùng kênh này. Sáu câu trả lời và thông tin cá nhân cung cấp bối cảnh tại cấp người tham gia. Đầu vào tuần tự của CNN hoặc GRU lấy từ các mảng số theo thời gian. Tài liệu dự án hiện dùng các kênh cổ tay và không đưa những trường hồ sơ trên vào tensor mô hình. [README của dự án, mục 2](../README.md)

## 4. S2_quest.csv

**[S2]** CSV lưu các trường dưới dạng văn bản. S2 dùng dấu chấm phẩy làm dấu phân tách, dù phần mở rộng là `.csv`. Có 24 hàng vật lý, mỗi hàng tách được 27 trường, gồm cột đầu dùng cho tên loại hàng và các cột để ghi giá trị. Bốn hàng chỉ có các trường trống để ngăn nhóm. Các dấu `;` cuối hàng tạo ô đệm trống. Ý nghĩa của một cột phụ thuộc nhóm hàng, nên không có một bộ tên cột đơn nhất áp dụng cho toàn tệp. [S2_quest.csv](../data/raw/WESAD/S2/S2_quest.csv)

| Hàng tệp | Trường đầu | Cách đọc | Ví dụ trích từ tệp |
|---|---|---|---|
| 1 | `# Subj` | Mã người tham gia ở trường kế tiếp | `S2` |
| 2–4 | `# ORDER`, `# START`, `# END` | Mỗi cột là một hoạt động với hai mốc, theo đồng hồ RespiBAN | `Base`, `7.08`, `26.32` tương ứng 7 phút 08 giây đến 26 phút 32 giây |
| 6–10 | `# PANAS` | 24 câu trả lời ở bốn lượt và 26 ở lượt TSST | Lượt Base bắt đầu bằng `1;1;3;2`. Lượt TSST bắt đầu bằng `3;2;4;1`. |
| 12–16 | `# STAI` | Sáu câu trả lời mỗi lượt | Lượt TSST có `1;3;2;1;3;1`. |
| 18–22 | `# DIM` | Hai chiều tự đánh giá, lần lượt là sắc thái cảm xúc và mức độ kích hoạt | `7;2` sau Base và `5;4` sau TSST |
| 24 | `# SSSQ` | Sáu câu bổ sung chỉ sau TSST | `5;5;4;4;3;5` |

**[PDF]** Các bảng hỏi được điền sau từng giai đoạn chính theo thứ tự `Base → TSST → Medi 1 → Fun → Medi 2` trong S2. Hai hoạt động đọc `sRead`, `fRead` có cột lịch riêng. PDF yêu cầu bỏ qua chúng trong bài toán của bộ dữ liệu và không liệt kê bảng tự báo cáo riêng cho chúng. Các thang dưới đây là điểm trên **từng mục**.

| Bảng hỏi | Nội dung và thang trả lời được mô tả |
|---|---|
| PANAS (*Positive and Negative Affect Schedule*) | Cảm xúc tích cực và tiêu cực. 20 mục gốc, cộng bốn mục về căng thẳng, bực bội, vui và buồn, dùng thang 1–5 từ không hề đến cực kỳ. TSST có thêm hai mục về tức giận và khó chịu, phục vụ SSSQ cùng một mục đã có sẵn trong PANAS. |
| STAI rút gọn (*State-Trait Anxiety Inventory*) | Sáu mục về lo âu trạng thái, như thấy lo lắng hoặc thư giãn. Thang 1–4 từ không hề đến rất nhiều. |
| SAM (*Self-Assessment Manikins*), ghi là `DIM` trong CSV | Hai đánh giá riêng về sắc thái dễ chịu và độ kích hoạt, mỗi đánh giá từ 1 đến 9, mức thấp đến mức cao. |
| SSSQ rút gọn (*Short Stress State Questionnaire*) | Các mục về sự tham gia vào nhiệm vụ, lo lắng và khó chịu sau TSST. Sáu mục riêng ở hàng cuối, ba mục còn lại lấy cùng PANAS theo PDF. Điểm thang 1–5. |

Ví dụ, mục 21 của PANAS là mức tự báo cáo về căng thẳng. Giá trị là `2` sau Base và `3` sau TSST. Các giá trị diễn tả hai câu trả lời hồi tưởng theo giai đoạn, còn `label` trong PKL ghi mã điều kiện thí nghiệm tại 700 Hz. Cùng một người có thể báo cáo cảm giác khác với mục tiêu của điều kiện được gán. Muốn dùng bảng hỏi làm đích dự đoán cần xác định đơn vị phân tích và quy tắc chấm điểm riêng. Đưa các câu trả lời có sau giai đoạn vào đầu vào của bài toán phân loại tín hiệu hiện tại sẽ đổi thông tin mà mô hình có thể sử dụng. [wesad_readme.pdf, mục III](../data/raw/WESAD/wesad_readme.pdf) · [Bài báo WESAD, mục 3.4](https://www.eti.uni-siegen.de/ubicomp/papers/ubi_icmi2018.pdf)

## 5. S2_respiban.txt

**[PDF] [S2]** Đây là dữ liệu gốc từ RespiBAN Professional. Tệp bắt đầu bằng `# OpenSignals Text File Format`, kế tiếp là một dòng metadata JSON đặt sau `#`, rồi `# EndOfHeader`. Metadata ghi thiết bị `RespiBan`, ngày `2017-5-22`, giờ `9:39:1.0` theo header, tám kênh có độ phân giải 16 bit và tần số 700 Hz. Tệp không ghi múi giờ cạnh trường giờ. Các dòng mẫu sau header dùng tab và có một tab cuối dòng. Có 10 trường dữ liệu có nghĩa, trong đó hai trường đầu là số thứ tự `nSeq` và trạng thái vào/ra số `DI`. PDF chỉ dẫn bỏ qua `DI` khi đọc tín hiệu WESAD. Đếm được **4.442.067 dòng mẫu**, tương ứng **6.345,81 giây** trên lưới 700 Hz. [wesad_readme.pdf, mục II.1](../data/raw/WESAD/wesad_readme.pdf) · [Đặc tả TXT OpenSignals của PLUX](https://support.pluxbiosignals.com/knowledge-base/opensignals-sensor-file-specifications-txt-format/)

| Vị trí cột | Tên trong header | Tín hiệu hoặc trường | Ba giá trị đầu |
|---:|---|---|---|
| 1 | `nSeq` | Số thứ tự mẫu | 0, 1, 2 |
| 2 | `DI` | Trạng thái vào/ra số | 0, 0, 0 |
| 3 | `CH1` | ECG | 31053, 30861, 30644 |
| 4 | `CH2` | EDA | 14694, 14680, 14684 |
| 5 | `CH3` | EMG | 32651, 32983, 32823 |
| 6 | `CH4` | Nhiệt độ | 29553, 29556, 29561 |
| 7 | `CH5` | ACC, thành phần thứ nhất của nhóm `XYZ` | 37415, 37417, 37409 |
| 8 | `CH6` | ACC, thành phần thứ hai | 32324, 32333, 32340 |
| 9 | `CH7` | ACC, thành phần thứ ba | 31659, 31663, 31655 |
| 10 | `CH8` | Hô hấp | 31364, 31379, 31367 |

**[PDF]** Các số nguyên gốc là mã số hóa. Công thức chuyển đơn vị của WESAD dùng `signal` cho mã gốc, `vcc = 3`, `chan_bit = 65536`. ECG và EMG dùng `(signal / chan_bit - 0.5) × vcc` để ra mV. EDA dùng `(signal / chan_bit) × vcc / 0.12` để ra µS. ACC dùng `(signal - 28000) / (38000 - 28000) × 2 - 1` để ra đơn vị `g`. Hô hấp dùng `(signal / chan_bit - 0.5) × 100` để ra phần trăm trên thang quy đổi của cảm biến. Nhiệt độ dùng công thức điện áp và điện trở nhiệt trong PDF trang 2 để ra °C. [wesad_readme.pdf, mục II.1, trang 1–2](../data/raw/WESAD/wesad_readme.pdf)

**[S2]** Hàng gốc `nSeq = 92017` có ECG `33236`, ra `0.02142333984375` mV theo công thức, đúng bằng `chest.ECG[0, 0]`. ACC ở hàng đó là `[37777, 31890, 30210]`, cho xấp xỉ `[0.9554, -0.2220, -0.5580]` g. Cùng phép đối chiếu khớp các mã gốc suy ngược của ECG, EDA, EMG, ba trục ACC và Resp tại 11 vị trí đầu, giữa và cuối PKL. Với Temp, sai số tuyệt đối tại các điểm kiểm tra tối đa khoảng `0.000031 °C`. Kết quả này hỗ trợ cách đọc đơn vị vật lý của `chest` trong PKL. Mốc bắt đầu tại hàng `92017` được tìm bằng dấu vết bảy kênh và kiểm chứng ở các vị trí sau. Phần ngực trong PKL có 4.255.300 mẫu mỗi kênh, tương ứng với khoảng quan sát `[92017, 4347317)` của dòng mẫu gốc, với 92.017 dòng trước và 94.750 dòng sau khoảng đó. Phép kiểm tra này lấy mẫu đối chiếu tại 11 vị trí và chưa tái lập toàn bộ quy trình tạo PKL.

## 6. S2_E4_Data.zip

**[S2]** Kho nén chứa tám tệp ở cấp gốc. Các tệp được đọc trực tiếp từ ZIP. Tất cả mục đều qua kiểm tra CRC, và `tags.csv` có kích thước 0 byte. Empatica E4 là vòng đeo có gia tốc kế, cảm biến quang đo mạch, điện cực đo độ dẫn da và cảm biến nhiệt độ da. **[PDF]** Bốn chuỗi đo chính được lưu ở tần số gốc. WESAD yêu cầu bỏ qua HR, IBI và tags trong tập tín hiệu chuẩn. [wesad_readme.pdf, mục II.2](../data/raw/WESAD/wesad_readme.pdf) · [Mô tả định dạng của Empatica](https://www.empatica.com/blog/decoding-wearable-sensor-signals-what-to-expect-from-your-e4-data/) · [Thông số kỹ thuật E4](https://box.empatica.com/documentation/20141119_E4_TechSpecs.pdf)

| Tệp con | Byte khi giải nén | Tổ chức và ý nghĩa | Tần số hoặc cách ghi | Đơn vị có căn cứ | Số dòng dữ liệu và ví dụ đầu |
|---|---:|---|---|---|---|
| `ACC.csv` | 2.329.669 | Hai dòng đầu là thời điểm bắt đầu và tần số, lặp trên ba cột. Các dòng sau lần lượt là X, Y, Z. | 32 Hz | Mỗi đơn vị số bằng `1/64 g` | 251.970, `29,-6,55` rồi `28,-6,55` |
| `BVP.csv` | 3.166.493 | Hai dòng đầu là mốc bắt đầu và tần số. Sau đó là một giá trị sóng mạch quang mỗi dòng. | 64 Hz | Chưa có đơn vị vật lý được xác nhận cho tệp WESAD này | 503.943, các giá trị đầu `-0.00` |
| `EDA.csv` | 283.473 | Hai dòng đầu là mốc bắt đầu và tần số. Sau đó là một giá trị độ dẫn điện da mỗi dòng. | 4 Hz | µS | 31.494, `0.000000`, `0.328021`, `0.410026` |
| `TEMP.csv` | 189.007 | Hai dòng đầu là mốc bắt đầu và tần số. Sau đó là một giá trị nhiệt độ mỗi dòng. | 4 Hz | °C theo định dạng | 31.496, bốn mẫu đầu `382.18`, kế tiếp `35.05` |
| `IBI.csv` | 72.018 | Dòng đầu `1495437325.000000, IBI`. Mỗi dòng sau chứa thời điểm phát hiện tính từ mốc đầu và độ dài khoảng giữa hai nhịp, suy ra từ BVP. | Theo sự kiện, không có hàng tần số | Giây cho cả hai cột | 3.458, `14.313155,0.765660` rồi `15.203821,0.890666` |
| `HR.csv` | 47.501 | Mốc đầu `1495437335.000000`, sau đó là dòng `1.000000` và chuỗi nhịp tim trung bình suy ra từ BVP. | 1 Hz theo header, hãng mô tả giá trị tính trên khoảng 10 giây | Nhịp/phút | 7.865, `101.00`, `106.50`, `95.33` |
| `info.txt` | 1.544 | Tài liệu đi kèm, mô tả cấu trúc dòng, đơn vị và ý nghĩa các tệp con. Hướng dẫn hai dòng chung có ngoại lệ được ghi riêng cho IBI và tags. | Không áp dụng | Không áp dụng | Ví dụ, ghi rõ thứ tự trục X, Y, Z của ACC |
| `tags.csv` | 0 | Theo `info.txt`, mỗi dòng nếu có sẽ ghi lúc nhấn nút đánh dấu trên thiết bị, theo Unix UTC. | Theo sự kiện | Giây Unix UTC | 0, tệp rỗng |

**[S2]** Bốn tệp đo chính có mốc chung `1495437325` giây Unix UTC, tức 07:15:25 UTC ngày 22/05/2017. Mốc HR muộn hơn 10 giây. Các mốc đầu của tệp gốc không được lưu dưới dạng trường riêng trong PKL. Bốn mẫu `382.18` ở đầu `TEMP.csv` vượt phạm vi đo nhiệt độ da được hãng công bố và nằm ngoài đoạn cổ tay được đưa vào PKL. Ý nghĩa và nguyên nhân của giá trị này chưa được xác minh. Ghi chú trong README áp dụng cho cảm biến RespiBAN ở ngực. [Thông số kỹ thuật E4](https://box.empatica.com/documentation/20141119_E4_TechSpecs.pdf)

**[S2]** Bỏ hai dòng đầu khi **đọc để đối chiếu**, mỗi mảng `wrist` trong PKL bằng chính xác đoạn liên tục dưới đây trong tệp tương ứng, từng giá trị và từng trục. Chỉ số đếm từ 0 trên phần mẫu đo trong CSV.

| Mảng PKL | Đoạn trong CSV gốc `[đầu, cuối)` | Mẫu trước đoạn | Mẫu sau đoạn | Giá trị ở đầu đoạn |
|---|---|---:|---:|---|
| `wrist.ACC` | `[49447, 243975)` | 49.447 | 7.995 | `[62, -21, 107]` |
| `wrist.BVP` | `[98894, 487950)` | 98.894 | 15.993 | `-59.37` |
| `wrist.EDA` | `[6180, 30496)` | 6.180 | 998 | `1.138257` |
| `wrist.TEMP` | `[6180, 30496)` | 6.180 | 1.000 | `35.41` |

**[Bổ sung]** Một lần lấy mẫu là một lần ghi giá trị trên lưới thời gian của cảm biến. Hãng chọn tần số theo nội dung biến đổi mà từng cảm biến cần ghi. BVP ghi hình dạng sóng mạch thường biến đổi nhanh hơn độ dẫn da, nên 64 Hz cho 16 lần lấy mẫu BVP trong mỗi khoảng có một lần lấy mẫu EDA ở 4 Hz. Ba trục ACC ghi ba thành phần vuông góc tại cùng lần lấy mẫu, vì chuyển động và hướng đặt cảm biến thay đổi trong không gian. [Tài liệu cảm biến E4 của Empatica](https://www.empatica.com/blog/decoding-wearable-sensor-signals-what-to-expect-from-your-e4-data/)

## 7. So sánh RespiBAN và Empatica E4

| Tiêu chí | RespiBAN Professional | Empatica E4 |
|---|---|---|
| Vị trí | Bộ ghi quanh ngực, các đầu đo gắn tại những vị trí cụ thể trên thân trên | Cổ tay không thuận |
| Loại tín hiệu trong PKL | ACC, ECG, EMG, EDA, Temp, Resp | ACC, BVP, EDA, TEMP |
| Số loại và số kênh | 6 loại, 8 kênh | 4 loại, 6 kênh |
| Tần số trong PKL | 700 Hz cho mọi kênh ngực | ACC 32 Hz, BVP 64 Hz, EDA và TEMP 4 Hz |
| Đại lượng ghi | Chuyển động thân, điện tim, điện cơ, độ dẫn da, nhiệt độ da và dao động hô hấp | Chuyển động cổ tay, dao động mạch ngoại vi bằng quang học, độ dẫn da và nhiệt độ da |
| Thông tin riêng có | ECG, EMG và Resp cho ba góc nhìn bổ sung về tim, cơ và hô hấp | BVP ghi dao động tưới máu ngoại vi, ACC ghi chuyển động cục bộ của tay |
| Giới hạn có căn cứ | Chất lượng phụ thuộc vị trí và độ tiếp xúc của các đầu đo. S2 có ghi chú về độ bám của cảm biến Temp. | Chuyển động cổ tay và độ chặt vòng có thể ảnh hưởng phép đo quang và tiếp xúc da. Chuỗi cổ tay gốc cần căn chỉnh với đồng hồ RespiBAN. |

**[PDF] [Bổ sung]** `chest` là nhóm dữ liệu của thiết bị đeo ngực. Một số đầu đo thuộc nhóm này được gắn ở bụng hoặc vai. Bảng so sánh cho thấy thông tin mỗi thiết bị cung cấp trong bố trí WESAD; khảo sát S2 chưa đo hiệu năng tương đối của các thiết bị trong phân loại. [Bài báo WESAD, mục 3.2 và hình 1](https://www.eti.uni-siegen.de/ubicomp/papers/ubi_icmi2018.pdf) · [Thông tin ảnh hưởng chuyển động và tiếp xúc của E4](https://www.empatica.com/blog/decoding-wearable-sensor-signals-what-to-expect-from-your-e4-data/)

## 8. Mười loại tín hiệu và ý nghĩa sinh lý

**[S2]** Bảng ghi hai mẫu đầu theo thời gian, làm tròn nếu cần. Đơn vị ngực dựa trên công thức PDF và phép đối chiếu ở chương 5. Đơn vị cổ tay dựa trên PDF, `info.txt` và các đoạn trùng khớp đã xác minh ở chương 6. Trong bảng, `g` là đơn vị gia tốc lấy theo gia tốc trọng trường chuẩn, mV là milivôn và µS là microsiemens.

| Tín hiệu | Tên đầy đủ và vị trí đo | Đại lượng và đơn vị đọc trong PKL | Hz | Shape | Hai mẫu đầu |
|---|---|---|---:|---|---|
| `chest.ACC` | *Acceleration*, gia tốc kế trên bộ RespiBAN ở ngực | Ba thành phần gia tốc, `g` | 700 | `(4255300, 3)` | `[0.955400, -0.222000, -0.558000]`, `[0.925800, -0.221600, -0.553800]` |
| `chest.ECG` | *Electrocardiogram*, điện tâm đồ từ bố trí ba điểm | Hiệu điện thế bề mặt liên quan hoạt động điện của tim, mV | 700 | `(4255300, 1)` | `0.021423`, `0.020325` |
| `chest.EMG` | *Electromyogram*, điện cơ đo ở vùng cơ thang trên hai bên cột sống | Hiệu điện thế bề mặt liên quan hoạt động cơ, mV | 700 | `(4255300, 1)` | `-0.004440`, `0.004349` |
| `chest.EDA` | *Electrodermal Activity*, hoạt động điện da đo tại vùng cơ thẳng bụng | Độ dẫn điện da, µS | 700 | `(4255300, 1)` | `5.250549`, `5.267334` |
| `chest.Temp` | *Skin Temperature*, nhiệt độ da tại xương ức | Nhiệt độ da tại vị trí tiếp xúc, °C | 700 | `(4255300, 1)` | `30.120758`, `30.129517` |
| `chest.Resp` | *Respiration*, hô hấp từ đai quanh thân | Biến thiên theo chu kỳ thở, % trên thang cảm biến | 700 | `(4255300, 1)` | `-1.148987`, `-1.124573` |
| `wrist.ACC` | *Acceleration*, gia tốc kế trên E4 | Ba thành phần gia tốc, mỗi đơn vị số bằng `1/64 g` | 32 | `(194528, 3)` | `[62, -21, 107]`, `[66, 13, 53]` |
| `wrist.BVP` | *Blood Volume Pulse*, sóng mạch thể tích máu cổ tay | Biên độ tín hiệu quang theo mạch, đơn vị vật lý chưa xác nhận | 64 | `(389056, 1)` | `-59.37`, `-53.42` |
| `wrist.EDA` | *Electrodermal Activity*, hoạt động điện da dưới điện cực cổ tay | Độ dẫn điện da, µS | 4 | `(24316, 1)` | `1.138257`, `1.125444` |
| `wrist.TEMP` | *Skin Temperature*, nhiệt độ da cổ tay | Nhiệt độ da ngoại vi, °C | 4 | `(24316, 1)` | `35.41`, `35.41` |

**[Bổ sung] ACC ngực và ACC cổ tay.** Vận tốc mô tả tốc độ và hướng di chuyển. Gia tốc mô tả mức thay đổi của vận tốc theo thời gian. Gia tốc kế đeo trên người phản ứng cả với chuyển động và thành phần tĩnh biểu kiến do trọng lực. Khi đứng yên, đổi hướng thiết bị làm thay đổi cách trọng lực chiếu lên ba trục X, Y, Z của thiết bị. Một mẫu gồm ba giá trị đồng thời. Trục của bộ đeo ngực và trục của vòng E4 có hệ tọa độ riêng. Ngực phản ánh chuyển động và tư thế thân, cổ tay nhạy với xoay tay và cử chỉ. ACC giúp nhận ra bối cảnh vận động hoặc tìm khả năng nhiễu ở BVP. Tư thế, gắn thiết bị, đọc, nói và cử động tay đều có thể ảnh hưởng gia tốc. ACC chỉ hỗ trợ diễn giải bối cảnh của stress, vì người tham gia có thể cử động trong bất kỳ giai đoạn nào. [Bài báo WESAD, mục 3.2 và 4](https://www.eti.uni-siegen.de/ubicomp/papers/ubi_icmi2018.pdf) · [Thông số gia tốc kế E4](https://www.empatica.com/blog/decoding-wearable-sensor-signals-what-to-expect-from-your-e4-data/)

**[Bổ sung] ECG ngực.** Các điện cực đo chênh lệch điện thế do tim hoạt động. Ở một chu kỳ ECG điển hình, sóng P liên quan khử cực nhĩ, phức bộ QRS liên quan khử cực thất và sóng T liên quan tái cực thất. Điện thế từng mẫu là một điểm trên dạng sóng. Nhịp tim cần suy ra từ khoảng thời gian giữa những nhịp được nhận diện, thường dùng các đỉnh R. Thay đổi nhịp và biến thiên khoảng nhịp có thể cung cấp thông tin về đáp ứng tự chủ trong giai đoạn stress. Hơi thở, vận động, tư thế và tiếp xúc điện cực cũng ảnh hưởng quan sát. Một điện thế riêng lẻ không cho biết nhịp/phút hay trạng thái tâm lý. [Tài liệu cảm biến ECG của PLUX](https://support.pluxbiosignals.com/knowledge-base/biosignalsplux-documentation/) · [Bài báo WESAD, mục 4](https://www.eti.uni-siegen.de/ubicomp/papers/ubi_icmi2018.pdf)

**[Bổ sung] EMG ngực.** Điện cực bề mặt ghi hoạt động điện tổng hợp khi các đơn vị vận động cơ được huy động. Dạng sóng dao động nhanh quanh mức nền, có thể xuất hiện những cụm biên độ lớn hơn khi cơ hoạt động. Độ lớn còn phụ thuộc lực co, tư thế, vị trí điện cực, chuyển động và hoạt động của cơ lân cận. Căng cơ vai có thể đi cùng stress, đồng thời cũng xảy ra do giữ tư thế, nhấc tay hoặc vận động. Dấu của một mẫu EMG thể hiện cực tính tại điểm đo, còn diễn giải độ hoạt hóa cơ cần xem nhiều mẫu. [Tài liệu cảm biến EMG của PLUX](https://support.pluxbiosignals.com/knowledge-base/biosignalsplux-documentation/) · [Bài báo WESAD, mục 3.2 và 4](https://www.eti.uni-siegen.de/ubicomp/papers/ubi_icmi2018.pdf)

**[Bổ sung] EDA ngực và EDA cổ tay.** Các điện cực đo độ dẫn điện qua vùng da tiếp xúc. Tuyến mồ hôi ngoại tiết do hệ thần kinh giao cảm chi phối. Khi độ ẩm ở da thay đổi, độ dẫn có thể đổi theo. Chuỗi thường có thành phần nền thay đổi chậm và những đáp ứng ngắn hơn. EDA có thể tăng khi mức kích hoạt sinh lý tăng, kể cả do căng thẳng, hào hứng, bất ngờ hoặc nóng. Vị trí vùng bụng và cổ tay khác nhau về da, tiếp xúc và môi trường cục bộ, đồng thời được lấy mẫu ở tần số khác nhau. Cùng đơn vị µS cho biết cùng đại lượng vật lý. Biên độ tuyệt đối giữa hai nơi cần được diễn giải với bố trí đo, mồ hôi, nhiệt độ, tư thế và chuyển động. Hai tín hiệu có thể bổ sung cho nghiên cứu về mức kích hoạt, nhưng một giá trị EDA riêng lẻ không xác định nguyên nhân kích hoạt hay cảm xúc cụ thể. [Nghiên cứu EDA theo vị trí cơ thể](https://pmc.ncbi.nlm.nih.gov/articles/PMC11946426/) · [Tài liệu EDA của Empatica](https://www.empatica.com/blog/decoding-wearable-sensor-signals-what-to-expect-from-your-e4-data/)

**[Bổ sung] Temp ngực và TEMP cổ tay.** Công thức RespiBAN dùng điện trở nhiệt để xác định nhiệt độ da tại xương ức. Thông số của E4 mô tả đầu đo nhiệt hồng ngoại hướng vào da cổ tay. Nhiệt độ da thường biến thiên chậm hơn sóng tim. Tưới máu ngoại vi, nhiệt độ phòng, quần áo và cách đeo cảm biến đều có thể ảnh hưởng. Co mạch trong một số đáp ứng stress có thể làm da ở vị trí ngoại vi lạnh hơn. Hướng và mức thay đổi phụ thuộc vị trí, nên số đo xương ức và cổ tay có thể diễn biến khác nhau. Giá trị nhiệt độ cục bộ đơn lẻ chưa cho biết thân nhiệt lõi hay mức stress. Với S2, cần tính đến ghi chú tiếp xúc của đầu đo xương ức và các mẫu đầu bất thường trong tệp E4 gốc. [Nghiên cứu thực nghiệm nhiệt độ theo vị trí trong stress cấp](https://pubmed.ncbi.nlm.nih.gov/23790072/) · [wesad_readme.pdf, mục II](../data/raw/WESAD/wesad_readme.pdf) · [Thông số kỹ thuật E4](https://box.empatica.com/documentation/20141119_E4_TechSpecs.pdf)

**[Bổ sung] Resp ngực.** Khi hít vào và thở ra, chu vi vùng đeo đai thay đổi, tạo dao động. Khoảng thời gian giữa các chu kỳ có thể cho biết nhịp thở, còn biên độ biểu thị độ thay đổi tương đối trên thang cảm biến. Bài báo mô tả phép đo bằng đai cảm ứng (*respiratory inductance plethysmography*). PDF dẫn datasheet cảm biến hô hấp áp điện bên cạnh công thức đổi sang phần trăm, nên tài liệu hiện có chưa đủ để xác minh mã phần cứng chính xác của S2. Hơi thở có thể đổi nhịp theo stress, nói, cử động hoặc thiền có hướng dẫn. Mẫu có dấu âm vẫn có thể là giá trị hợp lệ trên thang quy đổi. Chưa có hiệu chuẩn cho thể tích phổi tính bằng lít hoặc căn cứ để gán hít vào, thở ra từ dấu của một mẫu. [Bài báo WESAD, mục 3.2–3.3](https://www.eti.uni-siegen.de/ubicomp/papers/ubi_icmi2018.pdf) · [wesad_readme.pdf, mục II.1](../data/raw/WESAD/wesad_readme.pdf)

**[Bổ sung] BVP cổ tay.** Quang thể tích ký (*photoplethysmography*, PPG) chiếu ánh sáng vào mô và phát hiện thay đổi tín hiệu quang khi lượng máu ngoại vi dao động theo nhịp mạch. BVP biểu diễn sóng mạch đến cổ tay, trong khi ECG ghi sự kiện điện của tim. Với dạng sóng đủ rõ, khoảng giữa các nhịp mạch có thể dùng để suy ra nhịp và độ biến thiên. Cử động tay, vòng lỏng, áp lực tiếp xúc và tưới máu ngoại vi đều ảnh hưởng tín hiệu. Stress có thể đi cùng thay đổi nhịp hoặc tưới máu, nhưng dấu và biên độ của một điểm BVP chưa nói được lượng máu tuyệt đối, nhịp tim hay mức stress. Thông số phần cứng có độ phân giải đầu ra quang. PDF và `info.txt` chưa cho cách quy đổi từng giá trị trong bản CSV WESAD của S2 sang đơn vị vật lý. [Thông số kỹ thuật E4](https://box.empatica.com/documentation/20141119_E4_TechSpecs.pdf) · [Tài liệu BVP của Empatica](https://www.empatica.com/blog/decoding-wearable-sensor-signals-what-to-expect-from-your-e4-data/) · [Bài báo WESAD, mục 4](https://www.eti.uni-siegen.de/ubicomp/papers/ubi_icmi2018.pdf)

## 9. Cấu trúc kỹ thuật của S2.pkl

**[S2]** Đọc trực tiếp PKL cho một `dict` gốc gồm đúng ba khóa, theo thứ tự lưu là `signal`, `label`, `subject`. Các khóa phân biệt chữ hoa và chữ thường, ví dụ `chest["Temp"]` và `wrist["TEMP"]`.

```text
data (dict)
├── signal (dict)
│   ├── chest (dict)
│   │   ├── ACC  → ndarray (4255300, 3), float64
│   │   ├── ECG  → ndarray (4255300, 1), float64
│   │   ├── EMG  → ndarray (4255300, 1), float64
│   │   ├── EDA  → ndarray (4255300, 1), float64
│   │   ├── Temp → ndarray (4255300, 1), float32
│   │   └── Resp → ndarray (4255300, 1), float64
│   └── wrist (dict)
│       ├── ACC  → ndarray (194528, 3), float64
│       ├── BVP  → ndarray (389056, 1), float64
│       ├── EDA  → ndarray (24316, 1), float64
│       └── TEMP → ndarray (24316, 1), float64
├── label   → ndarray (4255300,), int32
└── subject → str 'S2'
```

| Khóa | Kiểu và ý nghĩa | Shape, số mẫu và số kênh | Giá trị minh họa |
|---|---|---|---|
| `signal` | `dict`, phân nhóm tín hiệu đo theo thiết bị | Hai dictionary con | `chest`, `wrist` |
| `signal.chest` | `dict` chứa sáu mảng RespiBAN | Mỗi mảng có 4.255.300 hàng. Tổng cộng tám kênh. | `chest["ECG"][0, 0] = 0.02142333984375` |
| `signal.wrist` | `dict` chứa bốn mảng E4 | Số hàng tùy tần số. Tổng cộng sáu kênh. | `wrist["ACC"][0] = [62.0, -21.0, 107.0]` |
| `label` | `numpy.ndarray`, mã điều kiện trên lưới 700 Hz | `(4255300,)`, `int32`, 4.255.300 giá trị đơn. Đây là dãy nhãn riêng với 14 kênh đo. | Tám phần tử đầu và tám phần tử cuối đều bằng `0`. |
| `subject` | `str`, mã một người tham gia | Một giá trị văn bản, không áp dụng shape, dtype NumPy, số kênh hay tần số | `'S2'` |

Bảng ở chương 8 ghi hai mẫu thật của từng mảng. Chương 10 ghi số hàng, số kênh qua shape và dtype của từng tín hiệu. Với `(N, 3)`, mỗi hàng chứa ba giá trị cùng lần lấy mẫu. `(N, 1)` vẫn là mảng hai chiều với một cột. `(N,)` chỉ có một chiều, mỗi chỉ số cho một giá trị vô hướng.

**[Bổ sung]** *Tuần tự hóa* là biểu diễn đối tượng trong một luồng có thể lưu và đọc lại. Pickle là định dạng tuần tự hóa của Python. Một dictionary ánh xạ khóa sang giá trị, trong trường hợp này giá trị có thể là dictionary con, mảng hoặc chuỗi. Mảng NumPy lưu các phần tử cùng kiểu số. `shape` mô tả kích thước các chiều, còn `dtype` mô tả cách lưu từng phần tử. Một bảng phẳng có chung số hàng cho mọi cột, trong khi các mảng trong PKL có thể có số hàng khác nhau. Các bảng trong tài liệu là hình thức trình bày, còn cấu trúc lưu thực tế là dictionary lồng nhau. [Tài liệu Python về Pickle](https://docs.python.org/3/library/pickle.html)

**[S2] [Dự án]** Tệp được nạp từ bản WESAD cục bộ bằng Python với `encoding="latin1"` để tương thích cách lưu cũ, cùng thiết lập mã hóa với bộ đọc của dự án. Bộ đọc kiểm tra trong khảo sát giới hạn các đối tượng dùng để khôi phục mảng NumPy và đọc đến đúng cuối tệp. Chỉ giải tuần tự hóa pickle từ nguồn tin cậy vì thao tác nạp pickle nói chung có thể thực thi mã. [Bộ đọc của dự án](../src/data/inspection.py) · [Tài liệu Python về an toàn khi nạp Pickle](https://docs.python.org/3/library/pickle.html)

## 10. Thống kê mô tả từng kênh trong S2.pkl

**[S2]** Các thống kê được tính trên toàn bộ mảng lưu, gồm cả các giai đoạn chuyển tiếp. Shape là của mảng chứa kênh, còn số mẫu và các thống kê áp dụng cho từng kênh. Đơn vị theo chương 8. Các giá trị thống kê được làm tròn đến sáu chữ số thập phân.

| Thiết bị | Tín hiệu | Kênh | Shape của mảng | Dtype | Hz | Số mẫu | Thời lượng (s) | Nhỏ nhất | Lớn nhất | Trung bình | Trung vị | Độ lệch chuẩn |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ngực | `ACC` | X | `(4255300, 3)` | `float64` | 700 | 4.255.300 | 6.079 | 0,215000 | 2,029800 | 0,753866 | 0,856800 | 0,175116 |
| Ngực | `ACC` | Y | `(4255300, 3)` | `float64` | 700 | 4.255.300 | 6.079 | -0,660000 | 0,539000 | -0,133188 | -0,125600 | 0,064173 |
| Ngực | `ACC` | Z | `(4255300, 3)` | `float64` | 700 | 4.255.300 | 6.079 | -1,135400 | 1,246800 | -0,422681 | -0,322200 | 0,300236 |
| Ngực | `ECG` | 1 | `(4255300, 1)` | `float64` | 700 | 4.255.300 | 6.079 | -1,499542 | 1,499313 | 0,001212 | 0,008240 | 0,154077 |
| Ngực | `EMG` | 1 | `(4255300, 1)` | `float64` | 700 | 4.255.300 | 6.079 | -0,414963 | 0,300980 | -0,002413 | -0,001602 | 0,011652 |
| Ngực | `EDA` | 1 | `(4255300, 1)` | `float64` | 700 | 4.255.300 | 6.079 | 0,263214 | 7,576752 | 1,528387 | 1,163101 | 1,238425 |
| Ngực | `Temp` | 1 | `(4255300, 1)` | `float32` | 700 | 4.255.300 | 6.079 | 28,045258 | 34,370392 | 30,876676 | 31,289917 | 1,281105 |
| Ngực | `Resp` | 1 | `(4255300, 1)` | `float64` | 700 | 4.255.300 | 6.079 | -27,903748 | 27,378845 | 0,054906 | -0,157166 | 2,917749 |
| Cổ tay | `ACC` | X | `(194528, 3)` | `float64` | 32 | 194.528 | 6.079 | -128,000000 | 127,000000 | 39,701452 | 44,000000 | 19,520848 |
| Cổ tay | `ACC` | Y | `(194528, 3)` | `float64` | 32 | 194.528 | 6.079 | -128,000000 | 104,000000 | -0,407381 | 5,000000 | 33,206585 |
| Cổ tay | `ACC` | Z | `(194528, 3)` | `float64` | 32 | 194.528 | 6.079 | -128,000000 | 127,000000 | 18,990526 | 17,000000 | 25,003438 |
| Cổ tay | `BVP` | 1 | `(389056, 1)` | `float64` | 64 | 389.056 | 6.079 | -873,670000 | 988,080000 | -0,000427 | 3,010000 | 75,871236 |
| Cổ tay | `EDA` | 1 | `(24316, 1)` | `float64` | 4 | 24.316 | 6.079 | 0,045113 | 1,717419 | 0,391743 | 0,236032 | 0,329229 |
| Cổ tay | `TEMP` | 1 | `(24316, 1)` | `float64` | 4 | 24.316 | 6.079 | 32,310000 | 35,970000 | 34,386945 | 34,250000 | 1,045318 |

Nhỏ nhất và lớn nhất là hai cực trị quan sát được. Trung bình tổng hợp giá trị trên toàn chuỗi, trung vị là trung điểm theo thứ tự giá trị và ít nhạy hơn với cực trị. Độ lệch chuẩn mô tả độ phân tán quanh trung bình, tính với mẫu số `N` (`ddof=0`). Cột thời lượng dùng `N/f` với tần số danh định `f`. Tất cả 14 kênh có giá trị số hữu hạn. Việc không gặp NaN hoặc ∞ chỉ xác nhận tính hữu hạn của số, còn độ tin cậy của phép đo cần kiểm tra riêng.

**[S2]** Một số trục ACC cổ tay chạm `-128` hoặc `127`, sát giới hạn mã xuất E4. Trục X ngực có cực đại khoảng `2.029800 g`, vượt mốc `+1 g` của khoảng tham chiếu dùng trong công thức tuyến tính của PDF. Những điểm này, cùng ghi chú Temp ngực và giá trị đầu TEMP gốc cổ tay, là câu hỏi cho EDA tiếp theo. Cực trị chưa cho biết nguyên nhân, tần suất theo giai đoạn hay sự cần thiết phải loại mẫu. [wesad_readme.pdf, mục II.1–II.2](../data/raw/WESAD/wesad_readme.pdf) · [Phạm vi gia tốc E4 theo Empatica](https://www.empatica.com/blog/decoding-wearable-sensor-signals-what-to-expect-from-your-e4-data/)

## 11. Tần số lấy mẫu và thời gian

Tần số lấy mẫu là số lần ghi trong một giây, đo bằng hertz (Hz). Tần số 700 Hz cho 700 lần ghi mỗi giây, cách danh định khoảng 1,429 ms. Với 32 Hz khoảng cách là 31,25 ms, với 64 Hz là 15,625 ms và với 4 Hz là 250 ms. Ba thành phần của một mẫu ACC được ghi cùng lần lấy mẫu, nên ACC 32 Hz có 32 hàng ba giá trị mỗi giây.

```text
thời lượng (giây) = số mẫu / tần số lấy mẫu (Hz)
thời điểm tương đối của mẫu i (giây) = i / tần số lấy mẫu (Hz)
```

Mẫu đầu có chỉ số `0`. Mẫu cuối có chỉ số `N - 1`, nên thời điểm của mẫu cuối là `(N - 1)/f`. Thời lượng `N/f` là cận phải của khoảng nửa mở khi tính cả khoảng mẫu cuối.

| Nhóm trong PKL | Tần số (Hz) | Số mẫu mỗi tín hiệu hoặc nhãn | Thời lượng (giây) | Thời lượng (phút) |
|---|---:|---:|---:|---:|
| `chest.ACC`, `ECG`, `EMG`, `EDA`, `Temp`, `Resp` và `label` | 700 | 4.255.300 | 6.079 | 101,316667 |
| `wrist.ACC` | 32 | 194.528 | 6.079 | 101,316667 |
| `wrist.BVP` | 64 | 389.056 | 6.079 | 101,316667 |
| `wrist.EDA`, `wrist.TEMP` | 4 | 24.316 | 6.079 | 101,316667 |

**[S2]** Tất cả các mảng trong PKL biểu diễn danh định **1 giờ 41 phút 19 giây**. BVP có số hàng gấp đôi ACC và gấp 16 lần EDA cổ tay vì các mảng có cùng thời lượng. Riêng dữ liệu gốc chứa thời lượng khác nhau. RespiBAN có 6.345,81 giây, còn ACC, BVP, EDA và TEMP gốc của E4 lần lượt có 7.874,0625, 7.874,109375, 7.873,5 và 7.874 giây khi tính theo số hàng và tần số.

**[S2] [PDF]** Để đối chiếu theo mốc tương đối của PKL, lấy chỉ số chia cho tần số của mảng. Ví dụ, ở `t = 1 s`, chỉ số danh định là `700` với nhãn và ngực, `32` với ACC cổ tay, `64` với BVP và `4` với EDA hoặc TEMP cổ tay. Dùng hàng `32` cho cả hai ACC sẽ so sánh khoảng `0,045714 s` của ngực với `1 s` của cổ tay. Với thời điểm nằm giữa hai lần lấy mẫu, cần quy định lấy điểm gần nhất hay chọn cửa sổ trước khi triển khai mô hình.

**[S2] [Chưa xác minh]** Khi truy ngược về tệp gốc, cộng mốc riêng của dãy. Đầu PKL ngực tương ứng `92017/700 ≈ 131,452857 s` sau đầu RespiBAN. Trong đồng hồ Unix UTC của E4 gốc, đoạn ACC và BVP bắt đầu sau `1545,21875 s` so với header của thiết bị, còn EDA và TEMP bắt đầu sau `1545 s`. Các mốc đầu của phần cổ tay suy từ header lệch `0,21875 s` giữa hai nhóm tần số. Đây là quan sát trên lưới gốc của từng tệp. Chưa xác minh quy tắc xử lý lệch này của tác giả hay sai số thời gian tuyệt đối giữa hai đồng hồ. [wesad_readme.pdf, mục II](../data/raw/WESAD/wesad_readme.pdf)

## 12. Phân bố nhãn của S2

**[PDF]** `data["label"]` là dãy mã điều kiện giao thức tại 700 Hz. Mã 0 là chưa xác định hoặc chuyển tiếp. Baseline là một giai đoạn được chỉ định riêng với mã 1. Một điều kiện kéo dài nhiều giây nên cùng mã có thể lặp hàng trăm nghìn lần. Phần trăm dưới đây dùng mẫu số 4.255.300, tính cả mã 0 và các mã được khuyến nghị bỏ qua.

| Mã gốc | Ý nghĩa theo PDF | Có trong S2 | Số mẫu | Tỷ lệ (%) | Thời lượng (s) | Thời lượng (phút) |
|---:|---|---|---:|---:|---:|---:|
| 0 | Chưa xác định hoặc chuyển tiếp | Có | 2.142.701 | 50,3537 | 3.061,001 | 51,017 |
| 1 | Baseline, điều kiện nền trung tính | Có | 800.800 | 18,8189 | 1.144,000 | 19,067 |
| 2 | Stress, điều kiện TSST | Có | 430.500 | 10,1168 | 615,000 | 10,250 |
| 3 | Amusement, điều kiện tạo hứng thú | Có | 253.400 | 5,9549 | 362,000 | 6,033 |
| 4 | Meditation, thiền có hướng dẫn | Có | 537.599 | 12,6336 | 767,999 | 12,800 |
| 5 | Bỏ qua theo WESAD | Vắng mặt | 0 | 0,0000 | 0,000 | 0,000 |
| 6 | Bỏ qua theo WESAD | Có | 45.500 | 1,0693 | 65,000 | 1,083 |
| 7 | Bỏ qua theo WESAD | Có | 44.800 | 1,0528 | 64,000 | 1,067 |
| | **Tổng** | | **4.255.300** | **100,0000** | **6.079,000** | **101,317** |

**[S2]** Có bảy mã xuất hiện thực tế, gồm `0, 1, 2, 3, 4, 6, 7`. Mã 5 có trong quy ước và vắng mặt ở S2. Mã 4 gộp hai lượt thiền trên timeline. Mã 0 chiếm khoảng 50,35% bản ghi đồng bộ. “Chưa xác định” nói về cách gán nhãn ở những đoạn đó, chưa phản ánh trạng thái trung tính hay mức độ căng thẳng của S2. [wesad_readme.pdf, mục II.3](../data/raw/WESAD/wesad_readme.pdf)

**[Dự án]** Tài liệu DL13-WESAD ghi quy ước khi huấn luyện ba lớp: giữ các mã gốc `1, 2, 3` và mã hóa đích thành `0, 1, 2` tương ứng Baseline, Stress và Amusement. Vì vậy `0` trong ngữ cảnh mô hình có ý nghĩa khác `0` trong PKL. Toàn bộ số liệu ở đây dùng mã gốc. [README của dự án, mục 2](../README.md)

## 13. Timeline trạng thái của S2

**[S2]** Mỗi đoạn bên dưới là một dãy nhãn liên tiếp không đổi. Điểm bắt đầu có mặt trong đoạn, điểm kết thúc thuộc đoạn tiếp theo. Chỉ số đếm từ 0 và khoảng là `[đầu, cuối)`. Giây/phút được tính từ đầu PKL bằng cách chia chỉ số cho `700` hoặc `42000`. Cận cuối của bản ghi là `4255300`.

| Đoạn | Chỉ số đầu | Chỉ số cuối | Giây đầu | Giây cuối | Phút đầu | Phút cuối | Thời lượng (s) | Nhãn | Ý nghĩa |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0 | 214.583 | 0,000000 | 306,547143 | 0,000000 | 5,109119 | 306,547143 | 0 | Chưa xác định trước Baseline |
| 2 | 214.583 | 1.015.383 | 306,547143 | 1.450,547143 | 5,109119 | 24,175786 | 1.144,000000 | 1 | Baseline |
| 3 | 1.015.383 | 1.591.482 | 1.450,547143 | 2.273,545714 | 24,175786 | 37,892429 | 822,998571 | 0 | Chưa xác định giữa Baseline và TSST |
| 4 | 1.591.482 | 2.021.982 | 2.273,545714 | 2.888,545714 | 37,892429 | 48,142429 | 615,000000 | 2 | Stress |
| 5 | 2.021.982 | 2.212.383 | 2.888,545714 | 3.160,547143 | 48,142429 | 52,675786 | 272,001429 | 0 | Chưa xác định sau TSST |
| 6 | 2.212.383 | 2.257.883 | 3.160,547143 | 3.225,547143 | 52,675786 | 53,759119 | 65,000000 | 6 | Bỏ qua, khớp mốc `sRead` |
| 7 | 2.257.883 | 2.868.283 | 3.225,547143 | 4.097,547143 | 53,759119 | 68,292452 | 872,000000 | 0 | Chưa xác định trước thiền 1 |
| 8 | 2.868.283 | 3.141.982 | 4.097,547143 | 4.488,545714 | 68,292452 | 74,809095 | 390,998571 | 4 | Thiền 1 |
| 9 | 3.141.982 | 3.334.483 | 4.488,545714 | 4.763,547143 | 74,809095 | 79,392452 | 275,001429 | 0 | Chưa xác định giữa thiền 1 và Amusement |
| 10 | 3.334.483 | 3.587.883 | 4.763,547143 | 5.125,547143 | 79,392452 | 85,425786 | 362,000000 | 3 | Amusement |
| 11 | 3.587.883 | 3.688.683 | 5.125,547143 | 5.269,547143 | 85,425786 | 87,825786 | 144,000000 | 0 | Chưa xác định sau Amusement |
| 12 | 3.688.683 | 3.733.483 | 5.269,547143 | 5.333,547143 | 87,825786 | 88,892452 | 64,000000 | 7 | Bỏ qua, khớp mốc `fRead` |
| 13 | 3.733.483 | 3.847.583 | 5.333,547143 | 5.496,547143 | 88,892452 | 91,609119 | 163,000000 | 0 | Chưa xác định trước thiền 2 |
| 14 | 3.847.583 | 4.111.483 | 5.496,547143 | 5.873,547143 | 91,609119 | 97,892452 | 377,000000 | 4 | Thiền 2 |
| 15 | 4.111.483 | 4.255.300 | 5.873,547143 | 6.079,000000 | 97,892452 | 101,316667 | 205,452857 | 0 | Chưa xác định sau thiền 2 |

Tính từ đầu PKL, Baseline kéo dài từ khoảng **05:06,547** đến **24:10,547**. Stress bắt đầu khoảng **37:53,546** và kéo dài **10 phút 15 giây**, tới **48:08,546**. Amusement kéo dài từ khoảng **79:23,547** đến **85:25,547**. Thiền diễn ra hai lần, khoảng **68:17,547–74:48,546** và **91:36,547–97:53,547**. Tám đoạn mã 0 xuất hiện ở đầu, giữa các đoạn có nhãn và cuối bản ghi.

### Đối chiếu với lịch trong S2_quest.csv

**[PDF] [S2]** Chuỗi `phút.giây` trong CSV dùng mốc 0 của RespiBAN gốc. Thời điểm `7.08` là 428 giây. Với `50.3` và `77.1`, cách đọc `50:30` và `77:10` được hỗ trợ bởi sự nhất quán với các biên nhãn sau khi quy về đồng hồ RespiBAN. Để so sánh, cộng `92017/700` vào giây trên PKL. [wesad_readme.pdf, mục III.1](../data/raw/WESAD/wesad_readme.pdf) · [S2_quest.csv](../data/raw/WESAD/S2/S2_quest.csv)

| Hoạt động CSV | Cặp mốc nguyên văn | Khoảng CSV so với đầu RespiBAN (s) | Khoảng nhãn PKL quy về đầu RespiBAN (s) | Mã trong PKL |
|---|---|---|---|---:|
| `Base` | `7.08` → `26.32` | `[428, 1592)` | `[438, 1582)` | 1 |
| `TSST` | `39.55` → `50.3` | `[2395, 3030)` | `[2404.998571, 3019.998571)` | 2 |
| `Medi 1` | `70.19` → `77.1` | `[4219, 4630)` | `[4229, 4619.998571)` | 4 |
| `Fun` | `81.25` → `87.47` | `[4885, 5267)` | `[4895, 5257)` | 3 |
| `Medi 2` | `93.38` → `100.15` | `[5618, 6015)` | `[5628, 6005)` | 4 |
| `sRead` | `54.42` → `56.07` | `[3282, 3367)` | `[3292, 3357)` | 6 |
| `fRead` | `89.51` → `91.15` | `[5391, 5475)` | `[5401, 5465)` | 7 |

**[S2] [Chưa xác minh]** Bảy đoạn khác 0 trong S2 bắt đầu muộn và kết thúc sớm khoảng 10 giây so với các cặp mốc tương ứng, với sai lệch so với mẫu hình này tối đa một mẫu 700 Hz. Sự trùng khớp về thứ tự và vị trí thời gian cho phép liên hệ mã 6 với `sRead` và mã 7 với `fRead` **trong S2**. PDF chỉ xác nhận các mã 5/6/7 thuộc nhóm bỏ qua. Phép đối chiếu không đủ để xác định lý do cắt biên, mã nguồn tạo nhãn hay quy tắc áp dụng cho tất cả người tham gia.

**[PDF] [S2]** Timeline biểu diễn lịch giai đoạn của nhóm WESAD. Ví dụ, `label=2` cho biết tại điểm đó dữ liệu nằm trong đoạn TSST đã được gán vào lớp Stress. Trải nghiệm tự báo cáo của S2 ở chương 4 có thể giúp diễn giải TSST nhưng chỉ có độ phân giải theo lần điền bảng hỏi. README của S2 không chỉ ra thời điểm cảm biến nhiệt độ tiếp xúc kém, nên chưa thể gắn ghi chú đó cho một đoạn cụ thể. [wesad_readme.pdf, mục II.3 và III](../data/raw/WESAD/wesad_readme.pdf) · [S2_readme.txt](../data/raw/WESAD/S2/S2_readme.txt)

## 14. Mối quan hệ giữa các tệp

```text
Phiên thí nghiệm của S2
│
├─ RespiBAN Professional ── S2_respiban.txt ── tín hiệu ngực ────────────┐
│                                                                    │
├─ Empatica E4 ──────────── S2_E4_Data.zip ── bốn tín hiệu cổ tay ───────┤
│                              └─ HR, IBI, info và tệp dấu sự kiện    │
│                                                                    ▼
│                              Căn hai thiết bị theo dấu gõ kép trên ACC
│                                               │
│                                               ▼
│                                   S2.pkl → signal.chest, signal.wrist
│                                          → label trên lưới 700 Hz
│                                          → subject là 'S2'
│                                               ▲
├─ Giao thức ───────────── Lịch điều kiện ───────┘ gán theo thời gian
│                             └─ Mốc lưu trong S2_quest.csv để đối chiếu
│
├─ Tự báo cáo của S2 ────── S2_quest.csv ── cách S2 đánh giá từng giai đoạn
│
└─ Hồ sơ và ghi chú thu ─── S2_readme.txt ── bối cảnh và chất lượng phép đo
```

**[PDF]** Khi thu dữ liệu, người tham gia dùng tay đeo E4 gõ hai lần vào ngực. Mẫu chuyển động đặc trưng hiện trên ACC hai thiết bị được dùng để căn chỉnh thủ công. Lịch giai đoạn dùng mốc bắt đầu của RespiBAN. PKL được WESAD phân phối sẵn như bản ghi hai thiết bị đã căn chỉnh và có nhãn. Sơ đồ biểu diễn quan hệ về nội dung và nguồn gốc. PDF không mô tả mã nguồn cụ thể đọc trực tiếp các tệp này để tạo PKL. [wesad_readme.pdf, mục II và III](../data/raw/WESAD/wesad_readme.pdf)

**[S2]** Dữ liệu E4 trong PKL đã được kiểm tra khớp giá trị từng mẫu với các đoạn tương ứng trong ZIP. Các điểm đối chiếu RespiBAN xác nhận offset và cách đọc đơn vị ở các vị trí đã kiểm tra. Biên nhãn trong PKL khớp với thứ tự hoạt động trong CSV, có chừa khoảng 10 giây ở mỗi đầu tại S2. Ghi chú nhiệt độ và tự báo cáo giúp giải thích chất lượng phép đo và trải nghiệm theo giai đoạn. Cấu trúc PKL không có trường lưu câu trả lời, hồ sơ hay ba tệp HR/IBI/tags. [Chương 5](#5-s2_respibantxt) · [Chương 6](#6-s2_e4_datazip) · [Chương 13](#13-timeline-trạng-thái-của-s2)

**[Chưa xác minh]** PDF xác nhận nguyên tắc gõ kép và việc căn chỉnh thủ công. Công cụ, thuật toán chọn mẫu gõ, quy tắc cắt đầu/cuối phiên, xử lý độ trôi đồng hồ và nội suy hoặc quy tắc làm tròn khi chọn biên chưa được tái lập. Kết quả khớp dữ liệu mô tả bản công bố của S2, chưa đủ để mô tả mọi thao tác của nhóm tác giả.

## 15. Vì sao S2.pkl là nguồn dữ liệu chính cho học sâu

**[Dự án]** DL13-WESAD nhằm nhận diện Baseline, Stress và Amusement bằng tín hiệu đeo cổ tay. Trong tài liệu E1/E2, đầu vào đang dùng là ACC ba trục, BVP và EDA, thành năm kênh. `S2.pkl` đã cung cấp những mảng này, thêm các kênh còn lại cho những câu hỏi sau và cung cấp dãy nhãn gốc với 4.255.300 phần tử. Bộ đọc dự án truy cập mảng NumPy bằng các khóa, kiểm tra mã `subject` và tra tần số theo từng tín hiệu. Đây là nguồn phù hợp cho bài toán hiện tại vì pha thí nghiệm và tín hiệu đã có một mốc phân tích chung, còn dữ liệu gốc vẫn sẵn để truy vết. [README của dự án, mục 2](../README.md) · [Tài liệu E1](../experiments/e1_cnn/README.md) · [Tài liệu E2](../experiments/e2_gru/README.md) · [Bộ đọc WESAD](../src/data/inspection.py)

| Công việc trong quy trình học sâu | PKL hỗ trợ như thế nào | Quyết định vẫn cần ở giai đoạn sau |
|---|---|---|
| Hiểu dữ liệu | Đọc thẳng dictionary và mảng, kiểm tra kiểu, shape, mẫu, nhãn và thời lượng | Giải thích đơn vị và chất lượng dựa trên PDF, bản gốc và ghi chú |
| Tiền xử lý | Có bản gốc đã căn theo thời gian và lưu kênh riêng | Chọn cách xử lý tần số, chất lượng, biên và chuẩn hóa theo mục tiêu |
| Chọn cảm biến | Truy cập từng nhóm theo khóa, chẳng hạn `signal["wrist"]["BVP"]` | Chọn nhóm tín hiệu dựa trên giả thuyết nghiên cứu, vị trí đeo và khả năng thu thập |
| Chia cửa sổ | Quy đổi khoảng thời gian sang chỉ số theo tần số từng mảng | Chọn độ dài, bước trượt và cách xử lý cửa sổ giao biên |
| Căn nhãn | Có nhãn 700 Hz cùng mốc danh định với dữ liệu, với cùng số hàng như các mảng ngực | Đặt quy tắc lấy nhãn cho cửa sổ cổ tay và xử lý mã chuyển tiếp hoặc mã bỏ qua |
| Huấn luyện CNN/GRU | Cung cấp chuỗi số và nhãn có thể chuyển thành tensor sau tiền xử lý | Chọn bố cục tensor, chia theo người và học tham số chuẩn hóa từ tập huấn luyện |
| Nghiên cứu lược bỏ (*ablation study*) | Có thể xác định hoặc bỏ chọn một nhóm như BVP hay EDA bằng khóa | Giữ giao thức đánh giá và điều kiện so sánh nhất quán khi đánh giá vai trò của nhóm tín hiệu |

Phép căn chỉnh do WESAD cung cấp giúp tránh làm lại việc phát hiện dấu gõ kép, đọc hai đồng hồ gốc và dựng nhãn từ đầu. Sự khác biệt tần số vẫn hiện trong PKL. Ngực và nhãn dùng 700 Hz, còn cổ tay dùng 32/64/4 Hz tùy mảng. Ghép ACC, BVP và EDA theo cùng chỉ số hàng sẽ đặt các thời điểm khác nhau vào cùng một vị trí tensor. Ghép theo thời gian vẫn cần quy tắc chọn lưới chung hoặc kiến trúc có thể nhận các chuỗi khác tần số. Mẫu rời rạc ở các đoạn chuyển tiếp cũng cần chính sách nhãn khi chia cửa sổ.

**[Dự án]** Một ví dụ về đầu vào đã định nghĩa trong dự án là mỗi cửa sổ 30 giây có 960 mốc trên lưới 32 Hz và năm kênh theo thứ tự ACC-x/y/z, BVP và EDA. Trước khi tạo tensor, pipeline hiện có quy tắc đưa tín hiệu về lưới đích, xác định các khoảng nhãn hợp lệ và chuẩn hóa theo tập huấn luyện. PKL chỉ cung cấp vật liệu đầu vào cho các bước đó. Ví dụ, `wrist.ACC` còn ở đơn vị `1/64 g`, `wrist.EDA` ở µS và cả hai kênh có trung bình khác 0 ngay trong thống kê S2. Shape của PKL còn theo từng tín hiệu, thời điểm và số kênh, chưa có trục cửa sổ hoặc trục lô của mô hình. [README của dự án, mục 2–3](../README.md)

| Tệp | Mục đích và dữ liệu | Độ thuận tiện khi sử dụng | Đồng bộ và khả năng truy cập nhãn | Vai trò và khi cần dùng |
|---|---|---|---|---|
| `S2.pkl` | Mảng hai thiết bị, mã người tham gia, nhãn | Mở một tệp và chọn mảng theo khóa. Cần Python/NumPy tương thích và lưu ý bộ nhớ. | Hai thiết bị trên mốc phân tích do WESAD cung cấp. `label` truy cập trực tiếp. | Nguồn chính cho bài toán dùng tín hiệu và nhãn giao thức đã công bố |
| `S2_respiban.txt` | Dữ liệu ngực gốc và header OpenSignals | Cần đọc tab và metadata, quy đổi mã số hóa để có đơn vị vật lý | Mốc gốc của lịch, ghép được với CSV. Tệp tín hiệu không chứa mảng nhãn. | Kiểm tra dữ liệu ngực gốc, đơn vị, nguồn gốc và biên bản ghi |
| `S2_E4_Data.zip` | Dữ liệu cổ tay gốc, tệp mô tả, HR/IBI và dấu sự kiện | Cần đọc từng tệp và xử lý quy ước header riêng | Có timestamp E4, cần căn về RespiBAN nếu dùng cùng dữ liệu ngực hoặc nhãn. | Kiểm tra dữ liệu cổ tay ngoài đoạn PKL và đối chiếu mốc, mã, định dạng |
| `S2_quest.csv` | Lịch và tự báo cáo theo giai đoạn | Nhỏ, dễ đọc bằng dấu `;`, vai trò cột thay đổi theo nhóm hàng | Có khoảng hoạt động trên đồng hồ RespiBAN và điểm tự báo cáo, không lưu dãy nhãn mẫu | Truy nguyên điều kiện, đối chiếu biên hoặc nghiên cứu tự báo cáo |
| `S2_readme.txt` | Hồ sơ S2 và ghi chú thu thập | Đọc trực tiếp dạng văn bản | Không có timeline hay nhãn mẫu | Tham chiếu bối cảnh và chất lượng cảm biến |

PKL ghi cả những giai đoạn ngoài ba lớp của dự án, đồng thời không có metadata như timestamp E4, cảnh báo tiếp xúc nhiệt độ, câu trả lời tự báo cáo hay bảng chú thích cho từng mã. Bản PKL của S2 có gần 1 GB trên đĩa, chứa khoảng 280,5 MB dữ liệu phần tử của các mảng, chưa tính bộ nhớ tạm khi nạp và xử lý. Đọc lại tệp thiết bị có giá trị khi cần bản ghi ngoài khoảng đồng bộ, tái tạo đồng bộ, truy xét biên hoặc kiểm nghiệm định dạng và chất lượng. Vì thế vai trò nguồn chính của PKL gắn với mục tiêu dùng bản ghi đồng bộ và nhãn giao thức của **nghiên cứu này**. [Chương 2](#2-cấu-trúc-toàn-bộ-thư-mục-s2) · [Chương 11](#11-tần-số-lấy-mẫu-và-thời-gian)

## 16. Sáu ví dụ đọc trực tiếp từ S2.pkl

**[S2]** Giả sử dictionary đã nạp là `data`. Với những thời điểm nguyên giây được chọn dưới đây, `chỉ số = t × f` là số nguyên cho cả bốn tần số. Thời gian tính từ đầu PKL. Nhãn đi kèm lấy từ `data["label"][t * 700]`. Sáu ví dụ dưới đây đều dùng giá trị đọc từ các mảng gốc của PKL.

| Ví dụ | Cách lấy dữ liệu | Thời điểm và nhãn | Giá trị thực tế | Có thể đọc được và giới hạn |
|---|---|---|---|---|
| 1. Một mẫu gia tốc cổ tay | `data["signal"]["wrist"]["ACC"][0]` | 0 s, `label[0]=0` | `[62.0, -21.0, 107.0]` | X, Y và Z là ba thành phần đồng thời, mỗi đơn vị số bằng `1/64 g`. Một mẫu còn chứa ảnh hưởng của hướng đặt và trọng lực, nên chưa xác định được chuyển động cụ thể. |
| 2. Một mẫu điện tim ngực | `data["signal"]["chest"]["ECG"][0, 0]` | 0 s, `label[0]=0` | `0.02142333984375` mV | Đây là một giá trị điện thế trên dạng sóng. Muốn tính nhịp/phút cần các vị trí nhịp trong một đoạn ECG dài hơn. |
| 3. Một mẫu sóng mạch cổ tay | `data["signal"]["wrist"]["BVP"][64, 0]` | 1 s, `label[700]=0` | `-117.88` | Đây là mẫu thứ 65 khi đếm từ 1, trên lưới 64 Hz. Dấu âm phụ thuộc thang tín hiệu. Chưa suy ra thể tích máu hay nhịp tim từ điểm này. |
| 4. Một thời điểm Baseline | `data["label"][420000]` và `data["signal"]["wrist"]["ACC"][19200]` | 600 s, 10:00, nhãn `1` | ACC `[59.0, 21.0, 8.0]` | Thời điểm thuộc điều kiện nền. Gia tốc có thể giúp diễn giải tư thế hoặc chuyển động khi xét cùng các mẫu lân cận. Chưa biết S2 cảm thấy thế nào ở đúng thời điểm đó. |
| 5. Một thời điểm Stress | `data["label"][1750000]` và `data["signal"]["chest"]["ECG"][1750000, 0]` | 2.500 s, 41:40, nhãn `2` | ECG `-0.1490936279296875` mV | Thời điểm thuộc TSST. Chưa suy ra mức stress, nhịp tim hay phần cụ thể của nhiệm vụ từ điện thế và nhãn này. |
| 6. Một thời điểm Amusement | `data["label"][3500000]` và `data["signal"]["wrist"]["BVP"][320000, 0]` | 5.000 s, 83:20, nhãn `3` | BVP `16.38` | Thời điểm thuộc điều kiện xem đoạn phim. Một điểm BVP chưa cho biết S2 đang cười, vui đến mức nào hay nhịp mạch cụ thể. |

## 17. Những hiểu lầm cần tránh

| Cách hiểu dễ nhầm | Cách đọc đúng với S2 |
|---|---|
| Coi một người tham gia là một mẫu | S2 là một người. Chỉ riêng ECG đã có 4.255.300 lần lấy mẫu. |
| Coi một dòng cảm biến là một mẫu huấn luyện hoàn chỉnh | Hàng `ACC[0]` có ba số cùng thời điểm. Một cửa sổ học sâu gồm nhiều hàng và có thể ghép nhiều loại tín hiệu theo thời gian. |
| Gán một nhãn duy nhất cho cả người | `label` của cùng S2 chuyển qua bảy mã ở những đoạn khác nhau. |
| Đếm mỗi loại tín hiệu là một kênh | Hai mảng ACC có ba cột X/Y/Z mỗi mảng. Tám loại còn lại có một cột mỗi mảng, tổng cộng 14 kênh. |
| Đọc sample index dưới dạng giây hoặc ghép cùng hàng | Hàng `32` là 1 s ở ACC cổ tay, 0,5 s ở BVP và khoảng 0,045714 s ở ngực. |
| Coi nhãn gốc 0 là Baseline | PKL dùng 0 cho chưa xác định hoặc chuyển tiếp, 1 cho Baseline. Mã 0 cho Baseline xuất hiện ở quy ước đích của dự án. |
| Coi định dạng PKL là đã chuẩn hóa | `wrist.ACC[0]` vẫn chứa các mã trên thang `1/64 g`, còn `chest.ECG[0, 0]` có đơn vị mV. Chuẩn hóa cho mô hình là bước riêng. |
| Coi đã đồng bộ là có cùng số lần đo mỗi giây | Các mảng PKL cùng bao phủ 6.079 giây danh định và vẫn có bốn tần số. |
| Coi nhãn Stress là phép đo cảm xúc ở từng giây | Nhãn cho biết điều kiện TSST. Bảng hỏi mô tả tự đánh giá sau giai đoạn. |
| Dùng một tín hiệu đơn lẻ để khẳng định trạng thái tâm lý | EDA có thể tăng khi hào hứng, căng thẳng hoặc nóng. Ngữ cảnh và chuỗi nhiều mẫu giúp kiểm tra các cách diễn giải. |
| Bỏ qua README và bảng hỏi | README cho biết cảm biến nhiệt độ ngực có vấn đề độ bám. CSV cho phép đối chiếu lịch và đọc câu trả lời sau từng giai đoạn. |

## 18. Tổng kết và câu hỏi cho giai đoạn tiếp theo

**[S2]** Hai tệp thiết bị giữ dữ liệu và metadata của từng đồng hồ. `S2_readme.txt` ghi hồ sơ và vấn đề độ bám của Temp ngực, còn `S2_quest.csv` ghi lịch và tự báo cáo. `S2.pkl` gộp 10 loại tín hiệu và 14 kênh từ hai thiết bị, phân theo `signal.chest` và `signal.wrist`, cùng mã người tham gia và nhãn 700 Hz. Khoảng đồng bộ dài 6.079 giây danh định tại các tần số 700, 64, 32 và 4 Hz. Nhãn có 15 đoạn liên tiếp và bảy mã thực tế. Trong 4.255.300 nhãn có 800.800 mẫu Baseline, 430.500 mẫu Stress và 253.400 mẫu Amusement. Các dạng đo bổ sung nhau về hô hấp, điện tim, điện cơ, mạch, hoạt động da và chuyển động. Đối với bài toán DL13-WESAD, PKL cung cấp chuỗi và nhãn có thể truy xuất theo thời gian, còn các tệp gốc và tài liệu giúp kiểm chứng các giả định khi sử dụng.

**[Chưa xác minh]** Còn thiếu công cụ tạo PKL và quy tắc chi tiết về căn đồng hồ, cắt phiên và cắt biên nhãn. Chưa xác minh đơn vị vật lý của BVP, mã phần cứng hô hấp cụ thể, nguyên nhân bốn mẫu TEMP gốc `382.18` hoặc các khoảng Temp ngực bị ảnh hưởng. Đồng bộ không cung cấp một phép đo độc lập để xác định cảm xúc ở từng thời điểm.

### Câu hỏi cho EDA

- Các cực trị ACC, biến động nhanh BVP và đoạn EDA ít biến thiên xảy ra lúc nào? Chúng có trùng chuyển động, đổi hoạt động hay biên tín hiệu không?
- Nhiệt độ xương ức biến thiên ở đâu theo thời gian và bằng chứng nào giúp phân biệt thay đổi sinh lý với thay đổi độ tiếp xúc?
- Sau khi so sánh theo thời gian trong phạm vi từng đoạn, các tín hiệu biến đổi thế nào ở ba giai đoạn mà dự án sử dụng? Sự khác biệt nào có thể đi cùng tư thế hoặc nhiệm vụ?

### Câu hỏi cho tiền xử lý

- Lưới thời gian và cách xử lý biên nào phù hợp cho ACC, BVP và EDA? Cần coi lệch dưới một mẫu của lưới cổ tay chậm như thế nào nếu nghiên cứu độ trễ?
- Quy tắc giữ cửa sổ nào bảo toàn nhãn gốc và tránh trộn điều kiện? Nên xử lý thiền, khoảng chuyển tiếp và mã bỏ qua ra sao trong từng bài toán?
- Nếu thêm kênh ngực hoặc TEMP, cần kiểm tra chất lượng nào trước khi đưa vào mô hình? Tham số chuẩn hóa sẽ được ước lượng trên phần dữ liệu nào?

Việc khảo sát chỉ đọc năm tệp của S2, tài liệu WESAD và tài liệu liên quan trong dự án. Thống kê và ví dụ giữ đơn vị, tần số, nhãn gốc. Các bước tiền xử lý, huấn luyện và phân tích chuyên sâu được để lại cho giai đoạn nghiên cứu tiếp theo.


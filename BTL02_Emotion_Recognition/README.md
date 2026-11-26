# BTL02 — Nhận dạng biểu cảm khuôn mặt

Đặt dữ liệu FER2013 đã giải nén tại `data/fer2013/fer2013.csv`. Có thể dùng
`--data` nếu CSV nằm ở vị trí khác.

Chạy các lệnh sau từ thư mục BTL02. Trên PowerShell, có thể dùng `py -3.12`
thay cho `python` nếu cần chọn bản Python 3.12.

```sh
python -m pip install -r requirements.txt
python code/train.py --epochs 12 --output results/baseline_new
```

Chương trình tự chọn CUDA nếu PyTorch có thể dùng GPU. Dùng `--device cpu`
để chạy trên CPU. Nếu thư mục kết quả đã tồn tại, dùng một tên mới với
`--output`, rồi truyền đường dẫn đó cho lệnh tạo tài liệu bằng `--results`.

## Thiết kế thực nghiệm

| Cấu hình | Lật ngang ngẫu nhiên khi train | Dropout |
|---|---|---:|
| baseline | Không | 0.25 |
| flip | Có | 0.25 |
| dropout | Không | 0.45 |
| flip_dropout | Có | 0.45 |

Khi bật augmentation, mỗi ảnh huấn luyện có xác suất 50% được lật ngang.
Mỗi cấu hình dùng cùng kiến trúc và seed. Chọn checkpoint có accuracy cao
nhất trên PublicTest, dùng loss thấp hơn khi hòa. Đo kết quả PrivateTest
cho cấu hình được chọn. Accuracy và F1 trong các tệp CSV/JSON được lưu
dưới dạng tỷ lệ từ 0 đến 1.

## Thử loss có trọng số và ngân sách 24 epoch

Hai cấu hình `flip` và `flip_weighted` cùng dùng lật ngang, dropout 0,25,
AdamW và seed 42. Ở cấu hình weighted, trọng số lớp tỷ lệ với nghịch đảo căn
bậc hai của số ảnh lớp đó trong **Training**. Chọn checkpoint theo macro F1
trên **PublicTest** và dùng validation loss để phân giải khi hòa. Cả hai
cấu hình dùng cross-entropy không trọng số khi đánh giá. So sánh
checkpoint trong 12 epoch đầu với checkpoint trong toàn bộ 24 epoch.

```sh
python code/train.py --configs flip flip_weighted --epochs 24 --selection-metric macro-f1 --output results/weighted_loss_new
python code/make_submission.py --results results/weighted_loss_new --output results/weighted_loss_new
```

Kết quả thực tế ngày 23/11 nằm trong `results/2026-11-23_weighted_loss`.
Lệnh tạo PDF chỉ ghi workbook khi truyền đường dẫn `--journal` riêng.
Các lượt mới được thêm vào `journal/1.xlsx` để đối chiếu và chép sang
sổ chính khi cần.

## Thử dời pooling sau khối convolution đầu tiên

Lượt 26/11 giữ công thức trọng số lớp, lật ngang, dropout, seed và ngân sách
24 epoch như `flip_weighted`. Cấu hình `flip_weighted_late_pool` đưa AvgPool
xuống sau khối đầu để convolution đầu xử lý ảnh 48×48. Từ khối thứ hai,
kích thước đặc trưng giống đối chứng.

```sh
python code/train.py --configs flip_weighted_late_pool --epochs 24 --selection-metric macro-f1 --validation-only --output results/late_pool_new
```

Kết quả gốc nằm trong `results/2026-11-26_late_pool`. So sánh với checkpoint
`flip_weighted` theo macro F1 trên PublicTest và dùng validation loss khi hòa.
Chỉ đo PrivateTest cho checkpoint mới nếu nó được chọn.

## Bộ nộp BTL02

- `report.pdf` và `slides.pdf` trong ZIP tổng hợp ba vòng thực nghiệm, có kết quả
  PublicTest, PrivateTest, ma trận nhầm lẫn và hình bộ lọc convolution cuối.
- `journal/detail.xlsx` trong ZIP là bản nhật ký từ `journal/1.xlsx` của workspace.
  Sheet **Thi nghiem** tóm tắt 7 lượt chạy, sheet **Ket qua tung epoch** có 120 dòng log.
- `code/train.py` chứa mã nguồn PyTorch. Log và hình của các vòng nằm trong `results/`.

Từ workspace, điền sheet **Phan cong** trong `journal/1.xlsx`, rồi tạo ZIP mới
với họ tên và MSSV của nhóm trưởng. Lệnh lấy nhật ký bạn chỉ định và ghi bản sao
vào ZIP. Chọn tên ZIP chưa tồn tại.

```sh
python code/final_report.py --output submission/2026-11-26 --journal journal/1.xlsx --archive submission/2026-11-26/BTL02_Emotion_HoTen_MaSV.zip
```

Hai bản PDF tổng hợp mới nhất được lưu tại `submission/2026-11-26/`. Để tạo lại riêng hai bản PDF
từ ZIP đã giải nén, chạy `python code/final_report.py`.

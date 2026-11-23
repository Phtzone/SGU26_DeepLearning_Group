# BTL02 — Nhận dạng biểu cảm khuôn mặt

Đặt dữ liệu FER2013 đã giải nén tại `data/fer2013/fer2013.csv`. Có thể dùng
`--data` nếu CSV nằm ở vị trí khác.

Chạy các lệnh sau từ thư mục BTL02. Trên PowerShell, có thể dùng `py -3.12`
thay cho `python` nếu cần chọn bản Python 3.12.

```sh
python -m pip install -r requirements.txt
python code/train.py --epochs 12
python code/make_submission.py
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

## Các tệp nộp

- `report.pdf` chứa phương pháp, kết quả và phân tích, cùng hình các bộ lọc convolution cuối.
- `slides.pdf` là bộ slide ngang, mỗi trang là một slide.
- `journal/detail.xlsx` ghi cấu hình, các epoch, kết quả test và bảng phân công.
- `code/train.py` chứa mã nguồn PyTorch. `code/make_submission.py` tạo lại tài liệu từ kết quả.

Khi chuẩn bị nộp, điền sheet **Phan cong** trong nhật ký rồi nén bộ tệp thành
`BTL02_Emotion_HoTen_MaSV.zip` theo thông tin nhóm trưởng. Nếu đã điền phân công, giữ bản
nhật ký đó khi chạy lại lệnh tạo tài liệu.

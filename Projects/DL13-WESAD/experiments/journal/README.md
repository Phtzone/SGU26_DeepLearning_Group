# Nhật ký nghiên cứu DL13

DL13 nhận diện ba trạng thái baseline, stress, amusement từ tín hiệu cổ tay WESAD: ACC, BVP và EDA. Có 15 người, 5 kênh ở 32 Hz; mỗi cửa sổ dài 30 giây, bước 15 giây. Đánh giá bằng grouped subject-wise 5-fold: mỗi fold có 9 người Train, 3 Validation, 3 Test. Chuẩn hóa chỉ fit trên Train; chọn checkpoint bằng Validation Macro-F1.

E1 CNN-1D đã hoàn tất 5 folds. Trung bình Accuracy = 0.5072 ± 0.1117, Macro-F1 = 0.4493 ± 0.0933. Đây là mean/std giữa folds, không phải pooled OOF. GRU, CNN-GRU và sensor ablation chưa có kết quả để báo cáo.

## Bắt đầu từ đâu?

Mở `research_journal.xlsx` trong thư mục này. Workbook chứa các sheet:

| Sheet | Ghi gì? |
|---|---|
| Huong_dan | Quy ước, ngày giờ và cách mở liên kết |
| Thi_nghiem | Một dòng cho mỗi GROUP_ID: ngày chạy, mục tiêu, cấu hình, script/notebook, kết quả, hình và log |
| Nhat_ky | Mỗi buổi làm một dòng: việc đã làm, kết quả, minh chứng, kế hoạch tiếp |
| Debug | Lỗi thực tế, cách tái hiện, nguyên nhân, cách sửa và kết quả kiểm chứng |
| Ke_hoach | Công việc chưa làm, điều kiện hoàn tất và trạng thái |
| Minh_chung | Các đường dẫn có thể bấm mở tới kết quả E1 hiện có |

Đường dẫn trong workbook tính từ thư mục dự án `Projects/DL13-WESAD`; hyperlink tính từ vị trí workbook. Giữ nguyên vị trí workbook để liên kết hoạt động. Excel có thể yêu cầu cho phép mở file cục bộ. Tên người thực hiện chưa xác minh được thì để trống, không tự điền.

Dòng E1 được tổng hợp hồi cứu từ artifacts. Ngày chạy lấy từ metadata từng fold và chuyển sang UTC+07:00; ngày bổ sung nhật ký là 13/11/2026. Không coi ngày bổ sung nhật ký là ngày huấn luyện. Các dòng kế hoạch chưa phải kết quả thí nghiệm.

## Chạy tiếp, thí nghiệm mới và bổ sung mô hình

Mở terminal tại:

```powershell
cd D:\SGU26_DeepLearning_Group\Projects\DL13-WESAD
```

Dùng Python có các dependencies trong `requirements.txt` và `requirements-tracking.txt`. Đường dẫn môi trường `D:\Deep_learning_group\...` trong README cũ là máy huấn luyện trước đây; cần chọn interpreter hiện có trên máy đang dùng. Thư mục `data/raw` tại workspace này hiện chỉ có `.gitkeep`; cần trỏ tới bản WESAD hợp lệ trước khi training:

```powershell
# Thay đường dẫn bằng nơi thực sự chứa S2/S2.pkl, S3/S3.pkl, ...
$env:WESAD_ROOT = 'D:\duong_dan_thuc_te\WESAD'
python -m scripts.run_grouped_cv --experiment experiments/e1_cnn/config.yaml --dry-run
```

Kiểm tra `missing_subjects` rỗng và đủ 15 người trong kết quả dry-run trước khi chạy. `.env.example` chỉ là tài liệu, runner không tự nạp `.env`.

- **Chạy CNN mới:** `python -m scripts.run_grouped_cv --experiment experiments/e1_cnn/config.yaml`. Runner tạo GROUP_ID mới và lưu logs/config/source/checkpoints. Dòng lệnh này dùng cấu hình E1 hiện tại.
- **Tiếp tục lượt chạy bị gián đoạn:** thêm `--resume --group-id GROUP_ID_DANG_DO`. Giữ nguyên cấu hình khoa học và split. Group E1 chính thức đã hoàn tất, không cần resume để train thêm.
- **Thay siêu tham số CNN:** sao chép YAML sang một thư mục thí nghiệm mới trong `experiments/`, ghi giả thuyết và thay đổi vào Excel, rồi chạy runner với YAML mới. Hiện runner vẫn ghi dưới `runs/experiment_e1` và `results/experiment_e1`, kể cả đổi experiment ID; dùng GROUP_ID để truy xuất chính xác.
- **Thêm GRU/CNN-GRU:** triển khai tại `src/models/gru.py` hoặc `src/models/cnn_gru.py`; cập nhật chỗ tạo model trong `src/training/trainer.py`, validation trong `src/utils/config.py`, và runner `scripts/run_grouped_cv.py` để định tuyến output theo experiment. YAML tương ứng nằm tại `experiments/e2_gru/config.yaml` và `experiments/e3_cnn_gru/config.yaml`. Hiện chỉ đổi `planned` thành `implemented` chưa đủ vì trainer đang khởi tạo CNN trực tiếp và config chỉ nhận `cnn1d`.
- **Bỏ từng cảm biến:** cấu hình dự kiến nằm trong `experiments/ablation/`; đọc và kiểm tra cấu hình trước khi chạy, ghi lượt chạy riêng.

Giữ nguyên baseline chính thức để so sánh. Thí nghiệm mới phải có GROUP_ID mới. Khi so mô hình, giữ cùng dữ liệu, splits, preprocessing và nguyên tắc chọn checkpoint; không chọn cấu hình theo Test score.

## Notebook nào?

E1 hiện chạy bằng `scripts/run_grouped_cv.py` (alias: `scripts/run_experiment.py`), không có notebook thí nghiệm. Thư mục `notebooks/` hiện chỉ có `.gitkeep`. Vì vậy dòng E1 trong Excel ghi rõ “Không dùng notebook”, kèm đường dẫn script.

Nếu sau này dùng notebook, tạo ở `notebooks/`, đặt tên dễ truy xuất như `E2_GRU_YYYYMMDD.ipynb`, ghi đường dẫn và cell/section thực thi trong Excel. Sau khi chạy, lưu cả output notebook và GROUP_ID. Nếu notebook gọi runner thì logs vẫn lấy ở `runs/`, không chỉ dựa vào ảnh chụp output notebook.

## Mỗi lần làm cần ghi những gì?

1. **Trước khi chạy:** ngày giờ UTC+07:00, người làm, mã thí nghiệm, mục tiêu/giả thuyết, thay đổi so với baseline, đường dẫn YAML và script/notebook, seed/folds và lệnh chạy.
2. **Trong khi chạy:** ghi GROUP_ID thực tế; lưu log, checkpoint và hình trong thư mục của lượt chạy. Nếu lỗi, giữ nguyên thông báo/traceback và đường dẫn log.
3. **Sau khi chạy:** ghi completed/partial/failed/interrupted, số folds đã xong, metrics và cách tổng hợp, nhận xét, liên kết hình/log/kết quả, kế hoạch tiếp theo.
4. **Khi debug:** mỗi lỗi một dòng trong sheet Debug, liên kết cùng GROUP_ID; ghi nguyên nhân đã xác minh hay chỉ là giả thuyết, sửa file nào, kiểm tra bằng lệnh nào và kết quả. Chỉ đánh dấu đã sửa sau khi có bằng chứng kiểm tra lại. Nếu không gặp lỗi, ghi “Không ghi nhận lỗi trong lần chạy”; không bịa một lỗi để điền bảng.

Ví dụ một thí nghiệm dự kiến: “Giảm learning rate của CNN từ 0.001 xuống 0.0005; giả thuyết giúp Validation ổn định hơn”. Trước khi chạy để trạng thái planned và để trống kết quả. Sau khi chạy mới điền GROUP_ID, ngày thực tế, metrics và đường dẫn minh chứng.

## Log và hình nằm ở đâu?

Với E1 chính thức, GROUP_ID là `20261108T154303Z_474b38f9`:

- `runs/experiment_e1/<GROUP_ID>/logs/`: log điều phối toàn lượt chạy.
- `runs/experiment_e1/<GROUP_ID>/seed_42/fold_01/logs/`: `iterations.csv` theo batch, `epochs.csv` theo epoch, `validation.csv`, `training.log`, `errors.log`, `events.jsonl`; tương tự fold_02 đến fold_05.
- `runs/experiment_e1/<GROUP_ID>/seed_42/fold_01/checkpoints/`: checkpoint của fold.
- `results/experiment_e1/<GROUP_ID>/`: `summary.json`, `fold_results.csv`, `subject_metrics.csv`, `oof_predictions.csv`.
- `results/experiment_e1/<GROUP_ID>/enhanced_visualizations/scientific-v2_44c886afdd32da86/`: hình E1 chính thức. Các lượt chạy sau có thư mục visualization riêng; điền đúng đường dẫn được tạo, không tái sử dụng tên thư mục E1.

Excel là phần giải thích quá trình; CSV/JSON/events là log máy. Cần giữ cả hai. Nếu lỗi xảy ra trước lúc logger khởi tạo, lưu thêm toàn bộ output terminal vào file riêng và liên kết trong Debug.

Để xem TensorBoard của các lượt đã lưu:

```powershell
python -m tensorboard.main --logdir runs --port 6006
```

Mở http://localhost:6006. Khi đưa hình vào báo cáo, ghi tên file, GROUP_ID và fold/seed tương ứng. Sheet Minh_chung đã có ví dụ liên kết thực tế cho E1.

## Tạo lại mẫu ở file khác

Workbook đã được tạo sẵn. Script hỗ trợ tạo một bản mẫu mới, cần `openpyxl` (chỉ phục vụ Excel, không phải dependency training):

```powershell
python experiments/journal/create_workbook.py --output experiments/journal/research_journal_new.xlsx
```

Script từ chối ghi đè file có sẵn để bảo vệ ghi chép thủ công. Nó chỉ đọc artifacts và tạo Excel; không huấn luyện mô hình hay sửa baseline.

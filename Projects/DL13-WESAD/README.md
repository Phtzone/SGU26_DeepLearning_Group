# DL13-WESAD — Nhận diện trạng thái stress từ tín hiệu sinh lý

Nhật ký làm việc và thí nghiệm: mở [sổ Excel](experiments/journal/research_journal.xlsx) và [hướng dẫn ghi nhật ký, debug, log, nơi chạy tiếp](experiments/journal/README.md). Sổ đã có dòng E1 tổng hợp từ artifacts; các lần làm tiếp cần bổ sung ngày thực tế, GROUP_ID và liên kết minh chứng.

Dự án sử dụng PyTorch để phân loại ba trạng thái **baseline, stress và amusement** từ tín hiệu cảm biến cổ tay WESAD. Mục tiêu là đánh giá khả năng tổng quát hóa sang người chưa xuất hiện trong huấn luyện, đồng thời xây dựng quy trình thực nghiệm có thể truy xuất từ cấu hình, split, logs đến dự đoán. TensorBoard trình bày diễn biến huấn luyện; dữ liệu gốc của kết quả vẫn là các CSV/JSON và checkpoints cục bộ.

## 1. Mục tiêu nghiên cứu và tiến độ

Các câu hỏi nghiên cứu gồm: CNN-1D có nhận diện được trạng thái của người mới không; GRU và CNN-GRU có cải thiện so với CNN-1D khi dùng cùng giao thức không; ACC, BVP và EDA đóng góp như thế nào qua các thí nghiệm loại bỏ cảm biến?

| Thành phần | Trạng thái thực tế |
|---|---|
| E1 CNN-1D | Đã huấn luyện từ đầu và hoàn tất 5 folds; group chính thức `20261108T154303Z_474b38f9` |
| E2 GRU | Chưa triển khai: module còn là placeholder, cấu hình `planned` |
| E3 CNN-GRU | Chưa triển khai: module còn là placeholder, cấu hình `planned` |
| Loại bỏ từng cảm biến | Đã có cấu hình dự kiến; chưa có kết quả để báo cáo |

[Registry E1](experiments/e1_cnn/baseline_registry.json) ghi trạng thái finalization và hash bằng chứng. Đây là baseline cố định để so sánh công bằng, không được chọn lại theo Test score. Xem [tài liệu E1](experiments/e1_cnn/README.md) để biết kiến trúc, splits và kết quả chi tiết.

## 2. Dữ liệu và tiền xử lý

Nguồn dữ liệu là [WESAD tại UCI](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection). Dự án sử dụng đủ **15 subjects: S2–S11 và S13–S17**. Các nhóm subjects được cố định trong [split manifest](splits/grouped_5fold_seed42.json).

| Đầu vào đang dùng | Tần số gốc được ghi trong metadata | Xử lý thực tế |
|---|---|---|
| ACC cổ tay, 3 trục | 32 Hz | Giữ nguyên tần số |
| BVP cổ tay, 1 kênh | 64 Hz | Giảm xuống 32 Hz bằng lọc đa pha chống aliasing |
| EDA cổ tay, 1 kênh | 4 Hz | Nội suy tuyến tính lên 32 Hz; giữ giá trị cuối ở đoạn biên |
| Nhãn trạng thái | 700 Hz | Chỉ giữ khoảng mẫu đích nằm hoàn toàn trong một vùng nhãn hợp lệ |

Pipeline hiện tại dùng **5 kênh**, thứ tự ACC-x/y/z → BVP → EDA. Không dùng TEMP hay các kênh ngực trong E1. Mỗi cửa sổ dài **30 giây = 960 mẫu**, stride **15 giây = 480 mẫu**, chồng lấp 50%. Cửa sổ được neo lại trong từng vùng nhãn liên tục; bỏ vùng nhãn không hợp lệ và không cho cửa sổ vượt qua chuyển trạng thái. Số cửa sổ giữ lại không được cộng với số loại bỏ của phép kiểm tra lưới toàn cục vì hai phép đếm dùng vị trí neo khác nhau.

Nhãn gốc `1, 2, 3` ánh xạ thành `0=baseline`, `1=stress`, `2=amusement`. Các nhãn gốc khác bị loại. Cache có tensor `[số cửa sổ, 5, 960]`, còn tensor đưa vào model có dạng `[batch, 5, 960]`. Lần E1 chính thức có **2.136 cửa sổ hợp lệ** trên toàn bộ subjects. Xem [resampling](src/data/resampling.py), [windowing](src/data/windowing.py) và [preprocessing](src/data/preprocessing.py).

## 3. Giao thức khoa học

Đánh giá dùng **grouped subject-wise 5-fold cross-validation**, split seed **42**. Mười lăm subjects được xáo trộn có seed thành năm nhóm, mỗi nhóm ba người. Với fold thứ `i`, nhóm thứ `i` là Test, nhóm kế tiếp theo vòng là Validation, ba nhóm còn lại là Train: **9 Train / 3 Validation / 3 Test**.

Các cửa sổ của cùng một người luôn thuộc cùng một tập trong mỗi fold. Random window split có thể đưa các cửa sổ chồng lấp hoặc đặc trưng cá nhân vào cả Train và Test, làm đánh giá quá lạc quan. Giao thức hiện tại không phải Leave-One-Subject-Out vì mỗi Test fold chứa ba người, không phải một người.

Mean/std của từng kênh chỉ được fit từ toàn bộ timesteps trong các cửa sổ **Train** của fold, kể cả phần chồng lấp; phương sai dùng `ddof=0`. Validation/Test chỉ được transform bằng scaler đó. Trọng số lớp của loss cũng chỉ tính từ nhãn Train. Mỗi fold khởi tạo CNN mới từ đầu với training seed 42.

Checkpoint được chọn theo **Validation Macro-F1 lớn nhất**, hòa điểm trong tolerance thì lấy Validation loss thấp hơn. Test chỉ được đánh giá sau khi chọn và nạp best checkpoint, không tham gia early stopping hay lựa chọn mô hình. OOF ghép dự đoán Test của năm folds; mỗi người và mỗi window xuất hiện đúng một lần trong OOF.

Checksum SHA256 của JSON canonical cho split: `239949dfebabd802091d14a4494d2216aee9809b0e2993dacdbdfbcdc7133f2c`. File [checksum](splits/grouped_5fold_seed42.sha256) được kiểm tra khi đọc manifest. Split seed xác định chia người; training seed xác định khởi tạo, dropout và thứ tự mini-batches.

## 4. Cấu trúc repository

```text
DL13-WESAD/
├── src/
│   ├── data/             # Kiểm tra, resampling, windows, cache, chuẩn hóa và splits
│   ├── models/           # CNN-1D; GRU/CNN-GRU hiện còn placeholder
│   ├── training/         # Trainer, optimizer, selection và checkpoints
│   ├── evaluation/       # Metrics, OOF, lịch sử và trực quan hóa
│   └── utils/            # Cấu hình, console, CSV/JSON và TensorBoard
├── scripts/              # Các lệnh điều phối, tổng hợp và hậu kiểm
├── configs/              # Cấu hình nền
├── experiments/          # Cấu hình/tài liệu E1, kế hoạch E2/E3 và ablation
├── splits/               # Manifest subject-wise dùng chung
├── tests/                # Unit và regression tests với dữ liệu tổng hợp
├── data/                 # Raw/interim/processed cục bộ; không đóng gói audit
├── runs/                 # Source snapshot, effective config, logs, events, checkpoints
├── results/              # Kết quả tổng hợp, OOF, figures và bằng chứng hậu kiểm
└── notebooks/            # Thư mục hiện chỉ có file giữ chỗ
```

[Cấu hình nền](configs/default.yaml) được hợp nhất với [cấu hình E1](experiments/e1_cnn/config.yaml). [Runner](scripts/run_grouped_cv.py) điều phối năm folds; [TensorBoardTracker](src/utils/tensorboard_tracker.py) ghi events. [Enhanced visualization](src/evaluation/enhanced_visualization.py) chỉ dùng artifacts đã lưu để tạo hình.

## 5. Môi trường và cài đặt trên Windows PowerShell

Môi trường huấn luyện được ghi tại [system.json](runs/experiment_e1/20261108T154303Z_474b38f9/system.json): **Python 3.14.6, PyTorch 2.14.0 CPU, TensorBoard 2.21.0**, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1 và Matplotlib 3.11.2. Interpreter là `D:\Deep_learning_group\.venv_new\Scripts\python.exe`; không dùng môi trường kiểm thử Python 3.13 cũ làm mô tả môi trường của lần chạy E1 này.

```powershell
cd D:\Deep_learning_group\Projects\DL13-WESAD
D:\Deep_learning_group\.venv_new\Scripts\Activate.ps1
python -c "import sys, torch, tensorboard; from torch.utils.tensorboard import SummaryWriter; print(sys.version); print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available()); print('TensorBoard:', tensorboard.__version__)"
python -m pip check
```

Môi trường trên đã tồn tại và vượt qua `pip check`. Khi tạo môi trường mới, cài [requirements.txt](requirements.txt) và [requirements-tracking.txt](requirements-tracking.txt); để đối chiếu phiên bản lần E1, dùng [requirements-runtime.txt](runs/experiment_e1/20261108T154303Z_474b38f9/requirements-runtime.txt) và metadata. Việc cài lại thư viện khác nền tảng không bảo đảm kết quả giống bitwise.

```powershell
python -m pip install -r requirements.txt -r requirements-tracking.txt
```

**Không cần TensorFlow** để huấn luyện bằng PyTorch hoặc ghi/xem TensorBoard. Backend hiện tại là TensorBoard cục bộ, không cần đăng nhập W&B hay API key. Các module W&B/importer còn trong source chỉ phục vụ truy xuất lịch sử, không thuộc active pipeline.

## 6. Chạy mới, chạy một fold và resume

Các lệnh dưới đây là hướng dẫn để người dùng tự thực hiện; quá trình finalization không chạy lại training.

```powershell
# Chạy mới, tuần tự đủ Fold 1 đến Fold 5
python -m scripts.run_grouped_cv --experiment experiments/e1_cnn/config.yaml

# Chạy mới chỉ Fold 1; kết quả chỉ là partial
python -m scripts.run_grouped_cv --experiment experiments/e1_cnn/config.yaml --folds 1

# Kiểm tra kế hoạch mà không chuẩn bị cache hoặc huấn luyện
python -m scripts.run_grouped_cv --experiment experiments/e1_cnn/config.yaml --dry-run

# Resume group của chính lượt chạy bị gián đoạn: thay GROUP_ID_CAN_RESUME
python -m scripts.run_grouped_cv --experiment experiments/e1_cnn/config.yaml --resume --group-id GROUP_ID_CAN_RESUME
```

Một lệnh điều phối cả pipeline; không cần mở từng file Python. Chạy mới tự tạo GROUP_ID từ thời gian UTC và hậu tố ngẫu nhiên; không tự resume và không ghi đè group có sẵn. `python -m scripts.run_experiment` là alias của cùng runner. Resume kiểm tra cấu hình khoa học, split, cache và completion hashes; fold hoàn tất được bỏ qua, fold dở dang khôi phục từ epoch đã commit trong `last.pt`. Với group partial, resume mặc định giữ lựa chọn folds cũ; thêm `--folds 1 2 3 4 5` nếu muốn mở rộng đủ năm folds.

Khi cần dừng training, nhấn Ctrl+C một lần và chờ runner đóng writers, ghi trạng thái, nhả lock. Không thay scientific config hoặc split khi resume. Muốn làm thí nghiệm khác phải tạo group mới. Không dùng canonical group làm nơi thử sửa cấu hình.

## 7. Local logging và TensorBoard

CSV/JSON logger ghi theo batch/epoch và flush từng batch. Train online loss được đo trong quá trình cập nhật weights; Train eval loss/Accuracy/Macro-F1/Balanced Accuracy được tính sau epoch ở chế độ đánh giá. Validation metrics dùng để giám sát tổng quát hóa và selection. Test metrics là kết quả cuối sau selection. Accuracy đo tỷ lệ dự đoán đúng; Macro-F1 lấy trung bình F1 ba lớp; Balanced Accuracy là trung bình recall các lớp có hỗ trợ theo implementation.

TensorBoard đọc event files, không trực tiếp huấn luyện model. Có thể mở trong terminal thứ hai khi training đang chạy hoặc mở lại sau khi training kết thúc:

```powershell
cd D:\Deep_learning_group\Projects\DL13-WESAD
D:\Deep_learning_group\.venv_new\Scripts\Activate.ps1

# Xem các groups của riêng E1
python -m tensorboard.main --logdir "D:\Deep_learning_group\Projects\DL13-WESAD\runs\experiment_e1" --port 6006

# Hoặc xem chung các experiments khi chúng có events
python -m tensorboard.main --logdir "D:\Deep_learning_group\Projects\DL13-WESAD\runs" --port 6006
```

Mở **http://localhost:6006**. Hai lệnh là hai lựa chọn trên cùng cổng; dừng server đang chạy bằng Ctrl+C trước khi đổi lệnh. Đổi `--logdir` chỉ thay phạm vi dữ liệu hiển thị, không ghi đè kết quả. Để chỉ xem canonical E1, thêm `\20261108T154303Z_474b38f9` vào cuối logdir E1. Mỗi fold có ba streams `tensorboard/epochs`, `tensorboard/batches`, `tensorboard/evaluation`.

## 8. Experiment ID, GROUP_ID và artifacts

Experiment ID mô tả thiết kế, ví dụ `E1_CNN`; GROUP_ID xác định một lượt thực nghiệm gồm các seeds/folds. Fold xác định phép chia người. Mỗi fold có run ID riêng, khởi tạo model riêng để không truyền thông tin huấn luyện giữa các Test folds. `runs/` giữ bằng chứng thực thi; `results/` giữ kết quả tổng hợp và hình từ các bằng chứng đó.

| Nội dung canonical E1 | Vị trí |
|---|---|
| Runs, config, source snapshot | [Group runs](runs/experiment_e1/20261108T154303Z_474b38f9/) |
| Logs và checkpoints từng fold | Trong `seed_42/fold_01` đến `fold_05`, các thư mục `logs/`, `checkpoints/` |
| Fold/subject metrics, OOF và summary | [Group results](results/experiment_e1/20261108T154303Z_474b38f9/) |
| Bộ 131 hình chính thức sau sửa nhãn | [Scientific-v2](results/experiment_e1/20261108T154303Z_474b38f9/enhanced_visualizations/scientific-v2_44c886afdd32da86/) |
| Kiểm chứng artifacts | [Baseline verification](results/experiment_e1/20261108T154303Z_474b38f9/baseline_verification.json) |
| Kiểm thử sau finalization | [Pytest XML](results/experiment_e1/20261108T154303Z_474b38f9/finalization_pytest.xml) |
| Trạng thái baseline và hash | [Baseline registry](experiments/e1_cnn/baseline_registry.json) |

Metadata runs/results đã thống nhất trạng thái kết thúc. Bộ hình v1 được giữ để đối chiếu; v2 sửa bốn learning overviews và ba biểu đồ metadata huấn luyện, không đổi số liệu đầu vào.

## 9. Kết quả E1 đã kiểm chứng

<!-- ACTUAL_RESULTS_START -->
Canonical GROUP_ID: `20261108T154303Z_474b38f9`. **5/5 folds, 15 Test subjects duy nhất, 137 epochs, 5.554 mini-batches, 2.136 OOF windows.**

| Cách tổng hợp | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---|---|---|
| Trung bình ± độ lệch chuẩn mẫu giữa 5 folds | 0.5072 ± 0.1117 | 0.4493 ± 0.0933 | 0.4567 ± 0.0886 |
| Pooled OOF trên 2.136 windows | 0.5066 | 0.4500 | 0.4547 |

Độ lệch chuẩn giữa folds dùng `ddof=1`. Pooled OOF tính trực tiếp trên tất cả dự đoán, không phải trung bình năm scores. [Bảng từng fold và phân tích E1](experiments/e1_cnn/README.md) · [Summary](results/experiment_e1/20261108T154303Z_474b38f9/summary.json).
<!-- ACTUAL_RESULTS_END -->

Chưa có kết quả E2/E3 để kết luận model nào tốt hơn. Registry chọn group đã được chỉ định trước để làm baseline, không lựa chọn group có Test score cao nhất.

## 10. Kiểm chứng và tái lập

Hậu kiểm đọc checkpoints bằng `weights_only=True` với danh sách NumPy RNG types cố định; không deserialize pickle không giới hạn. Đã đối chiếu strict state dictionary của CNN-1D, run identities, logs, split, OOF, scaler/class weights Train-only và 15 TensorBoard event files. Scalar TensorBoard dùng float32 nên so với CSV bằng `rtol=atol=1e-6`; metrics OOF được đối chiếu với sai số tuyệt đối `1e-12`.

```powershell
# Hậu kiểm chỉ đọc dữ liệu đã lưu, không training và không sửa run artifacts
python -m scripts.verify_e1_baseline --group-id 20261108T154303Z_474b38f9 --output results/experiment_e1/20261108T154303Z_474b38f9/baseline_verification.json

# Trong môi trường có pytest cùng các dependencies của dự án
python -m pytest -q
```

Kết quả kiểm thử sau thay đổi: **50 passed, 2 skipped, 0 failed**. Hai tests bỏ qua phụ thuộc source snapshot/artifacts E1 lịch sử đã được xóa, không phải group canonical hiện tại. Môi trường `.venv_new` của training chưa cài pytest; finalization dùng cùng interpreter/dependencies này và một bộ pytest 9.1.1 thuần Python cục bộ trong thư mục tạm, tắt tự nạp plugins ngoài dự án. Không thay dependencies huấn luyện. [Bằng chứng tests](results/experiment_e1/20261108T154303Z_474b38f9/finalization_pytest.xml) là kết quả sau thay đổi; các XML cũ khác có timestamp riêng.

Để tái lập cần dataset đúng nội dung, cache/preprocessing đúng phiên bản, split manifest/checksum, effective config, source snapshot và dependency versions của group. Source hiện tại có sửa lifecycle metadata/nhãn hình sau training; source snapshot trong group mới là bản đã dùng để tạo weights. Git commit trong system metadata kèm `git_dirty=true`, vì vậy riêng commit không đủ xác định source; dùng hashes của snapshot và registry.

## 11. Giới hạn

Mười lăm người trong bối cảnh thu thập WESAD và một training seed chưa đại diện cho mọi người hay bối cảnh sử dụng. Điểm số biến thiên rõ theo fold và subject; chưa có đánh giá độ ổn định qua nhiều training seeds hay systematic hyperparameter tuning. Chênh lệch Train/Validation gợi ý overfitting ở một số folds, nhưng chưa xác định nguyên nhân nhân quả. Baseline không được điều chỉnh vì riêng Fold 2 thấp.

## 12. Hướng phát triển

Triển khai E2 GRU, sau đó E3 CNN-GRU, dùng cùng preprocessing, split, selection và cách tổng hợp để so sánh với registry E1. Các sensor ablations và thí nghiệm điều chỉnh siêu tham số phải dùng experiment/group riêng. Chưa công bố bất kỳ metrics GRU/CNN-GRU hoặc ablation nào.

# E1 CNN-1D — Baseline chính thức trên WESAD

## 1. Vai trò và phạm vi

E1 là baseline CNN-1D được **huấn luyện từ đầu**, không dùng pretrained weights. Group chính thức là **`20261108T154303Z_474b38f9`**, hoàn tất đủ năm folds. Baseline giữ cố định thiết kế khoa học và kết quả để đối chiếu với E2 GRU/E3 CNN-GRU sau này. Chất lượng Fold 2 thấp không phải lý do chỉnh weights, splits hoặc chọn lại baseline theo Test score.

[Baseline registry](baseline_registry.json) là điểm truy xuất trạng thái finalization, paths và SHA256. [Hậu kiểm](../../results/experiment_e1/20261108T154303Z_474b38f9/baseline_verification.json) ghi bằng chứng đọc checkpoints, tính lại metrics và đối chiếu TensorBoard. E2/E3 hiện còn placeholders với cấu hình `planned`; không có kết quả E2/E3 để so sánh.

## 2. Dữ liệu, nhãn và tensor đầu vào

Sử dụng đủ 15 WESAD subjects: S2–S11, S13–S17. Dữ liệu cổ tay gồm ACC 3 trục ở 32 Hz, BVP ở 64 Hz và EDA ở 4 Hz. ACC giữ nguyên; BVP giảm mẫu bằng `resample_poly` chống aliasing; EDA nội suy tuyến tính. Timeline nhãn gốc 700 Hz được căn chỉnh về 32 Hz bằng các khoảng trạng thái liên tục. Không cắt tín hiệu âm thầm để ép độ dài; không clipping sinh lý hay đổi đơn vị.

| Nhãn WESAD | Nhãn model | Trạng thái |
|---|---|---|
| 1 | 0 | baseline |
| 2 | 1 | stress |
| 3 | 2 | amusement |

Các nhãn còn lại bị loại. Cửa sổ dài **30 giây**, stride **15 giây**, overlap 50%, chỉ nằm trong vùng nhãn hợp lệ liên tục; không gán nhãn theo đa số để vượt qua chuyển trạng thái. Thứ tự kênh là **ACC-x, ACC-y, ACC-z, BVP, EDA**. Cache và model dùng thứ tự `[N, C, T]` với `C=5`, `T=960`. Lần chạy có tổng **2.136 windows**.

Scaler được fit lại trong mỗi fold từ riêng Train: mean/std mỗi kênh trên tất cả timesteps của các Train windows, kể cả phần overlap, `ddof=0`; std nhỏ hơn `1e-8` được thay bằng 1 theo implementation. Validation/Test dùng cùng scaler Train, không fit lại. Trọng số lớp cũng chỉ dùng nhãn Train. Hậu kiểm đã tính lại mean/std, sample counts và class weights từ cache và đối chiếu đúng với artifacts.

## 3. Kiến trúc CNN-1D

Implementation: [CNN1D](../../src/models/cnn.py). Tổng số tham số trainable đã đối chiếu từ checkpoint: **45.059**.

| Thành phần | Chi tiết | Tensor đầu ra |
|---|---|---|
| Đầu vào | 5 kênh, 960 mẫu | `[B, 5, 960]` |
| Khối 1 | Conv1D 5→32, kernel 7, padding 3; BatchNorm; ReLU; MaxPool 2 | `[B, 32, 480]` |
| Khối 2 | Conv1D 32→64, kernel 5, padding 2; BatchNorm; ReLU; MaxPool 2 | `[B, 64, 240]` |
| Khối 3 | Conv1D 64→128, kernel 3, padding 1; BatchNorm; ReLU | `[B, 128, 240]` |
| Global Average Pooling | AdaptiveAvgPool1D(1), Flatten | `[B, 128]` |
| Bộ phân loại | Linear 128→64; ReLU; Dropout 0,3; Linear 64→3 | `[B, 3]` |

Model xuất **logits**, không softmax trước Cross-Entropy. Probabilities được xuất ở bước đánh giá. Conv1D học mẫu cục bộ theo thời gian; pooling giảm chiều dài; global average pooling tổng hợp thời gian; lớp fully connected tạo quyết định ba lớp. Mỗi fold tạo model mới, không chuyển weights từ fold trước.

## 4. Cấu hình chính thức

Nguồn: [cấu hình E1](config.yaml), [cấu hình nền](../../configs/default.yaml) và [effective config đã dùng](../../runs/experiment_e1/20261108T154303Z_474b38f9/config.yaml). Source/config trong group được giữ nguyên như lúc huấn luyện.

| Tham số | Giá trị |
|---|---|
| Split seed / training seed | 42 / 42 |
| Learning rate | 0.001 |
| Batch size | 32 |
| Optimizer | Adam |
| Weight decay | 0.0001 |
| Loss | Weighted Cross-Entropy |
| Trọng số lớp c | `N_train / (3 × N_train_class_c)` |
| Maximum epochs | 100 mỗi fold |
| Early stopping patience | 15 |
| Early stopping min_delta | 0.001 |
| Gradient clipping | L2 norm tối đa 1.0 |
| Dropout | 0.3 |
| Chọn checkpoint | Validation Macro-F1 lớn nhất; hòa trong `1e-12` thì Validation loss thấp hơn |
| Đánh giá Train/Validation | Sau mỗi epoch |
| Chạy folds | Tuần tự, không song song |
| Thiết bị đã dùng | CPU; cấu hình cho phép `auto` |

Selection best checkpoint và bộ đếm patience có hai quy tắc riêng: cải thiện Macro-F1 nhỏ vẫn có thể cập nhật best, nhưng patience chỉ reset khi vượt mốc cải thiện đáng kể hơn `min_delta`. Do đó best epoch không nhất thiết cách epoch dừng đúng 15 nếu có các cải thiện nhỏ. Không dùng Test trong hai cơ chế này.

## 5. Giao thức grouped subject-wise 5-fold

| Fold | Train (9 subjects) | Validation (3 subjects) | Test (3 subjects) |
|---|---|---|---|
| 1 | S7, S4, S11, S5, S6, S14, S2, S3, S13 | S8, S17, S15 | S10, S16, S9 |
| 2 | S10, S16, S9, S5, S6, S14, S2, S3, S13 | S7, S4, S11 | S8, S17, S15 |
| 3 | S10, S16, S9, S8, S17, S15, S2, S3, S13 | S5, S6, S14 | S7, S4, S11 |
| 4 | S10, S16, S9, S8, S17, S15, S7, S4, S11 | S2, S3, S13 | S5, S6, S14 |
| 5 | S8, S17, S15, S7, S4, S11, S5, S6, S14 | S10, S16, S9 | S2, S3, S13 |

Danh sách được đọc trực tiếp từ [manifest đã dùng](../../runs/experiment_e1/20261108T154303Z_474b38f9/split_manifest.json), khớp [manifest dùng chung](../../splits/grouped_5fold_seed42.json). Năm nhóm ba subjects quay vòng: nhóm Test của fold kế tiếp là Validation của fold hiện tại; ba nhóm còn lại làm Train. Mỗi subject là Test đúng một lần. Không có giao nhau Train/Validation/Test trong một fold.

Đây không phải LOSO và không phải random window split. Giữ cùng người trong cùng tập ngăn các cửa sổ chồng lấp và đặc trưng cá nhân rò rỉ sang Test. Checksum SHA256 JSON canonical: `239949dfebabd802091d14a4494d2216aee9809b0e2993dacdbdfbcdc7133f2c`.

## 6. Kết quả huấn luyện và OOF đã kiểm chứng

<!-- ACTUAL_RESULTS_START -->
GROUP_ID chính thức: `20261108T154303Z_474b38f9`.

| Fold | Seed | Epochs hoàn tất | Best epoch | Test Accuracy | Test Macro-F1 | Test Balanced Accuracy |
|---|---|---|---|---|---|---|
| 1 | 42 | 46 | 31 | 0.6079 | 0.5052 | 0.4927 |
| 2 | 42 | 17 | 2 | 0.3210 | 0.2897 | 0.3041 |
| 3 | 42 | 27 | 12 | 0.5282 | 0.5049 | 0.5145 |
| 4 | 42 | 23 | 8 | 0.5035 | 0.4423 | 0.4565 |
| 5 | 42 | 24 | 9 | 0.5752 | 0.5045 | 0.5155 |

Tổng cộng **137 epochs, 5.554 mini-batches, 2.136 OOF predictions**, 15 Test subjects duy nhất. Mỗi window identifier `(subject_id, window_id)` xuất hiện đúng một lần. Dự đoán từng fold khớp OOF tổng hợp; nhãn thật, xác suất, argmax và tập Test khớp cache/manifest.

| Cách tổng hợp | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---|---|---|
| Trung bình ± độ lệch chuẩn mẫu giữa 5 folds | 0.5072 ± 0.1117 | 0.4493 ± 0.0933 | 0.4567 ± 0.0886 |
| Pooled OOF trên 2.136 windows | 0.5066 | 0.4500 | 0.4547 |

**Trung bình ± độ lệch chuẩn giữa folds** coi năm fold scores là năm quan sát, độ lệch chuẩn mẫu `ddof=1`. **Pooled OOF** ghép 2.136 dự đoán rồi tính metrics; không lấy trung bình fold scores. Không thay thế hai đại lượng cho nhau. Subject-level metrics được lưu riêng; mỗi subject có trọng số như nhau khi tính trung bình subject scores.

Confusion matrix pooled OOF; hàng là nhãn thật, cột là dự đoán, thứ tự baseline → stress → amusement:

| Nhãn thật / dự đoán | baseline | stress | amusement |
|---|---|---|---|
| baseline | 655 | 244 | 251 |
| stress | 198 | 331 | 111 |
| amusement | 143 | 107 | 96 |

[Fold results](../../results/experiment_e1/20261108T154303Z_474b38f9/fold_results.csv) · [Subject metrics](../../results/experiment_e1/20261108T154303Z_474b38f9/subject_metrics.csv) · [OOF](../../results/experiment_e1/20261108T154303Z_474b38f9/oof_predictions.csv) · [Summary JSON](../../results/experiment_e1/20261108T154303Z_474b38f9/summary.json).
<!-- ACTUAL_RESULTS_END -->

## 7. Đọc learning curves

Train online loss được tính qua các mini-batches trong quá trình cập nhật weights, có dropout; Train eval loss và các Train scores được tính lại sau epoch ở chế độ đánh giá. Validation dùng dữ liệu người riêng và không cập nhật weights. Do khác chế độ và thời điểm đo, online loss và eval loss không nhất thiết trùng nhau.

Các learning curves thể hiện **Train/Validation theo epoch**, không phải Test OOF. Đường dọc đánh dấu best epoch theo Validation selection; epoch cuối là thời điểm dừng bởi patience hoặc giới hạn epochs. Accuracy phản ánh tỷ lệ đúng, Macro-F1 phản ánh chất lượng cân bằng giữa ba lớp, Balanced Accuracy phản ánh recall trung bình. Không đọc một điểm Validation như Test score.

| Fold | Train Macro-F1 tại best epoch | Validation Macro-F1 tại best epoch | Test Macro-F1 sau chọn checkpoint |
|---|---|---|---|
| 1 | 0.9514 | 0.4977 | 0.5052 |
| 2 | 0.8393 | 0.6874 | 0.2897 |
| 3 | 0.9322 | 0.5121 | 0.5049 |
| 4 | 0.9378 | 0.6399 | 0.4423 |
| 5 | 0.9028 | 0.6537 | 0.5045 |


Các giá trị Train/Validation trên lấy từ CSV ở best epoch; cột Test chỉ được tính sau khi nạp best checkpoint. [Learning Macro-F1](../../results/experiment_e1/20261108T154303Z_474b38f9/enhanced_visualizations/scientific-v2_44c886afdd32da86/cross_fold/seed_42/learning_macro_f1.png) và [learning Balanced Accuracy](../../results/experiment_e1/20261108T154303Z_474b38f9/enhanced_visualizations/scientific-v2_44c886afdd32da86/cross_fold/seed_42/learning_balanced_accuracy.png) đã được sửa tiền tố gây nhầm là OOF Test.

## 8. Phân tích và giới hạn diễn giải

Fold 2 có Accuracy, Macro-F1 và Balanced Accuracy thấp nhất trong năm folds. Các folds còn lại cũng khác nhau, cho thấy khả năng tổng quát hóa giữa người còn hạn chế. Chênh lệch Train/Validation tại best epoch ở bảng trên là dấu hiệu phù hợp với overfitting ở một số folds; dữ liệu này không đủ xác định nguyên nhân là cảm biến, phân bố subject hay kiến trúc.

Trong pooled OOF, F1 lớp baseline là 0.6104, stress là 0.5008, amusement là 0.2388; amusement khó phân loại hơn theo kết quả hiện tại. Subject Macro-F1 biến thiên từ **S8 (0.0303)** đến **S11 (0.7298)**. Không suy ra nguyên nhân chắc chắn từ khác biệt này và không điều chỉnh baseline theo Test outcomes.

Chỉ có một training seed và 15 người; chưa đánh giá độ ổn định nhiều seeds, systematic hyperparameter tuning hoặc khả năng tổng quát ngoài WESAD. E1 là baseline chưa tối ưu; E2/E3 và ablation phải là các experiments độc lập dùng cùng giao thức so sánh.

## 9. Artifacts, logging và phiên bản hình

Group runs: [`runs/experiment_e1/20261108T154303Z_474b38f9/`](../../runs/experiment_e1/20261108T154303Z_474b38f9/). Group results: [`results/experiment_e1/20261108T154303Z_474b38f9/`](../../results/experiment_e1/20261108T154303Z_474b38f9/).

```text
group runs/
├── experiment_manifest.json, config.yaml, split_manifest.json, split_manifest.sha256
├── system.json, requirements-runtime.txt, source_checksums.json
├── source/                       # Source/config đúng lúc chạy, không sửa sau training
├── logs/                         # Logs điều phối cấp group
└── seed_42/fold_01 ... fold_05/
    ├── metadata.json, config.yaml, normalization.json, tracking_identity.json
    ├── completion.json           # Hash artifacts gốc của fold
    ├── checkpoints/best.pt, last.pt
    ├── logs/iterations.csv, epochs.csv, validation.csv
    ├── logs/training.log, events.jsonl, errors.log
    ├── metrics/                  # Test, predictions, subject metrics, confusion
    ├── plots/                    # Hình gốc cấp fold
    └── tensorboard/epochs, batches, evaluation

group results/
├── experiment_manifest.json, summary.json, summary.md
├── fold_results.csv, subject_metrics.csv, oof_predictions.csv
├── aggregate_confusion_matrix.csv, iteration_audit.json
├── figures/                      # Hình tổng hợp ban đầu
├── enhanced_visualizations/      # Giữ v1 và phiên bản v2 chính thức
├── baseline_verification.json    # Kiểm chứng checkpoints, metrics, TensorBoard
└── finalization_pytest.xml        # Bằng chứng tests của lần finalization
```

Ví dụ Fold 1: [epoch logs](../../runs/experiment_e1/20261108T154303Z_474b38f9/seed_42/fold_01/logs/epochs.csv), [batch logs](../../runs/experiment_e1/20261108T154303Z_474b38f9/seed_42/fold_01/logs/iterations.csv), [best checkpoint](../../runs/experiment_e1/20261108T154303Z_474b38f9/seed_42/fold_01/checkpoints/best.pt), [last checkpoint](../../runs/experiment_e1/20261108T154303Z_474b38f9/seed_42/fold_01/checkpoints/last.pt) và [TensorBoard events](../../runs/experiment_e1/20261108T154303Z_474b38f9/seed_42/fold_01/tensorboard/). Thay `fold_01` bằng `fold_02`…`fold_05` cho các folds còn lại.

Phiên bản hình chính thức: [`scientific-v2_44c886afdd32da86`](../../results/experiment_e1/20261108T154303Z_474b38f9/enhanced_visualizations/scientific-v2_44c886afdd32da86/), gồm **131 PNG + 131 SVG** ở 220 DPI. Bản v1 được giữ nguyên. Bản v2 dựng lại **7 figures**: bốn learning overviews (loss, Accuracy, Macro-F1, Balanced Accuracy) đổi thành Train/Validation; ba biểu đồ best epoch, completed epochs và training duration đổi thành metadata Training. **124 figures còn lại được sao chép nguyên byte** sang phiên bản mới. Không đổi curves, predictions hoặc metrics. [Generation manifest](../../results/experiment_e1/20261108T154303Z_474b38f9/enhanced_visualizations/scientific-v2_44c886afdd32da86/generation_manifest.json) ghi hashes đầu vào/đầu ra và số hình tái sử dụng.

## 10. Tái lập, TensorBoard và resume

Môi trường E1 thực tế theo [system metadata](../../runs/experiment_e1/20261108T154303Z_474b38f9/system.json): Python **3.14.6**, PyTorch **2.14.0 CPU**, TensorBoard **2.21.0** trong `D:\Deep_learning_group\.venv_new`. Các phiên bản khác được lưu trong [requirements runtime](../../runs/experiment_e1/20261108T154303Z_474b38f9/requirements-runtime.txt). Không cần TensorFlow hoặc W&B để chạy pipeline hiện tại.

```powershell
cd D:\Deep_learning_group\Projects\DL13-WESAD
D:\Deep_learning_group\.venv_new\Scripts\Activate.ps1
python -c "import sys, torch, tensorboard; from torch.utils.tensorboard import SummaryWriter; print(sys.version); print(torch.__version__, torch.cuda.is_available(), tensorboard.__version__)"
python -m pip check

# Chỉ kiểm tra cấu hình/kế hoạch; không training
python -m scripts.run_grouped_cv --experiment experiments/e1_cnn/config.yaml --dry-run

# Người dùng tự chạy khi muốn tái lập bằng một group mới, đủ 5 folds
python -m scripts.run_grouped_cv --experiment experiments/e1_cnn/config.yaml

# Xem dữ liệu kết quả đã tồn tại
Get-Content results/experiment_e1/20261108T154303Z_474b38f9/summary.json
Get-ChildItem results/experiment_e1/20261108T154303Z_474b38f9/enhanced_visualizations
```

Trong terminal thứ hai, kích hoạt cùng venv rồi mở riêng canonical group:

```powershell
python -m tensorboard.main --logdir "D:\Deep_learning_group\Projects\DL13-WESAD\runs\experiment_e1\20261108T154303Z_474b38f9" --port 6006
```

Mở **http://localhost:6006**. Để xem tất cả groups E1 dùng logdir `...\runs\experiment_e1`; để xem các experiments tương lai dùng `...\runs`. Đổi logdir chỉ đổi phạm vi đọc. Có thể xem trực tiếp khi training hoặc mở lại sau đó; Ctrl+C trong terminal TensorBoard chỉ dừng dashboard.

Đã đọc được **15 event files**, mỗi fold ba streams. Epoch streams có Train/Validation Loss, Accuracy, Macro-F1, Balanced Accuracy; batch streams có loss, LR, gradient norm, duration; evaluation streams có Test scalars và ba images. Epoch steps tăng từ 1 đến epoch cuối, batch global steps liên tục trong từng fold; số điểm và giá trị khớp CSV với tolerance float32 `rtol=atol=1e-6`. Không chỉnh sửa event files đã hoàn tất.

Nếu một lượt chạy mới bị gián đoạn, nhấn Ctrl+C một lần và chờ cleanup, sau đó dùng đúng group ID của lượt đó:

```powershell
python -m scripts.run_grouped_cv --experiment experiments/e1_cnn/config.yaml --resume --group-id GROUP_ID_CAN_RESUME
```

Resume từ `last.pt`, xác minh scientific config/split/cache, khôi phục RNG/optimizer và epoch đã commit; không sửa checkpoint gốc thủ công. Folds completed được xác minh rồi skip. Muốn chạy riêng một fold mới dùng `--folds 1`; muốn mở rộng một group partial có thể resume kèm `--folds 1 2 3 4 5`. Chạy mới luôn có GROUP_ID mới, không nạp checkpoint E1 cũ. Canonical group hiện tại đã hoàn tất nên không cần resume để hậu kiểm.

## 11. Hậu kiểm, registry và bước tiếp theo

Đã xác minh đủ 10 checkpoints bằng PyTorch `weights_only=True`, chỉ cho phép chính xác các NumPy types cần cho RNG state; không dùng fallback `weights_only=False`. Model state được nạp strict vào kiến trúc CNN-1D và đối chiếu run ID → fold/seed metadata. Hash checkpoints, logs, predictions, events và source snapshot được giữ nguyên.

[Verifier](../../scripts/verify_e1_baseline.py) đọc lại cache để xác minh Train-only normalization, class weights, OOF labels/coverage và probabilities; không chạy preprocessing hoặc model training. Selection được phát lại từ Validation history. Source trainer chỉ tiêu thụ Test sau khi chọn/nạp best, có regression test instrumentation cho thứ tự này. Fold means/std và pooled metrics được tính lại độc lập từ saved predictions; sai số đối chiếu metrics là `1e-12`.

```powershell
python -m scripts.verify_e1_baseline --group-id 20261108T154303Z_474b38f9 --output results/experiment_e1/20261108T154303Z_474b38f9/baseline_verification.json
# Trong môi trường có pytest và dependencies dự án
python -m pytest -q
```

[Registry](baseline_registry.json) chỉ được đặt `finalized` khi checkpoints, CV/OOF, TensorBoard, metadata, README và automated tests cùng đạt. Timestamp finalization là thời điểm hậu kiểm thực tế, không phải thời điểm huấn luyện. Registry lưu source fingerprints thay vì coi Git commit của working tree đang dirty là đủ bằng chứng. Hai tests lịch sử có thể skip vì artifacts đã được xóa theo yêu cầu trước đây; không dựng lại kết quả lịch sử để làm tests pass.

Kết quả toàn bộ pytest sau thay đổi: **50 passed, 2 skipped, 0 failed**. Hai skips là kiểm tra source snapshot/artifacts lịch sử đã xóa. Môi trường `.venv_new` chưa cài pytest; lần finalization chạy cùng Python/dependencies E1 bằng pytest 9.1.1 thuần Python trong thư mục tạm và tắt plugins ngoài dự án, không sửa môi trường training. Kết quả tests và môi trường kiểm thử được lưu trong registry/XML.

Sau finalization, triển khai E2 GRU rồi E3 CNN-GRU; dùng cùng 15 subjects, preprocessing, split seed/training seed, Train-only normalization, validation selection và tổng hợp OOF để đối chiếu. Sensor ablations có group riêng. Không sửa baseline vì Test score thấp và chưa tuyên bố lợi ích của model chưa được triển khai.

ZIP audit trước finalization vẫn được giữ nguyên để truy xuất lịch sử; nó chưa chứa source/README/metadata/hình v2 và registry mới. Muốn gửi trạng thái finalized đi audit cần đóng gói lại sau khi nhiệm vụ này hoàn tất; tác vụ hiện tại không tự tạo ZIP mới.

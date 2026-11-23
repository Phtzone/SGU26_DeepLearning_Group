# Báo cáo CNN-1D trên WESAD 14 kênh ở 64 Hz

Group `20261123T072126Z_2c81daac` hoàn tất 5/5 fold. Macro-F1 trung bình trên 15 người đạt **0.2962 ± 0.1467** với độ lệch chuẩn mẫu. Mục tiêu là đo khả năng phân loại Baseline, Stress và Amusement trên những người chưa có trong Train.

## Dữ liệu và cửa sổ

Input lấy trực tiếp từ `data/processed/common_rate_64hz`, do [notebook 02](../../../notebooks/02_preprocessing_common_rate.ipynb) xuất. Có 15 mảng `float32` dạng `(N, 14)`, 8 kênh ngực và 6 kênh cổ tay theo thứ tự YAML. Fingerprint nguồn `9f2f09b7aee2ec94134a4734d7b9a1382e03936a49ecb9e8c5952e3b92760772`. Toàn bộ mảng được mở read-only bằng memory mapping. Runner dùng phần xuất 64 Hz có sẵn.

Đoạn nhãn 700 Hz có đầu phải loại trừ. Tại 64 Hz, lấy `ceil(start × 64/700)` đến `floor(end × 64/700)` theo khoảng nửa mở. Mỗi lưới cửa sổ được neo ở đầu vùng nhãn hợp lệ. Cửa sổ 30 giây, stride 15 giây, 1.920 mẫu, overlap 50%. Nhãn nguồn 1/2/3 thành 0/1/2. Mask `True` bắt buộc ở mọi mẫu của cửa sổ.

Có **2,134** cửa sổ hợp lệ: 1150 Baseline, 638 Stress và 346 Amusement. Mask loại 2 cửa sổ ứng viên. Các tệp nguồn, thời gian, định dạng và SHA256 có trong [kiểm tra dữ liệu](../../../runs/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac/window_index/common_rate_index.json) và [window manifest](../../../runs/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac/window_index/windows.csv).

| Người | Baseline | Stress | Amusement | Cửa sổ hợp lệ | Mẫu mask False | Cửa sổ bị loại |
| --- | --- | --- | --- | --- | --- | --- |
 | S2 | 75 | 39 | 23 | 137 | 0 | 0 | 
 | S3 | 74 | 39 | 23 | 136 | 265 | 2 | 
 | S4 | 76 | 41 | 23 | 140 | 0 | 0 | 
 | S5 | 78 | 41 | 23 | 142 | 0 | 0 | 
 | S6 | 77 | 42 | 23 | 142 | 0 | 0 | 
 | S7 | 78 | 41 | 23 | 142 | 0 | 0 | 
 | S8 | 76 | 43 | 23 | 142 | 0 | 0 | 
 | S9 | 77 | 41 | 23 | 141 | 0 | 0 | 
 | S10 | 77 | 47 | 23 | 147 | 0 | 0 | 
 | S11 | 77 | 44 | 23 | 144 | 0 | 0 | 
 | S13 | 77 | 43 | 24 | 144 | 0 | 0 | 
 | S14 | 77 | 43 | 23 | 143 | 0 | 0 | 
 | S15 | 77 | 44 | 23 | 144 | 0 | 0 | 
 | S16 | 77 | 43 | 23 | 143 | 0 | 0 | 
 | S17 | 77 | 47 | 23 | 147 | 0 | 0 | 

## Grouped 5-Fold và chuẩn hóa

Dùng [split manifest seed 42](../../../runs/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac/split_manifest.json) (`239949dfebabd802091d14a4494d2216aee9809b0e2993dacdbdfbcdc7133f2c`). Mỗi fold chứa 9 người Train, 3 Validation, 3 Test. Mọi người xuất hiện đúng một lần trong Test. Mean/std từng kênh tính từ các timestep cửa sổ Train, tính cả overlap, dùng `ddof=0`, `std < 1e-8` thay bằng 1. Validation/Test chỉ nhận phép biến đổi bằng scaler Train. Trọng số lớp là `N_train/(3*n_class_train)`.

| Fold | Train | Validation | Test |
| --- | --- | --- | --- |
 | 1 | S7, S4, S11, S5, S6, S14, S2, S3, S13 | S8, S17, S15 | S10, S16, S9 | 
 | 2 | S10, S16, S9, S5, S6, S14, S2, S3, S13 | S7, S4, S11 | S8, S17, S15 | 
 | 3 | S10, S16, S9, S8, S17, S15, S2, S3, S13 | S5, S6, S14 | S7, S4, S11 | 
 | 4 | S10, S16, S9, S8, S17, S15, S7, S4, S11 | S2, S3, S13 | S5, S6, S14 | 
 | 5 | S8, S17, S15, S7, S4, S11, S5, S6, S14 | S10, S16, S9 | S2, S3, S13 | 

## CNN-1D và huấn luyện

Model có 47,075 tham số trainable. Mỗi fold khởi tạo CNN và Adam từ seed 42, fit scaler riêng, đánh giá Validation sau mỗi epoch. Sau khi chọn best checkpoint, nạp weights của chính fold rồi đánh giá Test. 

| Lớp | Thiết lập | Shape đầu ra |
| --- | --- | --- |
 | Input | 14 kênh × 1.920 mẫu | [B, 14, 1920] | 
 | Khối 1 | Conv1d k=7 p=3, BatchNorm, ReLU, MaxPool 2 | [B, 32, 960] | 
 | Khối 2 | Conv1d k=5 p=2, BatchNorm, ReLU, MaxPool 2 | [B, 64, 480] | 
 | Khối 3 | Conv1d k=3 p=1, BatchNorm, ReLU | [B, 128, 480] | 
 | Global pooling | AdaptiveAvgPool1d(1), Flatten | [B, 128] | 
 | Head | Linear 128→64, ReLU, Dropout 0,3, Linear 64→3 | [B, 3] | 

[Effective config](../../../runs/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac/config.yaml) đặt Adam, batch 32, LR 0.001, weight decay 0.0001, weighted cross-entropy, gradient clip L2 1.0, tối đa 100 epochs. LR cố định. Chọn checkpoint bằng trung bình Macro-F1 của ba người Validation. Khi bằng trong `1e-12`, ưu tiên loss thấp hơn. Early stopping có patience 15 và min_delta 0.001. Tổng cộng 152 epochs, 6,175 mini-batches đã commit và thời lượng cộng dồn 1:02:48. Xem [môi trường](../../../runs/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac/system.json), [source snapshot](../../../runs/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac/source) và [huấn luyện từng fold](enhanced_visualizations/scientific-v2_a3d014ba1ef8351d/training/seed_42).

## Hiệu suất Test

Macro-F1 lấy trung bình F1 của nhãn cố định `[0,1,2]`. Balanced Accuracy lấy trung bình recall của ba nhãn. Lớp vắng mặt được gán điểm 0 khi tính trung bình theo lớp. Stress Recall là recall của nhãn 1. Mỗi người có một bộ dự đoán Test riêng.

| Fold | Cửa sổ T/V/Test | Epochs | Best | Val F1/người | Macro-F1 | Balanced Accuracy | Accuracy | Stress Recall | Checkpoint | Log |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
 | 1 | 1270/433/431 | 28 | 13 | 0.3595 | 0.2735 | 0.3824 | 0.3643 | 0.0000 | [best.pt](../../../runs/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac/seed_42/fold_01/checkpoints/best.pt) | [epoch log](../../../runs/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac/seed_42/fold_01/logs/epochs.csv) | 
 | 2 | 1275/426/433 | 29 | 14 | 0.3315 | 0.3567 | 0.3542 | 0.4734 | 0.4104 | [best.pt](../../../runs/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac/seed_42/fold_02/checkpoints/best.pt) | [epoch log](../../../runs/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac/seed_42/fold_02/logs/epochs.csv) | 
 | 3 | 1281/427/426 | 40 | 25 | 0.5367 | 0.3207 | 0.3372 | 0.3803 | 0.6825 | [best.pt](../../../runs/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac/seed_42/fold_03/checkpoints/best.pt) | [epoch log](../../../runs/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac/seed_42/fold_03/logs/epochs.csv) | 
 | 4 | 1290/417/427 | 23 | 8 | 0.6154 | 0.3777 | 0.4053 | 0.3653 | 0.3651 | [best.pt](../../../runs/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac/seed_42/fold_04/checkpoints/best.pt) | [epoch log](../../../runs/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac/seed_42/fold_04/logs/epochs.csv) | 
 | 5 | 1286/431/417 | 32 | 17 | 0.3731 | 0.4532 | 0.4527 | 0.5755 | 0.4463 | [best.pt](../../../runs/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac/seed_42/fold_05/checkpoints/best.pt) | [epoch log](../../../runs/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac/seed_42/fold_05/logs/epochs.csv) | 

| Cách tổng hợp | Macro-F1 | Balanced Accuracy | Accuracy | Stress Recall |
| --- | --- | --- | --- | --- |
 | Trung bình theo 15 người (chính) | 0.2962 ± 0.1467 | 0.3858 ± 0.1365 | 0.4325 ± 0.1733 | 0.3783 ± 0.4437 | 
 | Trung bình theo 5 fold | 0.3564 ± 0.0670 | 0.3864 ± 0.0453 | 0.4318 ± 0.0922 | 0.3809 ± 0.2457 | 
 | Pooled OOF | 0.3866 | 0.3847 | 0.4311 | 0.3777 | 

Các dòng trung bình dùng độ lệch chuẩn mẫu `ddof=1`. Pooled OOF tính chỉ số một lần trên toàn bộ dự đoán đã ghép. Có 2134 window ID duy nhất và 15 người Test. [OOF predictions](oof_predictions.csv) · [Summary JSON](summary.json).

| Fold | Subject | Windows | Macro-F1 | Balanced Accuracy | Accuracy | Stress Recall |
| --- | --- | --- | --- | --- | --- | --- |
 | 5 | S2 | 137 | 0.2358 | 0.3333 | 0.5474 | 0.0000 | 
 | 5 | S3 | 136 | 0.4991 | 0.4902 | 0.5809 | 0.2821 | 
 | 3 | S4 | 140 | 0.3021 | 0.3415 | 0.5500 | 0.0244 | 
 | 4 | S5 | 142 | 0.2963 | 0.3394 | 0.5211 | 0.1463 | 
 | 4 | S6 | 142 | 0.4518 | 0.6276 | 0.4507 | 0.9048 | 
 | 3 | S7 | 142 | 0.1494 | 0.3333 | 0.2887 | 1.0000 | 
 | 2 | S8 | 142 | 0.2971 | 0.3803 | 0.3944 | 0.9302 | 
 | 1 | S9 | 141 | 0.5624 | 0.6667 | 0.7092 | 0.0000 | 
 | 1 | S10 | 147 | 0.1266 | 0.1472 | 0.2313 | 0.0000 | 
 | 3 | S11 | 144 | 0.2529 | 0.3333 | 0.3056 | 1.0000 | 
 | 5 | S13 | 144 | 0.4346 | 0.5195 | 0.5972 | 1.0000 | 
 | 4 | S14 | 143 | 0.0976 | 0.2474 | 0.1259 | 0.0465 | 
 | 2 | S15 | 144 | 0.3842 | 0.3604 | 0.5000 | 0.3409 | 
 | 1 | S16 | 143 | 0.0924 | 0.3333 | 0.1608 | 0.0000 | 
 | 2 | S17 | 147 | 0.2606 | 0.3333 | 0.5238 | 0.0000 | 

## Lỗi dự đoán

| Nhãn thật / dự đoán | Baseline | Stress | Amusement |
| --- | --- | --- | --- |
 | Baseline | 587 | 177 | 386 | 
 | Stress | 234 | 241 | 163 | 
 | Amusement | 187 | 67 | 92 | 

| Lớp | Support | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
 | Baseline | 1150 | 0.5823 | 0.5104 | 0.5440 | 
 | Stress | 638 | 0.4969 | 0.3777 | 0.4292 | 
 | Amusement | 346 | 0.1435 | 0.2659 | 0.1864 | 

| Nhãn thật | Dự đoán sai thành | Cửa sổ | % trong lớp thật |
| --- | --- | --- | --- |
 | Baseline | Amusement | 386 | 33.6% | 
 | Stress | Baseline | 234 | 36.7% | 
 | Amusement | Baseline | 187 | 54.0% | 
 | Baseline | Stress | 177 | 15.4% | 

| Ba người có Macro-F1 thấp nhất | Fold | Windows | Macro-F1 | Stress Recall | Cặp nhầm nhiều nhất |
| --- | --- | --- | --- | --- | --- |
 | S16 | 1 | 143 | 0.0924 | 0.0000 | Baseline → Amusement (77 cửa sổ) | 
 | S14 | 4 | 143 | 0.0976 | 0.0465 | Baseline → Amusement (77 cửa sổ) | 
 | S10 | 1 | 147 | 0.1266 | 0.0000 | Stress → Baseline (45 cửa sổ) | 

Lỗi theo người phản ánh tập Test của fold tương ứng. Dữ liệu quan sát chỉ cho biết kiểu nhầm lẫn, chưa xác định được nguyên nhân ở chất lượng cảm biến, khác biệt cá nhân hay đặc trưng mô hình.

## Tracking và truy xuất

Dùng TrainingTracker, RunLogger, ConsoleLogger và TensorBoardTracker hiện có. Batch/epoch CSV, JSONL, best/last và completion hash giữ cùng quy ước. Bổ sung hai scalar theo tiêu chí Validation mới và metadata nguồn cho thí nghiệm Common-rate. Checkpoint resume lưu trạng thái optimizer và RNG ở ranh giới epoch. Các fold có completion hash hợp lệ được bỏ qua.

```powershell
python -m tensorboard.main --logdir runs/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac --port 6006
```

Mở http://localhost:6006. Kiểm chứng đã đọc 15 event files, 10 checkpoints và đối chiếu 268 file hình. [Bằng chứng](verification.json) · [Hình PNG/SVG](enhanced_visualizations/scientific-v2_a3d014ba1ef8351d) · [Research Journal](../../../experiments/common_rate64_cnn1d_grouped5fold_v1/Research_journal_common_rate64_cnn1d_grouped5fold_v1.xlsx).

## Giới hạn và nghiên cứu tiếp theo

Thí nghiệm dùng 15 người, một split seed và một training seed trong môi trường phòng thí nghiệm. Các cửa sổ chồng lấp trong một bản ghi có tương quan, nên độ lệch chuẩn giữa người phản ánh biến thiên quan sát được. Kết quả của nhánh 32 Hz khác cả đầu vào lẫn tiêu chí chọn checkpoint. So sánh CNN-1D, GRU và CNN-GRU cần cùng manifest, dữ liệu, tiêu chí lựa chọn và cách tổng hợp. Ablation tiếp theo cần giữ cùng window IDs và chỉ thay tập kênh trước khi fit scaler theo Train.

## Tài liệu tham khảo

- [Schmidt và cộng sự (2018) — Introducing WESAD](https://www.eti.uni-siegen.de/ubicomp/papers/ubi_icmi2018.pdf). Nguồn bộ dữ liệu. Khác cách trích đặc trưng và đơn vị chia fold.
- [Rashid và cộng sự (2021) — Feature Augmented Hybrid CNN](https://arxiv.org/abs/2108.03166). Khác số kênh, cửa sổ, đặc trưng và giao thức 9/3/3.
- [Gil-Martín và cộng sự (2022) — Human Stress Detection With Wearable Sensors Using CNNs](https://portalcientifico.upm.es/es/ipublic/item/9744363). Accuracy khác Macro-F1. Biểu diễn phổ và cách chia dữ liệu cũng khác.

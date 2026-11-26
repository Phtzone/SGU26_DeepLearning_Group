# Kiểm toán hậu thực nghiệm CNN-1D, WESAD 14 kênh ở 64 Hz

| | |
| --- | --- |
| Thí nghiệm | `common_rate64_cnn1d_grouped5fold_v1` |
| Group chính thức | `20261123T072126Z_2c81daac`, seed 42, hoàn thành 5/5 fold |
| Giao thức | 15 subject, 9 Train / 3 Validation / 3 Test mỗi fold, 30 s/cửa sổ, stride 15 s, 2.134 cửa sổ hợp lệ |
| Kết quả chính | Test subject-mean Macro-F1 **0,2962 ± 0,1467**, Balanced Accuracy **0,3858 ± 0,1365**, SD mẫu trên 15 subject |

## Phát hiện

**Z-score thực sự đến đầu vào CNN-1D.** Tự tính lại mean/std cho 14 kênh trên đúng Train ở từng fold cho kết quả khớp float32 đã lưu. Hậu kiểm 343 batch, tương ứng 10.670 lượt cửa sổ Train/Validation/Test qua 5 fold, xác nhận chính tensor `float32 [B,14,1920]` vào `CNN1D.forward()` trùng phép tính Z-score độc lập từng phần tử. Trên Train, mean gần 0 với sai số tối đa \(2,70\times10^{-6}\) và std gần 1 với sai số tối đa \(5,21\times10^{-8}\). Không phát hiện lỗi fit, transform, kênh, subject split hay dữ liệu Validation/Test đi vào scaler. [Kiểm toán normalization](NORMALIZATION_AUDIT.md) · [Thống kê chuẩn hóa](normalization_statistics.csv) · [Bằng chứng inference](replay_evidence.json).

**Windowing và đánh giá khớp hiện vật.** Tái dựng 2.134 cửa sổ, nhãn và vùng loại từ chuỗi 64 Hz. Hai cửa sổ Stress của S3 bị loại đúng mask. Cả 5 fold dùng Train-only class weights và cùng công thức weighted CE cho Train eval/Validation. Checkpoint được chọn theo Validation subject-mean Macro-F1, sau đó mới đánh giá Test. Tái tạo đủ 2.134 OOF nhãn và xác suất đã lưu bằng `best.pt`. Loss, confusion matrices, chỉ số theo fold/subject/pooled và 15 tệp TensorBoard khớp kiểm tra. [Kiểm toán training/evaluation](TRAINING_EVALUATION_AUDIT.md) · [Tái tạo metrics](checkpoint_replay.csv).

**Khoảng cách tổng quát hóa lớn ở cả 5 fold.** Train Macro-F1 đạt ít nhất 0,99 trong 2–4 epoch và bằng 1,000 ở epoch cuối. Validation loss của cả 5 fold chạm đáy tại epoch 1, tăng ở epoch 2 và còn dao động về sau. Inference cho thấy lỗi Validation/Test thường tự tin cao. Val subject-mean F1 tại best chỉ đạt 0,3315–0,6154. Train có 9 người mỗi fold và cửa sổ lặp nhiều nội dung. Dữ liệu khẳng định overfitting và mức tự tin cao khi sai. Cần đối chứng để xác định cơ chế CNN học mức nền theo người, capacity, regularization hay những yếu tố khác. [Phân tích overfitting và Test](OVERFITTING_ANALYSIS.md) · [Đường cong 5 fold](figures/overfitting_curves.svg).

**Dịch chuyển phân phối theo người được đo trực tiếp.** Điển hình `chest_Temp` ở Validation Fold 4 lệch −3,35 std Train, trong đó thành phần trong cùng lớp là −3,34 và thành phần do tỷ lệ lớp là −0,009. Cùng scaler, các subject có mức và độ phân tán rất khác nhau. Đây là bằng chứng có shift ở đầu vào. Mức ảnh hưởng của shift lên mỗi lỗi dự đoán cần đối chứng. [Bảng phân rã theo lớp](distribution_shift_decomposition.csv) · [Heatmap Test](figures/test_subject_channel_heatmap.svg) · [Boxplot Fold 1](figures/fold_01_subject_boxplots.svg).

**Test có các lỗi tập trung theo lớp và người.** Fold 1 nhận đúng 0/131 Stress và không dự đoán Stress cho cửa sổ nào. Pooled Amusement F1 là 0,1864, với 386 Baseline bị gán Amusement. Ba subject thấp nhất là S16 (0,0924), S14 (0,0976) và S10 (0,1266). Trong 1.214 dự đoán OOF sai, 1.006 có xác suất lớn nhất ≥ 0,90. [Bảng chẩn đoán Test](test_diagnostics.csv) · [Đồ thị Test](figures/test_predictions.png).

## Phân loại mức bằng chứng

| Mức | Phát hiện | Phạm vi hoặc việc cần làm |
| --- | --- | --- |
| **Confirmed issue** | S3 có đoạn tín hiệu bất thường xác nhận từ EDA, đã được mask loại khỏi cửa sổ. Khoảng cách tổng quát hóa, lỗi confidence cao và Stress Recall bằng 0 ở Fold 1 đều được xác nhận. | Tách lỗi chất lượng đã xử lý khỏi biểu hiện hiệu năng. Giữ S3 sau khi loại đúng hai cửa sổ. Chẩn đoán hiệu năng với dữ liệu và checkpoint gốc. |
| **Confirmed issue** | Byte của notebook 02 hiện tại khác SHA ghi trên run. Tìm được byte khớp trong Git, mã mọi cell vẫn giống. | Giữ hai SHA và commit trong evidence. Hai cell chỉ số 2 và 4 (đếm từ 0) có khác execution count hoặc outputs. Cả 33 file đầu vào xác minh khớp run. |
| **Verified correct** | Luồng 64 Hz, thứ tự 14 kênh, fit/transform từng fold, layout, nhãn, split, train/eval, chọn checkpoint, log, đường cong và OOF. | Chứng cứ kết hợp snapshot, độc lập tái tính số liệu và inference chỉ đọc. Shift trong cùng lớp và tương quan do overlap đã được đo. |
| **Suspected issue** | CNN có thể quá nhạy với offset/cảm biến hoặc dấu hiệu theo quy trình và người. | Phân tích mức lệch gắn với dự đoán theo người. Liên hệ mô tả hiện tại chưa chỉ ra quan hệ nhân quả. Đuôi lớn của BVP/EMG cần xem waveform và bối cảnh đo. |
| **Requires further experiment** | Mức đóng góp của từng kênh, overlap, capacity, class weights, LR và regularization. | Bắt đầu với một đối chứng giữ nguyên CNN-1D, chỉ center nhiệt độ ngực theo cửa sổ, chấm trên Validation theo subject. [Thiết kế kiểm chứng](RECOMMENDED_NEXT_EXPERIMENTS.md). |

## Nguồn đã đối chiếu

- Run snapshot có **61 tệp**. Cả 61 hash khớp source hiện tại và `source_checksums.json`. Effective config, split, index, checkpoint và predicted windows thuộc đúng group. Đã đối chiếu **31 tệp window index**, 33 tệp dataset và split checksum ghi trong bản chạy. [Manifest chính thức](../../../runs/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac/experiment_manifest.json) · [Source checksums](../../../runs/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac/source_checksums.json) · [Bằng chứng provenance](data_evidence.json).
- Đọc đề cương, chương dữ liệu, EDA và hướng dẫn codebase để đối chiếu thiết kế đầu vào và cách đánh giá. CNN-1D tiếp tục dùng ba block Conv1d, với convolution đầu nhận 14 kênh và input dài 1.920 mẫu từ nguồn common-rate. Giao thức thực thi được xác định theo manifest và snapshot. [project_DL.docx](../../../../project_DL.docx) · [Chương 3 hoàn chỉnh](../../../../Chuong_3_Bo_du_lieu_va_tien_xu_ly_Hoan_chinh.docx) · [EDA](../../../notebooks/preprocessing.ipynb) · [Notebook 02](../../../notebooks/02_preprocessing_common_rate.ipynb) · [Hướng dẫn codebase](../../../results/codebase_review/DL13-WESAD_Complete_Codebase_Learning_Guide.docx).
- So sánh ghi nhận thực nghiệm trong journal, báo cáo, verification chính thức với kết quả audit. [Research Journal gốc](../../../../Research_journal.xlsx) · [Journal của thí nghiệm](../../../experiments/common_rate64_cnn1d_grouped5fold_v1/Research_journal_common_rate64_cnn1d_grouped5fold_v1.xlsx) · [Báo cáo gốc](../../../results/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac/report.md) · [Verification gốc](../../../results/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac/verification.json).

## Bằng chứng và bảo toàn

Hai script hậu kiểm chạy với Python `.venv_new`, PyTorch 2.14.0+cpu và TensorBoard 2.21.0. `audit_data.py` tính lại toàn bộ thống kê từ mảng thật. `audit_replay.py` chỉ nạp checkpoint và chạy ở inference mode. Trong replay, toàn bộ parameters và BatchNorm buffers khớp bit trước/sau, 611 tệp được kiểm SHA-256 trước/sau và khớp. Kết quả của audit được ghi riêng trong thư mục này. [Script kiểm dữ liệu](audit_data.py) · [Script replay](audit_replay.py) · [Evidence JSON](replay_evidence.json).

Bộ hồ sơ gồm [kiểm toán normalization](NORMALIZATION_AUDIT.md), [kiểm toán training/evaluation](TRAINING_EVALUATION_AUDIT.md), [phân tích overfitting](OVERFITTING_ANALYSIS.md), [đề xuất đối chứng](RECOMMENDED_NEXT_EXPERIMENTS.md), [thống kê 210 hàng fold/tập/kênh](normalization_statistics.csv), [thống kê 4.200 hàng theo subject và lớp](subject_distribution_statistics.csv) và năm biểu đồ chẩn đoán với cả PNG và SVG. Cỡ mẫu diễn giải là 15 subject, dữ liệu trong nhiều cửa sổ có chồng lấp. Suy luận từ run này được giới hạn ở cohort, pipeline và seed đã kiểm.

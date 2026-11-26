# CNN-1D 64 Hz với Median Centering nhiệt độ ngực

**Experiment** `common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1` · **Training group** `20261126T062337Z_7760c683` · **Preprocessing group** `20261126T025058Z_fa1c92ea`.

**Hoàn tất 5/5 fold, 143 epoch và 5821 mini-batch đã được ghi vào checkpoint.** Mục tiêu là kiểm tra tác động của việc trừ median riêng kênh `chest_Temp` đến tổng quát hóa theo subject. So với baseline `20261123T072126Z_2c81daac`, điểm Validation subject-mean Macro-F1 tại checkpoint được chọn trung bình qua 5 fold là **0,4433 → 0,4181 (-0,0252)**. Test subject-mean Macro-F1 là **0,2962 → 0,3918 (+0,0956)**. Đây là đối chứng mô tả trên 15 người và một seed.

## Tiền xử lý, thiết lập và kiểm soát biến

Nguồn Common-rate 64 Hz có 14 kênh theo thứ tự đã lưu, 2.134 cửa sổ 1.920 mẫu, stride 960 mẫu. Giữ nguyên 15 subject, 9/3/3 trong mỗi fold, mask và nhãn 1/2/3 → 0/1/2. Mỗi subject được Test đúng một lần. Trừ median của 1.920 mẫu trong từng cửa sổ ở kênh `chest_Temp` (index 6). Fit Z-score 14 kênh trên cửa sổ Train đã center và sử dụng scaler đó cho cả ba tập. Cả 5 scaler dùng khi chạy trùng byte với group tiền xử lý. Mean/std của 13 kênh khác khớp baseline. Các cửa sổ cực trị được giữ theo mask hiện có. [Báo cáo tiền xử lý](../../20261126T025058Z_fa1c92ea/preprocessing_report.md) · [Kiểm toán cực trị](../../20261126T025058Z_fa1c92ea/extreme_value_audit/EXTREME_VALUES_AUDIT.md).

| Điều kiện | Baseline và lần chạy này |
| --- | --- |
| Kiến trúc | Cùng CNN-1D, 47.075 trainable parameters, tensor float32 `[B,14,1920]` |
| Môi trường | CPU, 2 threads PyTorch, cùng phiên bản Python/PyTorch/TensorBoard và các gói được chụp |
| Khởi tạo | Seed 42 cho từng fold, model và Adam khởi tạo mới. Các checkpoint dùng cùng GROUP_ID của lần chạy |
| Loss và optimizer | Cross-entropy trên logits có weight theo nghịch đảo tần suất lớp Train, Adam, learning rate 0.001, batch 32 |
| Regularization | Dropout 0.3, weight decay 0.0001, gradient norm limit 1, scheduler `None` |
| Dừng và chọn best | Tối đa 100 epoch. Patience 15, min delta 0.001. Tối đa Validation subject-mean Macro-F1, hòa thì dùng Validation loss thấp hơn |
| Test | Mỗi fold nạp best đã chọn. Replay xác suất từ best khớp các dự đoán OOF đã lưu |

Effective config chỉ khác định danh experiment và tùy chọn centering được chủ động bật. SHA của source snapshot baseline khác tại 8 đường dẫn. Trong code thực thi, khác biệt ở Dataset/Scaler và cách gắn vào trainer là median centering theo cửa sổ. Stress Recall được ghi thêm từ metric đang có sau mỗi epoch. Các khác biệt kỹ thuật còn lại là kiểm tra config, đọc lịch sử log và tùy chọn kiểm tra reference trong runner. Các module CNN, optimizer, seed, early stopping, evaluator, metrics và aggregation khớp bản baseline trong đối chiếu trực tiếp. [Bảng effective config](effective_config_comparison.csv) · [Preflight và danh sách source khác](../../../../runs/common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1/20261126T062337Z_7760c683/preflight_comparison.json) · [Source snapshot](../../../../runs/common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1/20261126T062337Z_7760c683/source_checksums.json).

## Kết quả theo fold

Các cặp trong bảng theo thứ tự **Baseline → Median Centering**. Điểm Validation là mean Macro-F1 theo subject tại best epoch. Các chỉ số Test là pooled trong từng fold.

| Fold | Test subjects | Epochs | Best epoch | Val subject-mean F1 | Test Macro-F1 | Test BA | Test Accuracy | Test Stress Recall |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | S10 S16 S9 | 28 → 24 | 13 → 9 | 0,3595 → 0,3110 | 0,2735 → 0,3294 | 0,3824 → 0,4329 | 0,3643 → 0,4455 | 0,0000 → 0,0000 |
| 2 | S8 S17 S15 | 29 → 18 | 14 → 3 | 0,3315 → 0,3686 | 0,3567 → 0,5848 | 0,3542 → 0,6067 | 0,4734 → 0,6212 | 0,4104 → 0,4403 |
| 3 | S7 S4 S11 | 40 → 26 | 25 → 11 | 0,5367 → 0,5073 | 0,3207 → 0,4398 | 0,3372 → 0,4901 | 0,3803 → 0,5939 | 0,6825 → 0,8254 |
| 4 | S5 S6 S14 | 23 → 23 | 8 → 8 | 0,6154 → 0,4868 | 0,3777 → 0,3753 | 0,4053 → 0,3702 | 0,3653 → 0,3911 | 0,3651 → 0,3651 |
| 5 | S2 S3 S13 | 32 → 52 | 17 → 37 | 0,3731 → 0,4165 | 0,4532 → 0,4719 | 0,4527 → 0,4826 | 0,5755 → 0,5276 | 0,4463 → 0,7025 |

[Bảng ghép cặp theo fold và các subject từng split](paired_fold_comparison.csv) · [Kết quả fold gốc](../fold_results.csv).

| Fold | Validation pooled F1 tại best | Validation BA | Validation Accuracy | Validation loss | Patience counter khi dừng |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0,3691 | 0,3443 | 0,4365 | 5,774 | 15 |
| 2 | 0,4484 | 0,4603 | 0,5164 | 2,414 | 15 |
| 3 | 0,5425 | 0,5646 | 0,6300 | 3,738 | 15 |
| 4 | 0,5869 | 0,5751 | 0,6499 | 5,019 | 15 |
| 5 | 0,4342 | 0,4359 | 0,5104 | 5,516 | 15 |

## Ba mức tổng hợp trên Test

Trung bình đều theo subject hoặc theo fold đi kèm SD mẫu. Pooled OOF tính một lần từ tất cả cửa sổ. Macro Recall trùng Balanced Accuracy do cùng lấy trung bình recall của ba nhãn cố định. Quy tắc `zero_division=0` và cách tính lớp vắng mặt giữ như baseline.

| Đơn vị | Metric | Baseline | Median Centering | Δ |
| --- | --- | ---: | ---: | ---: |
| Subject (15) | Macro-F1 | 0,2962 ± 0,1467 | 0,3918 ± 0,1783 | +0,0956 |
| Subject (15) | Balanced Accuracy | 0,3858 ± 0,1365 | 0,4765 ± 0,1770 | +0,0908 |
| Subject (15) | Accuracy | 0,4325 ± 0,1733 | 0,5159 ± 0,2034 | +0,0834 |
| Subject (15) | Stress Recall | 0,3783 ± 0,4437 | 0,4657 ± 0,4094 | +0,0873 |
| Subject (15) | Macro Precision | 0,3377 ± 0,1922 | 0,4355 ± 0,1924 | +0,0979 |
| Subject (15) | Macro Recall | 0,3858 ± 0,1365 | 0,4765 ± 0,1770 | +0,0908 |
| Fold (5) | Macro-F1 | 0,3564 ± 0,0670 | 0,4402 ± 0,0980 | +0,0839 |
| Fold (5) | Balanced Accuracy | 0,3864 ± 0,0453 | 0,4765 ± 0,0871 | +0,0901 |
| Fold (5) | Accuracy | 0,4318 ± 0,0922 | 0,5159 ± 0,0973 | +0,0841 |
| Fold (5) | Stress Recall | 0,3809 ± 0,2457 | 0,4667 ± 0,3214 | +0,0858 |
| Fold (5) | Macro Precision | 0,3942 ± 0,1021 | 0,4538 ± 0,1313 | +0,0597 |
| Fold (5) | Macro Recall | 0,3864 ± 0,0453 | 0,4765 ± 0,0871 | +0,0901 |
| Pooled OOF (2.134 cửa sổ) | Macro-F1 | 0,3866 | 0,4697 | +0,0831 |
| Pooled OOF (2.134 cửa sổ) | Balanced Accuracy | 0,3847 | 0,4744 | +0,0897 |
| Pooled OOF (2.134 cửa sổ) | Accuracy | 0,4311 | 0,5159 | +0,0848 |
| Pooled OOF (2.134 cửa sổ) | Stress Recall | 0,3777 | 0,4608 | +0,0831 |
| Pooled OOF (2.134 cửa sổ) | Macro Precision | 0,4076 | 0,4827 | +0,0751 |
| Pooled OOF (2.134 cửa sổ) | Macro Recall | 0,3847 | 0,4744 | +0,0897 |

[Bảng metrics đủ ba mức](metric_aggregation.csv) · [Bảng precision/recall/F1 và support theo lớp](class_metric_aggregation.csv) · [OOF predictions](../oof_predictions.csv).

### Chênh lệch ghép cặp theo subject

Macro-F1 tăng ở **11** subject, giảm ở **3** và giữ nguyên ở **1**. Trong nhóm có kết quả baseline thấp, Macro-F1 thay đổi ở S16 (0,0924 → 0,0924, +0,0000), S14 (0,0976 → 0,1562, +0,0587), S10 (0,1266 → 0,2171, +0,0905). Cột `stress_support` trong CSV cho biết số cửa sổ Stress dùng để tính recall từng subject.

| Subject | Fold | Cửa sổ | F1 baseline | F1 mới | Δ F1 | Δ BA | Δ Accuracy | Δ Stress Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S2 | 5 | 137 | 0,2358 | 0,3139 | +0,0781 | +0,0383 | +0,0292 | +0,1282 |
| S3 | 5 | 136 | 0,4991 | 0,4434 | -0,0557 | +0,0794 | -0,1397 | +0,6667 |
| S4 | 3 | 140 | 0,3021 | 0,5006 | +0,1985 | +0,1463 | +0,1286 | +0,4390 |
| S5 | 4 | 142 | 0,2963 | 0,2490 | -0,0473 | -0,0590 | -0,0845 | -0,0488 |
| S6 | 4 | 142 | 0,4518 | 0,6386 | +0,1867 | +0,0830 | +0,1831 | -0,2619 |
| S7 | 3 | 142 | 0,1494 | 0,1586 | +0,0093 | +0,0043 | +0,0070 | +0,0000 |
| S8 | 2 | 142 | 0,2971 | 0,4963 | +0,1992 | +0,1448 | +0,3028 | -0,3023 |
| S9 | 1 | 141 | 0,5624 | 0,5712 | +0,0089 | -0,0087 | -0,0142 | +0,0000 |
| S10 | 1 | 147 | 0,1266 | 0,2171 | +0,0905 | +0,1602 | +0,2517 | +0,0000 |
| S11 | 3 | 144 | 0,2529 | 0,5779 | +0,3251 | +0,3117 | +0,5000 | +0,0000 |
| S13 | 5 | 144 | 0,4346 | 0,4141 | -0,0204 | -0,0216 | -0,0347 | +0,0000 |
| S14 | 4 | 143 | 0,0976 | 0,1562 | +0,0587 | -0,1311 | -0,0210 | +0,3023 |
| S15 | 2 | 144 | 0,3842 | 0,5170 | +0,1328 | +0,2803 | -0,0139 | +0,3864 |
| S16 | 1 | 143 | 0,0924 | 0,0924 | +0,0000 | +0,0000 | +0,0000 | +0,0000 |
| S17 | 2 | 147 | 0,2606 | 0,5301 | +0,2695 | +0,3333 | +0,1565 | +0,0000 |

[Bảng đầy đủ theo subject](paired_subject_comparison.csv) · [Biểu đồ PNG](comparison_figures/paired_subject_differences.png) · [SVG](comparison_figures/paired_subject_differences.svg).

## Learning curves và khoảng cách tổng quát hóa

Gap ở đây là Train Macro-F1 trừ Validation pooled Macro-F1 tại cùng epoch. Tại best checkpoint, gap trung bình là **0,5017 → 0,5170 (+0,0153)**, thu hẹp ở 2/5 fold. Validation subject-mean F1 tăng ở 2/5 fold. Train F1 vẫn đạt ít nhất 0,99 ở 5/5 fold và Validation loss tăng từ epoch 1 sang 2 ở 5/5 fold. Các checkpoint best của hai phương pháp có thể ở epoch khác nhau.

| Fold | Epoch đầu Train F1 ≥ 0,99, cũ → mới | Epoch min Val loss, cũ → mới | Val loss epoch 1 → 2 cũ | Val loss epoch 1 → 2 mới | Gap tại best, cũ → mới | Gap khi dừng, cũ → mới |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 → 6 | 1 → 1 | 2,302 → 3,779 | 2,784 → 3,917 | 0,6034 → 0,6289 | 0,7599 → 0,7193 |
| 2 | 2 → 3 | 1 → 1 | 2,161 → 3,616 | 1,620 → 2,456 | 0,5876 → 0,5440 | 0,6290 → 0,5910 |
| 3 | 3 → 4 | 1 → 1 | 1,218 → 3,357 | 1,279 → 1,999 | 0,3857 → 0,4354 | 0,7999 → 0,5244 |
| 4 | 2 → 2 | 1 → 1 | 2,466 → 3,478 | 1,917 → 2,996 | 0,3400 → 0,4131 | 0,4034 → 0,5151 |
| 5 | 3 → 3 | 1 → 1 | 2,091 → 2,888 | 2,282 → 2,821 | 0,5916 → 0,5637 | 0,6480 → 0,6181 |

Learning rate giữ cố định 0.001. [Chi tiết loss, F1, early stopping và dao động Validation](learning_diagnostics.csv) · [Biểu đồ so sánh learning curves](comparison_figures/learning_curves_comparison.png) · [SVG](comparison_figures/learning_curves_comparison.svg).

## Sai số theo lớp và nhóm nhiệt độ đuôi

| Lớp | Support | Precision, cũ → mới | Recall, cũ → mới | F1, cũ → mới |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1150 | 0,5823 → 0,6469 | 0,5104 → 0,5896 | 0,5440 → 0,6169 |
| stress | 638 | 0,4969 → 0,5776 | 0,3777 → 0,4608 | 0,4292 → 0,5126 |
| amusement | 346 | 0,1435 → 0,2236 | 0,2659 → 0,3728 | 0,1864 → 0,2795 |

- **Baseline**. Số dự đoán theo thứ tự Baseline/Stress/Amusement là [1008, 485, 641]. Ba nhầm lẫn thường gặp nhất là baseline → amusement (386), stress → baseline (234), amusement → baseline (187). Confidence trung bình của 1214 cửa sổ sai là 0,9416, với 1006 cửa sổ có max softmax ≥ 0,90.
- **Median Centering + Z-score**. Số dự đoán theo thứ tự Baseline/Stress/Amusement là [1048, 509, 577]. Ba nhầm lẫn thường gặp nhất là baseline → amusement (321), stress → baseline (217), amusement → baseline (153). Confidence trung bình của 1033 cửa sổ sai là 0,9008, với 709 cửa sổ có max softmax ≥ 0,90.

Fold 1 có Test Stress Recall **0,0000 → 0,0000**. [Ma trận OOF chuẩn hóa và hiệu](comparison_figures/pooled_confusion_comparison.png) · [Ma trận OOF dạng đếm](../aggregate_confusion_matrix.csv) · [Bảng confidence](oof_confidence_comparison.csv) · [Biểu đồ confidence](comparison_figures/oof_confidence_comparison.png).

Các mức đuôi dưới đây dùng max trị tuyệt đối Z của kênh nhiệt độ trong mỗi cửa sổ Test, với scaler Train đúng fold. Bảng ghép hai dự đoán trên cùng cửa sổ. Nhóm trên 50 nằm trong nhóm trên 10.

| Mức | Cửa sổ | Chủ thể | Lỗi baseline | Lỗi Median Centering |
| --- | ---: | ---: | ---: | ---: |
| &#124;Z&#124;max <= 10 | 2110 | 15 | 1205 | 1027 |
| &#124;Z&#124;max > 10 | 24 | 3 | 9 | 6 |
| &#124;Z&#124;max > 50 | 6 | 1 | 0 | 1 |

`S2_84900_86820` thuộc **Test Fold 5**, nhãn **baseline**, max trị tuyệt đối Z trong lượt đánh giá **122,28**. Baseline dự đoán **baseline** (confidence 1,0000), Median Centering dự đoán **baseline** (1,0000). Giá trị 133,69 của audit dùng scaler Fold 4 khi cửa sổ ở Validation. Số lượng cửa sổ đuôi phụ thuộc subject/lớp và có overlap. Kết quả dự đoán tại đây chưa xác lập ảnh hưởng nhân quả của riêng các cực trị. [Số liệu từng cửa sổ](oof_temperature_context.csv).

## Xác minh và giới hạn

- 31 tệp window index khớp baseline. 624 tệp dữ liệu, baseline, preprocessing và journal XLSX khớp fingerprint trước chạy. Không có subject giao nhau trong từng fold. Class weights khớp baseline và được tính lại từ Train.
- 10 checkpoint, 15 TensorBoard event files và 134 hình scientific-v2 ở cả PNG/SVG đã được kiểm. 2.134 OOF windows khớp nhãn, subject và fold của baseline. Replay từ best đối chiếu cả xác suất. [Verification chuẩn](../verification.json) · [Verification đối chứng](training_verification.json).
- 0 lần resume. Các batch được kiểm có loss, learning rate, norm gradient và thời gian hữu hạn. Max norm trước clipping là 84,959, có 2603 batch vượt ngưỡng 1,0, sử dụng gradient clipping đã có trong cấu hình. Chi tiết lỗi kỹ thuật nằm ở verification và Research Journal.
- Validation được dùng để chọn best. Kết quả Test baseline đã được quan sát khi hình thành giả thuyết centering. So sánh hiện tại mang tính khám phá, trên cùng 15 người, một seed, và có cửa sổ chồng lấp. Các chênh lệch được báo cáo mô tả, chưa kiểm định ý nghĩa thống kê.

[Runs, logs và checkpoints](../../../../runs/common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1/20261126T062337Z_7760c683) · [Danh mục hình scientific-v2](../enhanced_visualizations/generation_manifest.json) · [Research Journal CSV](../../../../experiments/common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1/research_journal/experiments.csv).

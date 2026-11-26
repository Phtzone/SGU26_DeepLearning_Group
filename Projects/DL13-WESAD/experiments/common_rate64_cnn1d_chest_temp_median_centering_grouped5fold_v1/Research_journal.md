# Research Journal

## 1. Thí nghiệm

|  | Thí Nghiệm | Ngày chạy | Người làm | Nội dung | Kết quả | code/notebook | log | Trạng Thái |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | `common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1` · `20261126T062337Z_7760c683` | 26/11/2026 |  | Đánh giá khả năng tổng quát hóa theo subject của CNN-1D trên WESAD 64 Hz, 14 kênh, cửa sổ 30 giây, bước 15 giây. Grouped 5-fold (9/3/3), seed 42. Trừ median từng cửa sổ của `chest_Temp`, rồi fit Z-score trên Train của từng fold. Chọn best theo Validation subject-mean Macro-F1. | Test trên 15 subject, trung bình ± SD mẫu. Macro-F1 0,3918 ± 0,1783 (+0,0956 so với baseline), Balanced Accuracy 0,4765 ± 0,1770, Accuracy 0,5159 ± 0,2034 và Stress Recall 0,4657 ± 0,4094. | [Cấu hình thực thi](../../runs/common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1/20261126T062337Z_7760c683/config.yaml) · [Runner tại lúc chạy](../../runs/common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1/20261126T062337Z_7760c683/source/scripts/run_grouped_cv.py) | [Log nhóm](../../runs/common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1/20261126T062337Z_7760c683/logs/training.log) · [Báo cáo kết quả](../../results/common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1/20261126T062337Z_7760c683/comparison/training_report.md) | Hoàn thành Train, Validation và Test 5/5 fold |

## 2. Hình

|  | Thí nghiệm | mô tả ảnh | Dán ảnh tại đây |
| --- | --- | --- | --- |
|  | `20261126T062337Z_7760c683` | Nhiệt độ trước và sau centering, cực trị S2, đường cong Loss/F1 Train–Validation, chênh lệch ghép cặp theo subject và ma trận nhầm lẫn OOF. | [Phân phối nhiệt](../../results/common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1/20261126T025058Z_fa1c92ea/figures/temperature_shift_by_fold.svg) · [Tín hiệu S2](../../results/common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1/20261126T025058Z_fa1c92ea/extreme_value_audit/figures/s2_window_three_stages.svg) · [Learning curves](../../results/common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1/20261126T062337Z_7760c683/comparison/comparison_figures/learning_curves_comparison.svg) · [Chênh lệch theo subject](../../results/common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1/20261126T062337Z_7760c683/comparison/comparison_figures/paired_subject_differences.svg) · [Ma trận OOF](../../results/common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1/20261126T062337Z_7760c683/comparison/comparison_figures/pooled_confusion_comparison.svg) · [Danh mục scientific-v2](../../results/common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1/20261126T062337Z_7760c683/enhanced_visualizations/generation_manifest.json) |

## 3. Nhật ký

|  | Ngày | người làm | Thí nghiệm | đã làm | kết quả | Minh chứng | làm tiếp |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | 26/11/2026 |  | `20261126T062337Z_7760c683` · Tiền xử lý `20261126T025058Z_fa1c92ea` | Kiểm chứng tiền xử lý, chạy 143 epoch và 5.821 mini-batch, đánh giá trên Test rồi tái lập đủ 2.134 dự đoán OOF từ checkpoint best. | Test Macro-F1 tăng ở 11/15 subject. Validation subject-mean Macro-F1 giảm từ 0,4433 xuống 0,4181. Gap Train–Validation pooled F1 tại best tăng từ 0,5017 lên 0,5170. Overfitting vẫn hiện diện và Stress Recall Fold 1 vẫn bằng 0. | [Kiểm chứng tiền xử lý](../../results/common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1/20261126T025058Z_fa1c92ea/preprocessing_verification.json) · [Kiểm chứng đối chứng](../../results/common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1/20261126T062337Z_7760c683/comparison/training_verification.json) · [OOF đã lưu](../../results/common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1/20261126T062337Z_7760c683/oof_predictions.csv) | Rà soát sai số theo subject và lớp. |

## 4. Debug

|  | Ngày | Thí nghiệm | Lỗi | nguyên nhân | cách sửa | kiểm tra lại | Log |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | 26/11/2026 | `20261126T062337Z_7760c683` | `ImportError` khi preflight. Z = −133,69 tại cửa sổ `S2_84900_86820` ở Validation Fold 4. | Lệnh import dùng `RUN` thay vì `BASE_RUN`. Std Train của `chest_Temp` giảm từ 0,6514 xuống 0,0215 °C sau centering, làm tăng biên độ tính theo Z-score. | Đã sửa lệnh import và chạy lại preflight trước khi huấn luyện. Giữ cửa sổ S2 theo quality mask hiện có. | Preflight đạt. Loss và gradient norm trong log đều hữu hạn, 0 lần resume. Nguyên nhân biến thiên nhiệt ở S2 chưa xác định. | [Lỗi preflight](../../runs/common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1/20261126T062337Z_7760c683/preflight_retry.json) · [Kiểm toán cực trị](../../results/common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1/20261126T025058Z_fa1c92ea/extreme_value_audit/EXTREME_VALUES_AUDIT.md) |

## 5. Tham khảo

|  | Nghiên cứu | Bài toán / đầu vào | Cách làm | Đánh giá | Kết quả tiêu biểu | Điểm cần lưu ý khi so với nhóm |
| --- | --- | --- | --- | --- | --- | --- |
|  | [Kiểm toán baseline](../../docs/audits/common_rate64_cnn1d_grouped5fold_v1/AUDIT_REPORT.md) · [Thiết kế đối chứng](../../docs/audits/common_rate64_cnn1d_grouped5fold_v1/RECOMMENDED_NEXT_EXPERIMENTS.md) · [Chương 3 · Dữ liệu và tiền xử lý](../../../Chuong_3_Bo_du_lieu_va_tien_xu_ly_Hoan_chinh.docx) | WESAD Common-rate 64 Hz và lệch mức nền nhiệt độ ngực giữa subject. | Giữ giao thức dữ liệu và chọn checkpoint của baseline. Thí nghiệm này fit lại mean/std trên Train sau centering. | Dùng Validation để đánh giá hướng cải thiện và Test OOF để báo cáo kết quả. | Audit baseline ghi nhận mức nền `chest_Temp` lệch −3,35 std Train ở Validation Fold 4, làm cơ sở kiểm chứng centering. | Đề xuất ban đầu dùng scale Train trước centering. So sánh hiện tại mang tính khám phá trên 15 subject và một seed. Test baseline đã được quan sát khi hình thành giả thuyết. |

## 6. Việc cần làm

|  | vịêc cần làm | người làm | vị trí | hoàn tất khi | trạng thái |
| --- | --- | --- | --- | --- | --- |
|  | Phân tích lỗi Stress theo subject, kiểm tra nguồn cực trị S2 và thiết kế phép kiểm tiếp theo về tổng quát hóa. |  | [So sánh từng subject](../../results/common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1/20261126T062337Z_7760c683/comparison/paired_subject_comparison.csv) · [Phân tích OOF theo nhiệt độ](../../results/common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1/20261126T062337Z_7760c683/comparison/oof_temperature_context.csv) · [Chẩn đoán Train–Validation](../../results/common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1/20261126T062337Z_7760c683/comparison/learning_diagnostics.csv) | Có phân tích sai số và bằng chứng về cực trị. Tiêu chí Macro-F1 theo subject và Stress Recall trên Validation được chốt trước khi thử tiếp. | Chờ xem xét |

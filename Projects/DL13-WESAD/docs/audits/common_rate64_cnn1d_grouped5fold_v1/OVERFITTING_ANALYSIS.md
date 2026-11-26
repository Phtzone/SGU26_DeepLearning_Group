# Khoảng cách tổng quát hóa của CNN-1D

Phân tích chỉ dùng run `20261123T072126Z_2c81daac`. Loss trong bảng là weighted CE ở chế độ eval, tính tại cuối epoch. Ngưỡng Train F1 ≥ 0,99 được dùng để mô tả thời điểm mô hình khớp dữ liệu Train. [CSV tính từ lịch sử](overfitting_by_fold.csv) · [Hình loss và Macro-F1 của 5 fold](figures/overfitting_curves.svg).

## Diễn biến theo fold

| Fold | Train windows | Epoch đầu Train F1 ≥ 0,99 | Best / last epoch | Train F1 tại best | Val pooled F1 tại best | Val subject-mean F1 tại best | Val CE epoch 1 → best → last | Pooled F1 gap tại best → last |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 1.270 | 4 | 13 / 28 | 0,9941 | 0,3907 | 0,3595 | 2,302 → 7,040 → 8,003 | 0,603 → 0,760 |
| 2 | 1.275 | 2 | 14 / 29 | 1,0000 | 0,4124 | 0,3315 | 2,161 → 5,589 → 5,726 | 0,588 → 0,629 |
| 3 | 1.281 | 3 | 25 / 40 | 1,0000 | 0,6143 | 0,5367 | 1,218 → 4,467 → 7,346 | 0,386 → 0,800 |
| 4 | 1.290 | 2 | 8 / 23 | 1,0000 | 0,6600 | 0,6154 | 2,466 → 5,202 → 6,801 | 0,340 → 0,403 |
| 5 | 1.286 | 3 | 17 / 32 | 1,0000 | 0,4084 | 0,3731 | 2,091 → 6,008 → 8,617 | 0,592 → 0,648 |

**Overfitting quan sát được ở cả 5 fold.** Validation CE thấp nhất tại epoch 1, tăng lần đầu ở epoch 2 và dao động về sau. Train F1 ở last bằng 1,000 ở cả năm fold. Train eval CE ở last nằm trong 0,000025–0,001118 và Validation F1 theo subject thấp hơn best ở cả năm fold. LR ở mọi batch và epoch là 0,001. Dấu hiệu này tồn tại trong cùng cách tính loss và trong inference từ checkpoint. [Đối chiếu best/last](checkpoint_replay.csv) · [Kiểm toán cách ghi loss](TRAINING_EVALUATION_AUDIT.md).

Loss và F1 trả lời hai câu khác nhau. F1 dùng lớp có xác suất cao nhất. CE phạt khi xác suất của lớp thật thấp, đặc biệt với nhãn có trọng số cao. Vì vậy F1 có thể lên trong khi CE cũng lên. Qua 10 điểm Validation best/last, các dự đoán sai đóng góp khoảng 98,8% hoặc hơn của weighted CE. Fold 3 tại best có 133 lỗi, xác suất lớp dự đoán trung bình trên lỗi là 0,936. Tại last có 302 lỗi với trung bình 0,913, và phần CE do lỗi tăng từ 4,412 lên 7,323. Fold 4 có cả số lỗi tăng (104 → 135) và xác suất trung bình trên lỗi tăng (0,853 → 0,972). [Logits tái tạo và đóng góp CE](checkpoint_replay.csv) · [Guo và cộng sự, nghiên cứu hiệu chuẩn](https://proceedings.mlr.press/v70/guo17a.html).

Tập Train có khoảng 1.270–1.290 cửa sổ từ **9 người**. Trong toàn bộ benchmark, 48,92% lượt timestep lặp lại do cách dựng cửa sổ. CNN có 47.075 tham số và đầu pooling dồn đặc trưng cục bộ theo thời gian. Những điều kiện này giúp giải thích vì sao mô hình có thể đạt kết quả gần hoàn hảo trên các subject từng gặp. Ảnh hưởng của từng điều kiện đến khoảng cách với người chưa gặp cần đối chứng. [Định nghĩa CNN](../../../src/models/cnn.py#L5) · [Window overlap](window_overlap.csv).

## Test từ checkpoint đã chọn

| Cách tổng hợp | Macro-F1 | Balanced Accuracy | Accuracy |
| --- | ---: | ---: | ---: |
| Trung bình đều 15 subject, ± SD mẫu | 0,2962 ± 0,1467 | 0,3858 ± 0,1365 | 0,4325 ± 0,1733 |
| Trung bình 5 fold, ± SD mẫu | 0,3564 ± 0,0670 | 0,3864 ± 0,0453 | 0,4318 ± 0,0922 |
| Pooled 2.134 OOF windows | 0,3866 | 0,3847 | 0,4311 |

Chỉ số chính của run là subject-mean. Confusion matrix pooled ở dưới, với hàng là nhãn thật và cột là nhãn dự đoán. [OOF gốc](../../../results/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac/oof_predictions.csv) · [CSV hậu kiểm theo fold và người](test_diagnostics.csv).

| Nhãn thật | Baseline | Stress | Amusement | Tổng nhãn thật |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 587 | 177 | 386 | 1.150 |
| Stress | 234 | 241 | 163 | 638 |
| Amusement | 187 | 67 | 92 | 346 |
| Tổng dự đoán | 1.008 | 485 | 641 | 2.134 |

- **Fold 1.** Stress Recall = **0/131**. Model dự đoán 213 Baseline, 0 Stress và 218 Amusement. Trong 131 Stress thật, 79 thành Baseline và 52 thành Amusement. Tình trạng không có dự đoán Stress vẫn xuất hiện khi nạp đúng `best.pt`. [Fold 1 metrics](../../../runs/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac/seed_42/fold_01/metrics/test.json) · [Sai khác replay](replay_evidence.json).
- **Amusement.** Pooled precision 0,1435 (92/641), recall 0,2659 (92/346), F1 0,1864. Baseline → Amusement là lỗi ngoài đường chéo lớn nhất, 386/1.150 (33,6%). Cơ cấu dự đoán khác nhau theo fold. Số cửa sổ dự đoán Stress lần lượt 0, 105, 215, 62, 103. [Chẩn đoán Test](test_diagnostics.csv).
- **Confidence.** OOF có 920 đúng và 1.214 sai. Trung bình max softmax tương ứng 0,962 và 0,942. Có **1.006/1.214** lỗi (82,9%) với max softmax ≥ 0,90. Đó là tự tin cao khi sai, đo trên cửa sổ. Khi báo độ bất định cần xét phụ thuộc giữa cửa sổ của cùng người. [Đồ thị confidence và Macro-F1 theo subject](figures/test_predictions.png).

### Các subject có Macro-F1 thấp

| Subject, fold | Macro-F1 | Hành vi dự đoán | Ví dụ phân phối đầu vào so với Train |
| --- | ---: | --- | --- |
| S16, 1 | 0,0924 | Cả 143 cửa sổ thành Amusement, gồm 77 Baseline và 43 Stress. | `wrist_TEMP` mean \(z=-2,73\), 81,2% lượt timestep ngoài P01–P99 Train. |
| S14, 4 | 0,0976 | 134/143 được dự đoán Amusement, gồm toàn bộ 77 Baseline. Có 2/43 Stress đúng. | `wrist_ACC_X` mean \(z=-1,31\). `wrist_EDA` trong cửa sổ có mean 0,306 và std 0,027 ở 64 Hz, std sau scale 0,016. |
| S10, 1 | 0,1266 | 45/47 Stress thành Baseline, 43/77 Baseline thành Amusement, 0/23 Amusement đúng. | `chest_EDA` mean \(z=-1,17\), std \(z=0,13\). Với `chest_Temp`, 26,7% lượt timestep ngoài P01–P99 Train. |
| S9, 1 | 0,5624 | Đúng cả 77 Baseline và 23 Amusement, 0/41 Stress đúng. | Cùng fold và scaler với S10/S16. Khả năng phân loại hai lớp khác nhau rõ giữa ba người. |

Các subject trên có thay đổi khác nhau về offset, biên độ và lớp bị nhầm. EDA cũng đã ghi nhận khác biệt mức tín hiệu theo người, trong đó có wrist EDA của S14. Mask ghi vùng lỗi đã biết của S3 và không đánh dấu S10, S14 hay S16. Các thống kê trên chưa xác nhận hỏng cảm biến ở những người này. `chest_Temp` của Validation Fold 4 là dịch chuyển mean lớn nhất quan sát được, đồng thời Fold 4 có subject-mean F1 tại best cao nhất trên Validation. Thứ hạng này cho thấy một thống kê dịch chuyển của một kênh chưa dự báo được toàn bộ hiệu năng. [EDA](../../../notebooks/preprocessing.ipynb) · [Các bảng theo người và lớp](subject_distribution_statistics.csv) · [Vùng mask](quality_mask_intervals.csv).

## Giải thích theo mức bằng chứng

| Mức | Kết luận |
| --- | --- |
| **Confirmed issue** | Khoảng cách tổng quát hóa và lỗi Test nêu trên có thật trong predictions, checkpoint và logs. Các lỗi confidence cao góp phần làm CE cao. |
| **Verified correct** | Z-score, split, cách tổng hợp loss, nhãn, checkpoint và metric khớp hiện vật. Shift đầu vào có thể đo được và tồn tại trong các lớp, dù cả ba tập dùng đúng scaler Train. |
| **Suspected issue** | CNN có thể dựa quá nhiều vào mức tín hiệu riêng từng người, tư thế, tiếp xúc cảm biến hoặc mốc hoạt động theo quy trình. Overlap làm giảm sự đa dạng của mẫu, có thể khuếch đại việc học dấu hiệu đặc thù. Audit chưa đo biểu diễn nhận dạng người hay tác động nhân quả của overlap. |
| **Requires further experiment** | Mức đóng góp của capacity, regularization, LR, trọng số lớp, độ dài cửa sổ, chất lượng cảm biến và chiến lược chuẩn hóa vào từng lỗi. Trọng số Amusement cao có thể ảnh hưởng vùng quyết định, còn Fold 1 vẫn bỏ hẳn Stress dù Stress có trọng số >1. Cần đối chứng có kiểm soát trước khi quy lỗi. |

Kết quả Test trên trang này dùng để mô tả baseline đã chạy. Các lựa chọn phương pháp tiếp theo cần tiêu chí trên Validation và giao thức đánh giá ngoài đã ấn định. [Phương án kiểm chứng chỉ với CNN-1D](RECOMMENDED_NEXT_EXPERIMENTS.md).

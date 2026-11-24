# GRU 14 kênh — chuẩn bị đối chứng CNN

Đầu vào dự kiến: cùng 14 kênh Common-rate 64 Hz với CNN, cửa sổ 30 giây, stride 15 giây; cùng split seed 42, 9 Train / 3 Validation / 3 Test. Dùng GRU một lớp, hidden size 64, dropout 0,3; 19.715 tham số. Giữ nguyên loss, optimizer, ngân sách epoch/patience và tiêu chí chọn checkpoint theo trung bình Macro-F1 của ba người Validation.

## Trạng thái ngày 24/11/2026

**Đã hoàn tất và hậu kiểm đạt 5/5 folds:** group `20261124T100241Z_204cb671`, seed 42, tổng 98 epochs. Test Macro-F1 trung bình 15 người **0,2864**, Balanced Accuracy **0,3609**, Accuracy **0,4031**. Cùng dữ liệu/split/tiêu chí lựa chọn với CNN (Macro-F1 **0,2962**). Phát lại toàn bộ 2.134 dự đoán từ best checkpoint cho xác suất trùng chính xác CSV đã lưu. Một seed chưa đủ kết luận độ ổn định giữa các lần khởi tạo.

[Báo cáo CNN–GRU](../../results/common_rate64_gru_grouped5fold_v1/20261124T100241Z_204cb671/cnn_gru_comparison.md) · [Hậu kiểm](../../results/common_rate64_gru_grouped5fold_v1/20261124T100241Z_204cb671/verification.json) · [Phát lại dự đoán](../../results/common_rate64_gru_grouped5fold_v1/20261124T100241Z_204cb671/checkpoint_prediction_replay.json). Các đoạn tiến độ bên dưới được giữ làm lịch sử trước khi hoàn tất.

**Lượt chạy chính thức đang thực hiện:** `20261124T100241Z_204cb671`, đủ 5 folds, seed 42. Không đổi cấu hình giữa các folds. Đã khởi động `scripts.finish_common_rate_gru --wait`: khi group hoàn tất và đóng lock, tác vụ sẽ hậu kiểm, phát lại dự đoán GRU từ best checkpoint rồi tạo báo cáo CNN–GRU. Trạng thái tác vụ nằm tại `results/common_rate_review/20261124T100241Z_204cb671/status.json`; nếu lỗi, xem `review.log`. Chưa có báo cáo so sánh cuối cùng khi trạng thái còn `waiting_for_training`.

**Cập nhật sau khi nhận ZIP và đồng bộ:** 33/33 file nguồn chính đã khớp CNN, reference gate và smoke test mới đều đạt. Đã dọn ZIP, bản giải nén, cache cửa sổ lưu sẵn và pytest temp; giữ bản ghi khôi phục chính xác 6 file cũ. Xem `results/data_sync/20261124T095944Z_9631cdc4/cleanup_complete.json` và `results/common_rate_gru_preflight/20261124T100020Z_57c32682/`. Các gạch đầu dòng phía dưới ghi lịch sử kiểm tra trước đồng bộ, không còn là trạng thái chặn hiện tại.

- Có cấu hình chạy riêng tại `config.yaml`, dùng runner Common-rate mới của nhóm.
- Đã kiểm tra kỹ thuật: hai batch Train và một batch Validation của fold 1 trên dữ liệu cục bộ; gradient và loss hữu hạn, trọng số cập nhật, checkpoint nạp lại cho logits giống hệt. Không đọc Test và không tạo kết quả nghiên cứu từ smoke test.
- **Chưa chạy thí nghiệm GRU chính thức:** cache key của S4, S6, S7, S8, S11 và S15 khác lượt CNN tham chiếu `20261123T072126Z_2c81daac`. Danh sách 2.134 cửa sổ, nhãn và thời gian khớp, nhưng chưa xác nhận được nguồn tín hiệu/mask giống nhau từng byte.
- Mean/std Train của fold 1 khớp chính xác scaler CNN đã lưu; điều này không thay thế được kiểm tra dữ liệu nguồn.
- Bản pull thiếu `runs/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac/window_index/common_rate_index.json`. Cần bổ sung audit đó từ máy chạy CNN để đối chiếu SHA256 từng file nguồn; sau đó thống nhất bộ dữ liệu dùng chung. Không tự sửa hash hay kết quả CNN để vượt kiểm tra.

Bằng chứng: [reference check](../../results/common_rate_gru_preflight/20261124T090023Z_a3d7e9d7/reference_check.json), [smoke check](../../results/common_rate_gru_preflight/20261124T090023Z_a3d7e9d7/smoke_check.json), [log](../../results/common_rate_gru_preflight/20261124T090023Z_a3d7e9d7/preflight.log).

## Lệnh dùng tiếp

```powershell
# Kiểm tra nguồn và chạy thử kỹ thuật trên Train/Validation, không đọc Test
python -m scripts.preflight_common_rate_gru --smoke

# Kiểm tra kế hoạch; sẽ dừng nếu chưa khớp dữ liệu tham chiếu
python -m scripts.run_grouped_cv --experiment experiments/common_rate64_gru_grouped5fold_v1/config.yaml --dry-run

# Chỉ chạy sau khi reference check đạt
python -m scripts.run_grouped_cv --experiment experiments/common_rate64_gru_grouped5fold_v1/config.yaml
```

Runner kiểm tra reference trước khi tạo group hoặc gọi huấn luyện. Việc trùng số cửa sổ không đủ để vượt kiểm tra. Nhánh cấu hình trong `configs/common_rate_14ch.yaml` là bản đề xuất cũ; cấu hình hoạt động nằm trong `experiments/common_rate64_*`.

## Ghi nhật ký

Đã làm: chuẩn bị cấu hình GRU 14 kênh, đối chiếu với CNN và kiểm tra ngắn trên Train/Validation.

Kết quả: kiểm tra kỹ thuật đạt; 6 người chưa khớp mã dữ liệu tham chiếu nên chưa chạy chính thức. Không ghi Accuracy/F1 Test cho bước này.

Làm tiếp: lấy audit nguồn CNN, đối chiếu file khác biệt và thống nhất dữ liệu; sau khi kiểm tra đạt mới chạy GRU đủ 5 folds.

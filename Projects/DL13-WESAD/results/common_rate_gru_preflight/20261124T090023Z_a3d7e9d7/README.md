# Kiểm tra trước khi chạy GRU — 24/11/2026

**Kết luận: kiểm tra kỹ thuật GRU đạt; chưa chạy đủ 5 folds vì chưa khớp dữ liệu tham chiếu CNN.**

## Đã xác nhận

- Dữ liệu cục bộ có 15 người, 14 kênh, 64 Hz và 2.134 cửa sổ; loại 2 cửa sổ do mask. ID, nhãn và thời gian mọi cửa sổ khớp dự đoán đã lưu của CNN.
- Cấu hình GRU giữ nguyên các phần data/split/training/selection/runtime/tracking của cấu hình CNN; thay kiến trúc thành GRU hidden size 64 và dùng namespace riêng. Checkpoint chọn bằng Validation Macro-F1 trung bình theo người.
- Cache key khớp ở 9 người; không khớp ở **S4, S6, S7, S8, S11, S15**. Cache key phụ thuộc hash tín hiệu, mask, chính sách và các dòng cửa sổ. Việc trùng ID và số mẫu không chứng minh tín hiệu nguồn giống nhau.
- Fingerprint hiện tại: `2a1abcd2320035c3408707ec5c788e2848967b6be4bb73fea1740f94c6e83212`.
- Fingerprint lượt CNN: `9f2f09b7aee2ec94134a4734d7b9a1382e03936a49ecb9e8c5952e3b92760772`.
- Thiếu audit `window_index/common_rate_index.json` của lượt CNN trong bản pull; chưa đủ hash gốc từng file để xác định chính xác tệp/giá trị khác biệt. Không kết luận dữ liệu hỏng hay nguyên nhân khác biệt khi chưa có bằng chứng.

## Kiểm tra kỹ thuật

Dùng đúng người Train/Validation của fold 1 trên dữ liệu cục bộ. Fit scaler và trọng số lớp từ Train, chạy hai batch Train (32 mẫu/batch), một batch Validation. Không tạo Test Dataset, không đánh giá Test. Kiểm tra loss/gradient hữu hạn, trọng số cập nhật; checkpoint engineering nạp lại cho logits giống hệt.

Đầu vào `[32,14,1920]`, đầu ra `[32,3]`, 19.715 tham số. Mean/std Train của fold 1 khớp chính xác scaler CNN đã lưu, nhưng không dùng điều đó để bỏ qua cache-key mismatch. Đây không phải thí nghiệm chính, không có Accuracy/F1 Test để báo cáo.

Mười checkpoint best/last của CNN khớp hash completion và nạp được vào mô hình 14 kênh. Đã đối chiếu lại 482 file trong thư mục run/kết quả CNN với hash trước kiểm tra: không bị thay đổi.

## Cơ chế chặn

Đã kiểm tra lệnh runner thực, dùng mock để khẳng định `train_fold` không được gọi khi reference không khớp; không tạo group huấn luyện. Unit test kiểm tra thêm trường hợp lệch nhãn, thời gian, người, trùng cửa sổ và đổi giao thức. Chi tiết ở `runner_gate_check.json`, `pytest.log` và `pytest.xml`.

## Debug môi trường kiểm thử

Lượt pytest đầu không hoàn tất do thư mục tạm ngoài workspace không tạo được và lỗi native khi nạp thư viện tùy chọn pyarrow. Lượt kiểm tra sau dùng basetemp riêng tại `data/interim/` và coi pyarrow là không cài trong riêng tiến trình pytest (`sys.modules['pyarrow']=None`), đồng thời tắt plugin tự nạp. Không sửa/xóa/cài lại thư viện, không thay đổi thuật toán mô hình. Kết quả lượt sau được lưu trong pytest.log/XML.

## Làm tiếp

Lấy `runs/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac/window_index/common_rate_index.json` từ máy chạy CNN; đối chiếu `audit.source_file_sha256` với `current_source_audit.json`. Nếu tín hiệu/mask khác, cần thống nhất bộ dữ liệu nguồn trước khi huấn luyện GRU để so với CNN. Nếu chuẩn dữ liệu được thay đổi có chủ đích, phải chạy lại các mô hình đối chứng trên cùng chuẩn và lưu nhóm mới; không ghi đè kết quả cũ.

## File minh chứng

- `reference_check.json`, `subjects.csv`: đối chiếu từng người, cấu hình và cửa sổ.
- `current_source_audit.json`: SHA256 nguồn cục bộ và chính sách cửa sổ.
- `smoke_check.json`, `engineering_smoke.pt`, `normalization.json`: kiểm tra kỹ thuật, checkpoint và scaler.
- `preflight.log`: log các bước.
- `cnn_checkpoint_checks.json`, `reference_preservation.json`: kiểm tra checkpoint và bảo toàn kết quả CNN.
- `pytest.log`, `pytest.xml`: kết quả kiểm thử.

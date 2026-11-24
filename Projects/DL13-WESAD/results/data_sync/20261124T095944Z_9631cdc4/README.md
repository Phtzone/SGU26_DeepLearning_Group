# Đồng bộ dữ liệu và dọn bản trùng — 24/11/2026

Đã đồng bộ 6 file tín hiệu khác biệt vào `data/processed/common_rate_64hz`. Kiểm tra đủ 33 file nguồn khớp SHA256 của lượt CNN; giữ 2.134 cửa sổ và cùng manifest 5 folds 9/3/3.

## Dung lượng đã dọn

Đã bỏ khoảng **762,7 MiB dữ liệu dư có trước thao tác**: ZIP nhận được, thư mục giải nén riêng, cache cửa sổ 30/15 giây lưu sẵn và hai thư mục pytest tạm. Bản backup đầy đủ tạm tạo trong lúc đồng bộ cũng đã được bỏ sau khi kiểm tra lại khả năng khôi phục.

Giữ nguyên dữ liệu gốc, dữ liệu Common-rate chính, code/notebook, nhật ký, báo cáo, log và checkpoint của các thí nghiệm. Runner hiện lấy cửa sổ từ tín hiệu liên tục theo window index; không cần bản sao toàn bộ cửa sổ trên đĩa. Notebook 03 vẫn có thể tạo lại cache nếu cần.

## Khôi phục 6 file cũ nếu cần

`rollback_patches.json` ghi tên file, hash cũ/mới và từng byte đã thay. Đã kiểm tra áp dụng các byte cũ lên bản mới sẽ tạo đúng SHA256 của bản cũ trước khi bỏ backup đầy đủ. Muốn khôi phục phải kiểm tra file hiện tại khớp `new_sha256`, tạo bản sao làm việc rồi áp dụng `old_byte` tại từng `offset`, kiểm tra kết quả khớp `old_sha256`; không ghi đè dữ liệu đang dùng cho một lượt chạy.

## Minh chứng

- `before_source_audit.json`, `after_source_audit.json`: dữ liệu trước/sau.
- `rollback_patches.json`, `sync_progress.json`, `sync_complete.json`: đồng bộ và khôi phục.
- `after_reference_check.json`, `runner_dry_run.json`: nguồn GRU khớp đối chứng CNN.
- `cleanup_plan.json`, `cleanup_progress.json`, `cleanup_complete.json`: đường dẫn tuyệt đối, kích thước và kết quả dọn.
- `reference_preservation_after_cleanup.json`: 482 file CNN trước đó không bị thay đổi.
- `training_launch.json`: thông tin khởi động lượt GRU `20261124T100241Z_204cb671`; việc dọn hoàn thành trước khi bắt đầu lượt này.

Kiểm tra ngắn sau đồng bộ nằm tại `results/common_rate_gru_preflight/20261124T100020Z_57c32682/`: reference đạt, hai batch Train/một batch Validation đạt và checkpoint nạp lại khớp. Lượt kiểm tra ngắn không đánh giá Test.

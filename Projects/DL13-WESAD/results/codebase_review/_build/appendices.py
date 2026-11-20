NOTES={
'src/data/inspection.py':('Đọc cấu trúc raw và lập báo cáo từng subject.','Path gốc + subject ID → dict raw; dict tín hiệu/nhãn → báo cáo shape/dtype/rate/duration/finite/range.','Kiểm tra identity, labels 1D và miền 0..7, signal 2D và số kênh, finite theo block 100.000 hàng, thời lượng theo rate; cuối cùng yêu cầu wrist ACC/BVP/EDA. Lỗi đầu vào thường ValueError; file thiếu là FileNotFoundError. Không chuyển đơn vị hoặc lọc nhiễu sinh lý.'),
'src/data/resampling.py':('Đồng bộ tín hiệu và bảo toàn biên nhãn.','Signal [T,C] → float32 [target_length,C]; labels 700 Hz → int64 [target_length] với −1/0/1/2.','BVP giảm mẫu bằng resample_poly với gcd; EDA tăng mẫu bằng np.interp, endpoint hold; identity khi rate bằng nhau. Không crop/pad để che lệch thời lượng. Căn nhãn bằng ceil/floor trên vùng không đổi, không bỏ phiếu đa số.'),
'src/data/windowing.py':('Tạo cửa sổ trong từng vùng nhãn liên tục.','Signal [T,C] và labels [T] → (float32 [N,C,L], list rows, dict counts).','Tách boundaries; bỏ nhãn âm; neo lại start trong mỗi vùng; transpose từng cửa sổ; audit lưới toàn cục riêng. Cấu hình không hợp lệ, lệch chiều dài hoặc không có cửa sổ đầy đủ gây ValueError.'),
'src/data/preprocessing.py':('Điều phối cache chưa chuẩn hóa theo người.','Raw root + cache root + subject + data config → metadata và windows.npy/windows.csv.','Hash raw, tạo key từ version/data/hash; xác minh cache trước tái sử dụng. Khi cần, inspection→resampling→alignment→windowing; mảng ghi atomic và metadata commit cuối. prepare_dataset lặp 15 subjects. Đổi thuật toán cần xem CACHE_VERSION.'),
'src/data/dataset.py':('Dataset lười đọc memmap, giữ danh tính từng cửa sổ.','Cache + subjects + scaler tùy chọn → Dataset; __getitem__ → (Tensor[C,L], int label, index).','Mỗi subject mở mảng read-only; ghép rows/indices; chặn số rows khác số mảng. Copy một window trước transform. Không kiểm lại toàn bộ checksum ngay trong constructor; runner/preprocessing và completion chịu trách nhiệm integrity ở tầng khác.'),
'src/data/normalization.py':('Fit thống kê kênh trên windows Train.','Dataset không scaler → self chứa mean/std[C]; transform [C,L] → float32 [C,L]; save → JSON.','Block 64 windows float64, gộp moments ổn định; std population ddof=0; std cực nhỏ thay 1. Overlap vẫn góp nhiều lần. Fit Dataset đã chuẩn hóa bị từ chối; Train rỗng chưa có thông báo bảo vệ riêng.'),
'src/data/splits.py':('Manifest subject-wise đóng băng và bảo vệ checksum.','Seed → dict groups/folds; Path manifest → dict đã xác minh.','SUBJECTS cố định 15 ID; shuffle bằng random.Random rồi nhóm 3 và quay vòng Test/Val. validate_manifest so toàn bộ thuật toán lẫn coverage. ensure_manifest không ghi đè seed khác hoặc checksum mồ côi; load_manifest kiểm digest canonical.'),
'src/models/cnn.py':('CNN-1D ba khối hiện hành và E1 lịch sử.','Tensor[B,C,960] → logits[B,3]; C mặc định 5.','Conv-BN-ReLU: 32(k7),64(k5),128(k3); MaxPool sau hai khối đầu; GAP→Linear128/64→ReLU→Dropout .3→Linear64/3. 45.059 tham số khi C=5. Không có nhánh cấu hình hai khối.'),
'src/models/gru.py':('GRU E2 một chiều, một lớp.','Tensor[B,C,L] → logits[B,3]; hidden_size mặc định 64, dropout .3.','Transpose sang [B,L,C], chạy nn.GRU(batch_first=True), lấy hidden[-1], MLP64→64→3. Kiểm hidden_size int dương không bool và dropout [0,1). 17.987 tham số theo metadata/README E2.'),
'src/models/__init__.py':('Factory duy nhất cho models đã triển khai.','Config + input_channels → CNN1D hoặc GRUClassifier.','Nhánh theo experiment.model; CNN giữ constructor cũ, GRU lấy model kwargs. Tên khác gây ValueError. Factory không khởi tạo CNN-GRU planned.'),
'src/models/cnn_gru.py':('Placeholder E3.','Không có class, function hoặc forward để gọi.','Chỉ có module docstring nói planned; không được xem là hybrid đã triển khai. Config planned bị load_config chặn trước training.'),
'src/training/optimizer.py':('Adam và class weights chỉ từ Train.','Model + training config → optimizer; labels 0/1/2 → Tensor weights[3].','Adam dùng lr và weight_decay trong config. bincount minlength=3 và w=N/(3n_c); thiếu lớp Train gây ValueError. Không có scheduler hoặc AdamW ở đây.'),
'src/training/callbacks.py':('Tách best selection khỏi patience.','score, loss, epoch → (better:bool, stop:bool), đồng thời cập nhật state.','Best ưu tiên score trong tolerance 1e−12, tie loss thấp; anchor reset khi score vượt min_delta. Cải thiện nhỏ vẫn best nhưng có thể hết patience. Xem test_validation_selection_and_patience.'),
'src/training/checkpoint.py':('Lưu checkpoint atomic và nạp checkpoint cục bộ tin cậy.','Path + state dict → file; Path → state được torch.load trên CPU.','Ghi tạm UUID, flush/fsync, os.replace. load_checkpoint dùng weights_only=False, khác safe_load_checkpoint của hậu kiểm E1. Không gọi loader trong nhiệm vụ tài liệu.'),
'src/training/trainer.py':('Trung tâm vòng đời một fold.','Folder + cache + fold + seed + config + checksum → metadata; ghi logs/checkpoints/metrics/completion.','Guard completion→identity→RNG→Dataset/scaler/loaders→model/Adam/CE→resume→epochs→best Test→export→complete. Ngoại lệ ghi interrupted/failed và finally đóng tracker. Các đoạn trọng yếu được phân tích ở Chương 9.'),
'src/evaluation/metrics.py':('Metrics ba nhãn cố định.','Aligned labels/predictions [N] → dict scalar/per-class/support/confusion/absent.','Chặn rỗng, shape lệch và nhãn ngoài 0/1/2; sklearn zero_division=0. Macro-F1=mean F1 ba lớp; balanced accuracy=mean recall ba lớp kể cả lớp vắng; cảnh báo RuntimeWarning khi absent.'),
'src/evaluation/evaluator.py':('Suy luận không gradient và xuất kết quả Test có identity.','Model + loader + device + weights → metrics, probabilities[N,3], indices[N]; export → CSV/JSON.','eval/inference_mode, finite logits, weighted loss tổng đúng denominator. Argmax cho nhãn, softmax cho xác suất. Index nối về rows Dataset; per-subject metrics tính từ từng người. write_csv từ chối rows rỗng.'),
'src/evaluation/aggregation.py':('Kiểm provenance/coverage và tổng hợp OOF.','Group artifacts → summary/status; predictions → metrics; history checkpoint → iteration audit.','Xác minh split/config/completion; so expected rows với dự đoán; chặn duplicate/missing, sai nhãn/fold/probability. Lọc iterations đã commit theo attempt+epoch. Đủ seeds×5 folds mới báo cáo mean/std cuối; update_docs là cờ có tác dụng ghi README.'),
'src/evaluation/history.py':('Đọc history theo schema và epochs đã commit.','CSV text → records typed; GROUP_ID → history dict gồm FoldHistory, source hashes và result path.','parse_value xử lý null, ints, JSON lists, bool và floats finite. merged_epochs ghép legacy/v2 chỉ cho keys đã commit. resolve_group yêu cầu một namespace khớp duy nhất. read_group có mở checkpoint; không được gọi trong nhiệm vụ tài liệu.'),
'src/evaluation/visualization.py':('Hình cơ bản dùng backend không giao diện.','Matrix/history/labels-values → PNG.','Agg, confusion matrix có annotations; curves online/eval/val loss và F1; bar score trục 0..1. Đóng plt figure sau save. Hình cơ bản không cùng hệ version/manifest với enhanced visualization.'),
'src/evaluation/enhanced_visualization.py':('Bộ hình khoa học có provenance/version/reuse.','History và predictions thực → PNG+SVG, metrics_by_seed, generation_manifest.','FigureWriter không overwrite, hash từng file, 220 dpi. Learning thiếu trường thì skip; seed_figures tạo per-class/subject/fold/pooled và error analysis S8. Dataset figures có thể đọc memmap. generation_id gồm sources và generator hash; v2 sửa nhãn, không sửa v1.'),
'src/utils/config.py':('Config merge, validation, ROOT và raw path.','Optional YAML path → dict; config model → namespace; WESAD_ROOT env/default → Path.','ROOT dựa trên __file__; deep-copy recursive mapping merge, list thay nguyên. Guard protocol/model/selection/training/tracking; same_training_config bỏ logging/tracking. Không tự nạp .env; validation data params chưa là schema toàn diện.'),
'src/utils/tracking.py':('Tiện ích danh tính, canonical hash và atomic JSON.','Value → UTF-8 canonical bytes/digest; Path → hash/JSON; state → JSON file.','UTC now/identifier; digest dùng SHA256 canonical sort_keys/separators/allow_nan=False; file_hash streaming block 1 MiB. write_json tạm UUID rồi fsync/replace. Chỉ mô tả hash binary trong source, không gọi lên binary ở nhiệm vụ.'),
'src/utils/reproducibility.py':('Seed và khôi phục RNG tại ranh giới epoch.','Seed/deterministic/threads → trạng thái RNG; rng_state → dict; restore_rng nhận dict.','Seed Python, NumPy, Torch/CPU/CUDA khi có; deterministic algorithms, CuDNN flags, threads. Lưu RNG Python/NumPy/Torch/CUDA; không bảo đảm tái lập giữa mọi platform.'),
'src/utils/system_info.py':('Chụp môi trường thực thi.','Không tham số → dict Python/platform/CPU/RAM/CUDA/packages/git.','Đọc metadata phiên bản các packages và git commit/dirty; commit có thể None khi git không khả dụng. Không cài package, không pin hệ môi trường. Dictionary là báo cáo thời điểm chạy, không là môi trường hiện tại của người đọc.'),
'src/utils/logging.py':('Local CSV/JSONL và text logs có schema.','Records theo iteration/epoch/validation/event/error → các file append.','Headers cố định; schema mismatch gây ValueError. append_epoch idempotent theo attempt+epoch và legacy sidecar. Event có UTC timestamp; error ghi traceback; iteration flush ngay. Chưa có tự sửa partial CSV line do mất điện.'),
'src/utils/console.py':('Hiển thị theo TTY và tóm tắt dễ đọc.','Config console + metadata/rows → terminal text/progress.','number trả N/A khi thiếu; auto progress theo isatty; non-interactive không tạo bar. Header, epoch summary, fold/CV summary không dùng làm nguồn thay CSV. Tổng folds truyền vào để hiển thị đúng protocol.'),
'src/utils/experiment_tracking.py':('Phát một stream metrics tới ba kênh local/console/TensorBoard.','Log folder + config; callbacks records → tác dụng ghi/hiển thị.','RunLogger luôn hoạt động, TB tùy chọn; identity được ghi tracking_identity.json. log_epoch nối epoch/validation/local summary rồi TB flush. finish_run đóng bar và writer; warning giữ bằng chứng local.'),
'src/utils/tensorboard_tracker.py':('Backend TensorBoard cục bộ và committed-history recovery.','Config/directory/metadata/rows → ba streams epochs/batches/evaluation hoặc no-op local.','tracking_options chỉ nhận local/tensorboard và local logs bắt buộc. preserve_rng bọc calls; failure warning hoặc strict raise. Steps phải tăng; resume purge và replay committed CSV/history; test figures matrix/F1; cleanup idempotent.'),
'src/utils/wandb_tracker.py':('Adapter W&B lưu trữ, không active trong training hiện tại.','Historical importer config + records → SDK calls, identity, local receipts ở tầng importer.','Có redact, auth presence, env staging phục hồi, strict/fallback, offline/online identity và no raw upload. Chỉ đọc source, không đọc credential store/env secrets và không gọi online. SDK chưa có trong dependencies training bắt buộc.'),
'src/utils/history_import.py':('Replay lịch sử explicit-only sang W&B.','GROUP_ID, offline/online, dry_run, root/overrides/mock SDK → plan/import receipt.','read_group lấy history committed, không chế tạo missing metrics; dry_run dừng trước SDK. Lock PID/create_time, digest source và registry ngăn duplicate replay. Ép không upload raw/checkpoints/system metrics. Không chạy trong nhiệm vụ.'),
'scripts/run_grouped_cv.py':('CLI điều phối CNN/GRU, selected folds và resume.','Arguments → config/execution_plan → group Path hoặc plan khi dry-run.','Validate group ID/folds/version/config; kiểm raw presence; lock; snapshot .py/.yaml; prepare cache; train tuần tự; aggregate update_docs=False; sync manifests; enhanced figures. Error/interruption cập nhật lifecycle và finally nhả lock.'),
'scripts/run_experiment.py':('Alias mỏng cho runner hiện tại.','python -m scripts.run_experiment → scripts.run_grouped_cv.main.','Không có vòng training thứ hai. Docstring còn gọi manual E1 nhưng alias dùng cùng runner hỗ trợ GRU. Mặc định vẫn config E1 nếu không override.'),
'scripts/prepare_data.py':('CLI chuẩn bị cache và hàm chọn đường dẫn.','Config → cache Path có digest(data)[:16]; main → prepare_dataset và in tổng windows.','Mặc định E1, có --experiment. Có thể ghi cache, đọc raw và hash arrays; không được gọi trong nhiệm vụ tài liệu.'),
'scripts/inspect_dataset.py':('CLI inspection raw.','Không CLI option riêng; raw_root/default → inspection.json.','Gọi inspect_dataset và in số subject/status. Có ghi interim báo cáo; không phải read-only hoàn toàn với filesystem dù không sửa raw.'),
'scripts/generate_splits.py':('CLI tạo/kiểm manifest theo seed.','--seed mặc định 42 → splits/grouped_5fold_seed<seed>.json và checksum nếu chưa có.','Gọi ensure_manifest, in groups/hash/folds. Script cho seed khác nhưng load_config protocol chính thức chỉ seed 42; không dùng manifest khác để gán kết quả protocol chuẩn.'),
'scripts/aggregate_results.py':('CLI tổng hợp namespace CNN/GRU.','--group-id → resolve_group rồi aggregate_group(update_docs=False).','Chặn ID không hợp lệ. Không train, nhưng ghi result files/figures; không chạy chỉ vì muốn xem summary cũ. Bản snapshot E1 cũ có đường dẫn E1 cố định và default update_docs khác.'),
'scripts/generate_visualizations.py':('CLI bộ hình enhanced.','--group-id → generate_visualizations.','Không train hoặc overwrite original figures; vẫn đọc checkpoint/cache và có thể tạo results mới, ghi generation marker. Không chạy trong nhiệm vụ.'),
'scripts/verify_e1_baseline.py':('Hậu kiểm E1, có loader checkpoint hạn chế.','group_id/root → report dict; CLI --output ghi JSON.','Verify manifest/config/split/cache/checkpoints/selection/logs/OOF/scaler/TensorBoard/snapshots. safe_load_checkpoint cho weights_only=True và allowlist NumPy cố định; close dùng rtol=0. Verification gắn số đếm canonical; không là generic validator mọi CNN run.'),
'scripts/verify_history.py':('Hậu kiểm regression cho group lịch sử cũ.','Root + baseline JSON tùy chọn → report; CLI ghi output.','Mặc định tests/fixtures/historical_e1_integrity.json hiện không có; protected history group cũ không có trong checkout. Dùng ở hai tests skip; không lấy thiếu fixture làm bằng chứng E1 canonical hỏng.'),
'scripts/audit_preprocessing.py':('Audit cache và waveform có dữ liệu thật.','Output Path mới + subject mặc định S2 → CSV/plots/audit/README.','Kiểm raw/cache hashes, arrays shape/finite/labels, đếm flat channels; nạp raw một Train subject, tái tạo ba windows và vẽ 10 giây đầu. Không sửa data/training artifacts nhưng tạo report. Nhiệm vụ chỉ đọc báo cáo đã lưu.'),
'scripts/audit_experiment.py':('Công cụ audit/report lịch sử E1 có tác dụng ghi nhiều nơi.','GROUP_ID → aggregate, recompute scaler, checkpoint checks, artifact audit và IMPLEMENTATION_REPORT.md.','Không train, nhưng aggregate_group mặc định có thể cập nhật README/results; template còn nói GRU planned. Dùng recorded pytest-results.xml cũ. Không thực thi trong phạm vi bảo toàn hiện tại.'),
'scripts/smoke_tensorboard.py':('Smoke logging tổng hợp, không WESAD training.','Output folder mới tùy chọn → events/verification từ logits cố định.','Tạo vài scalars/images có chủ đích, kiểm steps/values rồi ghi smoke report cả ở root tensorboard_upgrade. Không có optimizer/model training; không coi smoke accuracy là kết quả stress recognition.'),
'scripts/import_history_to_wandb.py':('CLI importer W&B lưu trữ.','group-id/mode/dry-run/project/entity → import_history và JSON plan.','Offline mặc định; online là lựa chọn explicit. Không thuộc active TensorBoard training và không được chạy/export trong nhiệm vụ tài liệu.'),
'experiments/journal/create_workbook.py':('Tạo mẫu nhật ký Excel hồi cứu từ E1.','Output Path mới → workbook sáu sheets có hyperlinks.','Đọc summary/metadata và chuyển UTC sang +07, tạo hướng dẫn/thiết kế/nhật ký/debug/kế hoạch/minh chứng. Từ chối overwrite và thiếu evidence. Nội dung hồi cứu 13/11 còn GRU planned; cần phân biệt với tiến độ 17/11. openpyxl là dependency công cụ, không phải training.')
}

SPECIAL={
'load_subject':'Trả dict raw sau kiểm keys/identity; không gọi trong lần biên soạn.',
'inspect_subject':'Trả báo cáo cấu trúc và thống kê sau guard labels/signals/finite/duration; xem Chương 5.',
'inspect_dataset':'Duyệt SUBJECTS, lưu tiến độ/status và raw hash từng người; re-raise lỗi, finally lưu ended_at.',
'resample_signal':'Trả float32 [target_length,C]; nhánh giảm/tăng/giữ rate; từ chối mismatch.',
'align_labels':'Trả int64 [target_length]; khởi tạo −1, chỉ gán khoảng mẫu nằm trọn vùng nhãn 1/2/3.',
'make_windows':'Trả (windows[N,C,L], rows, report); start neo vùng nhãn, audit global grid riêng.',
'prepare_subject':'Tái sử dụng cache đã khớp key/hash, hoặc tái tạo và commit metadata sau cùng.',
'prepare_dataset':'Trả list metadata theo toàn bộ SUBJECTS; config data dùng chung, không fit scaler.',
'fit':'Fit mean/std[C] từ Dataset Train chưa scaler, gộp moments theo block; trả self.',
'transform':'Trả ((x−mean[:,None])/std[:,None]).astype(float32), shape [C,L].',
'class_weights':'Trả Tensor[3] float32 N/(3counts); thiếu lớp gây ValueError.',
'make_optimizer':'Trả torch.optim.Adam gắn model.parameters với lr/weight_decay config.',
'completed_fold':'Không marker → False; hashes sai → ValueError; đúng hashes → kiểm metadata.status.',
'recover_epoch_logs':'Gọi logger.epoch/validation cho records history, dựa vào writes idempotent.',
'train_fold':'Điều phối toàn fold; metadata return; tác dụng ghi nhiều artifacts. Xem Chương 9 và bảng phân rã dưới.',
'evaluate':'Trả metrics dict, NumPy probabilities[N,3] và indices[N]; model.eval/inference_mode.',
'export_test':'Ghi predictions, subject metrics, report và confusion; dùng indices để nối rows.',
'classification_metrics':'Trả scalar/per-class/support/confusion với ba nhãn cố định; vắng lớp cảnh báo và zero.',
'validate_predictions':'Kiểm keys/coverage/subject/fold/ground truth/probabilities rồi trả metrics tính lại.',
'audit_iterations':'Lọc committed attempt+epoch, kiểm đủ batch_index/sample count, trả đếm orphan và committed.',
'aggregate_group':'Trả summary; partial nếu thiếu fold; có ghi kết quả/figures và tùy cờ có sửa README.',
'stats':'Trả mean/std(ddof=1 hoặc None nếu n=1)/n cho ba metrics trên rows.',
'update_section':'Ghi phần giữa markers trong README; thiếu marker gây ValueError.',
'parse_value':'Chuyển text CSV theo tên field: null, identity text, bool, JSON, int hoặc float finite.',
'merged_epochs':'Hợp nhất legacy/v2 không rỗng theo epoch_key; duplicate trong một file gây ValueError.',
'resolve_group':'Trả đúng một group Path trong runs/*/<ID>; sai ID hoặc số khớp khác 1 gây ValueError.',
'read_group':'Trả history dict sau kiểm completion/logs/predictions; có nạp checkpoint, không gọi ở nhiệm vụ.',
'tracking_options':'Kiểm tùy chọn tracking chuyên biệt của module; active TensorBoard khác adapter W&B lưu trữ.',
'preserve_rng':'Context manager: chụp RNG, yield thao tác phụ, finally phục hồi RNG.',
'same_training_config':'So dict bỏ logging/tracking; không cho thay phần khoa học khi resume.',
'experiment_folder':'Map cnn1d→experiment_e1, gru→experiment_e2; không định tuyến E3 planned.',
'merge':'Deepcopy base, mapping merge đệ quy; list/scalar override thay nguyên giá trị.',
'load_config':'Đọc YAML nền/override, kiểm protocol/model/selection/training/tracking; trả effective dict.',
'raw_root':'Path từ WESAD_ROOT hoặc mặc định data/raw/WESAD, resolve; không nạp .env.',
'canonical':'Trả JSON canonical UTF-8 bytes: sort keys, separators cố định, không NaN.',
'digest':'Trả SHA256 hex của canonical(value), không phải hash raw byte file.',
'file_hash':'Trả SHA256 hex đọc file theo blocks 1 MiB; không gọi lên binary trong nhiệm vụ.',
'write_json':'Atomic UTF-8 JSON qua tệp UUID tạm, flush/fsync và os.replace.',
'read_json':'Trả object JSON sau đọc UTF-8; lỗi IO/JSON có thể truyền lên caller.',
'identifier':'Trả UTC timestamp + 8 ký tự UUID, dùng cho GROUP_ID/run/attempt.',
'now':'Trả ISO UTC datetime string.',
'rng_state':'Trả states Python/NumPy/Torch/CUDA để commit cùng checkpoint.',
'restore_rng':'Khôi phục trạng thái các RNG; CUDA chỉ khi có state và device khả dụng.',
'seed_everything':'Đặt seeds, threads và deterministic flags; không trả model hoặc dữ liệu.',
'row_normalize':'Chia từng hàng matrix cho support, dùng where để hàng tổng 0 thành 0.',
'prediction_metrics':'Lấy true_label/predicted_label từ rows và gọi classification_metrics.',
'generate_visualizations':'Đọc committed history, xác định generation fingerprint, reuse hoặc tạo hình, ghi manifest.',
'execution_plan':'Kiểm manifest/folds/resume/version/config và file raw tồn tại; trả split, selected, existing, plan.',
'write_group_manifest':'Ghi run manifest và result manifest nếu result dir đã có, giữ lifecycle đồng nhất.',
'run':'Điều phối CLI: dry-run dừng trước write/preprocess/train; thực thi thật dùng lock/snapshot/cache/folds/results.',
'cache_path':'Trả data/processed/digest(config[data])[:16]; không tạo thư mục trong hàm này.',
'safe_load_checkpoint':'Kiểm unsafe globals trước torch.load(weights_only=True), chỉ allowlist NumPy RNG cố định.',
'verify_scalars':'Đối chiếu TensorBoard steps và giá trị với rows bằng rtol=atol=1e−6.',
'compare_metrics':'Đối chiếu các metric được lưu và label order với giá trị tính lại.',
'close':'assert_allclose với rtol=0 và tolerance mặc định 1e−12.',
'import_history':'Tạo plan hoặc replay explicit; receipt/source digest chống trùng; không upload raw/checkpoints.',
'append_epoch':'Ghi idempotent theo attempt+epoch, giữ legacy header và thêm v2 sidecar khi cần.',
'append_csv':'Kiểm header khớp, ghi header nếu mới, serialize list/dict và flush một record.',
'redact':'Che trường hoặc chuỗi có secret theo chính sách; phân tích source, không đọc giá trị credential thật.',
'has_online_auth':'Kiểm sự hiện diện auth cho online adapter; không được gọi trong nhiệm vụ.',
'local_time':'Chuyển ISO timestamp thành biểu diễn UTC+07; không coi ngày ghi hồi cứu là ngày chạy.',
}

METHODS={
'__init__':'Khởi tạo state/thuộc tính theo tham số; các trường cụ thể được mô tả trong hợp đồng module.',
'__len__':'Trả len(rows), số windows trong Dataset đã ghép.',
'__getitem__':'Ánh xạ index→subject/offset, copy window, transform tùy chọn, trả tensor/label/index.',
'forward':'Tính logits theo kiến trúc module: CNN dùng network; GRU transpose và hidden cuối.',
'update':'Cập nhật best theo score/loss và anchor/patience độc lập; trả better/stop.',
'save':'Lưu scaler JSON hoặc FigureWriter PNG/SVG tùy class; kiểm hợp đồng riêng của module.',
'start_run':'Mở stream/ghi identity hoặc hiển thị metadata; khi resume có phục hồi history nếu backend hỗ trợ.',
'log_batch':'Ghi metrics theo global_step vào backend; local log luôn giữ mọi record trong active tracker.',
'log_epoch':'Ghi metrics sau epoch với trục epoch; không bịa giá trị thiếu của schema cũ.',
'log_validation':'Ghi per-class Validation theo epoch khi trường có sẵn.',
'log_checkpoint':'Ghi thông tin best checkpoint; active TB chỉ add_text, không upload weights.',
'log_test':'Ghi Test metrics và hình sau model selection; không dùng cho patience.',
'finish_run':'Flush/close tài nguyên theo trạng thái; nhánh đóng được thiết kế idempotent.',
'flush':'Yêu cầu writers flush nếu backend active/chưa failed.',
'_call':'Bọc thao tác backend bằng preserve_rng; warning/fail hoặc strict raise.',
'_scalar':'Chỉ add_scalar khi value khác None và finite, step ép int.',
'_batch':'Guard global_step tăng, lọc frequency và ghi các tags batch.',
'_epoch':'Guard epoch tăng và ánh xạ trường log sang tags train/val/selection.',
'_validation':'Guard epoch Validation tăng; ghi precision/recall/F1 từng lớp.',
'warning':'Ghi tracking_warning vào local và hiển thị console.',
'event':'Ghi sự kiện có timestamp/details; TrainingTracker chuyển tiếp và hiện một số events.',
'error':'Ghi traceback/failure local, hiển thị thông báo tương ứng.',
'epoch':'Chuyển record epoch sang logger local.',
'validation':'Chuyển record Validation sang logger local.',
'iteration':'Ghi batch row bằng ITERATION_FIELDS và flush.',
'message':'Append chữ thuần vào training.log, không có progress control codes.',
'write':'Hiển thị console nếu enabled, phối hợp với progress bar nếu có.',
'header':'Lưu context và in experiment/model/fold/subjects/windows/hyperparameters.',
'start_epoch':'Mở progress bar nếu TTY/chế độ cho phép, đóng bar cũ trước.',
'batch':'Cập nhật progress postfix loss/lr/step và tăng batch counter.',
'close_bar':'Đóng và bỏ tham chiếu bar nếu tồn tại.',
'epoch_summary':'In Train eval/Val/online loss, optimization và selection; dữ liệu thiếu là N/A.',
'fold_summary':'In epochs/best/Test scores/time/checkpoint của fold đã xong.',
'cv_summary':'In bảng mỗi fold/seed; không thay summary khoa học từ OOF.',
'skip':'Ghi figure/reason với status skipped thay vì chế tạo hình từ số thiếu.',
'line':'Vẽ đường nhiều series, trục/tên, mốc best/zero tùy chọn rồi save.',
'bars':'Vẽ một dãy cột, tùy chọn chặn trục score trong 0..1.',
'grouped':'Vẽ nhóm cột nhiều series, dùng width chia theo số nhóm.',
'heatmap':'Vẽ ma trận và nhãn ô, màu/tỉ lệ theo chế độ normalized.',
'confusion':'Vẽ confusion raw hoặc row-normalized; hàng thực/cột dự đoán.',
'environment':'Context manager đặt thư mục SDK staging riêng rồi phục hồi environment của caller.',
'main':'Phân tích các CLI arguments đúng như source rồi gọi hàm nghiệp vụ; không gọi khi import module.',
}

TESTMAP={
'test_aggregation.py':'Bất biến OOF, full/partial và tamper detection; test tích hợp có training fixture.',
'test_baseline_verification.py':'Allowlist loader, RNG NumPy và không đổi hash/safe-global state.',
'test_checkpoint.py':'Atomic write và hồi phục RNG.',
'test_dataset.py':'Lazy memmap, identity và thiếu raw file.',
'test_gru.py':'Sequence layout, isolation theo batch, học được, factory và options.',
'test_logging.py':'Live flush, append, sự kiện và traceback.',
'test_manual_runner.py':'Dry-run, selected folds, resume, manifest lifecycle và guard config.',
'test_metrics.py':'Metrics biết trước, absent class và selection/patience.',
'test_model.py':'CNN shape, số tham số và backward/optimizer.',
'test_normalization.py':'Train-only scaler và thống kê đúng.',
'test_resampling.py':'Anti-aliasing, interpolation/identity, duration và biên nhãn.',
'test_resume.py':'Resume CNN/GRU giống uninterrupted và Test sau selection.',
'test_splits.py':'Nhóm/coverage/hash/no overwrite.',
'test_tensorboard_tracking.py':'Event tags/steps/images, purge/replay, failure/cleanup và RNG.',
'test_tracking_upgrade.py':'Console, legacy schema, archived importer và regression lịch sử.',
'test_visualization_upgrade.py':'Chất lượng PNG/SVG, no overwrite, missing metrics và v1/v2 reuse.',
'test_windowing.py':'Không cắt qua trạng thái/invalid; vùng không đủ dài.',
}

def write_appendices(g):
    globals().update(g)
    callsites={}
    for caller in sorted(k for k in S if not k.startswith('runs/')):
        tree=ast.parse(C[caller]); aliases={}; local={n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        for n in ast.walk(tree):
            if isinstance(n,ast.ImportFrom) and n.module and n.module.startswith(('src','scripts')):
                for a in n.names:
                    modulepath=n.module.replace('.','/')+'.py'
                    if modulepath not in C: modulepath=n.module.replace('.','/')+'/__init__.py'
                    aliases[a.asname or a.name]=(modulepath,a.name)
            elif isinstance(n,ast.Import):
                for a in n.names:
                    if a.name.startswith(('src','scripts')):
                        aliases[a.asname or a.name]=(a.name.replace('.','/')+'.py',None)
        class Calls(ast.NodeVisitor):
            def __init__(self): self.stack=[]
            def visit_FunctionDef(self,n):
                self.stack.append(n.name); self.generic_visit(n); self.stack.pop()
            visit_AsyncFunctionDef=visit_FunctionDef
            def visit_Call(self,n):
                target=None
                if isinstance(n.func,ast.Name):
                    name=n.func.id
                    if name in aliases and aliases[name][1]: target=aliases[name]
                    elif name in local: target=(caller,name)
                elif isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name):
                    owner=n.func.value.id
                    if owner in aliases and aliases[owner][1] is None: target=(aliases[owner][0],n.func.attr)
                    elif owner=='self' and n.func.attr in local: target=(caller,n.func.attr)
                if target:
                    callsites.setdefault(target,[]).append(f'{Path(caller).name}:{n.lineno}'+(' — '+'.'.join(self.stack) if self.stack else ''))
                self.generic_visit(n)
        Calls().visit(tree)
    doc.add_page_break(); doc.add_heading('PHỤ LỤC A. TRA CỨU MÃ NGUỒN VÀ HỢP ĐỒNG HÀM/LỚP',1)
    p('Mỗi mục dưới đây có vai trò, đầu vào/đầu ra, thuật toán và trường hợp biên; bảng đi kèm chỉ ra tham số thực tế và dòng định nghĩa. Với class, phương thức nhận self không được nhắc lại như một tham số dữ liệu. Kiểu được nêu là hợp đồng suy ra từ thao tác source và tests; nhiều hàm chưa có type annotations tường minh.')
    p('Các nơi gọi được xác định từ imports của mã hiện tại và tests, không thực thi tracing. Đây là quan hệ tham chiếu tĩnh; gọi động qua đối tượng, aliases và callbacks có thể không được liệt kê đầy đủ. Bản snapshots được tra riêng ở phụ lục C. Không sao chép toàn bộ file Python.')
    for idx,(path,(role,contract,logic)) in enumerate(NOTES.items(),1):
        doc.add_heading(f'A.{idx}. {path}',2)
        p('Vai trò: '+role+' Hợp đồng dữ liệu: '+contract)
        p('Cách hoạt động và biên: '+logic)
        meta=S[path]; imports=sorted(set(x for x in meta['imports'] if x.startswith(('from src','from scripts','from torch','from numpy','from scipy','from sklearn','import torch','import numpy','import yaml','from openpyxl'))))
        if imports: p('Phụ thuộc chính: '+'; '.join(imports[:9])+'.','Bằng chứng')
        callers=[]
        module=path[:-3].replace('/','.')
        if module.endswith('.__init__'): module=module[:-9]
        for q,sm in S.items():
            if q==path or q.startswith('runs/'): continue
            if any(re.search(r'\b'+re.escape(module)+r'\b',x) for x in sm['imports']): callers.append(q)
        if callers: p('Tham chiếu import từ: '+'; '.join(callers)+'.','Bằng chứng')
        elif path.startswith('scripts/') or path.startswith('experiments/'): p('Điểm gọi chủ yếu: main/CLI hoặc công cụ thủ công; xem mô tả tác dụng ở Chương 3 và 15.','Bằng chứng')
        else: p('Không thấy module hiện tại import trực tiếp mục này; đó không phải bằng chứng mọi lời gọi động đều vắng.','Bằng chứng')
        rows=[]
        for sym in meta['symbols']:
            name=sym['name']
            used=list(dict.fromkeys(callsites.get((path,name),[])))
            where='\nGọi tĩnh: '+'; '.join(used[:3])+(' (còn các tham chiếu khác).' if len(used)>3 else '.') if used else '\nNơi gọi: qua giao thức đối tượng/CLI/callback hoặc chưa thấy lời gọi tĩnh trực tiếp; xem callers của module.'
            if sym['kind']=='ClassDef':
                rows.append((f'{name}\nDòng {sym["line"]}','Lớp chứa state và các phương thức liệt kê dưới; xem hợp đồng module ở trên.'+where,'Đối tượng lớp / trạng thái'))
                continue
            # Nested renderer/sheet helper retained but explicitly marked as local helper.
            desc=SPECIAL.get(name,METHODS.get(name))
            if desc is None:
                desc={
                  'learning_figures':'Tạo curves mỗi fold từ epoch/iteration đã commit; thiếu metric thì skip.',
                  'evaluation_figures':'Tạo Test confusion/class metrics từ predictions và trả metrics.',
                  'seed_figures':'Gộp các fold cùng seed, tạo bảng/hình subject/fold/pooled và trả metrics.',
                  'error_figures':'Phân tích S8 nếu có, confidence/timeline/so sánh; không đổi dự đoán.',
                  'dataset_figures':'Tạo thống kê từ unique window keys và ví dụ cache nếu có; có đọc mảng.',
                  'values':'Trả list float cho field; thiếu giá trị thành NaN để vẽ khoảng trống.',
                  'epoch_key':'Trả tuple(str(attempt_id),int(epoch)) làm khóa committed.',
                  'read_csv':'Đọc CSV thành records; module history có parse typed, aggregation giữ chuỗi.',
                  'read_rows':'Đọc CSV DictReader thành list rows.',
                  'write_csv':'Ghi rows với header từ khóa của row đầu; module cụ thể có guard rỗng khác nhau.',
                  'confusion_plot':'Vẽ ma trận raw true/predicted và giá trị từng ô rồi lưu PNG.',
                  'training_plots':'Vẽ loss và Macro-F1 theo epoch từ history.',
                  'bar_plot':'Vẽ cột score có trục 0..1.',
                  'save_checkpoint':'Atomic torch.save qua file tạm; flush/fsync rồi replace.',
                  'load_checkpoint':'torch.load map_location CPU, weights_only=False cho checkpoint tin cậy.',
                  'system_info':'Trả snapshot môi trường/packages/git tại thời điểm gọi.',
                  'number':'Format số với digits hoặc trả N/A nếu None/chuỗi rỗng.',
                  'verify':'Hậu kiểm theo hợp đồng của script, trả report; xem giới hạn module ở trên.',
                  'audit':'Chạy kiểm tra và ghi reports; phạm vi dữ liệu/tác động khác nhau theo hai scripts audit.',
                  'smoke':'Tạo events từ logits tổng hợp cố định và kiểm tags/steps; không train model.',
                  'create':'Tạo workbook hồi cứu E1 có links, từ chối overwrite/missing evidence.',
                  'sheet':'Hàm lồng trong create, dựng và định dạng một sheet, gắn hyperlink khi đường dẫn tồn tại.',
                  'render_tree':'Hàm lồng trong audit, biến cây dict thành cây thư mục text cho report.',
                }.get(name,'Hàm hỗ trợ nội bộ; đọc cùng module và các calls được đối chiếu ở source. Không suy ra khả năng đã chạy chỉ từ định nghĩa.')
            signature=name+'('+sym.get('args','')+')'
            signature=signature.replace(', ',',\n')
            returns=sym.get('returns') or 'Không có annotation trả về; xem hợp đồng module.'
            guards=sorted({re.match(r'\w+',s).group(0) for s in sym.get('exceptions',[]) if re.match(r'\w+',s)})
            if guards: returns+='\nNém trực tiếp: '+', '.join(guards)+'.'
            rows.append((signature+f'\nDòng {sym["line"]}',desc+where,returns))
        if rows: table('Hàm/lớp của '+path,['Khai báo và tham số','Xử lý / giá trị trả về','Kiểu khai báo / nhánh lỗi'],rows,[5.1,8,3.9])
        else: p('Không có định nghĩa hàm/lớp; xem trạng thái placeholder hoặc package ở phần vai trò.')
        source(path)
        if path=='src/training/trainer.py':
            table('Đọc train_fold theo khối, tránh lạc trong một hàm dài',['Khối','Điều phải theo dõi'],[
              ('Guard / metadata','completed_fold, effective_config, manifest_checksum, subjects, seed và attempts'),('Data / model','fit Train scaler trước gắn cả ba roles; class weights Train; model từ factory'),('Resume','last.pt authoritative, so cache_keys/run_id; khôi phục best từ state và RNG'),('Batch loop','zero_grad→forward→CE→finite→backward→clip→Adam.step→log'),('Epoch commit','Train eval/Val→selection→history→best khi better→last→logs'),('Test / completion','Nạp best→Test→export→plots→metadata completed→close tracker→completion hashes'),('Exception / finally','status interrupted/failed, ghi lỗi, re-raise và đóng tài nguyên')],[4,13])
    h2('A. Các tệp package không chứa nghiệp vụ riêng')
    p('src/__init__.py và scripts/__init__.py chỉ có docstring; src/data, training, evaluation, utils có __init__.py rỗng. src/models/__init__.py có build_model nên đã được phân tích riêng. Những tệp này vẫn nằm trong số 69 Python files hiện tại đã đọc.')

    doc.add_page_break(); doc.add_heading('PHỤ LỤC B. TRA CỨU TOÀN BỘ TEST FUNCTIONS',1)
    p('Danh sách lấy từ AST source thật, gồm cả các hàm được pytest parameterize. Số function definitions khác số test cases thực thi: một function có thể sinh nhiều cases, hoặc bị skip. Không chạy lại các tests trong nhiệm vụ này.')
    for path in sorted(k for k in S if k.startswith('tests/test_')):
        h2(path); p(TESTMAP[Path(path).name])
        rows=[]
        tree=ast.parse(C[path])
        for n in tree.body:
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name.startswith('test_'):
                assertions=[ast.unparse(x.test) for x in ast.walk(n) if isinstance(x,ast.Assert)]
                raises=[ast.unparse(x.items[0].context_expr) for x in ast.walk(n) if isinstance(x,ast.With) and x.items and 'pytest.' in ast.unparse(x.items[0].context_expr)]
                checks=assertions[:2]+raises[:2]
                short='; '.join(checks)
                if len(short)>650: short=short[:647]+'...'
                rows.append((n.name+f'\nDòng {n.lineno}',short or 'Dùng mocks / helper để kiểm hợp đồng; xem source.',', '.join(a.arg for a in n.args.args) or 'Không fixture tham số'))
        table('Assertions tiêu biểu, giữ nguyên biểu thức để đối chiếu',['Test function','Assertions / expected failure từ source','Fixture / tham số'],rows,[5.2,8.2,3.6]); source(path)
    h2('B. Fixtures dùng chung'); table('Fixtures không phải dữ liệu nghiên cứu',['Fixture','Đầu vào/đầu ra và giới hạn'],[
      ('tiny_cache','Tạo 15 subjects tổng hợp, 6 windows [5,32]/người, labels tuần hoàn; ghi trong tmp_path.'),('synthetic_history','Một fold schema cũ với predictions tổng hợp và checkpoint nhỏ; chỉ cho test công cụ.'),('fake_wandb','Mock SDK và run objects; không xác minh upload lên dịch vụ thật.'),('scientific_config_manifest','Reference dict độc lập trong test_manual_runner, giúp phát hiện thay config ngoài ý muốn.'),('sandbox_runner','Root giả có file raw rỗng và monkeypatch ROOT/raw_root/cache_path, tránh gọi raw thực.')],[5,12]); source('tests/conftest.py','tests/test_manual_runner.py')

    doc.add_page_break(); doc.add_heading('PHỤ LỤC C. PHẠM VI ĐỌC VÀ SOURCE SNAPSHOTS',1)
    table('Inventory trước khi tạo giáo trình, không tính codebase_review',['Hạng mục','Số lượng / kết quả'],[
      ('Tệp kiểm kê ngoài caches/venv bị loại',SCOPE['files']),('Tệp văn bản đọc/giải mã an toàn',SCOPE['text_read']),('Python tổng, đã phân tích AST',SCOPE['python_read']),('Python hiện tại',f"{SCOPE['active_python']} = 36 src + 14 scripts + 18 tests (17 test modules và conftest) + 1 journal generator"),('Python trong snapshots',199),('Cấu hình hiện tại',f"{SCOPE['configs']} YAML = 1 default + 7 experiment/ablation"),('YAML gồm snapshots',63),('Scripts trong scripts/', '14 tệp, gồm 13 tệp lệnh/hỗ trợ và __init__.py'),('Lỗi giải mã / AST trong phạm vi đọc','0; pytest.log UTF-16 được giải mã theo BOM'),('Notebook nội dung','0; chỉ tệp giữ chỗ .gitkeep'),('Tài liệu Markdown',12)],[9,8])
    p('Bộ đọc loại .git, __pycache__, .pytest_cache, .deepeval, .mypy_cache, .ruff_cache, node_modules, wandb và .venv*. Không mở .zip, .pkl, .npy/.npz, .pt, event binaries; không giải nén PDF/XLSX. Nguồn huấn luyện trong snapshots cũng được đọc, parse và so khác biệt với source hiện tại. Raw text respiban dung lượng lớn được loại khỏi corpus phân tích; các báo cáo số liệu sử dụng metadata đã lưu.')
    table('Bốn source snapshots được đối chiếu hash văn bản',['Group','Python / YAML','Tổng file source hash khớp'],[
      (x['path'].split('/')[2],str(sum(k.startswith(x['path'].rsplit('/',1)[0]+'/source/') and k.endswith('.py') for k in C))+' / '+str(sum(k.startswith(x['path'].rsplit('/',1)[0]+'/source/') and k.endswith('.yaml') for k in C)),x['files']) for x in CHECKS['source_snapshots']],[8.3,4.2,4.5])
    p('E1 canonical snapshot còn GRU planned và trainer tạo CNN trực tiếp; current source có factory/GRU. Current runner đồng bộ run/result lifecycle, aggregation định tuyến theo namespace và ghi primary metric subject-level, generator có v2/hash code. Snapshot E2 khớp source .py/.yaml hiện tại tại các file được lưu. Mọi phiên bản Python khác nhau đã được so đối chiếu; không chỉnh source snapshots để khớp source mới.')
    h2('C. Danh mục module Python bên trong snapshots')
    p('Mỗi hàng dùng đường dẫn tương đối sau source/. Cột cho biết snapshots nào chứa file; nhờ đó tra được đầy đủ 199 bản Python mà không lặp lại mô tả những nội dung giống hệt. Ký hiệu A=E1 canonical, B=CNN WD .001, C=CNN đối chứng, D=E2 GRU.')
    groups=[('A',f'runs/experiment_e1/{E1}/source/'),('B',f'runs/experiment_e1/{WD}/source/'),('C',f'runs/experiment_e1/{CNN}/source/'),('D',f'runs/experiment_e2/{E2}/source/')]
    rels=sorted({k.split('/source/')[1] for k in C if '/source/' in k and k.endswith('.py')})
    table('Tra tất cả bản Python được lưu',['Đường dẫn trong source/','Có trong snapshots','So với hiện tại'],[(rel,', '.join(label for label,pre in groups if pre+rel in C),'; '.join(label+(' giống' if C[pre+rel]==C.get(rel) else ' khác') for label,pre in groups if pre+rel in C)) for rel in rels],[8.3,3,5.7])
    h2('C. Các cấu hình và metadata khác')
    table('YAML hiện tại được đọc',['File','Vai trò / trạng thái'],[
      ('configs/default.yaml','Cấu hình nền khoa học/training/runtime/logging'),('experiments/e1_cnn/config.yaml','E1 cnn1d implemented; kế thừa nền'),('experiments/e1_cnn_wd_1e3/config.yaml','CNN implemented; override weight_decay=.001'),('experiments/e2_gru/config.yaml','GRU implemented; hidden=64, dropout=.3'),('experiments/e3_cnn_gru/config.yaml','cnn_gru planned'),('experiments/ablation/no_acc.yaml','planned; sensors BVP/EDA'),('experiments/ablation/no_bvp.yaml','planned; sensors ACC/EDA'),('experiments/ablation/no_eda.yaml','planned; sensors ACC/BVP')],[9,8])
    p('Các logs/metadata văn bản của runs/results được đọc vào corpus để đối chiếu schema, status, epochs, counts, probabilities, OOF và manifests. CSV nhiều dòng không được chép nguyên vào Word. Các manifest source_hashes có đường dẫn máy được sử dụng để hiểu provenance, không đưa thông tin tài khoản/người dùng vào giáo trình. Không đọc hoặc xuất giá trị secret/credential/token.')

    doc.add_page_break(); doc.add_heading('PHỤ LỤC D. TỪ ĐIỂN THUẬT NGỮ VÀ CÔNG THỨC',1)
    table('Tra nhanh các từ thường gặp',['Thuật ngữ','Diễn giải trong giáo trình'],[
      ('Subject / sensor / channel','Người tham gia / cảm biến / một chuỗi giá trị. ACC là một sensor có ba channels.'),('Sampling rate / timestep','Số mẫu mỗi giây / một vị trí thời gian trong chuỗi.'),('Window / stride / overlap','Đoạn làm một mẫu học / bước dịch / phần thời gian chia sẻ giữa windows.'),('Tensor [B,C,L]','Batch, kênh, chiều dài thời gian; GRU chuyển sang [B,L,C].'),('Cache / invalidation','Dữ liệu trung gian tái sử dụng / quy tắc buộc tạo lại khi nguồn hoặc thuật toán đổi.'),('Canonical JSON / SHA256','Cách tuần tự hóa ổn định / dấu vân tay nội dung; không phải mã hóa dữ liệu.'),('Fit / transform','Ước lượng thống kê từ Train / áp dụng thống kê đó lên mẫu bất kỳ.'),('Epoch / mini-batch / step','Một lượt Train / nhóm mẫu cho một cập nhật / số cập nhật optimizer.'),('Logits / softmax','Điểm đầu ra chưa chuẩn hóa / chuyển điểm thành xác suất tổng 1.'),('Checkpoint / state_dict','Trạng thái lưu để tiếp tục hoặc đánh giá / tham số và buffers của model.'),('Validation / Test','Dùng lựa chọn checkpoint / dùng báo cáo sau selection.'),('OOF / pooled','Dự đoán khi mẫu nằm Test của fold / tính metric một lần sau ghép dự đoán.'),('Macro-F1 / balanced accuracy','Mean F1 ba lớp / mean recall ba lớp cố định trong code.'),('Finalized / completed','Hồ sơ baseline được đóng băng / trạng thái thực thi hoàn tất ở một tầng cụ thể.'),('Reproducibility / provenance','Khả năng tái lập có điều kiện / truy xuất nguồn code-data-config-environment.')],[5.2,11.8])
    table('Công thức và điều kiện áp dụng',['Công thức','Ý nghĩa / lưu ý'],[
      ('T=N/f','Thời lượng giây khi N mẫu được mô tả ở f Hz; phân biệt khoảng giữa mốc cuối với tổng độ dài bản ghi.'),('L=rate×window_seconds; S=rate×stride_seconds','Code ép int; baseline L=960, S=480.'),('n=max(0,floor((R−L)/S)+1)','Số cửa sổ trong một vùng hợp lệ R mẫu; không áp vào toàn timeline nếu có ranh giới nhãn.'),('z=(x−μTrain)/σTrain','Broadcast theo kênh; σ rất nhỏ thay 1; không fit lại Val/Test.'),('w_c=NTrain/(3n_c)','Class weights Train-only; bắt buộc ba lớp đều có mẫu.'),('CE=−Σ w_y log p_y / Σw_y','Weighted mean đúng khi tổng hợp batch/epoch; không mean tùy ý các batch means.'),('F1_c=2TP/(2TP+FP+FN)','Quy ước zero_division=0; macro lấy trung bình đúng ba lớp.'),('s=√[Σ(x−mean)²/(n−1)]','Sample std của các scores folds/subjects; không phải std qua seeds nếu n đang là folds/subjects.')],[8,9])

    doc.add_page_break(); doc.add_heading('PHỤ LỤC E. BIÊN BẢN KIỂM TRA VÀ GIỚI HẠN BÀN GIAO',1)
    p('Tài liệu được tạo bằng python-docx thành gói Office Open XML .docx thật. Kiểm tra cấu trúc gồm 17 chương A–G, Heading 1/2/3, field mục lục TOC, field số trang PAGE, A4 dọc, font chính 11 pt, bảng có header lặp và tổng chiều rộng 17 cm; sơ đồ/biểu đồ xuất mới phục vụ giáo trình. Mục lục và số trang được cập nhật khi dàn trang bằng Microsoft Word nếu công cụ khả dụng.')
    table('Kiểm tra mới và giới hạn',['Hạng mục','Kết luận có thể khẳng định'],[
      ('Source','268 Python files parse AST; 808 tệp văn bản đọc/giải mã, không có lỗi giải mã còn tồn tại trong corpus.'),('Split / snapshot','Canonical split checksum khớp; vai trò rời nhau; hashes của 55/58/58/59 source text files trong bốn snapshots khớp manifest.'),('Processed metadata','15 CSV có tổng 2.136 rows, window IDs duy nhất trong người, duration=30 giây, label mapping và counts khớp metadata.'),('Kết quả văn bản','Bốn groups có 2.136 OOF windows/15 người; tính lại pooled và subject mean khớp stored summary trong tolerance 1e−12.'),('Chưa xác minh lại','Raw finite/signal quality, contents arrays/checkpoints/events, inference và tính đúng dữ liệu đo thực; chỉ dựa hồ sơ lịch sử tương ứng.'),('Tests','Không chạy pytest hoặc training trong nhiệm vụ. XML gần nhất là hồ sơ 58 pass/2 skip ngày 17/11.'),('Tệp khác','Không mở .zip hoặc giải nén audit/finalized. PDF/XLSX hiện có chỉ kiểm kê; không có notebook nội dung để đọc.'),('Bảo toàn','Không sửa source/config/tests/dataset/E1 artifacts, không training, không git commit/push; chỉ thêm DOCX và tệp tạm xuất tài liệu. So metadata của mọi tệp gốc và SHA256 của tệp văn bản trước/sau; binary chỉ stat theo giới hạn nhiệm vụ.')],[5.2,11.8])
    p('Đối với số trang và kiểm tra mở bằng Microsoft Word: xem thông tin bàn giao đi kèm. Nếu môi trường không cho phép ứng dụng Word tự động hoặc render PDF, giới hạn đó phải được nói rõ, không lấy số trang ước đoán làm số trang thật. Các thông tin kiểm tra dàn trang do bộ xuất lưu riêng trong tệp tạm, không chỉnh sửa bất kỳ hồ sơ E1 nào.')
    p('Giáo trình dùng bằng chứng source hiện tại để mô tả thực thi và dùng source snapshot để mô tả các lượt lịch sử. Khi README/nhật ký mâu thuẫn source/artifacts mới hơn, nội dung chương chỉ rõ mâu thuẫn thay vì âm thầm thay nghĩa. Roadmap chỉ là đề xuất, chưa được thực hiện.')

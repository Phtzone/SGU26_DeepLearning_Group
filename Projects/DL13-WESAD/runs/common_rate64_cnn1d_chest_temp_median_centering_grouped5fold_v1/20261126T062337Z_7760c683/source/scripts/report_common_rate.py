"""Write the Vietnamese report and workbook from one common-rate run's artifacts."""
import argparse
from collections import Counter
import copy
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = 'common_rate64_cnn1d_grouped5fold_v1'
EXPERIMENT_DIR = ROOT / 'experiments' / EXPERIMENT
JOURNAL_NAME = 'Research_journal_' + EXPERIMENT + '.xlsx'
SHEETS = ['Thí nghiệm', 'Hình', 'Nhật ký', 'Debug', 'Tham khảo', 'Việc cần làm']
LABELS = ('Baseline', 'Stress', 'Amusement')
BANGKOK = timezone(timedelta(hours=7))
REFERENCES = [
    ('Schmidt và cộng sự (2018) — Introducing WESAD',
     'https://www.eti.uni-siegen.de/ubicomp/papers/ubi_icmi2018.pdf',
     'Dữ liệu ngực và cổ tay, bài toán 2 lớp và 3 lớp.', 'Đặc trưng cảm biến và các bộ phân loại cổ điển.',
     'LOSO.', 'Accuracy tối đa khoảng 80% ở bài toán 3 lớp (tóm tắt).',
     'Nguồn bộ dữ liệu. Khác cách trích đặc trưng và đơn vị chia fold.'),
    ('Rashid và cộng sự (2021) — Feature Augmented Hybrid CNN', 'https://arxiv.org/abs/2108.03166',
     'BVP cổ tay WESAD, 2 và 3 lớp.', 'CNN kết hợp đặc trưng HR/HRV. Cửa sổ 60 giây, bước 5 giây.',
     'LOSO.', 'H-CNN 3 lớp: Accuracy 75,21%, Macro-F1 64,15% (Hình 2).',
     'Khác số kênh, cửa sổ, đặc trưng và giao thức 9/3/3.'),
    ('Gil-Martín và cộng sự (2022) — Human Stress Detection With Wearable Sensors Using CNNs',
     'https://portalcientifico.upm.es/es/ipublic/item/9744363',
     'WESAD, nhiều tổ hợp cảm biến. Bài toán 2, 3 và 5 lớp.', 'CNN trên biểu diễn tín hiệu miền tần số.',
     'LOSO.', 'Accuracy tốt nhất 96,6% (2 lớp) và 85,1% (3 lớp) theo tóm tắt.',
     'Accuracy khác Macro-F1. Biểu diễn phổ và cách chia dữ liệu cũng khác.'),
]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def local_time(value):
    return datetime.fromisoformat(value).astimezone(BANGKOK).replace(tzinfo=None)


def link(from_dir, path):
    return Path(os.path.relpath(path, from_dir)).as_posix()


def load_run(group_id):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', group_id):
        raise ValueError('Invalid group ID.')
    group = ROOT / 'runs' / EXPERIMENT / group_id
    result = ROOT / 'results' / EXPERIMENT / group_id
    manifest = read_json(group / 'experiment_manifest.json')
    if manifest['group_id'] != group_id or manifest['effective_config']['experiment']['id'] != EXPERIMENT:
        raise ValueError('Wrong experiment or group.')
    folds = {}
    for fold in sorted((group / 'seed_42').glob('fold_*')):
        path = fold / 'metadata.json'
        if path.is_file():
            folds[int(fold.name[-2:])] = (fold, read_json(path))
    completed = {fid: (path, m) for fid, (path, m) in folds.items()
                 if m['status'] == 'completed' and (path / 'completion.json').is_file()}
    verified = result / 'verification.json'
    final = (len(completed) == 5 and manifest['status'] == 'completed' and verified.is_file()
             and read_json(verified)['status'] == 'passed' and read_json(verified)['group_id'] == group_id)
    return group, result, manifest, folds, completed, final


def format_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *(' | '.join(['', *(str(v) for v in row), '']) for row in rows)])


def write_report(group, result, manifest, completed):
    audit = read_json(group / 'window_index/common_rate_index.json')['audit']
    split = read_json(group / 'split_manifest.json')
    summary = read_json(result / 'summary.json')['per_seed']['42']
    subjects = read_csv(result / 'subject_metrics.csv')
    predictions = read_csv(result / 'oof_predictions.csv')
    verification = read_json(result / 'verification.json')
    viz = ROOT / verification['figure_directory']
    cfg = manifest['effective_config']
    tc = cfg['training']

    def md(label, path):
        return f'[{label}]({link(result, path)})'

    def metric(value):
        return f'{float(value):.4f}'

    keys = ('macro_f1', 'balanced_accuracy', 'accuracy', 'stress_recall')
    fold_lines = []
    for fid, (folder, meta) in completed.items():
        volumes = meta['window_counts']
        fold_lines.append([fid, f"{volumes['train']}/{volumes['validation']}/{volumes['test']}", meta['completed_epochs'],
                           meta['best_epoch'], metric(meta['best_val_subject_macro_f1']),
                           *(metric(meta['test_metrics'][key]) for key in keys),
                           md('best.pt', folder / 'checkpoints/best.pt'), md('epoch log', folder / 'logs/epochs.csv')])
    subject_lines = [[r['fold_id'], r['subject_id'], r['windows'], *(metric(r[key]) for key in keys)]
                     for r in sorted(subjects, key=lambda row: int(row['subject_id'][1:]))]
    aggregation = []
    for description, level in [('Trung bình theo 15 người (chính)', 'subject_level'), ('Trung bình theo 5 fold', 'fold_level')]:
        aggregation.append([description, *(f"{metric(summary[level][k]['mean'])} ± {metric(summary[level][k]['std'])}" for k in keys)])
    pooled = summary['pooled_oof']
    aggregation.append(['Pooled OOF', *(metric(pooled[k]) for k in keys)])
    errors = Counter((int(r['true_label']), int(r['predicted_label'])) for r in predictions if r['true_label'] != r['predicted_label'])
    top_errors = [(LABELS[i], LABELS[j], count, f"{100 * count / pooled['support'][i]:.1f}%")
                  for (i, j), count in errors.most_common(4)]
    low_subjects = []
    for row in sorted(subjects, key=lambda r: float(r['macro_f1']))[:3]:
        subject = row['subject_id']
        wrong = Counter((r['true_label'], r['predicted_label']) for r in predictions
                        if r['subject_id'] == subject and r['true_label'] != r['predicted_label'])
        pair, count = wrong.most_common(1)[0] if wrong else ((None, None), 0)
        detail = f'{LABELS[int(pair[0])]} → {LABELS[int(pair[1])]} ({count} cửa sổ)' if count else 'Không có lỗi'
        low_subjects.append([subject, row['fold_id'], row['windows'], metric(row['macro_f1']), metric(row['stress_recall']), detail])

    weights = [sum(s['label_distribution'][str(cls)] for s in audit['subjects']) for cls in range(3)]
    totals = sum(v['epochs'] for v in verification['folds'])
    batches = sum(v['iterations']['committed_iterations'] for v in verification['folds'])
    elapsed = str(timedelta(seconds=round(summary['training_duration_seconds'])))
    p = completed[1][1]['trainable_parameters']
    lines = [
        '# Báo cáo CNN-1D trên WESAD 14 kênh ở 64 Hz', '',
        f"Group `{group.name}` hoàn tất {len(completed)}/5 fold. Macro-F1 trung bình trên 15 người đạt "
        f"**{metric(summary['subject_level']['macro_f1']['mean'])} ± {metric(summary['subject_level']['macro_f1']['std'])}** "
        'với độ lệch chuẩn mẫu. Mục tiêu là đo khả năng phân loại Baseline, Stress và Amusement trên những người chưa có trong Train.', '',
        '## Dữ liệu và cửa sổ', '',
        f"Input lấy trực tiếp từ `{cfg['data']['processed_dir']}`, do {md('notebook 02', ROOT / manifest['preprocessing_notebook']['path'])} xuất. "
        f"Có 15 mảng `float32` dạng `(N, 14)`, 8 kênh ngực và 6 kênh cổ tay theo thứ tự YAML. Fingerprint nguồn `{audit['dataset_fingerprint']}`. "
        'Toàn bộ mảng được mở read-only bằng memory mapping. Runner dùng phần xuất 64 Hz có sẵn.', '',
        'Đoạn nhãn 700 Hz có đầu phải loại trừ. Tại 64 Hz, lấy `ceil(start × 64/700)` đến `floor(end × 64/700)` theo khoảng nửa mở. '
        'Mỗi lưới cửa sổ được neo ở đầu vùng nhãn hợp lệ. Cửa sổ 30 giây, stride 15 giây, 1.920 mẫu, overlap 50%. '
        'Nhãn nguồn 1/2/3 thành 0/1/2. Mask `True` bắt buộc ở mọi mẫu của cửa sổ.', '',
        f"Có **{audit['windows']:,}** cửa sổ hợp lệ: {weights[0]} Baseline, {weights[1]} Stress và {weights[2]} Amusement. "
        f"Mask loại {audit['rejected_quality_windows']} cửa sổ ứng viên. Các tệp nguồn, thời gian, định dạng và SHA256 có trong "
        f"{md('kiểm tra dữ liệu', group / 'window_index/common_rate_index.json')} và {md('window manifest', group / 'window_index/windows.csv')}.", '',
        format_table(['Người', 'Baseline', 'Stress', 'Amusement', 'Cửa sổ hợp lệ', 'Mẫu mask False', 'Cửa sổ bị loại'],
                     [[s['subject_id'], *(s['label_distribution'][str(k)] for k in range(3)), s['shape'][0],
                       s['invalid_samples_64'], s['rejected_quality_windows']] for s in audit['subjects']]), '',
        '## Grouped 5-Fold và chuẩn hóa', '',
        f"Dùng {md('split manifest seed 42', group / 'split_manifest.json')} (`{manifest['manifest_checksum']}`). "
        'Mỗi fold chứa 9 người Train, 3 Validation, 3 Test. Mọi người xuất hiện đúng một lần trong Test. '
        'Mean/std từng kênh tính từ các timestep cửa sổ Train, tính cả overlap, dùng `ddof=0`, `std < 1e-8` thay bằng 1. '
        'Validation/Test chỉ nhận phép biến đổi bằng scaler Train. Trọng số lớp là `N_train/(3*n_class_train)`.', '',
        format_table(['Fold', 'Train', 'Validation', 'Test'], [[f['fold_id'], *(', '.join(f[x]) for x in ('train', 'validation', 'test'))] for f in split['folds']]), '',
        '## CNN-1D và huấn luyện', '',
        f"Model có {p:,} tham số trainable. Mỗi fold khởi tạo CNN và Adam từ seed 42, fit scaler riêng, đánh giá Validation sau mỗi epoch. "
        'Sau khi chọn best checkpoint, nạp weights của chính fold rồi đánh giá Test. ', '',
        format_table(['Lớp', 'Thiết lập', 'Shape đầu ra'], [
            ['Input', '14 kênh × 1.920 mẫu', '[B, 14, 1920]'],
            ['Khối 1', 'Conv1d k=7 p=3, BatchNorm, ReLU, MaxPool 2', '[B, 32, 960]'],
            ['Khối 2', 'Conv1d k=5 p=2, BatchNorm, ReLU, MaxPool 2', '[B, 64, 480]'],
            ['Khối 3', 'Conv1d k=3 p=1, BatchNorm, ReLU', '[B, 128, 480]'],
            ['Global pooling', 'AdaptiveAvgPool1d(1), Flatten', '[B, 128]'],
            ['Head', 'Linear 128→64, ReLU, Dropout 0,3, Linear 64→3', '[B, 3]'],
        ]), '',
        f"{md('Effective config', group / 'config.yaml')} đặt Adam, batch {tc['batch_size']}, LR {tc['learning_rate']}, "
        f"weight decay {tc['weight_decay']}, weighted cross-entropy, gradient clip L2 {tc['gradient_clip_norm']}, tối đa {tc['max_epochs']} epochs. "
        'LR cố định. Chọn checkpoint bằng trung bình Macro-F1 của ba người Validation. Khi bằng trong `1e-12`, ưu tiên loss thấp hơn. '
        f"Early stopping có patience {tc['early_stopping_patience']} và min_delta {tc['early_stopping_min_delta']}. Tổng cộng {totals} epochs, "
        f"{batches:,} mini-batches đã commit và thời lượng cộng dồn {elapsed}. "
        f"Xem {md('môi trường', group / 'system.json')}, {md('source snapshot', group / 'source')} và {md('huấn luyện từng fold', viz / 'training/seed_42')}.", '',
        '## Hiệu suất Test', '',
        'Macro-F1 lấy trung bình F1 của nhãn cố định `[0,1,2]`. Balanced Accuracy lấy trung bình recall của ba nhãn. '
        'Lớp vắng mặt được gán điểm 0 khi tính trung bình theo lớp. Stress Recall là recall của nhãn 1. Mỗi người có một bộ dự đoán Test riêng.', '',
        format_table(['Fold', 'Cửa sổ T/V/Test', 'Epochs', 'Best', 'Val F1/người', 'Macro-F1', 'Balanced Accuracy', 'Accuracy', 'Stress Recall', 'Checkpoint', 'Log'], fold_lines), '',
        format_table(['Cách tổng hợp', 'Macro-F1', 'Balanced Accuracy', 'Accuracy', 'Stress Recall'], aggregation), '',
        'Các dòng trung bình dùng độ lệch chuẩn mẫu `ddof=1`. Pooled OOF tính chỉ số một lần trên toàn bộ dự đoán đã ghép. '
        f"Có {verification['oof_windows']} window ID duy nhất và {verification['test_subjects']} người Test. "
        f"{md('OOF predictions', result / 'oof_predictions.csv')} · {md('Summary JSON', result / 'summary.json')}.", '',
        format_table(['Fold', 'Subject', 'Windows', 'Macro-F1', 'Balanced Accuracy', 'Accuracy', 'Stress Recall'], subject_lines), '',
        '## Lỗi dự đoán', '',
        format_table(['Nhãn thật / dự đoán', *LABELS], [[label, *row] for label, row in zip(LABELS, pooled['confusion_matrix'])]), '',
        format_table(['Lớp', 'Support', 'Precision', 'Recall', 'F1'],
                     [[label, pooled['support'][i], *(metric(pooled[k][i]) for k in ('precision_per_class', 'recall_per_class', 'f1_per_class'))]
                      for i, label in enumerate(LABELS)]), '',
        format_table(['Nhãn thật', 'Dự đoán sai thành', 'Cửa sổ', '% trong lớp thật'], top_errors), '',
        format_table(['Ba người có Macro-F1 thấp nhất', 'Fold', 'Windows', 'Macro-F1', 'Stress Recall', 'Cặp nhầm nhiều nhất'], low_subjects), '',
        'Lỗi theo người phản ánh tập Test của fold tương ứng. Dữ liệu quan sát chỉ cho biết kiểu nhầm lẫn, '
        'chưa xác định được nguyên nhân ở chất lượng cảm biến, khác biệt cá nhân hay đặc trưng mô hình.', '',
        '## Tracking và truy xuất', '',
        'Dùng TrainingTracker, RunLogger, ConsoleLogger và TensorBoardTracker hiện có. Batch/epoch CSV, JSONL, best/last và completion hash '
        'giữ cùng quy ước. Bổ sung hai scalar theo tiêu chí Validation mới và metadata nguồn cho thí nghiệm Common-rate. '
        'Checkpoint resume lưu trạng thái optimizer và RNG ở ranh giới epoch. Các fold có completion hash hợp lệ được bỏ qua.', '',
        '```powershell', f'python -m tensorboard.main --logdir runs/{EXPERIMENT}/{group.name} --port 6006', '```', '',
        f"Mở http://localhost:6006. Kiểm chứng đã đọc {verification['tensorboard_files']} event files, {verification['checkpoint_count']} checkpoints "
        f"và đối chiếu {verification['verified_figure_files']} file hình. {md('Bằng chứng', result / 'verification.json')} · "
        f"{md('Hình PNG/SVG', viz)} · {md('Research Journal', EXPERIMENT_DIR / JOURNAL_NAME)}.", '',
        '## Giới hạn và nghiên cứu tiếp theo', '',
        'Thí nghiệm dùng 15 người, một split seed và một training seed trong môi trường phòng thí nghiệm. Các cửa sổ chồng lấp trong một bản ghi '
        'có tương quan, nên độ lệch chuẩn giữa người phản ánh biến thiên quan sát được. Kết quả của nhánh 32 Hz khác cả đầu vào lẫn tiêu chí chọn checkpoint. '
        'So sánh CNN-1D, GRU và CNN-GRU cần cùng manifest, dữ liệu, tiêu chí lựa chọn và cách tổng hợp. '
        'Ablation tiếp theo cần giữ cùng window IDs và chỉ thay tập kênh trước khi fit scaler theo Train.', '',
        '## Tài liệu tham khảo', '',
        *[f'- [{ref[0]}]({ref[1]}). {ref[-1]}' for ref in REFERENCES], '',
    ]
    path = result / 'report.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path


def write_journal(template, output, run, failed_starts=()):
    from openpyxl import load_workbook

    group, result, manifest, folds, completed, final = run
    template_hash = hashlib.sha256(template.read_bytes()).hexdigest()
    workbook = load_workbook(template)
    if workbook.sheetnames != SHEETS or template.resolve() == output.resolve():
        raise ValueError('Expected the six-sheet team template and a separate output.')
    unit = EXPERIMENT + '\n' + group.name
    if output.exists():
        previous = load_workbook(output, read_only=True)
        try:
            if previous.sheetnames != SHEETS or previous['Thí nghiệm']['B2'].value != unit:
                raise FileExistsError('The journal belongs to another group or is not the generated journal.')
        finally:
            previous.close()
    styles = {sheet.title: [copy.copy(sheet.cell(3 if sheet.title in ('Thí nghiệm', 'Nhật ký') else 2, i)._style)
                            for i in range(2, sheet.max_column + 1)] for sheet in workbook}
    values = {sheet.title: tuple(sheet.cell(1, i).value for i in range(2, sheet.max_column + 1)) for sheet in workbook}
    for sheet in workbook:
        sheet.delete_rows(2, max(0, sheet.max_row - 1))

    def add(name, data, links=None, height=42):
        sheet = workbook[name]
        row = sheet.max_row + 1
        for i, value in enumerate(data, start=2):
            cell = sheet.cell(row, i, value)
            cell._style = copy.copy(styles[name][i - 2])
            alignment = copy.copy(cell.alignment)
            alignment.wrap_text, alignment.vertical = True, 'center'
            cell.alignment = alignment
            if isinstance(value, datetime):
                cell.number_format = 'dd/mm/yyyy hh:mm'
            if links and i in links:
                target = links[i]
                cell.hyperlink = target if isinstance(target, str) else link(output.parent, target)
                font = copy.copy(cell.font)
                font.color, font.underline = '0563C1', 'single'
                cell.font = font
        sheet.row_dimensions[row].height = height

    index = group / 'window_index/common_rate_index.json'
    audit = read_json(index)['audit']
    status = 'Completed' if final else ('Failed' if manifest['status'] == 'failed' else 'Partial')
    starts = [m['started_at'] for _, m in folds.values()]
    start = local_time(min(starts)) if starts else None
    count = len(completed)
    scores = f'{count}/5 fold có completion marker.'
    summary = None
    if final:
        summary = read_json(result / 'summary.json')['per_seed']['42']
        sm, ba = (summary['subject_level'][k] for k in ('macro_f1', 'balanced_accuracy'))
        scores += f" Macro-F1/người {sm['mean']:.4f} ± {sm['std']:.4f}. Balanced Accuracy/người {ba['mean']:.4f} ± {ba['std']:.4f}."
    add('Thí nghiệm', [unit, start, None, 'CNN-1D, đủ 14 kênh ở 64 Hz, cửa sổ 30 s / bước 15 s. Fold 9/3/3, seed 42.',
                        scores, 'YAML chính thức', 'Logs và checkpoints', status],
        {7: EXPERIMENT_DIR / 'config.yaml', 8: group}, height=72)

    audit_steps = [
        ('Khảo sát pipeline hiện có', 'Tái sử dụng split, Dataset, scaler, Trainer, checkpoint, aggregation và tracking. '
         'Cấu hình mới định tuyến run riêng.', ROOT / 'scripts/run_grouped_cv.py', 'Đối chiếu các tệp xuất notebook 02.'),
        ('Xác minh dữ liệu và lập Window Manifest', f"15 người, 14 cột float32, timeline 64 Hz, "
         f"{audit['windows']} cửa sổ hợp lệ. Mask loại {audit['rejected_quality_windows']} cửa sổ.", index, 'Dùng manifest để đối chiếu Test IDs.'),
        ('Kiểm tra scaler, split và CNN', 'Fit theo 9 người Train trong mỗi fold. Input 14 × 1920, 47.075 tham số. '
         'Validation chọn Macro-F1 trung bình theo người.', group / 'split_manifest.json', 'Chạy tuần tự cả năm fold.'),
    ]
    for action, outcome, evidence, next_step in audit_steps:
        add('Nhật ký', [start, None, EXPERIMENT, action, outcome, evidence.name, next_step], {7: evidence}, height=64)
    for fid, (folder, meta) in folds.items():
        done = fid in completed
        evidence = folder / ('metrics/test.json' if done else 'logs/training.log')
        if done:
            m = meta['test_metrics']
            outcome = (f"{meta['completed_epochs']} epochs, best {meta['best_epoch']}, "
                       f"Test Macro-F1 {m['macro_f1']:.4f}, Balanced Accuracy {m['balanced_accuracy']:.4f}, "
                       f"Stress Recall {m['stress_recall']:.4f}.")
        else:
            log = folder / 'logs/epochs.csv'
            epochs = read_csv(log) if log.is_file() else []
            outcome = f"Trạng thái {meta['status']}. Có {len(epochs)} epochs ghi vào CSV."
        follow_up = 'Chuyển fold kế tiếp.' if done and fid < 5 else ('Kiểm chứng đủ OOF.' if done else 'Theo dõi lượt chạy hiện tại.')
        add('Nhật ký', [local_time(meta.get('ended_at', meta['started_at'])), None, EXPERIMENT, f'Fold {fid}', outcome,
                        f'Fold {fid} · {evidence.name}', follow_up], {7: evidence} if evidence.is_file() else None, height=58)
    if final:
        verified = read_json(result / 'verification.json')
        add('Nhật ký', [local_time(verified['verified_at']), None, EXPERIMENT, 'Đối chiếu OOF, subject metrics, checkpoints và TensorBoard',
                        f"{verified['test_subjects']} người, {verified['oof_windows']} OOF, {verified['checkpoint_count']} checkpoints, "
                        f"{verified['tensorboard_files']} tệp sự kiện, {verified['verified_figures']} hình đã xác minh.",
                        'verification.json', 'Đối chiếu với GRU/CNN-GRU cùng giao thức.'], {7: result / 'verification.json'}, height=66)
        visual = ROOT / verified['figure_directory']
        for description, name in [
            ('Loss Train/Validation của năm fold', 'cross_fold/seed_42/learning_loss.png'),
            ('Macro-F1 Train/Validation theo epoch', 'cross_fold/seed_42/learning_macro_f1.png'),
            ('Macro-F1 theo người trên Validation, Fold 1', 'training/seed_42/fold_01/validation_subject_macro_f1.png'),
            ('Confusion Matrix OOF chuẩn hóa theo hàng', 'cross_fold/seed_42/aggregate_confusion_normalized.png'),
            ('Macro-F1 Test theo 15 người', 'subject_metrics/seed_42/macro_f1.png'),
            ('Macro-F1 Test theo fold', 'cross_fold/seed_42/macro_f1.png'),
            ('Precision/Recall/F1 theo lớp, Test Fold 1', 'class_metrics/seed_42/fold_01/precision_recall_f1.png'),
        ]:
            figure = visual / name
            if not figure.is_file():
                raise FileNotFoundError(figure)
            add('Hình', [EXPERIMENT, description, figure.name], {4: figure})
    for failed in failed_starts:
        failure = ROOT / 'runs' / EXPERIMENT / failed
        log = failure / 'logs/errors.log'
        first_line = log.read_text(encoding='utf-8').splitlines()[0]
        add('Debug', [local_time(first_line), EXPERIMENT, f'{failed}: FileNotFoundError trước Fold 1.',
                      'Biến đường dẫn dữ liệu trùng tên với biến vòng lặp sao lưu source.',
                      'Đổi biến trong vòng lặp thành source_file và khởi động group mới.',
                      f"Group {group.name} đã tạo window index hợp lệ, {count}/5 fold có completion marker.", 'Traceback gốc'],
            {8: log}, height=78)
    link_check = result / 'journal_link_check.json'
    if link_check.is_file():
        observation = read_json(link_check)
        add('Debug', [local_time(observation['observed_at']), EXPERIMENT,
                      'Kiểm tra hyperlink Excel báo PNG không tồn tại sau khi cả năm fold đã qua hậu kiểm.',
                      f"Trên Windows, đường dẫn chưa rút gọn dài {observation['unresolved_length']} ký tự. "
                      f"Đường dẫn đã rút gọn dài {observation['normalized_length']} ký tự và tệp tồn tại.",
                      'Chuẩn hóa đường dẫn trước khi kiểm tra, giữ hyperlink tương đối trong workbook.',
                      'Đọc lại workbook và kiểm tra tất cả hyperlink nội bộ.', 'Đối chiếu hai đường dẫn'],
            {8: link_check}, height=81)
    for label, url, *columns in REFERENCES:
        add('Tham khảo', [label, *columns], {2: url}, height=75)
    tasks = [
        ('Kiểm chứng năm fold, xuất báo cáo và nhật ký', result / 'report.md', 'Checkpoint, OOF, biểu đồ và event files đã kiểm tra.', status),
        ('Phân tích người có Macro-F1 thấp và cặp lớp nhầm', result / 'report.md', 'Có số lỗi theo người và lớp từ OOF.', status),
        ('Chạy GRU và CNN-GRU với cùng split và window IDs', EXPERIMENT_DIR / 'config.yaml', 'Đủ 5 fold mỗi kiến trúc và metrics theo 15 người.', 'Chưa làm'),
        ('Ablation theo nhóm cảm biến với cùng giao thức', EXPERIMENT_DIR / 'README.md', 'So sánh trên cùng cửa sổ, scaler fit trên Train từng fold.', 'Chưa làm'),
    ]
    for task, location, condition, state in tasks:
        add('Việc cần làm', [task, None, location.name, condition, state], {4: location} if location.is_file() else None, height=54)

    widths = {'Thí nghiệm': [33, 21, 18, 49, 57, 22, 25, 15], 'Hình': [32, 48, 40],
              'Nhật ký': [21, 18, 32, 46, 55, 32, 50], 'Debug': [21, 32, 44, 45, 45, 45, 25],
              'Tham khảo': [40, 38, 39, 31, 45, 47], 'Việc cần làm': [48, 18, 22, 52, 20]}
    for sheet in workbook:
        for i, width in enumerate(widths[sheet.title], start=2):
            letter = sheet.cell(1, i).column_letter
            sheet.column_dimensions[letter].width = max(width, sheet.column_dimensions[letter].width)
    with TemporaryDirectory(dir=output.parent, prefix='.journal-') as tmp:
        target = Path(tmp) / output.name
        workbook.save(target)
        saved = load_workbook(target)
        try:
            assert saved.sheetnames == SHEETS
            assert saved['Thí nghiệm']['I2'].value == status
            for sheet in saved:
                assert all(cell.value is None for cell in sheet['A'])
                assert tuple(sheet.cell(1, i).value for i in range(2, sheet.max_column + 1)) == values[sheet.title]
                for row in sheet.iter_rows():
                    for cell in row:
                        if cell.hyperlink and '://' not in cell.hyperlink.target:
                            target_path = Path(os.path.normpath(output.parent / cell.hyperlink.target))
                            assert target_path.exists(), cell.hyperlink.target
        finally:
            saved.close()
        assert hashlib.sha256(template.read_bytes()).hexdigest() == template_hash
        target.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group-id', required=True)
    parser.add_argument('--template', type=Path, default=ROOT.parent / 'Research_journal.xlsx')
    parser.add_argument('--failed-start', action='append', default=[])
    args = parser.parse_args()
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', value) for value in args.failed_start):
        parser.error('Invalid group ID for --failed-start.')
    run = load_run(args.group_id)
    group, result, manifest, _, completed, final = run
    if final:
        print(write_report(group, result, manifest, completed))
    journal = EXPERIMENT_DIR / JOURNAL_NAME
    write_journal(args.template, journal, run, args.failed_start)
    print(f"{journal} ({len(completed)}/5, {'verified' if final else manifest['status']})")


if __name__ == '__main__':
    main()

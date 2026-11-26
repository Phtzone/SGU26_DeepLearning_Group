"""Write a CSV research journal for a verified chest-temperature preprocessing group."""
import argparse
import csv
from collections import defaultdict
from datetime import timedelta
import re

from .report_common_rate import ROOT, link, local_time, read_json
from .research_journal import SCHEMA, cell_text, reference_text, write_journal as write_csv_journal


EXPERIMENT = 'common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1'
BASELINE = 'common_rate64_cnn1d_grouped5fold_v1'
BASELINE_GROUP = '20261123T072126Z_2c81daac'
EXPERIMENT_DIR = ROOT / 'experiments' / EXPERIMENT
ARCHIVE = EXPERIMENT_DIR / f'Research_journal_{EXPERIMENT}.xlsx'
ARCHIVE_GROUP = '20261126T025058Z_fa1c92ea'
STATUS = 'Preprocessing verified — Training not started'


def verified_group(group_id):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', group_id):
        raise ValueError('Invalid group ID.')
    result = ROOT / 'results' / EXPERIMENT / group_id
    evidence = result / 'preprocessing_verification.json'
    record = read_json(evidence)
    if (record['status'] != 'passed' or record['experiment_id'] != EXPERIMENT
            or record['group_id'] != group_id or record['baseline_group_id'] != BASELINE_GROUP
            or record['training_started'] is not False or record['training_epochs'] != 0):
        raise ValueError('The group must record completed preprocessing verification and zero training.')
    folds = sorted(record['fold_checks'], key=lambda fold: fold['fold_id'])
    if [fold['fold_id'] for fold in folds] != list(range(1, 6)):
        raise ValueError('The journal requires one verification record for each of the five folds.')
    return result, evidence, record, folds


def import_archived_workbook(group_id):
    if group_id != ARCHIVE_GROUP:
        raise ValueError(f'This workbook records only group {ARCHIVE_GROUP}.')
    verified_group(group_id)
    try:
        from openpyxl import load_workbook
    except ModuleNotFoundError as error:
        raise ValueError('Use a Python environment with openpyxl for the archived workbook import.') from error
    records = {}
    workbook = load_workbook(ARCHIVE, data_only=False)
    try:
        if workbook.sheetnames != [sheet for sheet, _ in SCHEMA.values()]:
            raise ValueError('Unexpected worksheet names or order in the archive.')
        if (workbook['Thí nghiệm']['B2'].value != f'{EXPERIMENT}\n{group_id}'
                or workbook['Thí nghiệm']['I2'].value != STATUS):
            raise ValueError(f'Unexpected group or status in {ARCHIVE}')
        for name, (sheet_name, header) in SCHEMA.items():
            sheet = workbook[sheet_name]
            if tuple(cell_text(c.value) for c in sheet[1]) != header:
                raise ValueError(f'Unexpected columns in {sheet_name}')
            records[name] = [
                tuple(reference_text(c.value, c.hyperlink.target if c.hyperlink else None) for c in row)
                for row in sheet.iter_rows(min_row=2, max_row=sheet.max_row, max_col=len(header))
            ]
    finally:
        workbook.close()
    return write_csv_journal(EXPERIMENT_DIR, records, import_archive=True)


def write_journal(group_id):
    result, evidence, record, folds = verified_group(group_id)
    unit = f'{EXPERIMENT}\n{group_id}'
    # Match the millisecond precision of timestamps already recorded in the archived workbook.
    rounded = local_time(record['verified_at']) + timedelta(microseconds=500)
    timestamp = rounded.replace(microsecond=rounded.microsecond // 1000 * 1000)
    files = {sheet: name for name, (sheet, _) in SCHEMA.items()}
    records = defaultdict(list)

    def append(name, values, links=None):
        row = [None, *values]
        for col, target in (links or {}).items():
            if not target.exists():
                raise FileNotFoundError(target)
            row[col - 1] = reference_text(row[col - 1], link(EXPERIMENT_DIR, target))
        records[files[name]].append(row)

    report = result / 'preprocessing_report.md'
    log = result / 'execution.log'
    normalization = result / 'normalization_statistics.csv'
    comparison = result / 'temperature_distribution_comparison.csv'
    stages = result / 'temperature_stage_statistics.csv'
    audit = ROOT / 'docs' / 'audits' / BASELINE
    baseline_index = ROOT / 'runs' / BASELINE / BASELINE_GROUP / 'window_index' / 'windows.csv'
    config = EXPERIMENT_DIR / 'config.yaml'

    append('Thí nghiệm', [unit, timestamp, None,
        'CNN-1D + Chest Temperature Median Centering + Z-score. '
        'Common-rate 64 Hz, 14 kênh, cửa sổ 30 giây và stride 15 giây. '
        'Giữ split 9/3/3 và seed 42. Trừ median từng cửa sổ trên chest_Temp rồi fit Z-score trên Train.',
        'Đã kiểm chứng tiền xử lý 5/5 fold và 2.134 cửa sổ duy nhất. Training epochs = 0.',
        'config.yaml', 'execution.log', STATUS], {7: config, 8: log})

    for description, outcome, target, next_step in [
        ('Chốt câu hỏi nghiên cứu', 'Baseline có mean chest_Temp ở Validation Fold 4 lệch khoảng −3,35 std Train.',
         audit / 'NORMALIZATION_AUDIT.md', 'Kiểm tra ảnh hưởng của offset và độ phân tán sau centering.'),
        ('Đối chiếu dữ liệu và giao thức',
         f'Dùng Window Manifest và split của baseline {BASELINE_GROUP}. Giữ nguyên nhãn và quality mask.',
         baseline_index, 'Xác minh từng cửa sổ qua Dataset và DataLoader.'),
        ('Kiểm tra trước và sau median centering',
         'Mười ba kênh còn lại, shape, ID và nhãn được giữ nguyên. Median chest_Temp của từng cửa sổ gần 0.',
         stages, 'Đo thay đổi scale sau khi fit Z-score.'),
    ]:
        append('Nhật ký', [timestamp, None, EXPERIMENT, description, outcome, target.name, next_step], {7: target})
    for fold in folds:
        number = fold['fold_id']
        lists = '/'.join(str(fold[f'{role}_windows']) for role in ('train', 'validation', 'test'))
        scaler = result / 'seed_42' / f'fold_{number:02d}' / 'normalization.json'
        outcome = (f"{lists} cửa sổ Train/Validation/Test. Max |mean z| Train {fold['train_max_abs_z_mean']:.2e}. "
                   f"Max |std z − 1| {fold['train_max_abs_z_std_error']:.2e}. "
                   f"Std chest_Temp Train {fold['train_chest_temp_baseline_std']:.6g} → "
                   f"{fold['train_chest_temp_centered_std']:.6g}.")
        append('Nhật ký', [timestamp, None, EXPERIMENT, f'Fit Train và xác minh tensor Fold {number}', outcome,
                           f'Fold {number:02d} · normalization.json', 'Dùng scaler Train để transform cả ba tập.'],
               {7: scaler})
    fold_04 = record['diagnostics']['fold_04']
    shift = (f"Fold 4 Validation − Train: mean z "
             f"{fold_04['baseline_validation_minus_train_mean_z']:.3f} → "
             f"{fold_04['centered_validation_minus_train_mean_z']:.3f}. "
             f"Hệ số 1/std tăng {fold_04['scale_amplification']:.2f} lần. "
             f"Tỷ lệ |z| > 5 trên Validation "
             f"{fold_04['baseline_validation_abs_z_gt_5_fraction']:.2%} → "
             f"{fold_04['centered_validation_abs_z_gt_5_fraction']:.2%}. "
             f"Max |z| {fold_04['baseline_validation_max_abs_z']:.2f} → "
             f"{fold_04['centered_validation_max_abs_z']:.2f}.")
    append('Nhật ký', [timestamp, None, EXPERIMENT, 'Đối chiếu phân phối Train và Validation', shift,
                       comparison.name, 'Đọc chênh lệch Fold 4 và độ lớn Z-score ở báo cáo.'], {7: comparison})
    append('Nhật ký', [timestamp, None, EXPERIMENT, 'Bàn giao tiền xử lý', 'Đã kiểm chứng 5/5 fold. Training epochs = 0.',
                       evidence.name, 'Đọc báo cáo và xác nhận trước khi huấn luyện.'], {7: evidence})

    figures = sorted((result / 'figures').glob('*.png'))
    if not figures:
        raise FileNotFoundError(result / 'figures')
    for figure in figures:
        for path in (figure, figure.with_suffix('.svg')):
            append('Hình', [EXPERIMENT, figure.stem.replace('_', ' '), path.name], {4: path})

    append('Tham khảo', ['Kiểm toán baseline CNN-1D', 'WESAD 64 Hz, 14 kênh',
                         'Dataset, split, scaler, checkpoint và phân phối', '5 fold theo subject',
                         'chest_Temp lệch rõ tại Validation Fold 4',
                         'Bằng chứng mô tả dịch chuyển, chưa cho biết tác động lên hiệu năng.'],
           {2: audit / 'AUDIT_REPORT.md'})
    append('Tham khảo', ['Thiết kế đối chứng centering', 'Nhiệt độ ngực theo cửa sổ',
                         'Trừ median trong cửa sổ và khảo sát tác động của mức nền', 'Đối chiếu Train và Validation',
                         'Mốc so sánh để thiết kế tiền xử lý',
                         'Thí nghiệm này fit lại cả mean và std Train sau centering theo yêu cầu mới.'],
           {2: audit / 'RECOMMENDED_NEXT_EXPERIMENTS.md'})
    append('Tham khảo', ['Chương 3 · Bộ dữ liệu và tiền xử lý', 'Đồng bộ tín hiệu WESAD ở 64 Hz',
                         'Windowing, quality mask và Z-score theo kênh', 'Grouped 5-fold, 9/3/3',
                         'Tài liệu thiết kế giao thức dùng lại',
                         'Đường dẫn hiện vật thực tế nằm trong báo cáo kiểm chứng của group.'],
           {2: ROOT.parent / 'Chuong_3_Bo_du_lieu_va_tien_xu_ly_Hoan_chinh.docx'})
    append('Việc cần làm', ['Xác minh pipeline trên năm fold', None, normalization.name,
                             'Cả 14 kênh có thống kê Train/Validation/Test cho từng fold.', STATUS],
           {4: normalization})
    append('Việc cần làm', ['Xem báo cáo và đánh giá rủi ro scale', None, report.name,
                             'Xác nhận cách đọc dịch chuyển Fold 4 và đuôi Z-score trước quyết định huấn luyện.',
                             'Chờ duyệt'], {4: report})
    return write_csv_journal(EXPERIMENT_DIR, records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group-id', required=True)
    parser.add_argument('--migrate-xlsx', action='store_true', help=f'Import the archived workbook for {ARCHIVE_GROUP}')
    args = parser.parse_args()
    action = import_archived_workbook if args.migrate_xlsx else write_journal
    try:
        print(action(args.group_id))
    except (OSError, ValueError, csv.Error, RuntimeError) as error:
        parser.exit(1, f'Could not write Research Journal: {error}\n')


if __name__ == '__main__':
    main()

"""Create a linked research journal from saved E1 evidence; never overwrite notes."""
import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

ROOT = Path(__file__).resolve().parents[2]
GROUP = '20261108T154303Z_474b38f9'
RUN = f'runs/experiment_e1/{GROUP}'
RESULT = f'results/experiment_e1/{GROUP}'
FIG = f'{RESULT}/enhanced_visualizations/scientific-v2_44c886afdd32da86'
REVIEW_DATE = '2026-11-13'


def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def local_time(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(
        timezone(timedelta(hours=7))).strftime('%Y-%m-%d %H:%M:%S +07:00')


def create(output):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing journal: {output}')
    summary = read_json(f'{RESULT}/summary.json')['per_seed']['42']
    metadata = [read_json(f'{RUN}/seed_42/fold_{fold:02d}/metadata.json')
                for fold in range(1, 6)]
    start = min(local_time(item['started_at']) for item in metadata)
    end = max(local_time(item['ended_at']) for item in metadata)
    wb = Workbook()
    wb.remove(wb.active)

    def sheet(name, headers, rows):
        ws = wb.create_sheet(name)
        ws.append(headers)
        for row in rows:
            ws.append(row)
        ws.freeze_panes = 'A2'
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.font = Font(bold=True, color='FFFFFF')
            cell.fill = PatternFill('solid', fgColor='24527A')
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical='top', wrap_text=True)
                if isinstance(cell.value, str) and (ROOT / cell.value).exists():
                    import os
                    cell.hyperlink = os.path.relpath(ROOT / cell.value, output.parent).replace('\\', '/')
                    cell.style = 'Hyperlink'
            ws.row_dimensions[row[0].row].height = 78
        for column in ws.columns:
            ws.column_dimensions[column[0].column_letter].width = 32
        return ws

    sheet('Huong_dan', ['Mục', 'Quy ước'], [
        ['Múi giờ', 'Asia/Saigon, UTC+07:00; ngày dạng YYYY-MM-DD.'],
        ['Phân biệt ngày', 'Ngày chạy lấy từ metadata. Ngày ghi hồi cứu E1: 2026-11-13.'],
        ['Thi_nghiem', 'Mỗi GROUP_ID một dòng; lần chạy thất bại/gián đoạn cũng phải ghi.'],
        ['Nhat_ky', 'Mỗi buổi làm một dòng, gồm việc đã làm và bước tiếp theo.'],
        ['Debug', 'Lưu lỗi thật, traceback, cách tái hiện, nguyên nhân, cách sửa và kiểm chứng.'],
        ['Đường dẫn', 'Nội dung tính từ gốc DL13-WESAD; hyperlink tính từ workbook. Giữ vị trí file.'],
        ['Notebook E1', 'Không dùng notebook. notebooks/ hiện chỉ có .gitkeep.'],
        ['Minh chứng lịch sử', 'Dòng E1 tổng hợp từ artifacts có sẵn; không tái dựng nhật ký debug cũ.'],
        ['Cách thêm dòng', 'Thêm dưới bảng và mở rộng vùng Filter; điền liên kết mới cho đúng GROUP_ID.'],
        ['Không ghi đè', 'Không chạy generator đè lên workbook đã ghi. Sao lưu Excel trước khi chỉnh sửa.'],
        ['Hướng dẫn đầy đủ', 'experiments/journal/README.md'],
    ])
    fold = summary['fold_level']
    sheet('Thi_nghiem', [
        'Experiment ID', 'GROUP_ID', 'Bắt đầu +07:00', 'Kết thúc +07:00',
        'Ngày ghi nhật ký', 'Người làm', 'Mục tiêu / giả thuyết', 'Thay đổi',
        'Config', 'Script', 'Notebook / cell', 'Lệnh chạy', 'Seed / folds',
        'Trạng thái', 'Accuracy mean', 'Accuracy std', 'Macro-F1 mean', 'Macro-F1 std',
        'Cách tổng hợp', 'Kết quả', 'Log', 'Hình', 'Nhận xét', 'Bước tiếp'], [[
        'E1_CNN', GROUP, start, end, REVIEW_DATE, None,
        'Baseline nhận diện ba trạng thái cho người chưa có trong Train',
        'Baseline chính thức; tổng hợp hồi cứu từ artifacts', f'{RUN}/config.yaml',
        'scripts/run_grouped_cv.py', 'Không dùng notebook',
        'Lệnh tái lập: python -m scripts.run_grouped_cv --experiment experiments/e1_cnn/config.yaml; không xác minh được nguyên văn lệnh lịch sử',
        'Training seed 42; split seed 42; đủ 5 folds', 'completed',
        fold['accuracy']['mean'], fold['accuracy']['std'],
        fold['macro_f1']['mean'], fold['macro_f1']['std'],
        'Mean/std giữa 5 Test folds; std ddof=1', f'{RESULT}/summary.json',
        f'{RUN}/logs/training.log', FIG,
        'Biến thiên giữa folds; chưa có kết quả GRU/CNN-GRU để so sánh',
        'Triển khai GRU, giữ nguyên giao thức đánh giá'], ])
    sheet('Nhat_ky', ['Ngày +07:00', 'Người làm', 'Experiment / GROUP_ID',
        'Việc đã làm', 'Kết quả', 'Minh chứng', 'Kế hoạch tiếp'], [[
        REVIEW_DATE, None, f'E1_CNN / {GROUP}',
        'Rà soát source, README và artifacts; tạo sổ nhật ký có liên kết',
        'E1 đã có kết quả 5 folds; notebooks trống; E2/E3 chưa triển khai',
        'experiments/journal/README.md', 'Bổ sung mô hình GRU và ghi nhật ký từng buổi'], ])
    sheet('Debug', ['Mã lỗi', 'Ngày phát hiện +07:00', 'GROUP_ID',
        'Lỗi / biểu hiện', 'Cách tái hiện', 'Log / traceback',
        'Nguyên nhân / mức xác minh', 'Cách khắc phục / file sửa',
        'Lệnh kiểm chứng', 'Kết quả kiểm chứng', 'Trạng thái'], [])
    sheet('Ke_hoach', ['Mã', 'Ngày lập', 'Việc cần làm', 'Vị trí',
        'Điều kiện hoàn tất', 'Trạng thái', 'Người làm'], [
        ['P01', REVIEW_DATE, 'Xác định interpreter và vị trí WESAD trên máy hiện tại',
         '.env.example', 'Dry-run báo đủ 15 subjects; môi trường import dependencies được', 'planned', None],
        ['P02', REVIEW_DATE, 'Triển khai GRU và nối model vào pipeline',
         'src/models/gru.py', 'Cập nhật config validation, trainer, runner; kiểm tra shape và lượt chạy nhỏ', 'planned', None],
        ['P03', REVIEW_DATE, 'Chạy E2 GRU theo cùng giao thức E1',
         'experiments/e2_gru/config.yaml', 'Sau P02: đủ 5 folds, có log, metrics, hình và nhật ký', 'planned', None],
        ['P04', REVIEW_DATE, 'Triển khai và đánh giá CNN-GRU',
         'src/models/cnn_gru.py', 'Đủ 5 folds với giao thức cố định, có minh chứng', 'planned', None],
        ['P05', REVIEW_DATE, 'Đánh giá vai trò từng cảm biến',
         'experiments/ablation', 'Mỗi thiết kế một group riêng, bảng so sánh và kết luận có bằng chứng', 'planned', None],
    ])
    evidence = [
        ['Cấu hình đã chạy', f'{RUN}/config.yaml'],
        ['Source đã chạy', f'{RUN}/source'],
        ['Môi trường lịch sử', f'{RUN}/system.json'],
        ['Summary', f'{RESULT}/summary.json'],
        ['Kết quả từng fold', f'{RESULT}/fold_results.csv'],
        ['Kết quả từng người', f'{RESULT}/subject_metrics.csv'],
        ['OOF predictions', f'{RESULT}/oof_predictions.csv'],
        ['Learning Macro-F1', f'{FIG}/cross_fold/seed_42/learning_macro_f1.png'],
        ['Confusion matrix', f'{RESULT}/figures/aggregate_confusion_matrix.png'],
    ]
    for index in range(1, 6):
        base = f'{RUN}/seed_42/fold_{index:02d}'
        for name in ['iterations.csv', 'epochs.csv', 'validation.csv',
                     'training.log', 'errors.log', 'events.jsonl']:
            evidence.append([f'Fold {index}: {name}', f'{base}/logs/{name}'])
        evidence.append([f'Fold {index}: checkpoint', f'{base}/checkpoints'])
    for _, relative in evidence:
        if not (ROOT / relative).exists():
            raise FileNotFoundError(f'Missing evidence: {relative}')
    sheet('Minh_chung', ['Nội dung', 'Đường dẫn từ gốc dự án'], evidence)
    output.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output)
    print(f'Created: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'experiments/journal/research_journal.xlsx')
    create(parser.parse_args().output.resolve())

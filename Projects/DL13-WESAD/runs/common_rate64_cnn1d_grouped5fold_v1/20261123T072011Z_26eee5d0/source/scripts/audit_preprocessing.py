"""Review existing WESAD preprocessing without changing data or training models."""
import argparse
from collections import Counter
import csv
import gc
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.prepare_data import cache_path
from src.data.inspection import load_subject, RATES
from src.data.resampling import resample_signal
from src.data.splits import SUBJECTS
from src.utils.config import ROOT, load_config, raw_root
from src.utils.tracking import file_hash, identifier, read_json, write_json

CLASSES = ['baseline', 'stress', 'amusement']
CHANNELS = ['ACC-x', 'ACC-y', 'ACC-z', 'BVP', 'EDA']
REFERENCE = '20261108T154303Z_474b38f9'


def read_rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(output, subject='S2'):
    config = load_config('experiments/e1_cnn/config.yaml')
    cache = cache_path(config)
    reference = ROOT / 'runs/experiment_e1' / REFERENCE / 'seed_42/fold_01'
    normalization = read_json(reference / 'normalization.json')
    if subject not in normalization['train_subjects']:
        raise ValueError('Waveform examples must come from the reference fold Train subjects.')
    if output.exists():
        raise FileExistsError('Use a new output directory; existing reports are preserved.')
    output.mkdir(parents=True)
    counts, quality, input_hashes = [], [], {}
    for sid in SUBJECTS:
        folder = cache / sid
        metadata = read_json(folder / 'metadata.json')
        raw_path = raw_root() / sid / (sid + '.pkl')
        raw_hash = file_hash(raw_path)
        assert raw_hash == metadata['raw_sha256'], (sid, 'raw hash mismatch')
        assert metadata['data_config'] == config['data'], (sid, 'cache config mismatch')
        for name, expected in metadata['files'].items():
            assert file_hash(folder / name) == expected, (sid, name, 'cache hash mismatch')
        windows = np.load(folder / 'windows.npy', mmap_mode='r', allow_pickle=False)
        rows = read_rows(folder / 'windows.csv')
        assert list(windows.shape) == metadata['shape']
        assert windows.shape[1:] == (5, 960) and len(rows) == len(windows)
        assert all(r['subject_id'] == sid and int(r['original_label']) == int(r['label']) + 1 for r in rows)
        distribution = Counter(int(row['label']) for row in rows)
        assert set(distribution) <= {0, 1, 2}
        assert {str(k): distribution[k] for k in range(3)} == metadata['label_distribution']
        counts.append({'subject': sid, **{name: distribution[i] for i, name in enumerate(CLASSES)},
                       'total': len(rows)})
        for channel, name in enumerate(CHANNELS):
            array = windows[:, channel, :]
            assert np.isfinite(array).all(), (sid, name, 'NaN/Inf')
            quality.append({'subject': sid, 'channel': name,
                            'minimum': float(array.min()), 'maximum': float(array.max()),
                            'constant_windows': int(np.count_nonzero(np.ptp(array, axis=1) == 0))})
        input_hashes[sid] = {'raw_sha256': raw_hash, 'cache_files': metadata['files']}
        del windows
        print(f'Checked {sid}: {len(rows)} windows', flush=True)

    totals = {name: sum(row[name] for row in counts) for name in CLASSES}
    total = sum(totals.values())
    proportions = [{'class': name, 'windows': value, 'percentage': value / total * 100}
                   for name, value in totals.items()]
    write_csv(output / 'windows_by_subject.csv', counts)
    write_csv(output / 'class_distribution.csv', proportions)
    write_csv(output / 'channel_checks.csv', quality)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    bottom = np.zeros(len(counts))
    for name in CLASSES:
        values = np.array([row[name] for row in counts])
        axes[0].bar([row['subject'] for row in counts], values, bottom=bottom, label=name)
        bottom += values
    axes[0].set(title='Retained windows per subject', ylabel='Windows')
    axes[0].legend()
    axes[1].bar(list(totals), list(totals.values()), color=['#4c78a8', '#e45756', '#72b7b2'])
    for i, row in enumerate(proportions):
        axes[1].text(i, row['windows'], f"{row['windows']} ({row['percentage']:.1f}%)", ha='center', va='bottom')
    axes[1].set(title='Class distribution (overlapping windows)', ylabel='Windows')
    axes[1].margins(y=.15)
    fig.savefig(output / 'class_distribution.png', dpi=160)
    plt.close(fig)

    print(f'Loading {subject} for waveform examples only', flush=True)
    data = load_subject(raw_root(), subject)
    wrist = data['signal']['wrist']
    length = int(round(len(data['label']) / 700 * 32))
    resampled = {name: resample_signal(wrist[name], RATES['wrist'][name], 32, length)
                 for name in ['ACC', 'BVP', 'EDA']}
    combined = np.concatenate(list(resampled.values()), axis=1)
    windows = np.load(cache / subject / 'windows.npy', mmap_mode='r', allow_pickle=False)
    rows = read_rows(cache / subject / 'windows.csv')
    examples = []
    for label, name in enumerate(CLASSES):
        index, row = next((i, r) for i, r in enumerate(rows) if int(r['label']) == label)
        start = int(round(float(row['start_time']) * 32))
        np.testing.assert_allclose(combined[start:start + 960].T, windows[index], rtol=1e-5, atol=1e-6)
        mean = np.asarray(normalization['mean'], dtype=np.float32)
        std = np.asarray(normalization['std'], dtype=np.float32)
        normalized = ((windows[index] - mean[:, None]) / std[:, None]).astype(np.float32)
        # First 10 seconds of a fixed, first-retained window; not chosen for appearance.
        begin, end = float(row['start_time']), float(row['start_time']) + 10
        fig, axes = plt.subplots(5, 3, figsize=(15, 11), layout='constrained')
        for channel, channel_name in enumerate(CHANNELS):
            sensor = 'ACC' if channel < 3 else channel_name
            axis = channel if channel < 3 else 0
            rate = RATES['wrist'][sensor]
            lo, hi = int(np.ceil(begin * rate)), int(np.ceil(end * rate))
            raw_x = np.arange(lo, hi) / rate - begin
            raw_y = wrist[sensor][lo:hi, axis]
            processed = windows[index, channel, :320]
            time = np.arange(320) / 32
            axes[channel, 0].plot(raw_x, raw_y, linewidth=.8)
            axes[channel, 1].plot(time, processed, linewidth=.8)
            axes[channel, 2].plot(time, normalized[channel, :320], linewidth=.8)
            low, high = min(raw_y.min(), processed.min()), max(raw_y.max(), processed.max())
            margin = max(float(high - low) * .05, 1e-6)
            for column in [0, 1]:
                axes[channel, column].set_ylim(float(low - margin), float(high + margin))
            axes[channel, 0].set_ylabel(channel_name + '\noriginal units')
            for column in range(3):
                axes[channel, column].grid(alpha=.2)
                axes[channel, column].set_xlim(0, 10)
                if channel == 4:
                    axes[channel, column].set_xlabel('Seconds from excerpt start')
        for column, title in enumerate(['Original sampling rates', 'Resampled to 32 Hz', 'Standardized (Fold 1 Train scaler)']):
            axes[0, column].set_title(title)
        fig.suptitle(f'{subject} / {name} / first retained window / excerpt starts at {begin:.4f}s')
        filename = f'{subject}_{name}_preprocessing.png'
        fig.savefig(output / filename, dpi=140)
        plt.close(fig)
        examples.append({'subject': subject, 'class': name, 'window_id': row['window_id'],
                         'excerpt_start_seconds': begin, 'excerpt_duration_seconds': 10,
                         'file': filename, 'cache_reconstruction_matches': True})
        print(f'Plotted {filename}; resampled data matches cache', flush=True)
    del data, wrist, resampled, combined, windows
    gc.collect()
    write_json(output / 'audit.json', {
        'status': 'passed', 'cache': str(cache), 'raw_root': str(raw_root()),
        'total_windows': total, 'class_distribution': proportions,
        'subject_counts': counts, 'input_hashes': input_hashes,
        'waveform_examples': examples, 'normalization_reference': str(reference / 'normalization.json'),
        'normalization_sha256': file_hash(reference / 'normalization.json'),
        'waveform_policy': 'First retained window per class from a Fold 1 Train subject; display first 10 seconds.',
        'limitations': 'Counts are overlapping windows, not independent observations. Waveform review covers only the displayed excerpts. No automated physiological artifact rejection or filter tuning.',
        'data_or_training_artifacts_modified': False,
    })
    lines = ['# Kiểm tra dữ liệu trước khi làm E2/E3', '',
             f'Có {len(counts)} người và {total} cửa sổ. Mỗi cửa sổ dài 30 giây, bước 15 giây.', '',
             '| Lớp | Số cửa sổ | Tỷ lệ |', '|---|---:|---:|']
    lines += [f"| {r['class']} | {r['windows']} | {r['percentage']:.2f}% |" for r in proportions]
    constant = sum(row['constant_windows'] for row in quality)
    constant_channels = sorted({row['channel'] for row in quality if row['constant_windows']})
    lines += ['', 'Đã kiểm tra hash dữ liệu gốc/cache, cấu hình cache, nhãn, kích thước và NaN/Inf cho đủ 15 người.',
              f'Có {constant} cặp cửa sổ–kênh có giá trị không đổi hoàn toàn; xem channel_checks.csv. Đây là thống kê, không tự động coi là lỗi hay loại dữ liệu.', '',
              'Các trường hợp này thuộc các kênh: ' + ', '.join(constant_channels) + '. Một cửa sổ có thể được đếm ở nhiều kênh.', '',
              f'Ba hình tín hiệu lấy từ {subject}, thuộc Train của Fold 1. Mỗi lớp lấy cửa sổ đầu tiên, hiển thị 10 giây đầu.',
              'Ba cột lần lượt là tín hiệu gốc, sau resampling về 32 Hz và sau chuẩn hóa bằng scaler Train đã lưu.',
              'Resampling được tính lại từ dữ liệu gốc và đối chiếu khớp cache cho cả ba cửa sổ.', '',
              '- [Bảng số cửa sổ từng người](windows_by_subject.csv)',
              '- [Phân bố ba lớp](class_distribution.png)',
              '- [Kiểm tra từng kênh](channel_checks.csv)']
    lines += [f"- [{r['class']} — {subject}]({r['file']})" for r in examples]
    lines += ['', 'Chưa đổi tiền xử lý hay chạy mô hình. Ảnh chỉ minh họa các đoạn đã chọn, chưa đủ để kết luận toàn bộ dữ liệu không có nhiễu.',
              'Các cửa sổ chồng lấp nên số cửa sổ không phải số quan sát độc lập. Khi báo cáo metric chính vẫn tính theo người.',
              'Nếu chưa thấy vấn đề cụ thể cần xử lý, giữ nguyên pipeline này khi so sánh CNN, GRU và CNN-GRU.']
    (output / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Report: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--subject', default='S2')
    args = parser.parse_args()
    destination = args.output or ROOT / 'results/preprocessing_review' / identifier()
    audit(destination.resolve(), args.subject)

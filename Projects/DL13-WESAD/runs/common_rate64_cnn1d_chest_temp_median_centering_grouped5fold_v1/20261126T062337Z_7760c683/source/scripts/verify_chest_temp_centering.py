"""Verify the chest-temperature preprocessing on the official index without training."""
from collections import Counter
from copy import deepcopy
import csv
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.common_rate import INDEX_METADATA
from src.data.dataset import WindowDataset
from src.data.normalization import ChannelScaler
from src.utils.config import ROOT, load_config
from src.utils.tracking import digest, file_hash, identifier, now, read_json, write_json


BASELINE = 'common_rate64_cnn1d_grouped5fold_v1'
BASELINE_GROUP = '20261123T072126Z_2c81daac'
EXPERIMENT = 'common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1'
BASE_RUN = ROOT / 'runs' / BASELINE / BASELINE_GROUP
BASE_RESULTS = ROOT / 'results' / BASELINE / BASELINE_GROUP
ROLES = ('train', 'validation', 'test')
CHANNEL = 'chest_Temp'
WINDOW = 1920
LOGGER = logging.getLogger(EXPERIMENT)


def read_rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def context():
    original = read_json(BASE_RUN / 'experiment_manifest.json')
    assert original['status'] == 'completed' and original['group_id'] == BASELINE_GROUP
    assert original == read_json(BASE_RESULTS / 'experiment_manifest.json')
    path = ROOT / 'experiments' / EXPERIMENT / 'config.yaml'
    config = load_config(path)
    centering = config['preprocessing']['window_centering']
    assert centering == {'enabled': True, 'method': 'median', 'channels': [CHANNEL]}
    comparable = deepcopy(config)
    del comparable['preprocessing']
    comparable['experiment'] = {
        **comparable['experiment'],
        'id': BASELINE,
        'status': original['effective_config']['experiment']['status'],
    }
    # The recorded baseline console progress differs because of --non-interactive.
    comparable['logging']['console']['progress'] = original['effective_config']['logging']['console']['progress']
    assert comparable == original['effective_config']
    assert config['experiment']['id'] == EXPERIMENT
    assert config['training']['seeds'] == [42] and config['split']['split_seed'] == 42
    source, index = Path(original['data_source']), Path(original['cache_path'])
    assert source == (ROOT / config['data']['processed_dir']).resolve() and index == BASE_RUN / 'window_index'
    index_meta = read_json(index / INDEX_METADATA)
    channels = read_json(source / 'config.json')['channels']
    assert channels == config['data']['channels'] == index_meta['audit']['policy']['channels']
    assert len(channels) == len(set(channels)) == 14
    split = read_json(BASE_RUN / 'split_manifest.json')
    assert digest(split) == original['manifest_checksum'] == (BASE_RUN / 'split_manifest.sha256').read_text().strip()
    assert read_json(ROOT / config['split']['manifest']) == split
    subjects = set(split['original_subject_order'])
    assert len(subjects) == 15 and [f['fold_id'] for f in split['folds']] == list(range(1, 6))
    for fold in split['folds']:
        parts = [set(fold[role]) for role in ROLES]
        assert [len(part) for part in parts] == [9, 3, 3]
        assert len(set.union(*parts)) == 15 and set.union(*parts) == subjects
    assert Counter(s for fold in split['folds'] for s in fold['test']) == Counter(subjects)
    return original, config, source, index, index_meta, split, channels


def protected_hashes(source, index_meta, config):
    folders = (BASE_RUN, BASE_RESULTS, ROOT / 'experiments' / BASELINE, ROOT / 'docs/audits' / BASELINE)
    paths = {p for directory in folders for p in directory.rglob('*') if p.is_file()}
    paths.update(source / name for name in index_meta['audit']['source_file_sha256'])
    paths.update((
        ROOT / config['split']['manifest'],
        ROOT / 'notebooks/02_preprocessing_common_rate.ipynb',
        ROOT.parent / 'Research_journal.xlsx',
        ROOT / 'experiments/journal/research_journal.xlsx',
    ))
    return {p.relative_to(ROOT.parent).as_posix(): file_hash(p) for p in sorted(paths)}


def inspect_windows(source, index, split, centering, channel):
    dataset = WindowDataset(index, split['original_subject_order'], window_centering=centering)
    rows = read_rows(index / 'windows.csv')
    by_id = {r['window_id']: r for r in rows}
    assert len(by_id) == len(rows) == len(dataset) == 2134
    timelines, temperature = {}, {}
    worst_median, worst_delta, worst_std, worst_range = 0.0, 0.0, 0.0, 0.0
    remainder = np.arange(14) != channel
    for subject in dataset.subjects:
        timeline = np.load(source / f'{subject}_X.npy', mmap_mode='r', allow_pickle=False)
        mask = np.load(source / f'{subject}_quality_mask.npy', mmap_mode='r', allow_pickle=False)
        timelines[subject] = timeline
        assert timeline.dtype == np.float32 and timeline.shape[1] == 14
        assert mask.dtype == np.bool_ and mask.shape == (len(timeline),)
        subject_rows = dataset.subject_rows[subject]
        raw_temperatures, centered_temperatures, labels = [], [], []
        for offset, row in enumerate(subject_rows):
            assert {k: str(v) for k, v in row.items()} == by_id[row['window_id']]
            start, end = row['start_index'], row['end_index']
            assert end - start == WINDOW and mask[start:end].all() and row['quality_status'] == 'valid'
            assert int(row['original_label']) == row['label'] + 1 and row['label'] in (0, 1, 2)
            raw = timeline[start:end].T
            x = raw[channel].astype(np.float64)
            middle = np.partition(x, (WINDOW // 2 - 1, WINDOW // 2))
            median = (middle[WINDOW // 2 - 1] + middle[WINDOW // 2]) / 2.0
            expected = (x - median).astype(np.float32)
            prepared = dataset.prepared_window(subject, offset)
            assert prepared.shape == (14, WINDOW) and prepared.dtype == np.float32
            assert np.isfinite(prepared).all()
            np.testing.assert_array_equal(prepared[remainder], raw[remainder])
            np.testing.assert_array_equal(prepared[channel], expected)
            residual = prepared[channel].astype(np.float64)
            worst_median = max(worst_median, abs(float(np.median(residual))))
            worst_delta = max(worst_delta, float(np.max(np.abs(np.diff(x) - np.diff(residual)))))
            worst_std = max(worst_std, abs(float(x.std() - residual.std())))
            worst_range = max(worst_range, abs(float(np.ptp(x) - np.ptp(residual))))
            raw_temperatures.append(raw[channel])
            centered_temperatures.append(expected)
            labels.append(row['label'])
        temperature[subject] = (np.stack(raw_temperatures), np.stack(centered_temperatures), np.array(labels))
    assert Counter(r['label'] for r in dataset.rows) == {0: 1150, 1: 638, 2: 346}
    assert all(value <= 1e-6 for value in (worst_median, worst_delta, worst_std, worst_range))
    return timelines, temperature, {
        'unique_windows': len(dataset), 'max_abs_window_median_after_centering': worst_median,
        'max_abs_adjacent_difference_error': worst_delta, 'max_abs_window_std_error': worst_std,
        'max_abs_peak_to_peak_error': worst_range, 'variance_tolerance': 1e-6,
        'unchanged_channels_at_centering': int(remainder.sum()), 'label_support': [1150, 638, 346],
    }


def reference_batch(dataset, indices, timelines, temperature, channel):
    raw = np.stack([timelines[row['subject_id']][row['start_index']:row['end_index']].T
                    for row in (dataset.rows[i] for i in indices)])
    prepared = raw.copy()
    prepared[:, channel] = np.stack([temperature[subject][1][offset]
                                      for subject, offset in (dataset.indices[i] for i in indices)])
    return raw, prepared


def add_moments(values, total, squares):
    values64 = values.astype(np.float64)
    total += values64.sum(axis=(0, 2))
    squares += np.square(values64).sum(axis=(0, 2))


def finish_moments(total, squares, count):
    mean = total / count
    return mean, np.sqrt(np.maximum(squares / count - np.square(mean), 0.0))


def independent_scaler(train, timelines, temperature, channel):
    total, squares = np.zeros(14), np.zeros(14)
    for start in range(0, len(train), 64):
        _, block = reference_batch(train, range(start, min(start + 64, len(train))), timelines, temperature, channel)
        add_moments(block, total, squares)
    mean, std = finish_moments(total, squares, len(train) * WINDOW)
    actual_std = std.astype(np.float32)
    actual_std[actual_std < 1e-8] = 1
    return mean.astype(np.float32), actual_std, std


def check_loader(fold_dir, role, dataset, baseline_dataset, baseline, timelines, temperature, channel, channels, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    previous_loader = DataLoader(baseline_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    mean, scale = dataset.scaler.mean, dataset.scaler.std
    old_mean, old_scale = baseline['mean'], baseline['std']
    remainder = np.arange(14) != channel
    total, squares = np.zeros(14), np.zeros(14)
    seen = 0
    for (x, y, indices), (old_x, old_y, old_indices) in zip(loader, previous_loader, strict=True):
        n = len(indices)
        assert x.shape == (n, 14, WINDOW) and x.dtype == torch.float32 and x.is_contiguous()
        assert torch.equal(y, old_y) and torch.equal(indices, old_indices)
        np.testing.assert_array_equal(indices.numpy(), np.arange(seen, seen + n))
        np.testing.assert_array_equal(y.numpy(), dataset.labels[seen:seen + n])
        raw, prepared = reference_batch(dataset, indices.tolist(), timelines, temperature, channel)
        expected = (prepared - mean[None, :, None]) / scale[None, :, None]
        old_expected = (raw - old_mean[None, :, None]) / old_scale[None, :, None]
        values = x.numpy()
        assert np.isfinite(values).all()
        np.testing.assert_array_equal(values, expected)
        np.testing.assert_array_equal(old_x.numpy(), old_expected)
        np.testing.assert_array_equal(values[:, remainder], old_x.numpy()[:, remainder])
        add_moments(values, total, squares)
        seen += n
    assert seen == len(dataset)
    z_mean, z_std = finish_moments(total, squares, seen * WINDOW)
    stats = [{'fold_id': int(fold_dir.name[-2:]), 'role': role, 'channel': name,
              'windows': seen, 'timestep_occurrences': seen * WINDOW,
              'baseline_train_scaler_mean': float(old_mean[c]), 'baseline_train_scaler_std': float(old_scale[c]),
              'new_train_scaler_mean': float(mean[c]), 'new_train_scaler_std': float(scale[c]),
              'post_z_mean': float(z_mean[c]), 'post_z_std': float(z_std[c]),
              'nan_count': 0, 'inf_count': 0}
             for c, name in enumerate(channels)]
    return stats, len(loader)


def verify_fold(output, config, index, fold, timelines, temperature, channels, channel):
    fid = fold['fold_id']
    fold_path = Path('seed_42') / f'fold_{fid:02d}'
    centering = config['preprocessing']['window_centering']
    old_artifact = read_json(BASE_RUN / fold_path / 'normalization.json')
    baseline = {k: np.array(old_artifact[k], dtype=np.float32) for k in ('mean', 'std')}
    assert old_artifact['train_subjects'] == fold['train']
    datasets = {role: WindowDataset(index, fold[role], window_centering=centering) for role in ROLES}
    disabled = {role: WindowDataset(index, fold[role]) for role in ROLES}
    train = datasets['train']
    scaler = ChannelScaler().fit(train)
    old_scaler = ChannelScaler().fit(disabled['train'])
    assert scaler.subjects == old_scaler.subjects == fold['train'] and scaler.count == len(train) * WINDOW
    assert old_artifact['sample_count_per_channel'] == scaler.count
    assert scaler.mean.shape == scaler.std.shape == (14,)
    assert np.isfinite(scaler.mean).all() and np.isfinite(scaler.std).all() and (scaler.std > 0).all()
    np.testing.assert_array_equal(old_scaler.mean, baseline['mean'])
    np.testing.assert_array_equal(old_scaler.std, baseline['std'])
    remainder = np.arange(14) != channel
    np.testing.assert_array_equal(scaler.mean[remainder], baseline['mean'][remainder])
    np.testing.assert_array_equal(scaler.std[remainder], baseline['std'][remainder])
    mean, std, unadjusted_std = independent_scaler(train, timelines, temperature, channel)
    np.testing.assert_array_equal(scaler.mean, mean)
    np.testing.assert_array_equal(scaler.std, std)
    stats, counts = [], {}
    for role in ROLES:
        ds, baseline_ds = datasets[role], disabled[role]
        assert ds.rows == baseline_ds.rows and ds.subjects == fold[role] and ds.input_channels == 14
        ds.scaler, baseline_ds.scaler = scaler, old_scaler
        role_stats, counts[role] = check_loader(fold_path, role, ds, baseline_ds, baseline,
                                               timelines, temperature, channel, channels, config['training']['batch_size'])
        stats.extend(role_stats)
    assert sum(len(ds) for ds in datasets.values()) == 2134
    train_stats = stats[:len(channels)]
    mean_error = max(abs(row['post_z_mean']) for row in train_stats)
    floor = [c for c, raw_std in zip(channels, unadjusted_std) if np.float32(raw_std) < 1e-8]
    std_error = max(abs(row['post_z_std'] - 1) for row in train_stats if row['channel'] not in floor)
    assert mean_error < 1e-5 and std_error < 1e-5
    scaler_path = output / fold_path / 'normalization.json'
    scaler.save(scaler_path)
    saved = read_json(scaler_path)
    assert saved['mean'] == scaler.mean.tolist() and saved['std'] == scaler.std.tolist()
    checks = {'fold_id': fid, **{f'{role}_windows': len(ds) for role, ds in datasets.items()},
              'train_subjects': fold['train'], 'validation_subjects': fold['validation'], 'test_subjects': fold['test'],
              'batches_verified': counts, 'scaler_verified_independently': True,
              'untouched_channels_match_baseline_exactly': 13, 'train_scaler_timestep_count_per_channel': scaler.count,
              'train_max_abs_z_mean': mean_error, 'train_max_abs_z_std_error': std_error,
              'train_chest_temp_centered_std': float(scaler.std[channel]),
              'train_chest_temp_baseline_std': float(baseline['std'][channel]),
              'floored_channels': floor, 'normalization_metadata': scaler_path.relative_to(output).as_posix()}
    LOGGER.info('Fold %d verified (%d Train / %d Validation / %d Test)', fid,
                checks['train_windows'], checks['validation_windows'], checks['test_windows'])
    return checks, stats, old_artifact, saved


def trace_extreme(output, evidence):
    peak = evidence['diagnostics']['largest_new_z']
    subject = peak['subject_id']
    channel = evidence['chest_temperature_channel_index_zero_based']
    source = ROOT / evidence['source']
    index = (ROOT / evidence['window_manifest']).parent
    scaler_path = evidence['fold_checks'][peak['fold_id'] - 1]['normalization_metadata']
    scale = read_json(output / scaler_path)
    mean, std = np.float32(scale['mean'][channel]), np.float32(scale['std'][channel])
    timeline = np.load(source / f'{subject}_X.npy', mmap_mode='r', allow_pickle=False)
    rows = read_rows(index / subject / 'windows.csv')
    values = np.stack([timeline[int(r['start_index']):int(r['end_index']), channel] for r in rows])
    values64 = values.astype(np.float64)
    medians = np.median(values64, axis=1, keepdims=True)
    z = ((values64 - medians).astype(np.float32) - mean) / std
    ordinal, timestep = np.unravel_index(np.abs(z).argmax(), z.shape)
    assert abs(float(z[ordinal, timestep])) == peak['max_abs_z']
    row = rows[ordinal]
    mask = np.load(source / f'{subject}_quality_mask.npy', mmap_mode='r', allow_pickle=False)
    assert mask[int(row['start_index']):int(row['end_index'])].all()
    return row['window_id'], (int(row['start_index']) + int(timestep)) / 64, float(z[ordinal, timestep])


def write_report(output, evidence):
    diagnostics = evidence['diagnostics']
    focus = diagnostics['fold_04']
    window = evidence['window_checks']
    n = lambda value, digits=4: f'{value:.{digits}f}'.replace('.', ',')
    pct = lambda fraction: n(100 * fraction, 2) + '%'
    stat_rows = []
    for fold in diagnostics['per_fold']:
        stat_rows.append('| {fold_id} | {old} | {new} | {factor}× | {old_z} → {new_z} | {old_max} → {new_max} |'.format(
            fold_id=fold['fold_id'], old=n(fold['baseline_train_scaler_std']), new=n(fold['centered_train_scaler_std']),
            factor=n(fold['scale_amplification'], 2), old_z=n(fold['baseline_validation_mean_z']),
            new_z=n(fold['centered_validation_mean_z']), old_max=n(fold['baseline_validation_max_abs_z'], 2),
            new_max=n(fold['centered_validation_max_abs_z'], 2)))
    by_subject = {(row['subject_id'], row['stage']): row for row in diagnostics['fold_04_validation_subjects']}
    subject_rows = []
    for subject in evidence['fold_checks'][3]['validation_subjects']:
        stages = [by_subject[(subject, stage)] for stage in ('raw', 'centered', 'baseline_z', 'centered_z')]
        subject_rows.append('| ' + ' | '.join([subject, *(n(row['mean']) for row in stages),
                                               n(stages[-1]['std']), n(max(abs(stages[-1]['min']), abs(stages[-1]['max'])), 2)]) + ' |')
    mean_error = max(row['train_max_abs_z_mean'] for row in evidence['fold_checks'])
    std_error = max(row['train_max_abs_z_std_error'] for row in evidence['fold_checks'])
    per_fold_md = '\n'.join(stat_rows)
    per_subject_md = '\n'.join(subject_rows)
    extreme_id, extreme_seconds, extreme_z = trace_extreme(output, evidence)
    extreme = diagnostics['largest_new_z']
    text = f"""# Kiểm chứng Median Centering và Z-score cho CNN-1D

**Preprocessing verified — Training not started**

Thí nghiệm `{evidence['experiment_id']}`, group `{evidence['group_id']}`, đối chiếu baseline `{evidence['baseline_group_id']}`. Dùng nguyên 15 subject, 14 kênh 64 Hz, cửa sổ 1.920 mẫu, stride 960, split 9/3/3 và seed 42. Hai manifest, mask và 33 tệp nguồn được đối chiếu fingerprint. [Định danh và bằng chứng](preprocessing_verification.json) · [Cấu hình](../../../experiments/{EXPERIMENT}/config.yaml).

## Phép biến đổi và kiểm chứng

`chest_Temp` là kênh thứ **{evidence['chest_temperature_channel_index_zero_based'] + 1}**, index **{evidence['chest_temperature_channel_index_zero_based']}** tính từ 0. Dataset tra tên kênh từ metadata, lấy một bản sao cửa sổ và chỉ trừ median của 1.920 mẫu nhiệt độ của cửa sổ đó. Tính median và trừ ở float64, rồi lưu kết quả vào tensor float32. `ChannelScaler` fit mean/std population trên chính các cửa sổ **Train đã center** của fold, gồm các timestep lặp do overlap. Cùng scaler được dùng để transform cả ba tập. [Dataset](../../../src/data/dataset.py) · [Scaler](../../../src/data/normalization.py).

- Kiểm trực tiếp **{window['unique_windows']:,} cửa sổ duy nhất** theo manifest. Median sau centering bằng 0 trong cả bộ. Chênh lệch tối đa về sai phân hai timestep là {window['max_abs_adjacent_difference_error']:.2g}, peak-to-peak là {window['max_abs_peak_to_peak_error']:.2g} và std mỗi cửa sổ là {window['max_abs_window_std_error']:.2g}. 13 kênh còn lại khớp từng phần tử. Nhãn, ID, shape và mask khớp baseline.
- Đã kiểm **{evidence['loader_batches_verified']} batch**, **{evidence['loader_windows_verified']:,} lượt cửa sổ** ở cả 5 fold. Tensor `[B,14,1920]` float32 khớp phép biến đổi độc lập cho mọi split. Cả 5 scaler khớp phép tính độc lập theo Train. Trên Train, sai số lớn nhất của mean so với 0 là {mean_error:.2e}, của std so với 1 là {std_error:.2e}. Không phát hiện NaN/Inf hay kênh phải dùng fallback std. [Thống kê 14 kênh](normalization_statistics.csv) · [Scaler Fold 4](seed_42/fold_04/normalization.json).
- Nhánh tắt centering cho kết quả scaler và tensor bằng baseline đã lưu. Ở nhánh bật centering, cả mean/std scaler lẫn tensor sau Z-score của 13 kênh còn lại khớp baseline. Checksum trước/sau của {evidence['protected_files_sha256_count_before_after']} tệp gốc khớp, gồm data, masks, index, split, logs, checkpoints và notebook 02. Test chỉ dùng xác minh transform.

## So sánh phân phối nhiệt độ ngực

| Fold | Std Train gốc (°C) | Std Train đã center (°C) | Tỷ số hai std | Mean Z Validation, cũ → mới | Max trị tuyệt đối Z Validation, cũ → mới |
| ---: | ---: | ---: | ---: | ---: | ---: |
{per_fold_md}

Ở **Validation Fold 4**, mean nhiệt độ gốc là **{n(focus['raw_validation_mean'], 3)} °C**, so với **{n(focus['raw_train_mean'], 3)} °C** ở Train. Chênh lệch mean Validation–Train giảm từ {n(focus['raw_validation_minus_train_mean'])} °C còn {n(focus['centered_validation_minus_train_mean'])} °C sau centering. Mean Train sau centering là {n(focus['centered_train_mean'])} °C, mean Validation là {n(focus['centered_validation_mean'])} °C. Median bằng 0 trong từng cửa sổ **trước** Z-score. Z-score sau đó trừ mean Train chung, nên median mỗi cửa sổ có thể lệch 0.

| Subject Validation Fold 4 | Mean gốc (°C) | Mean sau center (°C) | Mean Z baseline | Mean Z mới | Std Z mới | Max trị tuyệt đối Z mới |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{per_subject_md}

Scale mới làm mean Validation Fold 4 gần 0 hơn, đồng thời tăng std Z từ **{n(focus['baseline_validation_std_z'], 3)} lên {n(focus['centered_validation_std_z'], 3)}**. Tỷ lệ mẫu có trị tuyệt đối Z trên 5 giảm từ {pct(focus['baseline_validation_abs_z_gt_5_fraction'])} xuống {pct(focus['centered_validation_abs_z_gt_5_fraction'])}. Tỷ lệ trên 10 tăng từ {pct(focus['baseline_validation_abs_z_gt_10_fraction'])} lên {pct(focus['centered_validation_abs_z_gt_10_fraction'])}. P99 độ lớn sai phân hai timestep tăng từ {n(focus['baseline_validation_p99_abs_adjacent_diff_z'])} lên {n(focus['centered_validation_p99_abs_adjacent_diff_z'])} trong đơn vị Z-score.

Cực trị Train/Validation của toàn bộ kiểm chứng nằm ở `{extreme_id}`, fold {extreme['fold_id']} ({extreme['role']}), tại giây {n(extreme_seconds, 6)}, Z = **{n(extreme_z, 2)}**. Tất cả mẫu của cửa sổ này có mask True. Khi std Train nhỏ, cùng một mức dao động tính bằng °C chiếm nhiều đơn vị Z-score. Điều này có thể khuếch đại cả biến thiên sinh lý và nhiễu. Phân phối và mask hiện có chưa xác định được nguyên nhân của cực trị.

Các bảng phân tầng theo tập, từng người và từng lớp có đủ mean, std, median, IQR, min–max, phân vị và tỷ lệ ở đuôi. Báo cáo mô tả sự thay đổi phân phối ở các cửa sổ, gồm các lượt mẫu chồng lấp. Hiệu quả nhận diện Stress của phép biến đổi vẫn chưa được đo. [Bảng bốn trạng thái](temperature_stage_statistics.csv) · [So sánh theo fold](temperature_distribution_comparison.csv) · [Cửa sổ mẫu](figures/temperature_windows.png) · [Boxplot theo subject](figures/temperature_subject_boxplots.png) · [Density Fold 4](figures/temperature_density_fold_04.png) · [So sánh 5 fold](figures/temperature_shift_by_fold.png).

[Nhật ký kiểm chứng](execution.log) · [Research Journal CSV](../../../experiments/{EXPERIMENT}/research_journal/experiments.csv).
"""
    with (output / 'preprocessing_report.md').open('x', encoding='utf-8') as report:
        report.write(text)


def run(output, group):
    from scripts.chest_temp_centering_diagnostics import write_diagnostics

    original, config, source, index, index_meta, split, channels = context()
    torch.set_num_threads(config['runtime']['torch_threads'])
    channel = channels.index(CHANNEL)
    before = protected_hashes(source, index_meta, config)
    source_hashes = index_meta['audit']['source_file_sha256']
    assert digest(source_hashes) == original['dataset_fingerprint']
    assert all(before[(source / name).relative_to(ROOT.parent).as_posix()] == expected
               for name, expected in source_hashes.items())
    assert all(before[(index / name).relative_to(ROOT.parent).as_posix()] == expected
               for name, expected in index_meta['index_file_sha256'].items())
    LOGGER.info('Reusing official baseline index, split and %d verified source files', len(source_hashes))
    centering = config['preprocessing']['window_centering']
    timelines, temperature, window_checks = inspect_windows(source, index, split, centering, channel)
    folds, all_stats, baseline_scalers, new_scalers = [], [], {}, {}
    for fold in split['folds']:
        check, stats, previous, updated = verify_fold(output, config, index, fold, timelines, temperature, channels, channel)
        folds.append(check)
        all_stats.extend(stats)
        baseline_scalers[fold['fold_id']] = previous
        new_scalers[fold['fold_id']] = updated
    write_rows(output / 'normalization_statistics.csv', all_stats)
    diagnostics = write_diagnostics(output, temperature, baseline_scalers, new_scalers, split, channel)
    assert protected_hashes(source, index_meta, config) == before
    assert not (ROOT / 'runs' / EXPERIMENT / group).exists()
    evidence = {
        'status': 'passed', 'experiment_id': EXPERIMENT, 'group_id': group,
        'verified_at': now(), 'training_started': False, 'training_epochs': 0,
        'baseline_experiment_id': BASELINE, 'baseline_group_id': BASELINE_GROUP,
        'source': source.relative_to(ROOT).as_posix(), 'dataset_fingerprint': original['dataset_fingerprint'],
        'window_manifest': (index / 'windows.csv').relative_to(ROOT).as_posix(),
        'window_manifest_sha256': file_hash(index / 'windows.csv'),
        'split_manifest': (BASE_RUN / 'split_manifest.json').relative_to(ROOT).as_posix(),
        'split_manifest_checksum': original['manifest_checksum'], 'effective_config': config,
        'channels': channels, 'chest_temperature_channel': CHANNEL, 'chest_temperature_channel_index_zero_based': channel,
        'data_source_files_verified': len(source_hashes), 'window_index_files_verified': len(index_meta['index_file_sha256']),
        'window_checks': window_checks, 'fold_checks': folds,
        'loader_batches_verified': sum(sum(row['batches_verified'].values()) for row in folds),
        'loader_windows_verified': sum(sum(row[f'{r}_windows'] for r in ROLES) for row in folds),
        'diagnostics': diagnostics, 'protected_files_sha256_count_before_after': len(before),
        'protected_hash_collection_sha256': digest(before), 'numpy': np.__version__, 'torch': torch.__version__,
        'test_use': 'Transformation correctness verification only',
    }
    write_report(output, evidence)
    write_json(output / 'preprocessing_verification.json', evidence)
    LOGGER.info('PASS: %d loader windows and %d unchanged protected files',
                evidence['loader_windows_verified'], len(before))


def main():
    if not __debug__:
        raise RuntimeError('Run this verifier with assertions enabled.')
    group = identifier()
    output = ROOT / 'results' / EXPERIMENT / group
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s',
                        handlers=[logging.FileHandler(output / 'execution.log', encoding='utf-8'),
                                  logging.StreamHandler()])
    LOGGER.info('GROUP_ID=%s', group)
    try:
        run(output, group)
    except Exception:
        LOGGER.exception('Preprocessing verification failed')
        raise


if __name__ == '__main__':
    main()

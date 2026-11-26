"""Describe chest temperature after verified preprocessing on Train and Validation."""
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


ROLES = ('train', 'validation')
CLASSES = {0: 'baseline', 1: 'stress', 2: 'amusement'}
STAGES = {
    'raw': ('Gốc, 64 Hz', 'Nhiệt độ (°C)'),
    'centered': ('Sau trừ median', 'Chênh lệch nhiệt độ (°C)'),
    'baseline_z': ('Z-score baseline', 'Z-score'),
    'centered_z': ('Trừ median + Z-score', 'Z-score'),
}
COLORS = {'train': '#0072B2', 'validation': '#D55E00'}
BASELINE_COLOR = '#637489'
NEW_COLOR = '#00786D'


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(folder, figure, stem):
    for extension in ('png', 'svg'):
        figure.savefig(folder / f'{stem}.{extension}', dpi=170)
    plt.close(figure)


def fold_temperature(temperature, fold, baseline_mean, baseline_std, centered_mean, centered_std):
    result = {}
    for role in ROLES:
        subjects = {}
        for subject in fold[role]:
            raw, centered, labels = temperature[subject]
            subjects[subject] = {
                'labels': labels,
                'raw': raw,
                'centered': centered,
                'baseline_z': (raw - baseline_mean) / baseline_std,
                'centered_z': (centered - centered_mean) / centered_std,
            }
        keys = ['labels', *STAGES]
        combined = {key: np.concatenate([record[key] for record in subjects.values()]) for key in keys}
        result[role] = {'subjects': subjects, 'all': combined}
    return result


def describe(values, train_bounds, stage):
    q01, q05, q25, q50, q75, q95, q99 = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    lower, upper = train_bounds
    result = {
        'windows': int(values.shape[0]), 'timestep_occurrences': int(values.size),
        'mean': float(values.mean(dtype=np.float64)), 'std': float(values.std(dtype=np.float64)),
        'min': float(values.min()), 'max': float(values.max()),
        'p01': float(q01), 'p05': float(q05), 'p25': float(q25), 'median': float(q50),
        'p75': float(q75), 'p95': float(q95), 'p99': float(q99), 'iqr': float(q75 - q25),
        'outside_train_p01_p99_fraction': float(((values < lower) | (values > upper)).mean()),
    }
    absolute = np.abs(values) if stage.endswith('_z') else None
    for threshold in (3, 5, 10):
        result[f'abs_z_gt_{threshold}_fraction'] = (
            float((absolute > threshold).mean()) if absolute is not None else None
        )
    return result


def statistics_rows(fold_id, data):
    rows = []
    for stage in STAGES:
        train_bounds = np.percentile(data['train']['all'][stage], [1, 99])
        for role in ROLES:
            subjects = data[role]['subjects']
            for subject, group in [('all', data[role]['all']), *subjects.items()]:
                entries = [('all', group[stage])]
                entries.extend((name, group[stage][group['labels'] == label])
                               for label, name in CLASSES.items() if np.any(group['labels'] == label))
                for label, subset in entries:
                    scope = ('role' if subject == 'all' else 'subject') if label == 'all' else (
                        'class' if subject == 'all' else 'subject_class'
                    )
                    rows.append({
                        'fold_id': fold_id, 'role': role, 'stage': stage, 'scope': scope,
                        'subject_id': subject, 'label': label, 'train_p01': float(train_bounds[0]),
                        'train_p99': float(train_bounds[1]), **describe(subset, train_bounds, stage),
                    })
    return rows


def mean_shift_components(data, stage):
    train = data['train']['all']
    validation = data['validation']['all']
    mix, within = 0.0, 0.0
    for label in CLASSES:
        train_label = train['labels'] == label
        validation_label = validation['labels'] == label
        if not train_label.any():
            raise ValueError(f'Cannot estimate a Train class mean for label {label}.')
        train_mean = train[stage][train_label].mean(dtype=np.float64)
        validation_weight = validation_label.mean()
        mix += (validation_weight - train_label.mean()) * train_mean
        if validation_label.any():
            validation_mean = validation[stage][validation_label].mean(dtype=np.float64)
            within += validation_weight * (validation_mean - train_mean)
    return float(mix), float(within)


def compare_fold(fold_id, rows, data, baseline_mean, baseline_std, centered_mean, centered_std):
    result = {
        'fold_id': fold_id, 'train_windows': len(data['train']['all']['labels']),
        'validation_windows': len(data['validation']['all']['labels']),
        'baseline_train_scaler_mean': float(baseline_mean), 'baseline_train_scaler_std': float(baseline_std),
        'centered_train_scaler_mean': float(centered_mean), 'centered_train_scaler_std': float(centered_std),
        'scale_amplification': float(baseline_std) / float(centered_std),
    }
    aggregate = {(r['stage'], r['role']): r for r in rows if r['scope'] == 'role'}
    for stage in ('raw', 'centered'):
        train = aggregate[stage, 'train']
        validation = aggregate[stage, 'validation']
        for role, summary in [('train', train), ('validation', validation)]:
            result[f'{stage}_{role}_mean'] = summary['mean']
            result[f'{stage}_{role}_std'] = summary['std']
        result[f'{stage}_validation_minus_train_mean'] = validation['mean'] - train['mean']
    for prefix, stage in [('baseline', 'baseline_z'), ('centered', 'centered_z')]:
        mix, same_class = mean_shift_components(data, stage)
        shift = aggregate[stage, 'validation']['mean'] - aggregate[stage, 'train']['mean']
        np.testing.assert_allclose(mix + same_class, shift, rtol=0, atol=1e-10)
        result[f'{prefix}_validation_minus_train_mean_z'] = shift
        result[f'{prefix}_validation_label_mix_shift_z'] = mix
        result[f'{prefix}_validation_same_class_shift_z'] = same_class
        for role in ROLES:
            entry = aggregate[stage, role]
            differences = np.abs(np.diff(data[role]['all'][stage], axis=1))
            result.update({
                f'{prefix}_{role}_mean_z': entry['mean'],
                f'{prefix}_{role}_std_z': entry['std'],
                f'{prefix}_{role}_max_abs_z': max(abs(entry['min']), abs(entry['max'])),
                f'{prefix}_{role}_p99_abs_adjacent_diff_z': float(np.percentile(differences, 99)),
                f'{prefix}_{role}_outside_train_p01_p99_fraction': entry['outside_train_p01_p99_fraction'],
                **{f'{prefix}_{role}_abs_z_gt_{threshold}_fraction': entry[f'abs_z_gt_{threshold}_fraction']
                   for threshold in (3, 5, 10)},
            })
    return result


def plot_windows(folder, data, fold_id):
    subjects = data['validation']['subjects']
    examples = []
    for subject, group in subjects.items():
        amplitude = np.ptp(group['raw'], axis=1)
        ordinal = int(np.argmin(np.abs(amplitude - np.median(amplitude))))
        examples.append((subject, ordinal, None))
    extreme_subject = max(subjects, key=lambda s: np.abs(subjects[s]['centered_z']).max())
    extreme_z = np.abs(subjects[extreme_subject]['centered_z'])
    extreme_window, extreme_timestep = np.unravel_index(extreme_z.argmax(), extreme_z.shape)
    examples.append((extreme_subject, extreme_window, extreme_timestep))
    figure, axes = plt.subplots(len(examples), 2, figsize=(10, 2.55 * len(examples)),
                                squeeze=False, layout='constrained', sharex=True)
    for row, (subject, ordinal, extreme) in enumerate(examples):
        group = subjects[subject]
        time = np.arange(group['raw'].shape[1]) / 64
        for col, (stage, color) in enumerate([('raw', COLORS['train']), ('centered', NEW_COLOR)]):
            axis = axes[row, col]
            axis.plot(time, group[stage][ordinal], color=color, linewidth=1.25)
            title = f'{subject} · cửa sổ {ordinal + 1} · {STAGES[stage][0]}'
            if extreme is not None:
                peak = group['centered_z'][ordinal, extreme]
                title = f'{title} · z mới {peak:.1f}'
                axis.scatter(time[extreme], group[stage][ordinal, extreme], color=COLORS['validation'],
                             s=25, zorder=4)
            axis.set(title=title,
                     ylabel=STAGES[stage][1])
            axis.grid(axis='y', alpha=.23)
            if stage == 'centered':
                axis.axhline(0, color='#565D65', linewidth=.8, linestyle='--')
    for axis in axes[-1]:
        axis.set_xlabel('Thời gian trong cửa sổ (s)')
    figure.suptitle(f'Nhiệt độ ngực trong các cửa sổ Validation Fold {fold_id}\n'
                    'Các hàng đầu gần median peak-to-peak của từng người. Hàng cuối có |z| mới lớn nhất.', fontsize=13)
    save_figure(folder, figure, 'temperature_windows')


def plot_subject_boxes(folder, rows, subject_order, fold_id):
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for axis, (stage, (title, units)) in zip(axes.flat, STAGES.items()):
        subjects = {r['subject_id']: r for r in rows if r['stage'] == stage and r['scope'] == 'subject'}
        ordered = [subjects[s] for s in subject_order if s in subjects]
        boxes = [{'label': r['subject_id'], 'med': r['median'], 'q1': r['p25'], 'q3': r['p75'],
                  'whislo': r['p05'], 'whishi': r['p95']} for r in ordered]
        artists = axis.bxp(boxes, showfliers=False, patch_artist=True)
        for element, record in zip(artists['boxes'], ordered):
            element.set_facecolor(COLORS[record['role']])
            element.set_alpha(.65)
        for line in artists['medians']:
            line.set_color('#101820')
        axis.set(title=title, ylabel=units)
        axis.grid(axis='y', alpha=.2)
        if stage != 'raw':
            axis.axhline(0, color='#555E63', linewidth=.8, linestyle='--')
    figure.suptitle(f'Phân phối nhiệt độ ngực theo người · Fold {fold_id}\n'
                    'Hộp P25–P75, vạch median, râu P05–P95', fontsize=13)
    legend = [Patch(facecolor=color, alpha=.65, label=role.title()) for role, color in COLORS.items()]
    figure.legend(handles=legend, loc='outside lower center', ncol=2)
    save_figure(folder, figure, 'temperature_subject_boxplots')


def plot_density(folder, data, fold_id):
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    for axis, (stage, (title, units)) in zip(axes.flat, STAGES.items()):
        values = {role: data[role]['all'][stage] for role in ROLES}
        bounds = [np.percentile(signal, [.5, 99.5]) for signal in values.values()]
        low = min(bound[0] for bound in bounds)
        high = max(bound[1] for bound in bounds)
        if low == high:
            low, high = low - .5, high + .5
        bins = np.linspace(low, high, 81)
        for role, signal in values.items():
            histogram, _ = np.histogram(signal, bins)
            density = histogram / (signal.size * np.diff(bins))
            axis.stairs(density, bins, color=COLORS[role], label=role.title(), linewidth=1.6)
        axis.set(title=title, xlabel=units, ylabel='Mật độ')
        axis.grid(axis='y', alpha=.18)
    axes[0, 0].legend(frameon=False)
    figure.suptitle(f'Nhiệt độ ngực · Train và Validation Fold {fold_id}\n'
                    'Mật độ theo tất cả lượt timestep. Trục ngang bao phủ P0,5–P99,5 của cả hai tập.', fontsize=13)
    save_figure(folder, figure, f'temperature_density_fold_{fold_id:02d}')


def plot_fold_comparison(folder, summaries):
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    axes = axes.ravel()
    fold_ids = np.array([row['fold_id'] for row in summaries])
    for offset, (prefix, color, label) in zip(
        (-.18, .18), [('baseline', BASELINE_COLOR, 'Z-score baseline'), ('centered', NEW_COLOR, 'Trừ median + Z-score')]
    ):
        axes[0].bar(fold_ids + offset, [r[f'{prefix}_validation_minus_train_mean_z'] for r in summaries],
                    color=color, width=.34, label=label)
        axes[1].bar(fold_ids + offset, [100 * r[f'{prefix}_validation_abs_z_gt_5_fraction'] for r in summaries],
                    color=color, width=.34, label=label)
        axes[3].bar(fold_ids + offset, [r[f'{prefix}_validation_max_abs_z'] for r in summaries],
                    color=color, width=.34, label=label)
    axes[2].plot(fold_ids, [r['scale_amplification'] for r in summaries], color=NEW_COLOR, marker='o')
    axes[0].set(title='Độ lệch trung bình Validation', ylabel='Mean Validation − Train (z)')
    axes[0].axhline(0, color='#555E63', linewidth=.8)
    axes[1].set(title='Đuôi phân phối Validation', ylabel='Tỷ lệ |z| > 5 (%)')
    axes[2].set(title='Hệ số khuếch đại scale', ylabel='Std Train baseline / Std Train sau centering')
    axes[3].set(title='Cực trị Validation', ylabel='Giá trị |z| lớn nhất')
    for axis in axes:
        axis.set(xlabel='Fold', xticks=fold_ids)
        axis.grid(axis='y', alpha=.22)
        axis.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles=handles, labels=labels, loc='outside lower center', ncol=2)
    figure.suptitle('Dịch chuyển và độ phân tán nhiệt độ ngực sau chuẩn hóa', fontsize=13)
    save_figure(folder, figure, 'temperature_shift_by_fold')


def describe_extreme(rows, temperature, baseline_scalers, new_scalers, channel_index):
    peak = max((r for r in rows if r['stage'] == 'centered_z' and r['scope'] == 'subject'),
               key=lambda r: max(abs(r['min']), abs(r['max'])))
    fold_id, subject = peak['fold_id'], peak['subject_id']
    raw, centered, labels = temperature[subject]
    new = new_scalers[fold_id]
    centered_z = (centered - np.float32(new['mean'][channel_index])) / np.float32(new['std'][channel_index])
    window, timestep = np.unravel_index(np.abs(centered_z).argmax(), centered_z.shape)
    baseline = baseline_scalers[fold_id]
    raw_point, centered_point = raw[window, timestep], centered[window, timestep]
    baseline_z = (raw_point - np.float32(baseline['mean'][channel_index])) / np.float32(baseline['std'][channel_index])
    return {
        'max_abs_z': abs(float(centered_z[window, timestep])), 'fold_id': fold_id, 'role': peak['role'],
        'subject_id': subject, 'window_number_1_based': int(window + 1), 'timestep_index_0_based': int(timestep),
        'label': CLASSES[int(labels[window])], 'raw_temperature': float(raw_point),
        'centered_temperature': float(centered_point), 'window_temperature_median': float(np.median(raw[window].astype(np.float64))),
        'baseline_z': float(baseline_z), 'centered_z': float(centered_z[window, timestep]),
    }


def write_diagnostics(output, temperature, baseline_scalers, new_scalers, split, channel_index):
    """Write paired preprocessing diagnostics and return JSON-ready descriptive evidence."""
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    all_rows, summaries = [], []
    for fold in split['folds']:
        fold_id = fold['fold_id']
        baseline, centered = baseline_scalers[fold_id], new_scalers[fold_id]
        parameters = tuple(np.float32(scaler[key][channel_index])
                           for scaler in (baseline, centered) for key in ('mean', 'std'))
        data = fold_temperature(temperature, fold, *parameters)
        rows = statistics_rows(fold_id, data)
        summaries.append(compare_fold(fold_id, rows, data, *parameters))
        all_rows.extend(rows)
        if fold_id == 4:
            plot_windows(figures, data, fold_id)
            plot_subject_boxes(figures, rows, split['original_subject_order'], fold_id)
            plot_density(figures, data, fold_id)
    plot_fold_comparison(figures, summaries)
    write_csv(output / 'temperature_stage_statistics.csv', all_rows)
    write_csv(output / 'temperature_distribution_comparison.csv', summaries)
    return {
        'per_fold': summaries,
        'fold_04': next((row for row in summaries if row['fold_id'] == 4), None),
        'fold_04_validation_subjects': [
            r for r in all_rows if r['fold_id'] == 4 and r['role'] == 'validation' and r['scope'] == 'subject'
        ],
        'largest_new_z': describe_extreme(all_rows, temperature, baseline_scalers, new_scalers, channel_index),
        'figure_stems': [name.stem for name in sorted(figures.glob('temperature_*.png'))],
        'statistics_rows': len(all_rows),
    }

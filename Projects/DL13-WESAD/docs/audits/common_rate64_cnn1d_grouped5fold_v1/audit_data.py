"""Read the completed CNN run and write diagnostic data summaries in this audit directory."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXPERIMENT = 'common_rate64_cnn1d_grouped5fold_v1'
GROUP_ID = '20261123T072126Z_2c81daac'
RUN = ROOT / 'runs' / EXPERIMENT / GROUP_ID
RESULT = ROOT / 'results' / EXPERIMENT / GROUP_ID
ROLES = ('train', 'validation', 'test')
LABELS = ('Baseline', 'Stress', 'Amusement')
PERCENTILES = (1, 5, 25, 50, 75, 95, 99)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    with (HERE / name).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def json_hash(value):
    canonical = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def load_context():
    manifest = read_json(RUN / 'experiment_manifest.json')
    assert manifest['group_id'] == GROUP_ID and manifest['status'] == 'completed'
    assert manifest == read_json(RESULT / 'experiment_manifest.json')
    config = manifest['effective_config']
    assert config['experiment']['id'] == EXPERIMENT and config['experiment']['model'] == 'cnn1d'
    split = read_json(RUN / 'split_manifest.json')
    assert json_hash(split) == manifest['manifest_checksum']
    assert (RUN / 'split_manifest.sha256').read_text().strip() == manifest['manifest_checksum']
    assert split == read_json(ROOT / config['split']['manifest'])
    subjects = split['original_subject_order']
    assert len(subjects) == len(set(subjects)) == 15
    assert [fold['fold_id'] for fold in split['folds']] == list(range(1, 6))
    for fold in split['folds']:
        groups = [set(fold[role]) for role in ROLES]
        assert [len(group) for group in groups] == [9, 3, 3]
        assert set.union(*groups) == set(subjects) and sum(map(len, groups)) == len(set.union(*groups))
    assert Counter(subject for fold in split['folds'] for subject in fold['test']) == Counter(subjects)
    return manifest, config, split


def stats(values):
    finite = np.isfinite(values)
    clean = values if finite.all() else values[finite]
    q = np.percentile(clean, PERCENTILES)
    return {'mean': float(clean.mean(dtype=np.float64)), 'std': float(clean.std(dtype=np.float64)),
            'min': float(clean.min()), 'max': float(clean.max()),
            **{f'p{p:02}': float(x) for p, x in zip(PERCENTILES, q)},
            'iqr': float(q[4] - q[2]), 'nan': int(np.isnan(values).sum()), 'inf': int(np.isinf(values).sum())}


def provenance(manifest):
    source = Path(manifest['data_source'])
    index = Path(manifest['cache_path'])
    meta = read_json(index / 'common_rate_index.json')
    stored = meta['audit']['source_file_sha256']
    assert {name: sha256(source / name) for name in stored} == stored
    assert json_hash(stored) == manifest['dataset_fingerprint'] == meta['audit']['dataset_fingerprint']
    assert all(sha256(index / name) == expected for name, expected in meta['index_file_sha256'].items())
    snapshot = read_json(RUN / 'source_checksums.json')
    assert all(sha256(RUN / 'source' / name) == expected for name, expected in snapshot.items())
    different = [name for name, expected in snapshot.items() if not (ROOT / name).is_file() or sha256(ROOT / name) != expected]

    notebook = manifest['preprocessing_notebook']
    path = ROOT / notebook['path']
    git_commit = read_json(RUN / 'system.json')['git_commit']
    repo_rel = f'Projects/DL13-WESAD/{notebook["path"]}'
    git = subprocess.run(['git', 'show', f'{git_commit}:{repo_rel}'], cwd=ROOT, capture_output=True, check=False)
    found = git.returncode == 0 and hashlib.sha256(git.stdout).hexdigest() == notebook['sha256']
    live, old = read_json(path), json.loads(git.stdout) if found else None
    cell_sources = lambda nb: [(cell['cell_type'], cell['source']) for cell in nb['cells']]
    output_changes = [i for i, (a, b) in enumerate(zip(live['cells'], old['cells']))
                      if a.get('outputs') != b.get('outputs') or a.get('execution_count') != b.get('execution_count')] if found else []
    return {'data_files_verified': len(stored), 'source_files_verified': len(snapshot), 'source_files_changed': different,
            'index_files_verified': len(meta['index_file_sha256']), 'dataset_fingerprint': manifest['dataset_fingerprint'],
            'split_sha256': manifest['manifest_checksum'], 'notebook_expected_sha256': notebook['sha256'],
            'notebook_current_sha256': sha256(path), 'notebook_reference_commit': git_commit if found else None,
            'notebook_cell_sources_equal': cell_sources(live) == cell_sources(old) if found else None,
            'notebook_execution_or_output_changes': output_changes}


def manifest_rows(index):
    return {(r['subject_id'], int(r['start_index'])): r for r in read_csv(index / 'windows.csv')}


def inspect_timelines(manifest, config, split):
    source = Path(manifest['data_source'])
    index = Path(manifest['cache_path'])
    source_config = read_json(source / 'config.json')
    channels = config['data']['channels']
    assert channels == source_config['channels'] and len(channels) == 14
    assert source_config['target_fs'] == 64 and source_config['label_source_fs'] == 700
    assert not source_config['normalized'] and not source_config['windowed']
    segments = defaultdict(list)
    for row in read_csv(source / 'label_segments.csv'):
        segments[row['Subject']].append(row)
    saved = manifest_rows(index)
    assert len(saved) == len({r['window_id'] for r in saved.values()}) == manifest['window_count'] == 2134
    lengths = {r['Subject']: int(r['Output_samples']) for r in read_csv(source / 'subjects_summary.csv')}
    windows, labels, continuous, masks, rejected, overlap = {}, {}, [], [], [], []
    keys = set()
    for subject in split['original_subject_order']:
        array = np.load(source / f'{subject}_X.npy', mmap_mode='r', allow_pickle=False)
        mask = np.load(source / f'{subject}_quality_mask.npy', mmap_mode='r', allow_pickle=False)
        assert array.dtype == np.float32 and array.shape == (lengths[subject], 14)
        assert mask.dtype == np.bool_ and mask.shape == (len(array),)
        mask_breaks = np.flatnonzero(np.diff(np.r_[False, ~mask, False])).reshape(-1, 2)
        for start, end in mask_breaks:
            masks.append({'subject': subject, 'start': int(start), 'end_exclusive': int(end),
                          'start_sec': start / 64, 'end_sec': end / 64, 'samples': int(end - start)})

        accepted = []
        previous_end = 0
        for segment in segments[subject]:
            original = int(segment['Label'])
            lo, hi = int(segment['Start_sample_700']), int(segment['End_sample_700'])
            assert lo == previous_end and hi > lo
            assert abs(float(segment['Start_sec']) - lo / 700) < 1e-8
            assert abs(float(segment['End_sec']) - hi / 700) < 1e-8
            previous_end = hi
            if original not in (1, 2, 3):
                continue
            quotient, remainder = divmod(lo * 64, 700)
            first, limit = quotient + bool(remainder), min(len(array), (hi * 64) // 700)
            for start in range(first, limit - 1920 + 1, 960):
                end = start + 1920
                assert start * 700 >= lo * 64 and end * 700 <= hi * 64
                mask_count = int(np.count_nonzero(~mask[start:end]))
                key = (subject, start)
                if mask_count:
                    assert key not in saved
                    rejected.append({'subject': subject, 'original_label': original, 'start': start,
                                     'end_exclusive': end, 'masked_samples': mask_count})
                else:
                    r = saved[key]
                    assert r['window_id'] == f'{subject}_{start}_{end}' and r['quality_status'] == 'valid'
                    assert (int(r['original_label']), int(r['label']), int(r['end_index'])) == (original, original - 1, end)
                    assert (float(r['start_time']), float(r['end_time'])) == (start / 64, end / 64)
                    keys.add(key)
                    accepted.append((start, end, original - 1))
        assert round(previous_end * 64 / 700) == len(array)
        subject_records = read_csv(index / subject / 'windows.csv')
        assert subject_records == [saved[(subject, start)] for start, _, _ in accepted]
        labels[subject] = np.array([label for _, _, label in accepted], dtype=np.int64)
        blocks = np.stack([array[start:end].T for start, end, _ in accepted])  # windows, channels, time
        windows[subject] = blocks

        coverage = np.zeros(len(array) + 1, dtype=np.int16)
        starts, ends = zip(*[(start, end) for start, end, _ in accepted])
        np.add.at(coverage, list(starts), 1)
        np.add.at(coverage, list(ends), -1)
        coverage = np.cumsum(coverage[:-1])
        assert coverage.max() == 2 and not np.any((coverage > 0) & ~mask)
        pairs = np.flatnonzero(np.diff(starts) == 960)
        assert np.array_equal(blocks[pairs, :, 960:], blocks[pairs + 1, :, :960])
        repeated = int(np.count_nonzero(coverage == 2))
        assert repeated == len(pairs) * 960
        a, b = blocks[pairs].astype(np.float64), blocks[pairs + 1].astype(np.float64)
        a -= a.mean(axis=-1, keepdims=True)
        b -= b.mean(axis=-1, keepdims=True)
        denominator = np.sqrt((a * a).sum(axis=-1) * (b * b).sum(axis=-1))
        corr = np.divide((a * b).sum(axis=-1), denominator, out=np.full_like(denominator, np.nan), where=denominator > 0)
        for i, channel in enumerate(channels):
            overlap.append({'subject': subject, 'channel': channel, 'windows': len(accepted), 'adjacent_overlap_pairs': len(pairs),
                            'unique_timesteps': int(np.count_nonzero(coverage)), 'timesteps_in_windows': len(accepted) * 1920,
                            'repeated_timesteps': repeated, 'finite_correlation_pairs': int(np.isfinite(corr[:, i]).sum()),
                            'median_pearson_at_lag_15s': float(np.nanmedian(corr[:, i])),
                            'p25_pearson_at_lag_15s': float(np.nanpercentile(corr[:, i], 25)),
                            'p75_pearson_at_lag_15s': float(np.nanpercentile(corr[:, i], 75))})
            selection = [('continuous_all', array[:, i]), ('continuous_mask_true', array[mask, i]),
                         ('valid_windows', blocks[:, i, :].reshape(-1))]
            for scope, samples in selection:
                s = stats(samples)
                robust_tail = (samples < s['p25'] - 3 * s['iqr']) | (samples > s['p75'] + 3 * s['iqr'])
                continuous.append({'subject': subject, 'channel': channel, 'scope': scope, 'count': len(samples),
                                   'invalid_mask_samples_in_full_timeline': int(np.count_nonzero(~mask)),
                                   **s, 'pct_outside_outer_tukey_fence': 100 * int(robust_tail.sum()) / len(samples)})
        print(f'{subject}: {len(accepted)} valid windows, {len(pairs)} overlapping pairs', flush=True)
    assert keys == set(saved)
    assert [(row['subject'], row['original_label']) for row in rejected] == [('S3', 2), ('S3', 2)]
    write_csv('pre_normalization_statistics.csv', continuous)
    write_csv('window_overlap.csv', overlap)
    write_csv('quality_mask_intervals.csv', masks)
    write_csv('rejected_windows.csv', rejected)
    return windows, labels, continuous, overlap, rejected, masks


def prefixed(prefix, values):
    return {prefix + k: v for k, v in values.items()}


def distribution(config, split, windows, labels):
    channels = config['data']['channels']
    normalization, by_subject, shift, checks = [], [], [], []
    for fold in split['folds']:
        fid = fold['fold_id']
        folder = RUN / 'seed_42' / f'fold_{fid:02}'
        scaler = read_json(folder / 'normalization.json')
        assert scaler['train_subjects'] == fold['train'] and scaler['ddof'] == 0
        means = np.asarray(scaler['mean'], dtype=np.float32)
        scales = np.asarray(scaler['std'], dtype=np.float32)
        assert means.shape == scales.shape == (14,)
        by_role = {role: np.concatenate([windows[s] for s in fold[role]]) for role in ROLES}
        role_labels = {role: np.concatenate([labels[s] for s in fold[role]]) for role in ROLES}
        assert len(by_role['train']) * 1920 == scaler['sample_count_per_channel']
        train_f64 = by_role['train'].astype(np.float64)
        ref_mean = train_f64.mean(axis=(0, 2))
        ref_std = train_f64.std(axis=(0, 2), ddof=0)
        del train_f64
        np.testing.assert_array_equal(means, ref_mean.astype(np.float32))
        np.testing.assert_array_equal(scales, np.where(ref_std < 1e-8, 1, ref_std).astype(np.float32))
        class_proportions = {role: np.bincount(y, minlength=3) / len(y) for role, y in role_labels.items()}
        class_means = {role: np.array([by_role[role][y == label].mean(axis=(0, 2), dtype=np.float64)
                                       for label in range(3)]) for role, y in role_labels.items()}
        max_mean_error, max_std_error = 0.0, 0.0
        for i, channel in enumerate(channels):
            raw_train = by_role['train'][:, i].reshape(-1)
            lo, hi = np.percentile(raw_train, [1, 99])
            for role in ROLES:
                raw = by_role[role][:, i].reshape(-1)
                scaled = (raw - means[i]) / scales[i]
                before, after = stats(raw), stats(scaled)
                outside = (raw < lo) | (raw > hi)
                tail = {f'pct_abs_z_gt_{k}': 100 * float((np.abs(scaled) > k).mean()) for k in (3, 5, 10)}
                normalization.append({'fold': fid, 'role': role, 'channel': channel, 'windows': len(by_role[role]),
                                      'timesteps_per_channel': len(raw), 'stored_train_mean': float(means[i]),
                                      'stored_train_std': float(scales[i]), 'independent_train_mean': float(ref_mean[i]),
                                      'independent_train_std': float(ref_std[i]), 'std_floor_applied': bool(ref_std[i] < 1e-8),
                                      'train_raw_p01': float(lo), 'train_raw_p99': float(hi), 'pct_outside_train_p01_p99': 100 * float(outside.mean()),
                                      **tail, **prefixed('raw_', before), **prefixed('z_', after)})
                if role == 'train':
                    max_mean_error = max(max_mean_error, abs(after['mean']))
                    max_std_error = max(max_std_error, abs(after['std'] - 1))
                else:
                    probs, train_probs = class_proportions[role], class_proportions['train']
                    mu_train, mu_eval = class_means['train'][:, i], class_means[role][:, i]
                    label_mix = (probs - train_probs) @ mu_train / scales[i]
                    conditional = probs @ (mu_eval - mu_train) / scales[i]
                    direct = (before['mean'] - ref_mean[i]) / scales[i]
                    assert np.isclose(label_mix + conditional, direct, rtol=0, atol=1e-10)
                    shift.append({'fold': fid, 'role': role, 'channel': channel, 'mean_shift_z': float(direct),
                                  'label_mix_component_z': float(label_mix), 'within_class_component_z': float(conditional),
                                  **{f'train_prop_{name.lower()}': float(p) for name, p in zip(LABELS, train_probs)},
                                  **{f'eval_prop_{name.lower()}': float(p) for name, p in zip(LABELS, probs)},
                                  **{f'{name.lower()}_conditional_mean_shift_z': float(g) for name, g in zip(LABELS, (mu_eval - mu_train) / scales[i])}})
                for subject in fold[role]:
                    for label, name in [(-1, 'All'), *enumerate(LABELS)]:
                        mask = labels[subject] == label if label >= 0 else slice(None)
                        samples = windows[subject][mask, i].reshape(-1)
                        assert len(samples) > 0
                        scaled_subject = (samples - means[i]) / scales[i]
                        by_subject.append({'fold': fid, 'role': role, 'subject': subject, 'label': name, 'channel': channel,
                                           'windows': len(samples) // 1920, 'timestep_occurrences': len(samples),
                                           'pct_outside_train_p01_p99': 100 * float(((samples < lo) | (samples > hi)).mean()),
                                           'pct_abs_z_gt_5': 100 * float((abs(scaled_subject) > 5).mean()),
                                           **prefixed('raw_', stats(samples)), **prefixed('z_', stats(scaled_subject))})
        checks.append({'fold': fid, 'train_timestep_occurrences': scaler['sample_count_per_channel'],
                       'max_abs_train_z_mean': max_mean_error, 'max_abs_train_z_std_minus_one': max_std_error,
                       'floored_channels': [channel for channel, std in zip(channels, ref_std) if std < 1e-8],
                       'mean_std_match_float32': True})
        print(f'Fold {fid}: independent scaler matches; Train/Validation/Test distributions measured.', flush=True)
    write_csv('normalization_statistics.csv', normalization)
    write_csv('subject_distribution_statistics.csv', by_subject)
    write_csv('distribution_shift_decomposition.csv', shift)
    return normalization, by_subject, shift, checks


def save_figure(fig, name):
    import matplotlib.pyplot as plt

    figures = HERE / 'figures'
    figures.mkdir(exist_ok=True)
    for suffix in ('png', 'svg'):
        fig.savefig(figures / f'{name}.{suffix}', dpi=170, bbox_inches='tight')
    plt.close(fig)


def plot_distributions(config, split, windows, subjects, shift):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    channels = config['data']['channels']
    colors = {'train': '#0072B2', 'validation': '#E69F00', 'test': '#D55E00'}
    subject_ids = split['original_subject_order']
    test_rows = {(r['subject'], r['channel']): r for r in subjects if r['role'] == 'test' and r['label'] == 'All'}
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), layout='constrained')
    for ax, title, key, scale in [(axes[0], 'Mean in Train standard deviations', 'z_mean', lambda v: v),
                                 (axes[1], 'Log₂ of std divided by Train std', 'z_std', np.log2)]:
        matrix = scale([[test_rows[s, c][key] for c in channels] for s in subject_ids])
        limit = np.max(np.abs(matrix))
        image = ax.imshow(matrix, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
        ax.set(xticks=range(14), xticklabels=channels, yticks=range(15), yticklabels=subject_ids, title=title)
        ax.tick_params(axis='x', labelrotation=60, labelsize=8)
        fig.colorbar(image, ax=ax, shrink=.75)
    fig.suptitle('Test subject distributions using each subject’s held-out fold scaler', fontsize=13)
    save_figure(fig, 'test_subject_channel_heatmap')

    fold = split['folds'][0]
    fold1_shift = sorted((r for r in shift if r['fold'] == 1), key=lambda x: abs(x['within_class_component_z']), reverse=True)
    examples = list(dict.fromkeys(row['channel'] for row in fold1_shift))[:3]
    fig, axes = plt.subplots(len(examples), 2, figsize=(15, 9), layout='constrained', squeeze=False)
    lookup = {(r['subject'], r['channel']): r for r in subjects if r['fold'] == 1 and r['label'] == 'All'}
    for i, channel in enumerate(examples):
        for j, (prefix, title) in enumerate([('raw_', 'Raw 64 Hz units'), ('z_', 'Z-score using Fold 1 Train')]):
            boxes = [{'label': subject, 'med': lookup[subject, channel][prefix + 'p50'],
                      'q1': lookup[subject, channel][prefix + 'p25'], 'q3': lookup[subject, channel][prefix + 'p75'],
                      'whislo': lookup[subject, channel][prefix + 'p05'], 'whishi': lookup[subject, channel][prefix + 'p95']}
                     for subject in subject_ids]
            parts = axes[i, j].bxp(boxes, showfliers=False, patch_artist=True)
            for subject, box in zip(subject_ids, parts['boxes']):
                box.set_facecolor(colors[lookup[subject, channel]['role']])
                box.set_alpha(.55)
            axes[i, j].set_title(f'{channel} · {title}')
            axes[i, j].grid(axis='y', alpha=.2)
    fig.legend(handles=[Patch(facecolor=color, alpha=.55, label=role.title()) for role, color in colors.items()], loc='outside lower center', ncol=3)
    fig.suptitle('Fold 1 subject boxplots · Q25–Q75, median and P05–P95 whiskers')
    save_figure(fig, 'fold_01_subject_boxplots')

    example = max(shift, key=lambda r: abs(r['within_class_component_z']))
    fid, channel = example['fold'], example['channel']
    fold = split['folds'][fid - 1]
    scaler = read_json(RUN / 'seed_42' / f'fold_{fid:02}' / 'normalization.json')
    i = channels.index(channel)
    role_values = {role: np.concatenate([windows[s][:, i].reshape(-1) for s in fold[role]]) for role in ROLES}
    lo = min(float(np.percentile(v, 1)) for v in role_values.values())
    hi = max(float(np.percentile(v, 99)) for v in role_values.values())
    mean, scale = scaler['mean'][i], scaler['std'][i]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for ax, after in zip(axes, (False, True)):
        for role, raw in role_values.items():
            values = (raw - np.float32(mean)) / np.float32(scale) if after else raw
            bounds = ((lo - mean) / scale, (hi - mean) / scale) if after else (lo, hi)
            bins = np.linspace(*bounds, 80)
            counts, _ = np.histogram(values, bins=bins)
            ax.stairs(counts / (len(values) * np.diff(bins)), bins, color=colors[role], label=role.title())
        ax.set(xlabel='Z-score' if after else 'Raw 64 Hz units', ylabel='Density / all timesteps')
        ax.legend()
    fig.suptitle(f'Fold {fid:02} / {channel} · largest absolute within-class mean-shift component')
    save_figure(fig, 'distribution_density_example')


def main():
    manifest, config, split = load_context()
    verified = provenance(manifest)
    windows, labels, pre, overlap, rejected, masks = inspect_timelines(manifest, config, split)
    norm, subjects, shift, checks = distribution(config, split, windows, labels)
    plot_distributions(config, split, windows, subjects, shift)
    one_row_per_subject = [r for r in overlap if r['channel'] == config['data']['channels'][0]]
    repeated = sum(r['repeated_timesteps'] for r in one_row_per_subject)
    write_json('data_evidence.json', {**verified, 'group_id': GROUP_ID, 'window_count': sum(map(len, windows.values())),
               'class_counts': np.bincount(np.concatenate(list(labels.values())), minlength=3).tolist(),
               'mask_intervals': masks, 'rejected_windows': rejected, 'scaler_checks': checks,
               'pre_nonfinite_count': sum(r['nan'] + r['inf'] for r in pre if r['scope'] == 'continuous_all'),
               'timesteps_per_channel_in_all_windows': sum(r['timesteps_in_windows'] for r in one_row_per_subject),
               'unique_timesteps_covered': sum(r['unique_timesteps'] for r in one_row_per_subject),
               'repeated_timesteps': repeated, 'adjacent_overlap_pairs': sum(r['adjacent_overlap_pairs'] for r in one_row_per_subject),
               'outside_train_range_max': max((r for r in norm if r['role'] != 'train'), key=lambda r: r['pct_outside_train_p01_p99']),
               'max_abs_eval_z': max(max(abs(r['z_min']), abs(r['z_max'])) for r in norm if r['role'] != 'train'),
               'strongest_shifts': sorted(shift, key=lambda r: abs(r['within_class_component_z']), reverse=True)[:10]})
    print(f'Audit wrote {len(norm)} fold/role/channel rows and {len(subjects)} subject/label/channel rows.', flush=True)


if __name__ == '__main__':
    main()

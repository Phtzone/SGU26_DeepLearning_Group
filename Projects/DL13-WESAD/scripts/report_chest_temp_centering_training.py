"""Verify and compare the completed chest-temperature CNN-1D training group."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.report_common_rate import link, local_time
from scripts.research_journal import reference_text, write_journal
from scripts.verify_chest_temp_centering import (BASELINE, BASELINE_GROUP, BASE_RESULTS, BASE_RUN,
                                                EXPERIMENT, ROOT, context, protected_hashes, read_rows)
from scripts.verify_e1_baseline import safe_load_checkpoint
from src.data.dataset import WindowDataset
from src.data.normalization import ChannelScaler
from src.evaluation.evaluator import evaluate
from src.evaluation.history import merged_epochs
from src.evaluation.metrics import LABEL_NAMES, classification_metrics
from src.models import build_model
from src.training.optimizer import class_weights
from src.utils.tracking import digest, file_hash, now, read_json, write_json


PREPROCESSING_GROUP = '20261126T025058Z_fa1c92ea'
PREPROCESSING = ROOT / 'results' / EXPERIMENT / PREPROCESSING_GROUP
TARGET_WINDOW = 'S2_84900_86820'
METRICS = ('macro_f1', 'balanced_accuracy', 'accuracy', 'stress_recall', 'macro_precision', 'macro_recall')
PROBABILITIES = tuple('p_' + label for label in LABEL_NAMES)
METHODS = ('baseline', 'median_centered')
DISPLAY = {'baseline': 'Baseline', 'median_centered': 'Median Centering + Z-score'}
COLORS = {'baseline': '#466381', 'median_centered': '#B24B2F'}


def save_csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(folder, name, figure):
    for extension in ('png', 'svg'):
        with (folder / f'{name}.{extension}').open('xb') as stream:
            figure.savefig(stream, format=extension, dpi=150)
    plt.close(figure)


def flatten_config(value, prefix=''):
    flat = {}
    for key, field in value.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(field, dict):
            flat.update(flatten_config(field, name))
        else:
            flat[name] = field
    return flat


def config_rows(before, after):
    old, new = flatten_config(before), flatten_config(after)
    return [{'field': key, 'baseline': json.dumps(old[key], ensure_ascii=False) if key in old else '',
             'median_centered': json.dumps(new[key], ensure_ascii=False) if key in new else '',
             'identical': key in old and key in new and old[key] == new[key]}
            for key in sorted(old.keys() | new.keys())]


def verify_controls(run, result, preflight):
    _, _, source, baseline_index, index_meta, split, channels = context()
    manifest = read_json(run / 'experiment_manifest.json')
    verification = read_json(result / 'verification.json')
    assert verification['status'] == 'passed' and verification['group_id'] == run.name
    assert manifest['status'] == 'completed' and verification['verified_folds'] == 5
    assert verification['oof_windows'] == 2134 and verification['test_subjects'] == 15
    assert manifest == read_json(result / 'experiment_manifest.json')
    assert manifest['effective_config'] == preflight['config']
    assert manifest['dataset_fingerprint'] == preflight['dataset_fingerprint'] == verification['data_fingerprint']
    assert manifest['manifest_checksum'] == preflight['split_manifest_checksum'] == verification['split_manifest_sha256']
    assert manifest['cache_keys'] == preflight['cache_keys']
    assert file_hash(PREPROCESSING / 'preprocessing_verification.json') == preflight['preprocessing_verification_sha256']
    assert read_json(run / 'split_manifest.json') == split
    index = Path(manifest['cache_path'])
    assert index == run / 'window_index'
    index_matches = {key: file_hash(index / key) == file_hash(baseline_index / key)
                     for key in index_meta['index_file_sha256']}
    assert index_matches and all(index_matches.values())
    assert file_hash(index / 'windows.csv') == preflight['window_manifest_sha256']
    assert read_json(index / 'common_rate_index.json')['audit'] == index_meta['audit']
    protected = protected_hashes(source, index_meta, preflight['config'])
    protected.update({p.relative_to(ROOT.parent).as_posix(): file_hash(p)
                      for p in PREPROCESSING.rglob('*') if p.is_file()})
    archive = ROOT / 'experiments' / EXPERIMENT / f'Research_journal_{EXPERIMENT}.xlsx'
    if archive.is_file():
        protected[archive.relative_to(ROOT.parent).as_posix()] = file_hash(archive)
    protected.update({p.relative_to(ROOT.parent).as_posix(): file_hash(p)
                      for p in (ROOT / 'experiments' / BASELINE).glob('*.xlsx')})
    assert len(protected) == preflight['protected_original_file_count']
    assert digest(protected) == preflight['protected_original_sha256']
    system = read_json(run / 'system.json')
    assert system['packages'] == preflight['baseline_system']['packages']
    assert system['python'] == preflight['baseline_system']['python'] and not system['cuda_version']
    configs = config_rows(preflight['baseline_config'], preflight['config'])
    assert {r['field'] for r in configs if not r['identical']} == {
        'experiment.id', 'preprocessing.window_centering.channels',
        'preprocessing.window_centering.enabled', 'preprocessing.window_centering.method'}
    return manifest, verification, split, channels, configs, len(protected), len(index_matches)


def index_predictions(path):
    rows = read_rows(path)
    keyed = {row['window_id']: row for row in rows}
    assert len(rows) == len(keyed) == 2134
    return keyed


def validate_oof(base, centered):
    assert base.keys() == centered.keys()
    identities = ('training_seed', 'fold_id', 'subject_id', 'true_label', 'original_label',
                  'start_index', 'end_index', 'start_time', 'end_time', 'quality_status')
    assert all(all(base[w][k] == centered[w][k] for k in identities) for w in base)
    assert len({row['subject_id'] for row in centered.values()}) == 15
    assert Counter(r['true_label'] for r in centered.values()) == {'0': 1150, '1': 638, '2': 346}


def replay_test(run, split, channels, current, baseline, preflight):
    config = preflight['config']
    seed, = config['training']['seeds']
    torch.set_num_threads(config['runtime']['torch_threads'])
    torch.use_deterministic_algorithms(config['runtime']['deterministic'])
    rows, checks = [], []
    channel = channels.index('chest_Temp')
    rest = np.arange(len(channels)) != channel
    for fold in split['folds']:
        fid = fold['fold_id']
        relative = Path(f'seed_{seed}/fold_{fid:02d}')
        folder = run / relative
        meta = read_json(folder / 'metadata.json')
        old_meta = read_json(BASE_RUN / relative / 'metadata.json')
        assert meta['status'] == 'completed' and meta['device'] == old_meta['device'] == 'cpu'
        assert meta['input_shape'] == old_meta['input_shape'] == [14, 1920]
        assert meta['trainable_parameters'] == old_meta['trainable_parameters'] == 47075
        assert meta['subjects'] == old_meta['subjects'] == fold and meta['training_seed'] == seed
        assert meta['architecture'] == old_meta['architecture']
        assert meta['label_distribution'] == old_meta['label_distribution']
        assert meta['class_weights'] == old_meta['class_weights']
        saved_scale = folder / 'normalization.json'
        expected_scale = PREPROCESSING / relative / 'normalization.json'
        assert saved_scale.read_bytes() == expected_scale.read_bytes()
        scale = read_json(saved_scale)
        old_scale = read_json(BASE_RUN / relative / 'normalization.json')
        assert scale['train_subjects'] == fold['train'] and scale['window_centering'] == config['preprocessing']['window_centering']
        for name in ('mean', 'std'):
            np.testing.assert_array_equal(np.asarray(scale[name])[rest], np.asarray(old_scale[name])[rest])
        train = WindowDataset(run / 'window_index', fold['train'], window_centering=scale['window_centering'])
        scaler = ChannelScaler().fit(train)
        np.testing.assert_array_equal(scaler.mean, np.asarray(scale['mean'], dtype=np.float32))
        np.testing.assert_array_equal(scaler.std, np.asarray(scale['std'], dtype=np.float32))
        weights = class_weights(train.labels)
        assert weights.tolist() == meta['class_weights']
        test = WindowDataset(run / 'window_index', fold['test'], scaler=scaler, window_centering=scale['window_centering'])
        model = build_model(config, len(channels))
        best = safe_load_checkpoint(folder / 'checkpoints/best.pt')
        assert best['epoch'] == meta['best_epoch'] and best['checkpoint_kind'] == 'best'
        assert best['scheduler'] is None
        model.load_state_dict(best['model'])
        loader = DataLoader(test, batch_size=config['training']['batch_size'], shuffle=False, num_workers=0)
        maxima = {}

        def checked_batches():
            for x, y, indices in loader:
                assert x.dtype == torch.float32 and tuple(x.shape) == (len(y), 14, 1920)
                assert torch.isfinite(x).all()
                for i, value in zip(indices.tolist(), x[:, channel].abs().amax(dim=1).tolist(), strict=True):
                    maxima[i] = value
                yield x, y, indices

        test_metrics, probabilities, indices = evaluate(model, checked_batches(), torch.device('cpu'), weights)
        np.testing.assert_allclose(test_metrics['loss'], meta['test_metrics']['loss'], atol=1e-6, rtol=1e-6)
        assert not model.training
        assert all(torch.equal(value, best['model'][k]) for k, value in model.state_dict().items())
        saved = [current[test.rows[i]['window_id']] for i in indices]
        assert len(saved) == len(test) == len(maxima)
        expected = np.asarray([[float(row[p]) for p in PROBABILITIES] for row in saved])
        np.testing.assert_allclose(probabilities, expected, atol=1e-6, rtol=1e-6)
        np.testing.assert_array_equal(probabilities.argmax(axis=1), [int(row['predicted_label']) for row in saved])
        for index, row in zip(indices, saved, strict=True):
            key = row['window_id']
            prior = baseline[key]
            rows.append({'fold_id': fid, 'window_id': key, 'subject_id': row['subject_id'],
                         'label': LABEL_NAMES[int(row['true_label'])],
                         'max_abs_chest_temp_z': float(maxima[int(index)]),
                         'baseline_prediction': LABEL_NAMES[int(prior['predicted_label'])],
                         'centered_prediction': LABEL_NAMES[int(row['predicted_label'])],
                         'baseline_correct': prior['predicted_label'] == row['true_label'],
                         'centered_correct': row['predicted_label'] == row['true_label'],
                         'baseline_confidence': max(float(prior[p]) for p in PROBABILITIES),
                         'centered_confidence': max(float(row[p]) for p in PROBABILITIES)})
        difference = float(np.max(np.abs(np.asarray(probabilities) - expected)))
        checks.append({'fold_id': fid, 'scaler_equal_preprocessing': True,
                       'scaler_sha256': file_hash(saved_scale), 'unmodified_channels': 13,
                       'scheduler': None, 'replayed_test_windows': len(test),
                       'max_test_probability_difference': difference, 'replayed_test_loss': test_metrics['loss']})
        print(f'Fold {fid}: best checkpoint reproduced {len(test)} OOF predictions', flush=True)
    assert len(rows) == len({row['window_id'] for row in rows}) == 2134
    return rows, checks


def metrics(rows):
    result = classification_metrics([int(r['true_label']) for r in rows], [int(r['predicted_label']) for r in rows])
    result['macro_precision'] = float(np.mean(result['precision_per_class']))
    result['macro_recall'] = float(np.mean(result['recall_per_class']))
    result['stress_recall'] = result['recall_per_class'][1]
    return result


def by_key(rows, key):
    result = defaultdict(list)
    for row in rows:
        result[row[key]].append(row)
    return result


def metric_breakdown(predictions):
    return {'pooled_oof': metrics(list(predictions.values())),
            'fold_level': {int(k): metrics(rows) for k, rows in by_key(predictions.values(), 'fold_id').items()},
            'subject_level': {k: metrics(rows) for k, rows in by_key(predictions.values(), 'subject_id').items()}}


def summary_stat(vals):
    return {'mean': float(np.mean(vals)), 'sample_std': float(np.std(vals, ddof=1)) if len(vals) > 1 else None}


def aggregate_metric_rows(arms):
    summaries, classes = [], []
    for method, breakdown in arms.items():
        for level in ('subject_level', 'fold_level', 'pooled_oof'):
            results = list(breakdown[level].values()) if level != 'pooled_oof' else [breakdown[level]]
            for metric in METRICS:
                summaries.append({'method': method, 'aggregation': level, 'n': len(results), 'metric': metric,
                                  **summary_stat([m[metric] for m in results])})
            for i, label in enumerate(LABEL_NAMES):
                for metric in ('precision', 'recall', 'f1'):
                    classes.append({'method': method, 'aggregation': level, 'class': label, 'metric': metric,
                                    'n': len(results), 'support': sum(m['support'][i] for m in results),
                                    **summary_stat([m[f'{metric}_per_class'][i] for m in results])})
    return summaries, classes


def load_learning(run, fold_id):
    folder = run / 'seed_42' / f'fold_{fold_id:02d}'
    meta = read_json(folder / 'metadata.json')
    saved = safe_load_checkpoint(folder / 'checkpoints/last.pt')
    keys = {(str(row['attempt_id']), int(row['epoch'])) for row in saved['history']}
    history = merged_epochs(folder / 'logs/epochs.csv', keys)
    assert len(history) == meta['completed_epochs']
    first = history[0]
    best = history[meta['best_epoch'] - 1]
    last = history[-1]
    overview = {'run_id': meta['run_id'], 'started_at': meta['started_at'], 'ended_at': meta['ended_at'],
                'epochs': len(history), 'batches': saved['global_step'], 'best_epoch': best['epoch'],
                'last_early_stopping_counter': last['early_stopping_counter'],
                'first_train_f1_099_epoch': next((e['epoch'] for e in history if e['train_macro_f1'] >= .99), None),
                'val_loss_epoch_1': first['val_loss'], 'val_loss_epoch_2': history[1]['val_loss'],
                'val_loss_at_best': best['val_loss'], 'val_loss_at_last': last['val_loss'],
                'min_val_loss_epoch': min(history, key=lambda row: row['val_loss'])['epoch'],
                'best_val_subject_macro_f1': best['val_subject_macro_f1'],
                'best_val_pooled_macro_f1': best['val_macro_f1'], 'best_val_accuracy': best['val_accuracy'],
                'best_val_balanced_accuracy': best['val_balanced_accuracy'],
                'train_macro_f1_at_best': best['train_macro_f1'], 'train_macro_f1_at_last': last['train_macro_f1'],
                'gap_at_best': best['train_macro_f1'] - best['val_macro_f1'],
                'gap_at_last': last['train_macro_f1'] - last['val_macro_f1'],
                'val_pooled_macro_f1_epoch_sd': float(np.std([row['val_macro_f1'] for row in history], ddof=1)),
                'learning_rate_min': min(row['learning_rate'] for row in history),
                'learning_rate_max': max(row['learning_rate'] for row in history)}
    return history, overview


def paired_records(arms, split, histories):
    paired_folds, paired_subjects, learning = [], [], []
    old, new = (arms[key] for key in METHODS)
    for fold in split['folds']:
        fid = fold['fold_id']
        row = {'fold_id': fid, **{role + '_subjects': ' '.join(fold[role]) for role in ('train', 'validation', 'test')}}
        for method, breakdown in zip(METHODS, (old, new), strict=True):
            overview = histories[method][fid][1]
            learning.append({'fold_id': fid, 'method': method, **overview})
            row.update({f'{method}_{key}': overview[key] for key in
                        ('epochs', 'batches', 'best_epoch', 'best_val_subject_macro_f1', 'best_val_pooled_macro_f1', 'gap_at_best')})
            row.update({f'{method}_test_{key}': breakdown['fold_level'][fid][key] for key in METRICS})
        for key in [*(f'test_{metric}' for metric in METRICS), 'best_val_subject_macro_f1', 'gap_at_best']:
            row[f'delta_{key}'] = row[f'median_centered_{key}'] - row[f'baseline_{key}']
        paired_folds.append(row)
    for subject in split['original_subject_order']:
        fold = next(f for f in split['folds'] if subject in f['test'])
        support = old['subject_level'][subject]['support']
        row = {'subject_id': subject, 'fold_id': fold['fold_id'], 'windows': sum(support),
               'stress_support': support[1]}
        for key in METRICS:
            row[f'baseline_{key}'] = old['subject_level'][subject][key]
            row[f'median_centered_{key}'] = new['subject_level'][subject][key]
            row[f'delta_{key}'] = row[f'median_centered_{key}'] - row[f'baseline_{key}']
        paired_subjects.append(row)
    return paired_folds, paired_subjects, learning


def confidence_comparison(arms):
    output, distributions = [], {}
    for method, predictions in arms.items():
        for status in ('correct', 'incorrect'):
            correct = status == 'correct'
            confidence = [max(float(record[key]) for key in PROBABILITIES) for record in predictions.values()
                          if (record['true_label'] == record['predicted_label']) == correct]
            distributions[method, status] = confidence
            output.append({'method': method, 'prediction_status': status, 'windows': len(confidence),
                           'mean_max_probability': float(np.mean(confidence)) if confidence else None,
                           'median_max_probability': float(np.median(confidence)) if confidence else None,
                           'confidence_at_least_0_90': sum(p >= .9 for p in confidence)})
    return output, distributions


def tail_comparison(rows):
    summary = []
    for lo, hi in ((None, 10), (10, None), (50, None)):
        chosen = [r for r in rows if (hi is None or r['max_abs_chest_temp_z'] <= hi)
                  and (lo is None or r['max_abs_chest_temp_z'] > lo)]
        count = len(chosen)
        summary.append({'group': f'|Z|max <= {hi}' if hi is not None else f'|Z|max > {lo}', 'windows': count,
                        'subjects': len({r['subject_id'] for r in chosen}),
                        'labels': dict(Counter(r['label'] for r in chosen)),
                        'baseline_errors': sum(not r['baseline_correct'] for r in chosen),
                        'centered_errors': sum(not r['centered_correct'] for r in chosen),
                        'baseline_error_rate': sum(not r['baseline_correct'] for r in chosen) / count if count else None,
                        'centered_error_rate': sum(not r['centered_correct'] for r in chosen) / count if count else None})
    return summary


def numerics_and_errors(run, checks):
    evidence = []
    for check in checks:
        fid = check['fold_id']
        folder = run / 'seed_42' / f'fold_{fid:02d}'
        iterations = read_rows(folder / 'logs/iterations.csv')
        strict = np.array([[float(row[field]) for field in ('train_loss', 'learning_rate', 'gradient_norm', 'batch_duration_ms')]
                           for row in iterations])
        assert np.isfinite(strict).all()
        text = (folder / 'logs/errors.log').read_text(encoding='utf-8')
        metadata = read_json(folder / 'metadata.json')
        evidence.append({'fold_id': fid, 'observed_iterations': len(iterations),
                         'resume_attempts': len(metadata['attempts']) - 1, 'error_log_chars': len(text),
                         'max_logged_gradient_norm_before_clipping': float(strict[:, 2].max()),
                         'batches_with_gradient_norm_above_1': int(np.count_nonzero(strict[:, 2] > 1)),
                         'errors_log': (folder / 'logs/errors.log').relative_to(ROOT).as_posix()})
    return evidence


def plot_comparison(folder, histories, subjects, arms, confidence, group_id):
    caption = f'{EXPERIMENT} · {group_id}'
    figure, axes = plt.subplots(4, 5, figsize=(16.6, 11), layout='constrained', sharex='col')
    for column, fid in enumerate(range(1, 6)):
        axes[0, column].set_title(f'Fold {fid}')
        for method in METHODS:
            history, overview = histories[method][fid]
            x = [row['epoch'] for row in history]
            for index, train, val in [(0, 'train_eval_loss', 'val_loss'), (1, 'train_macro_f1', 'val_macro_f1')]:
                ax = axes[index, column]
                ax.plot(x, [row[train] for row in history], color=COLORS[method], alpha=.8, ls='--', label=f'{DISPLAY[method]} Train')
                ax.plot(x, [row[val] for row in history], color=COLORS[method], label=f'{DISPLAY[method]} Validation')
            axes[2, column].plot(x, [row['val_subject_macro_f1'] for row in history], color=COLORS[method], label=DISPLAY[method])
            axes[2, column].scatter(overview['best_epoch'], overview['best_val_subject_macro_f1'], color=COLORS[method], s=14)
            axes[3, column].plot(x, [r['train_macro_f1'] - r['val_macro_f1'] for r in history], color=COLORS[method])
        axes[0, column].set_yscale('symlog', linthresh=.001)
        axes[1, column].set_ylim(0, 1.04)
        axes[2, column].set_ylim(0, 1.04)
        axes[3, column].set_ylim(-.03, 1)
        axes[3, column].set_xlabel('Epoch')
        for ax in axes[:, column]:
            ax.grid(axis='y', alpha=.25)
    for ax, label in zip(axes[:, 0], ('Weighted CE', 'Pooled Macro-F1', 'Validation subject-mean F1', 'Pooled F1 gap'), strict=True):
        ax.set_ylabel(label)
    axes[0, -1].legend(loc='best', fontsize=6.7)
    figure.suptitle(f'CNN-1D 64 Hz · so sánh Train và Validation (seed 42)\n{caption}', fontsize=12)
    save_figure(folder, 'learning_curves_comparison', figure)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6), layout='constrained', sharex=True)
    names = [r['subject_id'] for r in subjects]
    for ax, metric, title in zip(axes.flat, METRICS[:4], ('Macro-F1', 'Balanced Accuracy', 'Accuracy', 'Stress Recall'), strict=True):
        values = [r[f'delta_{metric}'] for r in subjects]
        ax.bar(names, values, color=['#10766D' if a >= 0 else '#AB492F' for a in values])
        ax.axhline(0, color='.35', linewidth=.7)
        ax.set(title=title, ylabel='Median Centering − Baseline')
        ax.grid(axis='y', alpha=.2)
    fig.suptitle(f'Chênh lệch Test ghép cặp trên cùng 15 subject\n{caption}', fontsize=11)
    save_figure(folder, 'paired_subject_differences', fig)

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.4), layout='constrained', sharey=True)
    matrices = [np.array(arms[method]['pooled_oof']['confusion_matrix']) for method in METHODS]
    normalized = [100 * matrix / matrix.sum(axis=1, keepdims=True) for matrix in matrices]
    values = normalized + [normalized[1] - normalized[0]]
    for ax, matrix, label, cmap in zip(axes, values, (*[DISPLAY[m] for m in METHODS], 'Δ điểm phần trăm'),
                                       ('Blues', 'Blues', 'RdBu'), strict=True):
        ax.imshow(matrix, vmin=-100 if cmap == 'RdBu' else 0, vmax=100, cmap=cmap, aspect='equal')
        for i, j in np.ndindex(matrix.shape):
            ax.text(j, i, f'{matrix[i,j]:.1f}' if cmap != 'RdBu' else f'{matrix[i,j]:+.1f}',
                    ha='center', va='center', color='white' if abs(matrix[i,j]) > 66 else '#18222A')
        ax.set(title=label, xticks=range(3), yticks=range(3), xticklabels=LABEL_NAMES,
               yticklabels=LABEL_NAMES, xlabel='Dự đoán')
    axes[0].set_ylabel('Nhãn thật')
    fig.suptitle(f'Confusion matrix OOF chuẩn hóa theo lớp thật (%)\n{caption}', fontsize=11)
    save_figure(folder, 'pooled_confusion_comparison', fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout='constrained', sharex=True, sharey=True)
    bins = np.linspace(0.3, 1, 15)
    for ax, (status, label) in zip(axes, (('correct', 'Dự đoán đúng'), ('incorrect', 'Dự đoán sai')), strict=True):
        for method in METHODS:
            scores = confidence[method, status]
            if scores:
                counts = np.histogram(scores, bins=bins)[0]
                ax.stairs(100 * counts / len(scores), bins, color=COLORS[method], linewidth=1.8, label=DISPLAY[method])
        ax.set(title=label, xlabel='Xác suất softmax lớn nhất', xlim=(0.3, 1))
        ax.grid(axis='y', alpha=.25)
    axes[0].set_ylabel('Tỷ lệ trong nhóm (%)')
    axes[0].legend()
    fig.suptitle(f'Phân bố xác suất dự đoán qua OOF\n{caption}', fontsize=11)
    save_figure(folder, 'oof_confidence_comparison', fig)


def counts_change(rows, key):
    delta = [row[key] for row in rows]
    return {'increase': sum(x > 1e-12 for x in delta), 'decrease': sum(x < -1e-12 for x in delta),
            'unchanged': sum(abs(x) <= 1e-12 for x in delta)}


def most_common_errors(matrix):
    pairs = [(matrix[i][j], LABEL_NAMES[i], LABEL_NAMES[j]) for i, j in np.ndindex((3, 3)) if i != j]
    return sorted(pairs, reverse=True)[:3]


def write_report(out, result, run, evidence, histories, paired, subjects, aggregates, arms, confidence, tails, target, preflight):
    def number(value, places=4):
        return f'{value:.{places}f}'.replace('.', ',')

    def ratio(value):
        return number(value * 100, 2) + '%'

    def change(value):
        return f'{value:+.4f}'.replace('.', ',')

    def md(label, path):
        return f'[{label}]({link(result / "comparison", path)})'

    lookup = {(row['method'], row['aggregation'], row['metric']): row for row in aggregates}
    def table_value(method, level, key):
        item = lookup[method, level, key]
        return number(item['mean']) + (f" ± {number(item['sample_std'])}" if item['sample_std'] is not None else '')

    vn_labels = {'macro_f1': 'Macro-F1', 'balanced_accuracy': 'Balanced Accuracy', 'accuracy': 'Accuracy',
                 'stress_recall': 'Stress Recall', 'macro_precision': 'Macro Precision', 'macro_recall': 'Macro Recall'}
    labels = {'subject_level': 'Subject (15)', 'fold_level': 'Fold (5)', 'pooled_oof': 'Pooled OOF (2.134 cửa sổ)'}
    metric_lines = []
    for level in labels:
        for metric in METRICS:
            delta = lookup['median_centered', level, metric]['mean'] - lookup['baseline', level, metric]['mean']
            metric_lines.append(f'| {labels[level]} | {vn_labels[metric]} | {table_value("baseline", level, metric)} | '
                                f'{table_value("median_centered", level, metric)} | {change(delta)} |')
    fold_lines, learn_lines, subject_lines = [], [], []
    val_lines = []
    for row in paired:
        fid = row['fold_id']
        old, new = (histories[key][fid][1] for key in METHODS)
        fold_lines.append(f"| {fid} | {row['test_subjects']} | {old['epochs']} → {new['epochs']} | "
                          f"{old['best_epoch']} → {new['best_epoch']} | {number(old['best_val_subject_macro_f1'])} → "
                          f"{number(new['best_val_subject_macro_f1'])} | " + ' | '.join(
                              f"{number(row['baseline_test_'+key])} → {number(row['median_centered_test_'+key])}"
                              for key in METRICS[:4]) + ' |')
        learn_lines.append(f"| {fid} | {old['first_train_f1_099_epoch'] or '—'} → {new['first_train_f1_099_epoch'] or '—'} | "
                           f"{old['min_val_loss_epoch']} → {new['min_val_loss_epoch']} | "
                           f"{number(old['val_loss_epoch_1'],3)} → {number(old['val_loss_epoch_2'],3)} | "
                           f"{number(new['val_loss_epoch_1'],3)} → {number(new['val_loss_epoch_2'],3)} | "
                           f"{number(old['gap_at_best'])} → {number(new['gap_at_best'])} | "
                           f"{number(old['gap_at_last'])} → {number(new['gap_at_last'])} |")
        val_lines.append(f"| {fid} | {number(new['best_val_pooled_macro_f1'])} | {number(new['best_val_balanced_accuracy'])} | "
                         f"{number(new['best_val_accuracy'])} | {number(new['val_loss_at_best'], 3)} | "
                         f"{int(new['last_early_stopping_counter'])} |")
    for row in subjects:
        subject_lines.append(f"| {row['subject_id']} | {row['fold_id']} | {row['windows']} | "
                             f"{number(row['baseline_macro_f1'])} | {number(row['median_centered_macro_f1'])} | "
                             + ' | '.join(change(row[f'delta_{key}']) for key in METRICS[:4]) + ' |')
    class_lines = []
    for i, label in enumerate(LABEL_NAMES):
        old, new = (arms[m]['pooled_oof'] for m in METHODS)
        class_lines.append(f"| {label} | {new['support'][i]} | " + ' | '.join(
            f'{number(old[metric][i])} → {number(new[metric][i])}'
            for metric in ('precision_per_class', 'recall_per_class', 'f1_per_class')) + ' |')
    totals = evidence['totals']
    old_avg, new_avg = (lookup[m, 'subject_level', 'macro_f1']['mean'] for m in METHODS)
    comparison = evidence['comparison']
    old_val, new_val = (comparison['mean_best_validation_subject_macro_f1'][method] for method in METHODS)
    old_gap, new_gap = (comparison['mean_pooled_gap_at_best'][method] for method in METHODS)
    val_delta = comparison['mean_delta_best_validation_subject_macro_f1']
    gap_delta = comparison['mean_delta_pooled_gap_at_best']
    gaps = counts_change(paired, 'delta_gap_at_best')
    val_changes = counts_change(paired, 'delta_best_val_subject_macro_f1')
    train_reached = sum(info[1]['first_train_f1_099_epoch'] is not None for info in histories['median_centered'].values())
    val_loss_rises = sum(info[1]['val_loss_epoch_2'] > info[1]['val_loss_epoch_1']
                         for info in histories['median_centered'].values())
    impaired = counts_change(subjects, 'delta_macro_f1')
    focus_subjects = []
    for subject in ('S16', 'S14', 'S10'):
        row = next(record for record in subjects if record['subject_id'] == subject)
        focus_subjects.append(f"{subject} ({number(row['baseline_macro_f1'])} → "
                              f"{number(row['median_centered_macro_f1'])}, {change(row['delta_macro_f1'])})")
    after_train = evidence['numerics']
    attempts = sum(row['resume_attempts'] for row in after_train)
    max_gradient = max(row['max_logged_gradient_norm_before_clipping'] for row in after_train)
    clip_count = sum(row['batches_with_gradient_norm_above_1'] for row in after_train)
    errors = []
    for method in METHODS:
        matrix = arms[method]['pooled_oof']['confusion_matrix']
        top = ', '.join(f'{label} → {prediction} ({count})' for count, label, prediction in most_common_errors(matrix))
        predicted = np.sum(matrix, axis=0).tolist()
        bad = next(row for row in confidence if row['method'] == method and row['prediction_status'] == 'incorrect')
        errors.append(f"- **{DISPLAY[method]}**. Số dự đoán theo thứ tự Baseline/Stress/Amusement là {predicted}. "
                      f"Ba nhầm lẫn thường gặp nhất là {top}. Confidence trung bình của {bad['windows']} cửa sổ sai "
                      f"là {number(bad['mean_max_probability']) if bad['windows'] else '—'}, với "
                      f"{bad['confidence_at_least_0_90']} cửa sổ có max softmax ≥ 0,90.")
    tail_lines = [f"| {r['group'].replace('|','&#124;')} | {r['windows']} | {r['subjects']} | {r['baseline_errors']} | {r['centered_errors']} |"
                  for r in tails]
    seed = preflight['config']['training']['seeds'][0]
    staged = preflight['config']['training']
    output = f"""# CNN-1D 64 Hz với Median Centering nhiệt độ ngực

**Experiment** `{EXPERIMENT}` · **Training group** `{run.name}` · **Preprocessing group** `{PREPROCESSING_GROUP}`.

**Hoàn tất 5/5 fold, {totals['epochs']} epoch và {totals['committed_batches']} mini-batch đã được ghi vào checkpoint.** Mục tiêu là kiểm tra tác động của việc trừ median riêng kênh `chest_Temp` đến tổng quát hóa theo subject. So với baseline `{BASELINE_GROUP}`, điểm Validation subject-mean Macro-F1 tại checkpoint được chọn trung bình qua 5 fold là **{number(old_val)} → {number(new_val)} ({change(val_delta)})**. Test subject-mean Macro-F1 là **{number(old_avg)} → {number(new_avg)} ({change(new_avg-old_avg)})**. Đây là đối chứng mô tả trên 15 người và một seed.

## Tiền xử lý, thiết lập và kiểm soát biến

Nguồn Common-rate 64 Hz có 14 kênh theo thứ tự đã lưu, 2.134 cửa sổ 1.920 mẫu, stride 960 mẫu. Giữ nguyên 15 subject, 9/3/3 trong mỗi fold, mask và nhãn 1/2/3 → 0/1/2. Mỗi subject được Test đúng một lần. Trừ median của 1.920 mẫu trong từng cửa sổ ở kênh `chest_Temp` (index 6). Fit Z-score 14 kênh trên cửa sổ Train đã center và sử dụng scaler đó cho cả ba tập. Cả 5 scaler dùng khi chạy trùng byte với group tiền xử lý. Mean/std của 13 kênh khác khớp baseline. Các cửa sổ cực trị được giữ theo mask hiện có. {md('Báo cáo tiền xử lý', PREPROCESSING / 'preprocessing_report.md')} · {md('Kiểm toán cực trị', PREPROCESSING / 'extreme_value_audit/EXTREME_VALUES_AUDIT.md')}.

| Điều kiện | Baseline và lần chạy này |
| --- | --- |
| Kiến trúc | Cùng CNN-1D, 47.075 trainable parameters, tensor float32 `[B,14,1920]` |
| Môi trường | CPU, 2 threads PyTorch, cùng phiên bản Python/PyTorch/TensorBoard và các gói được chụp |
| Khởi tạo | Seed {seed} cho từng fold, model và Adam khởi tạo mới. Các checkpoint dùng cùng GROUP_ID của lần chạy |
| Loss và optimizer | Cross-entropy trên logits có weight theo nghịch đảo tần suất lớp Train, Adam, learning rate {staged['learning_rate']}, batch {staged['batch_size']} |
| Regularization | Dropout {preflight['config']['model']['dropout']}, weight decay {staged['weight_decay']}, gradient norm limit {staged['gradient_clip_norm']}, scheduler `None` |
| Dừng và chọn best | Tối đa {staged['max_epochs']} epoch. Patience {staged['early_stopping_patience']}, min delta {staged['early_stopping_min_delta']}. Tối đa Validation subject-mean Macro-F1, hòa thì dùng Validation loss thấp hơn |
| Test | Mỗi fold nạp best đã chọn. Replay xác suất từ best khớp các dự đoán OOF đã lưu |

Effective config chỉ khác định danh experiment và tùy chọn centering được chủ động bật. SHA của source snapshot baseline khác tại {len(preflight['changed_files_vs_baseline_snapshot'])} đường dẫn. Trong code thực thi, khác biệt ở Dataset/Scaler và cách gắn vào trainer là median centering theo cửa sổ. Stress Recall được ghi thêm từ metric đang có sau mỗi epoch. Các khác biệt kỹ thuật còn lại là kiểm tra config, đọc lịch sử log và tùy chọn kiểm tra reference trong runner. Các module CNN, optimizer, seed, early stopping, evaluator, metrics và aggregation khớp bản baseline trong đối chiếu trực tiếp. [Bảng effective config](effective_config_comparison.csv) · {md('Preflight và danh sách source khác', run/'preflight_comparison.json')} · {md('Source snapshot', run/'source_checksums.json')}.

## Kết quả theo fold

Các cặp trong bảng theo thứ tự **Baseline → Median Centering**. Điểm Validation là mean Macro-F1 theo subject tại best epoch. Các chỉ số Test là pooled trong từng fold.

| Fold | Test subjects | Epochs | Best epoch | Val subject-mean F1 | Test Macro-F1 | Test BA | Test Accuracy | Test Stress Recall |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(fold_lines)}

[Bảng ghép cặp theo fold và các subject từng split](paired_fold_comparison.csv) · [Kết quả fold gốc](../fold_results.csv).

| Fold | Validation pooled F1 tại best | Validation BA | Validation Accuracy | Validation loss | Patience counter khi dừng |
| ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(val_lines)}

## Ba mức tổng hợp trên Test

Trung bình đều theo subject hoặc theo fold đi kèm SD mẫu. Pooled OOF tính một lần từ tất cả cửa sổ. Macro Recall trùng Balanced Accuracy do cùng lấy trung bình recall của ba nhãn cố định. Quy tắc `zero_division=0` và cách tính lớp vắng mặt giữ như baseline.

| Đơn vị | Metric | Baseline | Median Centering | Δ |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(metric_lines)}

[Bảng metrics đủ ba mức](metric_aggregation.csv) · [Bảng precision/recall/F1 và support theo lớp](class_metric_aggregation.csv) · [OOF predictions](../oof_predictions.csv).

### Chênh lệch ghép cặp theo subject

Macro-F1 tăng ở **{impaired['increase']}** subject, giảm ở **{impaired['decrease']}** và giữ nguyên ở **{impaired['unchanged']}**. Trong nhóm có kết quả baseline thấp, Macro-F1 thay đổi ở {', '.join(focus_subjects)}. Cột `stress_support` trong CSV cho biết số cửa sổ Stress dùng để tính recall từng subject.

| Subject | Fold | Cửa sổ | F1 baseline | F1 mới | Δ F1 | Δ BA | Δ Accuracy | Δ Stress Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(subject_lines)}

[Bảng đầy đủ theo subject](paired_subject_comparison.csv) · [Biểu đồ PNG](comparison_figures/paired_subject_differences.png) · [SVG](comparison_figures/paired_subject_differences.svg).

## Learning curves và khoảng cách tổng quát hóa

Gap ở đây là Train Macro-F1 trừ Validation pooled Macro-F1 tại cùng epoch. Tại best checkpoint, gap trung bình là **{number(old_gap)} → {number(new_gap)} ({change(gap_delta)})**, thu hẹp ở {gaps['decrease']}/5 fold. Validation subject-mean F1 tăng ở {val_changes['increase']}/5 fold. Train F1 vẫn đạt ít nhất 0,99 ở {train_reached}/5 fold và Validation loss tăng từ epoch 1 sang 2 ở {val_loss_rises}/5 fold. Các checkpoint best của hai phương pháp có thể ở epoch khác nhau.

| Fold | Epoch đầu Train F1 ≥ 0,99, cũ → mới | Epoch min Val loss, cũ → mới | Val loss epoch 1 → 2 cũ | Val loss epoch 1 → 2 mới | Gap tại best, cũ → mới | Gap khi dừng, cũ → mới |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(learn_lines)}

Learning rate giữ cố định {staged['learning_rate']}. [Chi tiết loss, F1, early stopping và dao động Validation](learning_diagnostics.csv) · [Biểu đồ so sánh learning curves](comparison_figures/learning_curves_comparison.png) · [SVG](comparison_figures/learning_curves_comparison.svg).

## Sai số theo lớp và nhóm nhiệt độ đuôi

| Lớp | Support | Precision, cũ → mới | Recall, cũ → mới | F1, cũ → mới |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(class_lines)}

{chr(10).join(errors)}

Fold 1 có Test Stress Recall **{number(paired[0]['baseline_test_stress_recall'])} → {number(paired[0]['median_centered_test_stress_recall'])}**. [Ma trận OOF chuẩn hóa và hiệu](comparison_figures/pooled_confusion_comparison.png) · [Ma trận OOF dạng đếm](../aggregate_confusion_matrix.csv) · [Bảng confidence](oof_confidence_comparison.csv) · [Biểu đồ confidence](comparison_figures/oof_confidence_comparison.png).

Các mức đuôi dưới đây dùng max trị tuyệt đối Z của kênh nhiệt độ trong mỗi cửa sổ Test, với scaler Train đúng fold. Bảng ghép hai dự đoán trên cùng cửa sổ. Nhóm trên 50 nằm trong nhóm trên 10.

| Mức | Cửa sổ | Chủ thể | Lỗi baseline | Lỗi Median Centering |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(tail_lines)}

`{TARGET_WINDOW}` thuộc **Test Fold {target['fold_id']}**, nhãn **{target['label']}**, max trị tuyệt đối Z trong lượt đánh giá **{number(target['max_abs_chest_temp_z'], 2)}**. Baseline dự đoán **{target['baseline_prediction']}** (confidence {number(target['baseline_confidence'])}), Median Centering dự đoán **{target['centered_prediction']}** ({number(target['centered_confidence'])}). Giá trị 133,69 của audit dùng scaler Fold 4 khi cửa sổ ở Validation. Số lượng cửa sổ đuôi phụ thuộc subject/lớp và có overlap. Kết quả dự đoán tại đây chưa xác lập ảnh hưởng nhân quả của riêng các cực trị. [Số liệu từng cửa sổ](oof_temperature_context.csv).

## Xác minh và giới hạn

- {evidence['verified_index_files']} tệp window index khớp baseline. {evidence['protected_original_file_count']} tệp dữ liệu, baseline, preprocessing và journal XLSX khớp fingerprint trước chạy. Không có subject giao nhau trong từng fold. Class weights khớp baseline và được tính lại từ Train.
- {evidence['standard_verification']['checkpoint_count']} checkpoint, {evidence['standard_verification']['tensorboard_files']} TensorBoard event files và {evidence['standard_verification']['verified_figures']} hình scientific-v2 ở cả PNG/SVG đã được kiểm. 2.134 OOF windows khớp nhãn, subject và fold của baseline. Replay từ best đối chiếu cả xác suất. [Verification chuẩn](../verification.json) · [Verification đối chứng](training_verification.json).
- {attempts} lần resume. Các batch được kiểm có loss, learning rate, norm gradient và thời gian hữu hạn. Max norm trước clipping là {number(max_gradient, 3)}, có {clip_count} batch vượt ngưỡng 1,0, sử dụng gradient clipping đã có trong cấu hình. Chi tiết lỗi kỹ thuật nằm ở verification và Research Journal.
- Validation được dùng để chọn best. Kết quả Test baseline đã được quan sát khi hình thành giả thuyết centering. So sánh hiện tại mang tính khám phá, trên cùng 15 người, một seed, và có cửa sổ chồng lấp. Các chênh lệch được báo cáo mô tả, chưa kiểm định ý nghĩa thống kê.

{md('Runs, logs và checkpoints', run)} · {md('Danh mục hình scientific-v2', result/'enhanced_visualizations/generation_manifest.json')} · {md('Research Journal CSV', ROOT/'experiments'/EXPERIMENT/'research_journal'/'experiments.csv')}.
"""
    with (out / 'training_report.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(output)


def write_training_journal(run, result, evidence):
    journal = ROOT / 'experiments' / EXPERIMENT
    comparison = result / 'comparison'
    unit = f'{EXPERIMENT} | {run.name}'
    events = [json.loads(line) for line in (run / 'logs/events.jsonl').read_text(encoding='utf-8').splitlines() if line]
    closed = next(row for row in reversed(events) if row['event'] == 'experiment_end')
    finished = local_time(closed['timestamp'])
    totals = evidence['totals']
    primary = evidence['test']['median_centered']['subject_level']['macro_f1']
    result_description = (f"5/5 fold, {totals['epochs']} epoch, {totals['committed_batches']} mini-batch. "
                          f"Test Macro-F1 theo subject {primary['mean']:.4f} ± {primary['sample_std']:.4f}.")

    def ref(path, name=None):
        return reference_text(name or path.name, link(journal, path))

    records = {'experiments.csv': [[None, unit, finished, None,
                f'CNN-1D + Median Centering chest_Temp + Z-score. Seed 42. Preprocessing group {PREPROCESSING_GROUP}.',
                result_description, ref(run / 'config.yaml'), ref(run / 'logs/training.log'), 'Completed 5/5']],
               'activity_log.csv': [], 'figures.csv': [], 'debug.csv': [],
               'todo.csv': [[None, f'Xem xét kết quả đối chứng {run.name}', None, ref(comparison / 'training_report.md'),
                             'Đối chiếu Validation, sai số Test và các giới hạn đã ghi.', 'Chờ xem xét']]}
    for fold in evidence['paired_folds']:
        fid = fold['fold_id']
        folder = run / 'seed_42' / f'fold_{fid:02d}'
        meta = read_json(folder / 'metadata.json')
        outcomes = (f"{fold['median_centered_epochs']} epoch, best {fold['median_centered_best_epoch']}. "
                    f"Validation subject-mean F1 {fold['median_centered_best_val_subject_macro_f1']:.4f}. "
                    f"Test F1 {fold['median_centered_test_macro_f1']:.4f}, BA "
                    f"{fold['median_centered_test_balanced_accuracy']:.4f}, Accuracy {fold['median_centered_test_accuracy']:.4f}, "
                    f"Stress Recall {fold['median_centered_test_stress_recall']:.4f}.")
        records['activity_log.csv'].append([None, local_time(meta['ended_at']), None, unit, f'Hoàn tất Fold {fid}',
                                            outcomes, ref(folder/'metadata.json'), 'Đối chiếu artifacts của fold.'])
        error_path = folder / 'logs/errors.log'
        if error_path.stat().st_size:
            records['debug.csv'].append([None, None, unit, f'Fold {fid} có nội dung trong errors.log',
                                         None, None, 'Kiểm chứng artifacts hoàn thành đạt.', ref(error_path)])
    records['activity_log.csv'].extend([
        [None, finished, None, unit, 'Runner hoàn tất Grouped 5-fold', result_description,
         ref(result / 'summary.json'), 'Kiểm chứng các hiện vật của group.'],
        [None, local_time(evidence['verified_at']), None, unit, 'Kiểm chứng, replay và so sánh với baseline',
         'Đạt 5/5 fold và 2.134 dự đoán OOF trên 15 subject.', ref(comparison / 'training_report.md'),
         'Xem kết quả so sánh và quyết định hướng nghiên cứu.']])
    for image in sorted((comparison/'comparison_figures').iterdir()):
        records['figures.csv'].append([None, unit, image.stem.replace('_',' '), ref(image)])
    records['figures.csv'].append([None, unit, 'Danh mục hình scientific-v2',
                                   ref(result/'enhanced_visualizations/generation_manifest.json')])
    preflight_error = run / 'preflight_retry.json'
    if preflight_error.is_file():
        issue = read_json(preflight_error)
        records['debug.csv'].append([None, None, unit, issue['error'], issue['cause'], issue['recovery'],
                                     'Preflight đạt trước khi runner bắt đầu.', ref(preflight_error)])
    return write_journal(journal, records)


def build_comparison(run, result):
    preflight = read_json(run / 'preflight_comparison.json')
    assert preflight['status'] == 'passed' and preflight['training_group_id'] == run.name
    assert preflight['preprocessing_group_id'] == PREPROCESSING_GROUP and preflight['baseline_group_id'] == BASELINE_GROUP
    manifest, verification, split, channels, configs, protected, index_count = verify_controls(run, result, preflight)
    base = index_predictions(BASE_RESULTS / 'oof_predictions.csv')
    current = index_predictions(result / 'oof_predictions.csv')
    validate_oof(base, current)
    tail_rows, checks = replay_test(run, split, channels, current, base, preflight)
    predictions = {'baseline': base, 'median_centered': current}
    arms = {method: metric_breakdown(rows) for method, rows in predictions.items()}
    aggregates, class_rows = aggregate_metric_rows(arms)
    histories = {method: {fold['fold_id']: load_learning(group, fold['fold_id']) for fold in split['folds']}
                 for method, group in zip(METHODS, (BASE_RUN, run), strict=True)}
    paired, subjects, learning = paired_records(arms, split, histories)
    confidence, distributions = confidence_comparison(predictions)
    tails = tail_comparison(tail_rows)
    target, = (row for row in tail_rows if row['window_id'] == TARGET_WINDOW)
    numerics = numerics_and_errors(run, checks)
    totals = {name: sum(row[key] for row in paired) for name, key in
              [('epochs', 'median_centered_epochs'), ('committed_batches', 'median_centered_batches'),
               ('baseline_epochs', 'baseline_epochs'), ('baseline_committed_batches', 'baseline_batches')]}
    assert totals['epochs'] == sum(row['epochs'] for row in verification['folds'])
    assert totals['committed_batches'] == sum(r['iterations']['committed_iterations'] for r in verification['folds'])
    summary = {}
    for method, breakdown in arms.items():
        summary[method] = {level: {row['metric']: {k: row[k] for k in ('mean', 'sample_std', 'n')}
                                  for row in aggregates if row['method'] == method and row['aggregation'] == level}
                           for level in breakdown}
        summary[method]['pooled_confusion_matrix'] = breakdown['pooled_oof']['confusion_matrix']
    evidence = {
        'status': 'passed', 'training_group_id': run.name, 'preprocessing_group_id': PREPROCESSING_GROUP,
        'baseline_group_id': BASELINE_GROUP, 'verified_at': now(), 'totals': totals,
        'source_commit': read_json(run/'system.json')['git_commit'],
        'analysis_source_sha256': file_hash(Path(__file__)),
        'window_manifest_sha256': preflight['window_manifest_sha256'], 'verified_index_files': index_count,
        'protected_original_file_count': protected, 'protected_original_sha256': preflight['protected_original_sha256'],
        'training_config': manifest['effective_config'], 'standard_verification': verification,
        'fold_replays': checks, 'numerics': numerics, 'test': summary,
        'paired_folds': paired, 'paired_subjects': subjects, 'target_window': target, 'temperature_tail': tails,
        'comparison': {
            'mean_best_validation_subject_macro_f1': {
                method: float(np.mean([row[f'{method}_best_val_subject_macro_f1'] for row in paired]))
                for method in METHODS},
            'mean_pooled_gap_at_best': {
                method: float(np.mean([row[f'{method}_gap_at_best'] for row in paired])) for method in METHODS},
            'mean_delta_best_validation_subject_macro_f1': float(np.mean([r['delta_best_val_subject_macro_f1'] for r in paired])),
            'mean_delta_pooled_gap_at_best': float(np.mean([r['delta_gap_at_best'] for r in paired])),
            'subject_changes': {key: counts_change(subjects, f'delta_{key}') for key in METRICS},
        },
    }
    destination = result / 'comparison'
    with TemporaryDirectory(dir=result, prefix='.centered-report-') as tmp:
        stage = Path(tmp).resolve()
        assert stage.parent == result.resolve()
        output = stage / 'comparison'
        figures = output / 'comparison_figures'
        figures.mkdir(parents=True)
        for filename, rows in [('effective_config_comparison.csv', configs), ('metric_aggregation.csv', aggregates),
                               ('class_metric_aggregation.csv', class_rows), ('paired_fold_comparison.csv', paired),
                               ('paired_subject_comparison.csv', subjects), ('learning_diagnostics.csv', learning),
                               ('oof_confidence_comparison.csv', confidence), ('oof_temperature_context.csv', tail_rows)]:
            save_csv(output / filename, rows)
        plot_comparison(figures, histories, subjects, arms, distributions, run.name)
        evidence['comparison_figures'] = {p.relative_to(output).as_posix(): file_hash(p) for p in sorted(figures.iterdir())}
        write_json(output / 'training_verification.json', evidence)
        write_report(output, result, run, evidence, histories, paired, subjects, aggregates, arms, confidence, tails, target, preflight)
        if destination.exists():
            raise FileExistsError(destination)
        output.rename(destination)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group-id', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.group_id):
        parser.error('Invalid training group ID.')
    run = ROOT / 'runs' / EXPERIMENT / args.group_id
    result = ROOT / 'results' / EXPERIMENT / args.group_id
    output = result / 'comparison' / 'training_verification.json'
    evidence = read_json(output) if output.exists() else build_comparison(run, result)
    assert evidence['status'] == 'passed' and evidence['training_group_id'] == run.name
    journal = write_training_journal(run, result, evidence)
    print(result / 'comparison' / 'training_report.md')
    print(journal)


if __name__ == '__main__':
    main()

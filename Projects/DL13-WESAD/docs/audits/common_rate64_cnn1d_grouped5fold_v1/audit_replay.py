"""Audit the saved CNN-1D folds with read-only CPU inference and metric checks."""
from itertools import groupby
from pathlib import Path
from types import SimpleNamespace
import json
import math
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.data import DataLoader
import yaml

from audit_data import (GROUP_ID, LABELS, RESULT, ROLES, ROOT, RUN, json_hash,
                        load_context, read_csv, read_json, save_figure, sha256,
                        write_csv, write_json)

sys.path.insert(0, str(RUN / 'source'))
from src.data.dataset import WindowDataset
from src.data.normalization import ChannelScaler
from src.evaluation.enhanced_visualization import learning_figures
from src.evaluation.evaluator import evaluate
from src.models.cnn import CNN1D


PROBABILITY_FIELDS = [f'p_{label.lower()}' for label in LABELS]
METRICS = ('macro_f1', 'balanced_accuracy', 'accuracy')
EPOCH_TAGS = {
    'train_online_loss': 'train/online_loss', 'train_eval_loss': 'train/eval_loss',
    'train_accuracy': 'train/accuracy', 'train_macro_f1': 'train/macro_f1',
    'train_balanced_accuracy': 'train/balanced_accuracy', 'val_loss': 'val/loss',
    'val_accuracy': 'val/accuracy', 'val_macro_f1': 'val/macro_f1',
    'val_balanced_accuracy': 'val/balanced_accuracy',
    'learning_rate': 'optimization/learning_rate',
    'epoch_duration_seconds': 'optimization/epoch_duration_seconds',
    'best_epoch': 'selection/best_epoch', 'best_val_macro_f1': 'selection/best_val_macro_f1',
    'val_subject_macro_f1': 'val/subject_macro_f1',
    'best_val_subject_macro_f1': 'selection/best_val_subject_macro_f1',
    'early_stopping_counter': 'selection/early_stopping_counter',
}
BATCH_TAGS = {'train_loss': 'batch/loss', 'learning_rate': 'batch/learning_rate',
              'gradient_norm': 'batch/gradient_norm', 'batch_duration_ms': 'batch/duration_ms'}


def fingerprint(manifest, config):
    """Hash protected inputs and official artifacts without invoking the writers."""
    data = Path(manifest['data_source'])
    index = read_json(RUN / 'window_index/common_rate_index.json')['audit']
    paths = {p for folder in (RUN, RESULT) for p in folder.rglob('*') if p.is_file()}
    paths.update(data / name for name in index['source_file_sha256'])
    paths.update(ROOT / name for name in read_json(RUN / 'source_checksums.json'))
    paths.add(ROOT / manifest['preprocessing_notebook']['path'])
    paths.add(ROOT / config['split']['manifest'])
    paths.add(ROOT.parent / 'Research_journal.xlsx')
    return {p.relative_to(ROOT.parent).as_posix(): sha256(p) for p in sorted(paths) if p.is_file()}


def check_metric_values(actual, expected, keys=METRICS):
    for key in keys:
        np.testing.assert_allclose(actual[key], float(expected[key]), rtol=0, atol=1e-12, err_msg=key)


def confusion_metrics(labels, predictions):
    cm = np.zeros((3, 3), dtype=np.int64)
    np.add.at(cm, (labels, predictions), 1)
    tp = cm.diagonal()
    support, pred_counts = cm.sum(axis=1), cm.sum(axis=0)
    recall = np.divide(tp, support, out=np.zeros(3), where=support > 0)
    precision = np.divide(tp, pred_counts, out=np.zeros(3), where=pred_counts > 0)
    f1 = np.divide(2 * tp, support + pred_counts, out=np.zeros(3), where=support + pred_counts > 0)
    return {'confusion_matrix': cm.tolist(), 'support': support.tolist(),
            'predicted_counts': pred_counts.tolist(), 'recall_per_class': recall.tolist(),
            'precision_per_class': precision.tolist(), 'f1_per_class': f1.tolist(),
            'accuracy': float(tp.sum() / cm.sum()), 'macro_f1': float(f1.mean()),
            'balanced_accuracy': float(recall.mean()), 'stress_recall': float(recall[1])}


def open_checkpoint(folder, kind, config, manifest, meta, epoch):
    path = folder / 'checkpoints' / f'{kind}.pt'
    allowed = [np.ndarray, np.dtype, np._core.multiarray._reconstruct, np.dtypes.UInt32DType]
    with torch.serialization.safe_globals(allowed):
        state = torch.load(path, map_location='cpu', weights_only=True)
    assert state['checkpoint_kind'] == kind and state['epoch'] == epoch
    assert state['effective_config'] == config and state['manifest_checksum'] == manifest['manifest_checksum']
    assert state['cache_keys'] == meta['cache_keys'] and state['run_id'] == meta['run_id']
    assert state['best_epoch'] == meta['best_epoch'] and state['scheduler'] is None
    assert state['best_validation_score'] == meta['best_val_subject_macro_f1']
    (optimizer,) = state['optimizer']['param_groups']
    assert optimizer['lr'] == config['training']['learning_rate']
    assert optimizer['weight_decay'] == config['training']['weight_decay']
    return state


def check_history(epochs, validation, batches, state, config, meta, folder):
    assert len(epochs) == len(validation) == len(state['history']) == meta['completed_epochs']
    assert len(batches) == state['global_step']
    assert [int(b['global_step']) for b in batches] == list(range(1, len(batches) + 1))
    assert all(b['run_id'] == meta['run_id'] for b in batches)
    assert all(float(b['learning_rate']) == config['training']['learning_rate'] for b in batches)
    assert all(float(e['learning_rate']) == config['training']['learning_rate'] for e in epochs)
    batches_by_epoch = {epoch: list(rows) for epoch, rows in groupby(batches, lambda b: int(b['epoch']))}
    training_log = (folder / 'logs/training.log').read_text(encoding='utf-8')
    summaries = [dict(field.split('=', 1) for field in line.split()[2:])
                 for line in training_log.splitlines() if line.startswith('EPOCH SUMMARY ')]
    assert len(summaries) == len(epochs)
    batch_size = config['training']['batch_size']
    full, partial = divmod(meta['window_counts']['train'], batch_size)
    expected_batch_sizes = [batch_size] * full + ([partial] if partial else [])
    best, best_score, best_loss = 0, -1.0, float('inf')
    anchor, bad_epochs = -1.0, 0
    for epoch, (e, v, h) in enumerate(zip(epochs, validation, state['history']), 1):
        assert int(e['epoch']) == int(v['epoch']) == h['epoch'] == epoch
        for key in EPOCH_TAGS:
            assert float(e[key]) == h['epoch_log'][key], (epoch, key)
        assert all(float(e[key]) == float(value) for key, value in summaries[epoch - 1].items())
        for key in ('val_loss', 'val_macro_f1', 'val_accuracy', 'val_balanced_accuracy', 'val_subject_macro_f1'):
            assert float(e[key]) == float(v[key]) == h['validation_log'][key]
        subject_f1 = json.loads(v['val_macro_f1_by_subject'])
        assert set(subject_f1) == set(meta['subjects']['validation'])
        assert subject_f1 == h['validation_log']['val_macro_f1_by_subject']
        score, loss = float(e['val_subject_macro_f1']), float(e['val_loss'])
        assert math.isclose(score, np.mean(list(subject_f1.values())), abs_tol=1e-12)
        better = score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and loss < best_loss)
        assert better == (v['is_best_checkpoint'] == 'True') == h['validation_log']['is_best_checkpoint']
        if better:
            best, best_score, best_loss = epoch, score, loss
        if score > anchor + config['training']['early_stopping_min_delta']:
            anchor, bad_epochs = score, 0
        else:
            bad_epochs += 1
        assert int(e['best_epoch']) == best and int(e['early_stopping_counter']) == bad_epochs
        assert float(e['best_val_subject_macro_f1']) == best_score
        assert float(e['best_val_macro_f1']) == float(epochs[best - 1]['val_macro_f1'])
        eb = batches_by_epoch[epoch]
        assert [int(b['batch_index']) for b in eb] == list(range(len(eb)))
        assert [int(b['batch_size']) for b in eb] == expected_batch_sizes
        assert all(math.isfinite(float(b[k])) for b in eb for k in ('train_loss', 'gradient_norm'))
    assert best == meta['best_epoch'] == state['best_epoch']
    assert bad_epochs == config['training']['early_stopping_patience']
    assert read_json(folder / 'metrics/best_validation.json') == state['history'][best - 1]['validation_log']
    events = [json.loads(line) for line in (folder / 'logs/events.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [event['epoch'] for event in events if event.get('event') == 'epoch_end'] == list(range(1, len(epochs) + 1))
    assert [(event['epoch'], event['best']) for event in events if event.get('event') == 'checkpoint_saved'] == [
        (int(v['epoch']), v['is_best_checkpoint'] == 'True') for v in validation]
    last_selection = max(i for i, event in enumerate(events) if event.get('event') == 'checkpoint_saved')
    testing = [i for i, event in enumerate(events) if event.get('event') == 'test_completed']
    assert len(testing) == 1 and testing[0] > last_selection
    assert (folder / 'logs/errors.log').stat().st_size == 0


def check_scalar_tags(accumulator, mapping, rows, step):
    for source, tag in mapping.items():
        events = accumulator.Scalars(tag)
        assert [e.step for e in events] == [int(r[step]) for r in rows], tag
        np.testing.assert_array_equal([e.value for e in events],
                                      np.asarray([float(r[source]) for r in rows], dtype=np.float32),
                                      err_msg=tag)


def check_tensorboard(folder, epochs, validation, batches, test):
    readers = {}
    for stream in ('epochs', 'batches', 'evaluation'):
        stream_folder = folder / 'tensorboard' / stream
        assert len(list(stream_folder.glob('events.out.tfevents.*'))) == 1
        readers[stream] = EventAccumulator(str(stream_folder), size_guidance={'scalars': 0}).Reload()
    check_scalar_tags(readers['epochs'], EPOCH_TAGS, epochs, 'epoch')
    check_scalar_tags(readers['batches'], BATCH_TAGS, batches, 'global_step')
    for metric in ('precision', 'recall', 'f1'):
        values = [json.loads(v[f'val_{metric}_per_class']) for v in validation]
        for label in range(3):
            rows = [{'epoch': v['epoch'], 'value': x[label]} for v, x in zip(validation, values)]
            check_scalar_tags(readers['epochs'], {'value': f'val/class_{label}/{metric}'}, rows, 'epoch')
    check_scalar_tags(readers['evaluation'], {k: f'test/{k}' for k in (*METRICS, 'loss')},
                      [test | {'step': len(epochs)}], 'step')
    return sum(len(reader.Scalars(tag)) for reader in readers.values() for tag in reader.Tags()['scalars'])


class CurveCapture:
    def __init__(self):
        self.series = {}

    def line(self, stem, title, series, xlabel, ylabel, best=None, zero=False):
        self.series[Path(stem).name] = (series, xlabel, best)

    def save(self, figure, stem, title):
        plt.close(figure)

    def skip(self, stem, reason):
        raise AssertionError((stem, reason))


def check_curve_inputs(meta, epochs, batches):
    capture = CurveCapture()
    learning_figures(capture, SimpleNamespace(metadata=meta, epochs=epochs, iterations=batches))
    expected = {
        'loss': [('Train eval', 'train_eval_loss'), ('Validation', 'val_loss'), ('Train online', 'train_online_loss')],
        'macro_f1': [('Train', 'train_macro_f1'), ('Validation', 'val_macro_f1')],
        'accuracy': [('Train', 'train_accuracy'), ('Validation', 'val_accuracy')],
        'balanced_accuracy': [('Train', 'train_balanced_accuracy'), ('Validation', 'val_balanced_accuracy')],
        'validation_subject_macro_f1': [('Mean over Validation subjects', 'val_subject_macro_f1')],
    }
    for name, fields in expected.items():
        series, xlabel, best = capture.series[name]
        assert xlabel == 'Epoch' and best == meta['best_epoch']
        assert series == [(label, [r['epoch'] for r in epochs], [float(r[column]) for r in epochs])
                          for label, column in fields]
    return len(expected)


def replay(model, state, data, weights, batch_size, scaler, verify_inputs):
    model.load_state_dict(state['model'])
    initial = {name: v.clone() for name, v in model.state_dict().items()}
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0)
    pending = {}
    logits = []
    seen_batches = 0

    def batches():
        for x, y, indices in loader:
            if verify_inputs:
                rows = [data.rows[int(i)] for i in indices]
                original = np.stack([data.arrays[r['subject_id']][r['start_index']:r['end_index']].T for r in rows])
                assert original.dtype == np.float32 and original.shape == (len(rows), 14, 1920)
                expected = (original - scaler[0][None, :, None]) / scaler[1][None, :, None]
                assert np.isfinite(expected).all()
                np.testing.assert_array_equal(x.numpy(), expected)
                np.testing.assert_array_equal(y.numpy(), [r['label'] for r in rows])
            pending['x'] = x
            yield x, y, indices
            assert 'x' not in pending

    def pre_hook(module, inputs):
        nonlocal seen_batches
        (x,) = inputs
        assert not any(layer.training for layer in module.modules())
        assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled() and not x.requires_grad
        assert x.shape[1:] == (14, 1920) and x.dtype == torch.float32 and x.is_contiguous()
        assert x is pending.pop('x')
        seen_batches += 1

    def collect_logits(module, inputs, output):
        logits.append(output.cpu().numpy())

    first = model.register_forward_pre_hook(pre_hook)
    last = model.register_forward_hook(collect_logits)
    try:
        actual, probs, indices = evaluate(model, batches(), torch.device('cpu'), weights)
    finally:
        first.remove()
        last.remove()
    assert seen_batches == len(loader)
    assert all(torch.equal(before, model.state_dict()[k]) for k, before in initial.items())
    assert all(p.grad is None for p in model.parameters())
    np.testing.assert_array_equal(indices, np.arange(len(data)))
    y = data.labels
    z = np.concatenate(logits).astype(np.float64)
    z -= z.max(axis=1, keepdims=True)
    nll = np.log(np.exp(z).sum(axis=1)) - z[np.arange(len(y)), y]
    weighted = np.average(nll, weights=weights.numpy()[y].astype(np.float64))
    np.testing.assert_allclose(actual['loss'], weighted, atol=1e-7, rtol=5e-7)
    exp = np.exp(z)
    np.testing.assert_allclose(probs, exp / exp.sum(axis=1, keepdims=True), atol=2e-7, rtol=0)
    independent = confusion_metrics(y, probs.argmax(axis=1))
    check_metric_values(actual, independent)
    np.testing.assert_array_equal(actual['confusion_matrix'], independent['confusion_matrix'])
    return actual, probs, nll, seen_batches


def confidence(labels, probability, nll=None):
    predicted = probability.argmax(axis=1)
    correct = predicted == labels
    maximum = probability.max(axis=1)
    error = ~correct
    stats = {'correct_count': int(correct.sum()), 'error_count': int(error.sum()),
             'mean_max_probability_correct': float(maximum[correct].mean()) if correct.any() else None,
             'mean_max_probability_wrong': float(maximum[error].mean()) if error.any() else None,
             'wrong_pmax_at_least_0_9': int((error & (maximum >= 0.9)).sum())}
    if nll is not None:
        stats.update(mean_nll=float(nll.mean()), mean_nll_wrong=float(nll[error].mean()) if error.any() else None)
    return stats


def role_row(folder, role, kind, state, epochs, validation, dataset, actual, probs, nll, batch_count, weights):
    recorded = epochs[state['epoch'] - 1]
    column = 'train' if role == 'train' else 'val'
    target = {key: recorded[f'{column}_{key}'] for key in METRICS} if role != 'test' else read_json(folder / 'metrics/test.json')
    loss = (float(recorded['train_eval_loss' if role == 'train' else 'val_loss'])
            if role != 'test' else float(target['loss']))
    check_metric_values(actual, target)
    np.testing.assert_allclose(actual['loss'], loss, rtol=1e-6, atol=1e-7)
    subject_mean = None
    if role == 'validation':
        subjects = np.asarray([r['subject_id'] for r in dataset.rows])
        predictions = probs.argmax(axis=1)
        by_subject = {}
        for subject in dataset.subjects:
            mask = subjects == subject
            by_subject[subject] = confusion_metrics(dataset.labels[mask], predictions[mask])['macro_f1']
        v = validation[state['epoch'] - 1]
        check_metric_values(by_subject, json.loads(v['val_macro_f1_by_subject']), by_subject)
        subject_mean = float(np.mean(list(by_subject.values())))
        np.testing.assert_allclose(subject_mean, float(recorded['val_subject_macro_f1']), rtol=0, atol=1e-12)
    wrong = probs.argmax(axis=1) != dataset.labels
    per_row_weight = weights.numpy()[dataset.labels].astype(np.float64)
    contribution = nll * per_row_weight / per_row_weight.sum()
    return {'fold_id': int(folder.name[-2:]), 'checkpoint': kind, 'epoch': state['epoch'], 'role': role,
            'windows': len(dataset), 'batches': batch_count, 'loss': actual['loss'], 'recorded_loss': loss,
            **{k: actual[k] for k in METRICS}, 'subject_mean_macro_f1': subject_mean,
            'weighted_ce_from_wrong': float(contribution[wrong].sum()),
            'weighted_ce_from_correct': float(contribution[~wrong].sum()),
            **confidence(dataset.labels, probs, nll)}


def check_test_rows(folder, fold_id, data, probs):
    rows = read_csv(folder / 'metrics/predictions.csv')
    assert len(rows) == len(data)
    for predicted, source, p in zip(rows, data.rows, probs):
        assert all(str(source[key]) == predicted[key] for key in ('subject_id', 'window_id', 'start_index',
                   'end_index', 'original_label', 'quality_status', 'start_time', 'end_time'))
        assert int(predicted['fold_id']) == fold_id and int(predicted['true_label']) == source['label']
        assert int(predicted['predicted_label']) == p.argmax()
    saved = np.array([[float(r[key]) for key in PROBABILITY_FIELDS] for r in rows])
    np.testing.assert_allclose(saved, probs, rtol=0, atol=2e-6)
    return rows, float(np.abs(saved - probs).max())


def audit_fold(manifest, config, fold):
    fid = fold['fold_id']
    folder = RUN / 'seed_42' / f'fold_{fid:02d}'
    meta = read_json(folder / 'metadata.json')
    assert meta['group_id'] == GROUP_ID and meta['status'] == 'completed' and meta['subjects'] == fold
    assert meta['effective_config'] == config == yaml.safe_load((folder / 'config.yaml').read_text(encoding='utf-8'))
    assert meta['selection_metric'] == 'val_subject_macro_f1'
    checksums = read_json(folder / 'completion.json')['files']
    for relative, expected in checksums.items():
        assert sha256(folder / relative) == expected, relative
    epochs, validation, batches = (read_csv(folder / 'logs' / name) for name in ('epochs.csv', 'validation.csv', 'iterations.csv'))
    for e in epochs:
        e['epoch'] = int(e['epoch'])
    for b in batches:
        b['global_step'] = int(b['global_step'])
    states = {kind: open_checkpoint(folder, kind, config, manifest, meta, meta[key]) for kind, key in
              [('best', 'best_epoch'), ('last', 'completed_epochs')]}
    assert all(torch.equal(value, states['last']['best_model'][name]) for name, value in states['best']['model'].items())
    check_history(epochs, validation, batches, states['last'], config, meta, folder)
    scalar_values = check_tensorboard(folder, epochs, validation, batches, read_json(folder / 'metrics/test.json'))
    checked_curves = check_curve_inputs(meta, epochs, batches)
    data = {role: WindowDataset(Path(manifest['cache_path']), fold[role]) for role in ROLES}
    for role, ds in data.items():
        assert ds.timeline and ds.input_channels == 14 and ds.source == Path(manifest['data_source'])
        assert ds.subjects == fold[role] and len(ds) == meta['window_counts'][role]
        np.testing.assert_array_equal(np.bincount(ds.labels, minlength=3), meta['label_distribution'][role])
    scaler = ChannelScaler().fit(data['train'])
    stored = read_json(folder / 'normalization.json')
    arrays = [np.asarray(stored[k], dtype=np.float32) for k in ('mean', 'std')]
    np.testing.assert_array_equal(np.stack([scaler.mean, scaler.std]), np.stack(arrays))
    for ds in data.values():
        ds.scaler = scaler
    counts = np.bincount(data['train'].labels, minlength=3)
    weights = torch.from_numpy((len(data['train']) / (3 * counts)).astype(np.float32))
    np.testing.assert_array_equal(weights.numpy(), meta['class_weights'])
    net = CNN1D(input_channels=14, num_classes=3, dropout=config['model']['dropout'])
    assert sum(p.numel() for p in net.parameters()) == meta['trainable_parameters'] == 47075
    replay_rows = []
    test_rows = None
    for kind, state in states.items():
        for role in ROLES if kind == 'best' else ROLES[:2]:
            ds = data[role]
            actual, probs, nll, number = replay(net, state, ds, weights, config['training']['batch_size'],
                                                arrays, verify_inputs=kind == 'best')
            replay_rows.append(role_row(folder, role, kind, state, epochs, validation, ds, actual, probs, nll, number, weights))
            if role == 'test':
                test_rows, max_error = check_test_rows(folder, fid, ds, probs)
    evidence = {'fold_id': fid, 'completion_file_hashes': len(checksums), 'tensorboard_scalar_values': scalar_values,
                'epochs': len(epochs), 'batches': len(batches), 'checked_curve_series_groups': checked_curves,
                'best_epoch': meta['best_epoch'], 'max_abs_test_probability_difference': max_error,
                'input_batches_checked': sum(r['batches'] for r in replay_rows if r['checkpoint'] == 'best'),
                'input_windows_checked': sum(r['windows'] for r in replay_rows if r['checkpoint'] == 'best')}
    print(f"Fold {fid}: {evidence['input_windows_checked']} checked inputs, {len(test_rows)} reproduced Test rows", flush=True)
    return meta, epochs, replay_rows, test_rows, evidence


def check_visualization():
    folder = RESULT / 'enhanced_visualizations'
    active = read_json(folder / 'generation_manifest.json')
    assert active['group_id'] == GROUP_ID and active['status'] == 'completed'
    generation = folder / active['generation_directory']
    manifest = read_json(generation / 'generation_manifest.json')
    assert active == manifest | {'generation_directory': generation.name}
    for name, expected in manifest['source_hashes'].items():
        assert sha256(ROOT / name) == expected, name
    entries = manifest['images']
    assert len(entries) == manifest['figure_count']
    for entry in entries:
        assert {Path(target['path']).suffix for target in entry['files']} == {'.png', '.svg'}
        for target in entry['files']:
            assert sha256(generation / target['path']) == target['sha256'], target
    return {'directory': generation.relative_to(ROOT).as_posix(), 'source_files': len(manifest['source_hashes']),
            'figure_pairs': len(entries)}


def prediction_arrays(rows):
    y = np.array([int(r['true_label']) for r in rows])
    p = np.array([[float(r[field]) for field in PROBABILITY_FIELDS] for r in rows])
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=2e-7, rtol=0)
    np.testing.assert_array_equal(p.argmax(axis=1), [int(r['predicted_label']) for r in rows])
    return y, p


def diagnostic_row(level, key, rows):
    y, p = prediction_arrays(rows)
    return {'level': level, 'group': key, 'windows': len(y), **confusion_metrics(y, p.argmax(axis=1)),
            **confidence(y, p)}


def check_test_aggregation(folds, by_window, manifest):
    oof = read_csv(RESULT / 'oof_predictions.csv')
    assert len(oof) == len({r['window_id'] for r in oof}) == len(by_window) == 2134
    for row in oof:
        assert row == {'training_seed': '42', **by_window[row['window_id']]}
    index_rows = read_csv(Path(manifest['cache_path']) / 'windows.csv')
    assert {r['window_id'] for r in oof} == {r['window_id'] for r in index_rows}
    pooled = diagnostic_row('pooled_oof', 'all', oof)
    fold_rows = {int(r['fold_id']): r for r in read_csv(RESULT / 'fold_results.csv')}
    by_fold = []
    for fid, rows in folds.items():
        folder = RUN / 'seed_42' / f'fold_{fid:02d}'
        d = diagnostic_row('fold', fid, rows)
        by_fold.append(d)
        test = read_json(folder / 'metrics/test.json')
        check_metric_values(d, test, (*METRICS, 'stress_recall'))
        for field in ('support', 'confusion_matrix', 'precision_per_class', 'recall_per_class', 'f1_per_class'):
            np.testing.assert_allclose(d[field], test[field], atol=1e-12, rtol=0, err_msg=field)
        check_metric_values(d, fold_rows[fid], (*METRICS, 'stress_recall'))
        np.testing.assert_array_equal(d['confusion_matrix'], np.loadtxt(folder / 'metrics/confusion_matrix.csv',
                                                                       delimiter=',', skiprows=1, dtype=int))
        class_report = read_json(folder / 'metrics/classification_report.json')
        for j, label in enumerate(LABELS):
            for key in ('precision', 'recall', 'f1'):
                np.testing.assert_allclose(class_report[label.lower()][key], d[f'{key}_per_class'][j], atol=1e-12, rtol=0)
            assert class_report[label.lower()]['support'] == d['support'][j]
    subjects = {r['subject_id']: r for r in read_csv(RESULT / 'subject_metrics.csv')}
    assert len(subjects) == 15
    by_subject = []
    for subject, expected in subjects.items():
        rows = [r for r in oof if r['subject_id'] == subject]
        assert {r['fold_id'] for r in rows} == {expected['fold_id']}
        d = diagnostic_row('subject', subject, rows)
        assert d['windows'] == int(expected['windows'])
        check_metric_values(d, expected, (*METRICS, 'stress_recall'))
        folder = RUN / 'seed_42' / f"fold_{int(expected['fold_id']):02d}" / 'metrics'
        local = next(r for r in read_csv(folder / 'subject_metrics.csv') if r['subject_id'] == subject)
        check_metric_values(d, local, (*METRICS, 'stress_recall'))
        by_subject.append(d)
    summary = read_json(RESULT / 'summary.json')
    assert summary['group_id'] == GROUP_ID and summary['status'] == 'completed' and summary['completed_folds'] == 5
    seed = summary['per_seed']['42']
    for level, values in [('fold_level', by_fold), ('subject_level', by_subject)]:
        for key in (*METRICS, 'stress_recall'):
            vals = [r[key] for r in values]
            assert seed[level][key]['n'] == len(vals)
            np.testing.assert_allclose([np.mean(vals), np.std(vals, ddof=1)],
                                       [seed[level][key]['mean'], seed[level][key]['std']], atol=1e-12, rtol=0)
    assert seed['oof_windows'] == len(oof)
    check_metric_values(pooled, seed['pooled_oof'], (*METRICS, 'stress_recall'))
    np.testing.assert_array_equal(pooled['confusion_matrix'], np.loadtxt(RESULT / 'aggregate_confusion_matrix.csv',
                                                                        delimiter=',', skiprows=1, dtype=int))
    np.testing.assert_array_equal(pooled['confusion_matrix'], seed['pooled_oof']['confusion_matrix'])
    diagnostics = [pooled, *by_fold, *by_subject]
    write_csv('test_diagnostics.csv', [{k: json.dumps(v) if isinstance(v, list) else v for k, v in r.items()}
                                       for r in diagnostics])
    return diagnostics, seed, oof


def history_row(meta, epochs):
    best, last = epochs[meta['best_epoch'] - 1], epochs[-1]
    min_loss = min(epochs, key=lambda r: float(r['val_loss']))
    first_increase = next((end['epoch'] for start, end in zip(epochs, epochs[1:])
                           if float(end['val_loss']) > float(start['val_loss'])), None)
    first_train_99 = next((e['epoch'] for e in epochs if float(e['train_macro_f1']) >= .99), None)
    row = {'fold_id': meta['fold_id'], 'train_subjects': len(meta['subjects']['train']),
           'train_windows': meta['window_counts']['train'], 'validation_windows': meta['window_counts']['validation'],
           'test_windows': meta['window_counts']['test'], 'first_train_macro_f1_ge_0_99': first_train_99,
           'first_val_loss_increase_epoch': first_increase, 'min_val_loss_epoch': min_loss['epoch'],
           'min_val_loss': float(min_loss['val_loss']), 'best_epoch': best['epoch'], 'last_epoch': last['epoch']}
    for phase, e in [('best', best), ('last', last)]:
        for metric in ('train_eval_loss', 'train_macro_f1', 'val_loss', 'val_macro_f1', 'val_subject_macro_f1', 'learning_rate'):
            row[f'{phase}_{metric}'] = float(e[metric])
        row[f'{phase}_pooled_macro_f1_gap'] = float(e['train_macro_f1']) - float(e['val_macro_f1'])
    return row


def plot_history(folds):
    fig, axes = plt.subplots(2, 5, figsize=(16, 7), layout='constrained', sharex='col', sharey='row')
    for i, (meta, epochs) in enumerate(folds):
        x = [r['epoch'] for r in epochs]
        ax = axes[0, i]
        for key, label, color in [('train_eval_loss', 'Train eval', '#0072B2'),
                                  ('val_loss', 'Validation', '#D55E00')]:
            ax.plot(x, [float(r[key]) for r in epochs], label=label, color=color)
        ax.set(yscale='log', title=f"Fold {meta['fold_id']} · best epoch {meta['best_epoch']}")
        ax = axes[1, i]
        for key, label, color, style in [('train_macro_f1', 'Train eval', '#0072B2', '-'),
                                         ('val_macro_f1', 'Validation pooled', '#D55E00', '-'),
                                         ('val_subject_macro_f1', 'Validation subject mean', '#009E73', '--')]:
            ax.plot(x, [float(r[key]) for r in epochs], label=label, color=color, linestyle=style)
        ax.set(xlabel='Epoch', ylim=(0, 1.03))
        for ax in axes[:, i]:
            ax.axvline(meta['best_epoch'], color='.35', linewidth=1, linestyle=':')
            ax.grid(axis='y', alpha=.25)
    axes[0, -1].legend(fontsize=8, loc='lower right')
    axes[1, -1].legend(fontsize=8, loc='lower right')
    axes[0, 0].set_ylabel('Weighted cross-entropy')
    axes[1, 0].set_ylabel('Macro-F1')
    fig.suptitle('CNN-1D · training and validation on all five folds')
    save_figure(fig, 'overfitting_curves')


def plot_test(diagnostics, oof):
    y, p = prediction_arrays(oof)
    correct = p.argmax(axis=1) == y
    maximum = p.max(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    bins = np.linspace(1 / 3, 1, 18)
    for mask, label, color in [(correct, 'Correct', '#0072B2'), (~correct, 'Wrong', '#D55E00')]:
        axes[0].hist(maximum[mask], bins=bins, histtype='step', linewidth=2, color=color,
                     label=f'{label} (n={int(mask.sum())})', weights=np.ones(mask.sum()) / mask.sum())
    axes[0].set(xlabel='Maximum softmax probability', ylabel='Fraction within each group', title='Pooled Test OOF confidence')
    axes[0].legend()
    subjects = sorted((d for d in diagnostics if d['level'] == 'subject'), key=lambda d: d['macro_f1'])
    axes[1].bar([d['group'] for d in subjects], [d['macro_f1'] for d in subjects], color='#0072B2')
    axes[1].axhline(np.mean([d['macro_f1'] for d in subjects]), color='#D55E00', linestyle='--', label='Subject mean')
    axes[1].set(ylim=(0, 1), ylabel='Macro-F1', title='Test performance by held-out subject')
    axes[1].tick_params(axis='x', labelrotation=45)
    axes[1].legend()
    save_figure(fig, 'test_predictions')


def main():
    manifest, config, split = load_context()
    torch.set_num_threads(config['runtime']['torch_threads'])
    torch.use_deterministic_algorithms(config['runtime']['deterministic'])
    before = fingerprint(manifest, config)
    visual = check_visualization()
    evidence, replay_rows, training, folds, by_window = [], [], [], {}, {}
    for fold in split['folds']:
        meta, epochs, rows, test_rows, proof = audit_fold(manifest, config, fold)
        evidence.append(proof)
        replay_rows.extend(rows)
        training.append((meta, epochs))
        folds[fold['fold_id']] = test_rows
        for row in test_rows:
            assert row['window_id'] not in by_window
            by_window[row['window_id']] = row
    diagnostics, summary, oof = check_test_aggregation(folds, by_window, manifest)
    write_csv('checkpoint_replay.csv', replay_rows)
    write_csv('overfitting_by_fold.csv', [history_row(meta, epochs) for meta, epochs in training])
    plot_history(training)
    plot_test(diagnostics, oof)
    after = fingerprint(manifest, config)
    assert before == after, 'A protected file changed during the replay'
    write_json('replay_evidence.json', {
        'group_id': GROUP_ID, 'torch': torch.__version__, 'device': 'cpu',
        'folds': evidence, 'tensorboard_event_files': len(evidence) * 3,
        'checkpoint_files': len(evidence) * 2, 'enhanced_visualizations': visual,
        'oof_windows': len(oof), 'primary_subject_mean': summary['subject_level'],
        'fold_summary': summary['fold_level'], 'pooled_test': diagnostics[0],
        'all_weights_and_buffers_unchanged_by_inference': True,
        'protected_file_count_before_after': len(before), 'protected_file_hash_collection': json_hash(before),
    })
    print(f"Verified {len(oof)} Test rows and {len(before)} protected file hashes before and after replay.", flush=True)


if __name__ == '__main__':
    main()

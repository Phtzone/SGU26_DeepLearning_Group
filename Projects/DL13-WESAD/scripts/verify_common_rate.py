"""Check a completed common-rate run against its index, checkpoints and event streams."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from scripts.verify_e1_baseline import close, compare_metrics, safe_load_checkpoint, verify_scalars
from src.data.common_rate import INDEX_METADATA, build_index
from src.data.dataset import WindowDataset
from src.data.normalization import ChannelScaler
from src.data.splits import load_manifest
from src.evaluation.aggregation import audit_iterations, read_csv as read_raw_csv, stats, validate_predictions
from src.evaluation.history import read_csv, resolve_group
from src.evaluation.metrics import classification_metrics, metrics_by_subject
from src.models import build_model
from src.training.callbacks import SelectionState
from src.training.optimizer import class_weights
from src.training.trainer import completed_fold
from src.utils.config import ROOT, experiment_folder, same_training_config
from src.utils.tracking import digest, file_hash, now, read_json, write_json


EPOCH_TAGS = {
    'train/online_loss': 'train_online_loss', 'train/eval_loss': 'train_eval_loss',
    'train/accuracy': 'train_accuracy', 'train/macro_f1': 'train_macro_f1',
    'train/balanced_accuracy': 'train_balanced_accuracy', 'val/loss': 'val_loss',
    'val/accuracy': 'val_accuracy', 'val/macro_f1': 'val_macro_f1',
    'val/balanced_accuracy': 'val_balanced_accuracy', 'val/subject_macro_f1': 'val_subject_macro_f1',
    'selection/best_epoch': 'best_epoch', 'selection/best_val_subject_macro_f1': 'best_val_subject_macro_f1',
    'selection/early_stopping_counter': 'early_stopping_counter',
}
BATCH_TAGS = {'batch/loss': 'train_loss', 'batch/learning_rate': 'learning_rate',
              'batch/gradient_norm': 'gradient_norm', 'batch/duration_ms': 'batch_duration_ms'}
TEST_TAGS = {'test/' + key: key for key in ('loss', 'accuracy', 'macro_f1', 'balanced_accuracy')}


def verify_fold(folder, group_id, fold, seed, config, checksum, cache):
    assert completed_fold(folder), folder
    meta = read_json(folder / 'metadata.json')
    assert meta['subjects'] == fold and meta['training_seed'] == seed and meta['group_id'] == group_id
    assert meta['input_shape'] == [len(config['data']['channels']), 1920]
    assert meta['manifest_checksum'] == checksum and same_training_config(meta['effective_config'], config)
    train = WindowDataset(cache, fold['train'])
    scaler = ChannelScaler().fit(train)
    normalization = read_json(folder / 'normalization.json')
    assert normalization['train_subjects'] == fold['train'] and normalization['sample_count_per_channel'] == scaler.count
    close(normalization['mean'], scaler.mean, 1e-7)
    close(normalization['std'], scaler.std, 1e-7)
    close(meta['class_weights'], class_weights(train.labels).numpy(), 1e-7)

    model = build_model(config, train.input_channels)
    checkpoints, states = {}, {}
    for kind in ('best', 'last'):
        path = folder / 'checkpoints' / (kind + '.pt')
        state = safe_load_checkpoint(path)
        assert state['checkpoint_kind'] == kind and state['run_id'] == meta['run_id']
        assert state['manifest_checksum'] == checksum and state['cache_keys'] == meta['cache_keys']
        assert same_training_config(state['effective_config'], config)
        assert state['epoch'] == meta['best_epoch' if kind == 'best' else 'completed_epochs']
        model.load_state_dict(state['model'], strict=True)
        assert all(torch.isfinite(value).all() for value in state['model'].values())
        checkpoints[kind] = {'path': path.relative_to(ROOT).as_posix(), 'sha256': file_hash(path),
                             'epoch': state['epoch'], 'size_bytes': path.stat().st_size}
        states[kind] = state
    last = states['last']
    assert all(torch.equal(value, last['best_model'][key]) for key, value in states['best']['model'].items())
    epochs = read_csv(folder / 'logs/epochs.csv')
    validation = read_csv(folder / 'logs/validation.csv')
    tc = config['training']
    selection = SelectionState(tc['early_stopping_patience'], tc['early_stopping_min_delta'])
    assert len(epochs) == len(validation) == len(last['history']) == meta['completed_epochs']
    committed = {(row['attempt_id'], row['epoch']) for row in last['history']}
    assert [r['epoch'] for r in epochs] == list(range(1, len(epochs) + 1))
    for row, val, saved in zip(epochs, validation, last['history']):
        assert row['epoch'] == val['epoch'] == saved['epoch'] and row['attempt_id'] == val['attempt_id'] == saved['attempt_id']
        scores = val['val_macro_f1_by_subject']
        assert set(scores) == set(fold['validation'])
        close(row['val_subject_macro_f1'], np.mean(list(scores.values())))
        close(val['val_subject_macro_f1'], row['val_subject_macro_f1'])
        for field in EPOCH_TAGS.values():
            close(row[field], saved[field])
        better, stop = selection.update(row['val_subject_macro_f1'], row['val_loss'], row['epoch'])
        assert better == val['is_best_checkpoint'] and row['early_stopping_counter'] == selection.bad_epochs
        assert not stop or row['epoch'] == len(epochs)
    assert selection.best_epoch == meta['best_epoch'] and last['best_epoch'] == selection.best_epoch
    close(selection.best_score, meta['best_val_subject_macro_f1'])
    log_audit = audit_iterations(folder, last, len(train), tc['batch_size'])
    iterations = [row for row in read_csv(folder / 'logs/iterations.csv') if (row['attempt_id'], row['epoch']) in committed]
    assert [r['global_step'] for r in iterations] == list(range(1, last['global_step'] + 1))
    assert {row['run_id'] for row in iterations} == {meta['run_id']}

    predictions = read_raw_csv(folder / 'metrics/predictions.csv')
    expected = [r for subject in fold['test'] for r in read_raw_csv(cache / subject / 'windows.csv')]
    metrics = validate_predictions(predictions, fold, expected)
    saved_metrics = read_json(folder / 'metrics/test.json')
    compare_metrics(metrics, saved_metrics)
    compare_metrics(metrics, meta['test_metrics'])
    close(saved_metrics['stress_recall'], metrics['recall_per_class'][1])

    identity = read_json(folder / 'tracking_identity.json')
    assert identity['group_id'] == group_id and identity['run_id'] == meta['run_id'] and identity['backend'] == 'tensorboard'
    tb = folder / 'tensorboard'
    verify_scalars(tb / 'epochs', EPOCH_TAGS, epochs, 'epoch')
    verify_scalars(tb / 'batches', BATCH_TAGS, iterations, 'global_step')
    evaluations = verify_scalars(tb / 'evaluation', TEST_TAGS, [dict(saved_metrics, epoch=len(epochs))], 'epoch')
    assert set(evaluations.Tags()['images']) == {'test/confusion_raw', 'test/confusion_normalized', 'test/per_class_f1'}
    files = [p.relative_to(ROOT).as_posix() for p in sorted(tb.rglob('events.out.tfevents.*'))]
    assert len(files) >= 3
    return {'fold_id': fold['fold_id'], 'run_id': meta['run_id'], 'started_at': meta['started_at'],
            'ended_at': meta['ended_at'], 'window_counts': meta['window_counts'], 'best_epoch': meta['best_epoch'],
            'epochs': len(epochs), 'trainable_parameters': meta['trainable_parameters'],
            'normalized_on_train': True, 'validation_selection_replayed': True, 'checkpoints': checkpoints,
            'iterations': log_audit, 'tensorboard_files': files, 'test_metrics': saved_metrics}, predictions


def verify(group_id):
    group = resolve_group(group_id)
    result = ROOT / 'results' / group.parent.name / group.name
    manifest = read_json(group / 'experiment_manifest.json')
    summary = read_json(result / 'summary.json')
    config = manifest['effective_config']
    assert group.parent.name == experiment_folder(config) and config['data']['source'] == 'common_rate'
    assert manifest == read_json(result / 'experiment_manifest.json')
    assert manifest['status'] == summary['status'] == 'completed'
    assert manifest['group_id'] == group_id and manifest['requested_folds'] == [1, 2, 3, 4, 5]
    assert summary['completed_folds'] == summary['expected_folds'] == 5 and not summary['missing_folds']

    split = load_manifest(group / 'split_manifest.json')
    split_checksum = digest(split)
    assert split_checksum == manifest['manifest_checksum'] == summary['manifest_checksum']
    assert split == load_manifest(ROOT / config['split']['manifest'])
    cache = Path(manifest['cache_path'])
    index = read_json(cache / INDEX_METADATA)
    audit, rebuilt = build_index(manifest['data_source'], config['data'], split['original_subject_order'])
    assert manifest['dataset_fingerprint'] == audit['dataset_fingerprint'] and index['audit'] == audit
    assert file_hash(ROOT / manifest['preprocessing_notebook']['path']) == manifest['preprocessing_notebook']['sha256']
    for relative, expected in index['index_file_sha256'].items():
        assert file_hash(cache / relative) == expected
    manifest_rows = read_raw_csv(cache / 'windows.csv')
    assert manifest_rows == [{k: str(v) for k, v in row.items()} for rows in rebuilt.values() for row in rows]
    source = read_json(group / 'source_checksums.json')
    assert all(file_hash(group / 'source' / name) == checksum for name, checksum in source.items())
    seed, = config['training']['seeds']

    torch.set_num_threads(config['runtime']['torch_threads'])
    folds, original = [], []
    for fold in split['folds']:
        folder = group / f'seed_{seed}' / f"fold_{fold['fold_id']:02d}"
        verification, predictions = verify_fold(folder, group_id, fold, seed, config, split_checksum, cache)
        folds.append(verification)
        original.extend({'training_seed': str(seed), **row} for row in predictions)
        print(f"Fold {fold['fold_id']}: artifacts, checkpoint selection and TensorBoard verified.", flush=True)
    assert len({r['run_id'] for r in folds}) == 5
    oof = read_raw_csv(result / 'oof_predictions.csv')
    assert oof == original and len(oof) == audit['windows']
    assert len({r['window_id'] for r in oof}) == len(oof)
    test_subjects = Counter(subject for fold in split['folds'] for subject in fold['test'])
    assert test_subjects == Counter(split['original_subject_order'])
    labels, predicted = ([int(row[k]) for row in oof] for k in ('true_label', 'predicted_label'))
    pooled = classification_metrics(labels, predicted)
    breakdown = summary['per_seed'][str(seed)]
    compare_metrics(pooled, breakdown['pooled_oof'])
    close(pooled['recall_per_class'][1], breakdown['pooled_oof']['stress_recall'])
    close(pooled['confusion_matrix'], np.loadtxt(result / 'aggregate_confusion_matrix.csv', delimiter=',', skiprows=1))
    by_subject = metrics_by_subject(labels, predicted, [r['subject_id'] for r in oof])
    subjects = read_csv(result / 'subject_metrics.csv')
    assert len(subjects) == len(by_subject) == 15 and {r['subject_id'] for r in subjects} == set(test_subjects)
    test_fold = {subject: fold['fold_id'] for fold in split['folds'] for subject in fold['test']}
    for row in subjects:
        subject = row['subject_id']
        metrics = by_subject[subject]
        assert row['fold_id'] == test_fold[subject] and row['windows'] == sum(metrics['support'])
        compare_metrics(metrics, row)
        close(metrics['recall_per_class'][1], row['stress_recall'])
    exported_folds = read_csv(result / 'fold_results.csv')
    assert len(exported_folds) == len(folds)
    for outcome, row in zip(folds, exported_folds):
        assert outcome['fold_id'] == row['fold_id'] and outcome['run_id'] == row['run_id']
        assert outcome['best_epoch'] == row['best_epoch'] and outcome['epochs'] == row['completed_epochs']
        compare_metrics(outcome['test_metrics'], row)
        close(outcome['test_metrics']['stress_recall'], row['stress_recall'])
    for level, rows in [('subject_level', subjects), ('fold_level', exported_folds)]:
        for metric, data in stats(rows).items():
            for key, value in data.items():
                close(value, breakdown[level][metric][key])

    visual = result / 'enhanced_visualizations'
    figures = read_json(visual / 'generation_manifest.json')
    assert figures['status'] == 'completed' and figures['group_id'] == group_id
    files = [item for image in figures['images'] for item in image['files']]
    assert len(files) == figures['file_count'] == 2 * figures['figure_count']
    for item in files:
        path = visual / figures['generation_directory'] / item['path']
        assert path.stat().st_size > 0 and file_hash(path) == item['sha256']
    evidence = {'status': 'passed', 'verified_at': now(), 'group_id': group_id, 'data_fingerprint': audit['dataset_fingerprint'],
                'split_manifest_sha256': split_checksum, 'source_snapshot_files': len(source),
                'verified_folds': len(folds), 'folds': folds, 'oof_windows': len(oof), 'test_subjects': len(test_subjects),
                'tensorboard_files': sum(len(fold['tensorboard_files']) for fold in folds),
                'checkpoint_count': sum(len(fold['checkpoints']) for fold in folds),
                'verified_figures': figures['figure_count'], 'verified_figure_files': len(files),
                'figure_directory': (visual / figures['generation_directory']).relative_to(ROOT).as_posix()}
    write_json(result / 'verification.json', evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group-id', required=True)
    args = parser.parse_args()
    report = verify(args.group_id)
    print(f"PASS: {report['verified_folds']} folds, {report['test_subjects']} Test subjects, "
          f"{report['oof_windows']} OOF windows, {report['checkpoint_count']} checkpoints, "
          f"{report['tensorboard_files']} TensorBoard files, {report['verified_figures']} figures.")


if __name__ == '__main__':
    main()

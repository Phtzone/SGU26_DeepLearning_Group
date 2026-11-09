"""Read-only E1 post-training verification; never trains or rewrites run artifacts."""
import argparse
from collections import Counter
from pathlib import Path
import re
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from src.data.dataset import WindowDataset
from src.data.normalization import ChannelScaler
from src.data.splits import load_manifest
from src.evaluation.aggregation import audit_iterations, stats, validate_predictions
from src.evaluation.history import read_csv
from src.evaluation.metrics import classification_metrics
from src.models.cnn import CNN1D
from src.training.callbacks import SelectionState
from src.training.optimizer import class_weights
from src.utils.config import ROOT, load_config, same_training_config
from src.utils.tracking import digest, file_hash, now, read_json, write_json


def safe_load_checkpoint(path):
    """Only tensors, primitives and the fixed NumPy RNG-state representation."""
    allowed = {'numpy.ndarray', 'numpy.dtype', 'numpy._core.multiarray._reconstruct'}
    unexpected = set(torch.serialization.get_unsafe_globals_in_checkpoint(path)) - allowed
    if unexpected:
        raise ValueError(f'Unexpected checkpoint globals: {sorted(unexpected)}')
    # These exact NumPy types encode the saved MT19937 RNG state. No fallback
    # to weights_only=False or arbitrary checkpoint-provided globals is allowed.
    with torch.serialization.safe_globals([
            np.ndarray, np.dtype, np._core.multiarray._reconstruct, np.dtypes.UInt32DType]):
        return torch.load(path, map_location='cpu', weights_only=True)


def close(actual, expected, tolerance=1e-12):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)


def compare_metrics(actual, stored):
    for key in ('accuracy', 'macro_f1', 'balanced_accuracy', 'confusion_matrix',
                'precision_per_class', 'recall_per_class', 'f1_per_class', 'support'):
        if key in stored:
            close(actual[key], stored[key])
    if 'label_order' in stored:
        assert actual['label_order'] == stored['label_order']


def verify_scalars(directory, mapping, rows, step_key):
    acc = EventAccumulator(str(directory), size_guidance={'scalars': 0, 'images': 0}).Reload()
    for tag, key in mapping.items():
        events = acc.Scalars(tag)
        assert [e.step for e in events] == [int(row[step_key]) for row in rows], (directory, tag)
        # TensorBoard serializes scalar values as float32; CSV retains float64.
        np.testing.assert_allclose([e.value for e in events], [row[key] for row in rows],
                                   rtol=1e-6, atol=1e-6, err_msg=str((directory, tag)))
    return acc


def verify(group_id, root=ROOT):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', group_id):
        raise ValueError('Invalid group ID')
    root = Path(root)
    group = root / 'runs/experiment_e1' / group_id
    result = root / 'results/experiment_e1' / group_id
    manifest = read_json(group / 'experiment_manifest.json')
    result_manifest = read_json(result / 'experiment_manifest.json')
    summary = read_json(result / 'summary.json')
    assert manifest == result_manifest, 'Run/result manifests differ'
    assert manifest['group_id'] == summary['group_id'] == group_id
    assert manifest['status'] == summary['status'] == 'completed'
    config = manifest['effective_config']
    assert same_training_config(config, load_config('experiments/e1_cnn/config.yaml'))
    assert config['training']['seeds'] == [42] and config['tracking']['backend'] == 'tensorboard'
    split = load_manifest(group / 'split_manifest.json')
    assert split == load_manifest(root / config['split']['manifest'])
    assert digest(split) == manifest['manifest_checksum'] == summary['manifest_checksum']
    assert manifest['requested_folds'] == [1, 2, 3, 4, 5]
    assert summary['completed_folds'] == summary['expected_folds'] == 5 and not summary['missing_folds']
    test_subjects = []
    for fold in split['folds']:
        tr, va, te = (set(fold[k]) for k in ('train', 'validation', 'test'))
        assert (len(tr), len(va), len(te)) == (9, 3, 3)
        assert not (tr & va or tr & te or va & te)
        assert tr | va | te == set(split['original_subject_order'])
        test_subjects.extend(fold['test'])
    assert len(test_subjects) == 15 and set(Counter(test_subjects).values()) == {1}
    cache = Path(manifest['cache_path'])
    cache_records = []
    for subject in split['original_subject_order']:
        metadata = read_json(cache / subject / 'metadata.json')
        assert metadata['cache_key'] == manifest['cache_keys'][subject]
        assert metadata['data_config'] == config['data']
        for filename, expected in metadata['files'].items():
            assert file_hash(cache / subject / filename) == expected
        cache_records.append({'subject_id': subject, 'cache_key': metadata['cache_key'],
                              'preprocessing_version': metadata['preprocessing_version'],
                              'raw_sha256_recorded_at_preprocessing': metadata['raw_sha256'],
                              'cache_files': metadata['files'], 'shape': metadata['shape']})
    folds, checkpoints, subjects, all_predictions, tensorboard = [], [], [], [], []
    protected = {}
    epoch_mapping = {'train/online_loss': 'train_online_loss', 'train/eval_loss': 'train_eval_loss',
                     'train/accuracy': 'train_accuracy', 'train/macro_f1': 'train_macro_f1',
                     'train/balanced_accuracy': 'train_balanced_accuracy', 'val/loss': 'val_loss',
                     'val/accuracy': 'val_accuracy', 'val/macro_f1': 'val_macro_f1',
                     'val/balanced_accuracy': 'val_balanced_accuracy'}
    total_epochs = total_batches = 0
    for fold in split['folds']:
        fid = fold['fold_id']
        folder = group / 'seed_42' / f'fold_{fid:02d}'
        metadata = read_json(folder / 'metadata.json')
        assert metadata['status'] == 'completed' and metadata['training_seed'] == 42
        assert metadata['subjects'] == fold and metadata['fold_id'] == fid
        assert metadata['manifest_checksum'] == digest(split)
        assert metadata['effective_config'] == config
        for rel, expected in read_json(folder / 'completion.json')['files'].items():
            assert file_hash(folder / rel) == expected, str(folder / rel)
            protected[(folder / rel).relative_to(root).as_posix()] = expected
        loaded = {}
        for kind in ('best', 'last'):
            path = folder / f'checkpoints/{kind}.pt'
            before = file_hash(path)
            assert path.stat().st_size > 0
            state = safe_load_checkpoint(path)
            assert file_hash(path) == before
            assert state['checkpoint_kind'] == kind and state['run_id'] == metadata['run_id']
            assert state['effective_config'] == config and state['manifest_checksum'] == digest(split)
            assert state['cache_keys'] == metadata['cache_keys']
            assert set(state['cache_keys']) == set(fold['train'] + fold['validation'] + fold['test'])
            model = CNN1D()
            model.load_state_dict(state['model'], strict=True)
            assert sum(p.numel() for p in model.parameters()) == metadata['trainable_parameters'] == 45059
            assert str(model) == metadata['architecture']
            assert all(torch.isfinite(value).all() for value in state['model'].values())
            expected_epoch = metadata['best_epoch'] if kind == 'best' else metadata['completed_epochs']
            assert state['epoch'] == expected_epoch and state['best_epoch'] == metadata['best_epoch']
            checkpoints.append({'path': path.relative_to(root).as_posix(), 'size_bytes': path.stat().st_size,
                                'sha256': before, 'kind': kind, 'epoch': state['epoch'],
                                'fold_id': fid, 'training_seed': 42, 'run_id': state['run_id'],
                                'weights_only_load': True, 'strict_architecture_match': True})
            loaded[kind] = state
        last, best = loaded['last'], loaded['best']
        assert all(torch.equal(best['model'][k], last['best_model'][k]) for k in best['model'])
        epochs = read_csv(folder / 'logs/epochs.csv')
        validation = read_csv(folder / 'logs/validation.csv')
        batches = read_csv(folder / 'logs/iterations.csv')
        assert len(epochs) == len(validation) == len(last['history']) == metadata['completed_epochs']
        assert [r['epoch'] for r in epochs] == list(range(1, len(epochs) + 1))
        assert [r['global_step'] for r in batches] == list(range(1, len(batches) + 1))
        assert last['global_step'] == len(batches)
        selection = SelectionState(config['training']['early_stopping_patience'], config['training']['early_stopping_min_delta'])
        for index, (row, vrow, committed) in enumerate(zip(epochs, validation, last['history'])):
            assert row['epoch'] == vrow['epoch'] == committed['epoch'] == index + 1
            assert row['attempt_id'] == vrow['attempt_id'] == committed['attempt_id']
            for key in epoch_mapping.values():
                close(row[key], committed[key])
            for key in ('val_loss', 'val_macro_f1', 'val_balanced_accuracy', 'val_accuracy'):
                close(vrow[key], row[key])
            better, stop = selection.update(row['val_macro_f1'], row['val_loss'], row['epoch'])
            assert vrow['is_best_checkpoint'] == better
            assert not stop or index == len(epochs) - 1
        assert selection.best_epoch == metadata['best_epoch'] == last['best_epoch']
        close(selection.best_score, metadata['best_val_macro_f1'])
        dataset = WindowDataset(cache, fold['train'])
        scaler = ChannelScaler().fit(dataset)
        normalization = read_json(folder / 'normalization.json')
        assert normalization['train_subjects'] == fold['train']
        np.testing.assert_allclose(normalization['mean'], scaler.mean, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(normalization['std'], scaler.std, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(metadata['class_weights'], class_weights(dataset.labels).numpy(), rtol=1e-6)
        assert normalization['sample_count_per_channel'] == scaler.count
        iteration_audit = audit_iterations(folder, last, len(dataset), config['training']['batch_size'])
        assert iteration_audit['orphan_or_replayed_iterations'] == 0
        assert iteration_audit == read_json(result / 'iteration_audit.json')[f'42/fold_{fid:02d}']
        predictions = read_csv(folder / 'metrics/predictions.csv')
        expected = [r for subject in fold['test'] for r in read_csv(cache / subject / 'windows.csv')]
        metrics = validate_predictions(predictions, fold, expected)
        compare_metrics(metrics, metadata['test_metrics'])
        compare_metrics(metrics, read_json(folder / 'metrics/test.json'))
        close(metrics['confusion_matrix'], np.loadtxt(folder / 'metrics/confusion_matrix.csv', delimiter=',', skiprows=1))
        for subject in fold['test']:
            rows = [r for r in predictions if r['subject_id'] == subject]
            sm = classification_metrics([r['true_label'] for r in rows], [r['predicted_label'] for r in rows])
            subjects.append({'subject_id': subject, 'fold_id': fid, 'training_seed': 42,
                             'windows': len(rows), **{k: sm[k] for k in ('accuracy', 'macro_f1', 'balanced_accuracy')}})
        identity = read_json(folder / 'tracking_identity.json')
        assert identity['group_id'] == group_id and identity['run_id'] == metadata['run_id']
        assert identity['backend'] == 'tensorboard'
        verify_scalars(folder / 'tensorboard/epochs', epoch_mapping, epochs, 'epoch')
        verify_scalars(folder / 'tensorboard/batches', {'batch/loss': 'train_loss',
                       'batch/learning_rate': 'learning_rate', 'batch/gradient_norm': 'gradient_norm',
                       'batch/duration_ms': 'batch_duration_ms'}, batches, 'global_step')
        evaluation = verify_scalars(folder / 'tensorboard/evaluation',
                                    {'test/' + k: k for k in ('loss', 'accuracy', 'macro_f1', 'balanced_accuracy')},
                                    [{**metadata['test_metrics'], 'epoch': len(epochs)}], 'epoch')
        assert set(evaluation.Tags()['images']) == {'test/confusion_raw', 'test/confusion_normalized', 'test/per_class_f1'}
        files = sorted((folder / 'tensorboard').rglob('events.out.tfevents.*'))
        assert len(files) == 3
        tensorboard.append({'fold_id': fid, 'event_files': [p.relative_to(root).as_posix() for p in files],
                            'epoch_steps': len(epochs), 'batch_steps': len(batches),
                            'scalar_values_match_csv': True, 'tolerance': {'rtol': 1e-6, 'atol': 1e-6}})
        folds.append({'fold_id': fid, 'training_seed': 42, 'run_id': metadata['run_id'],
                      'completed_epochs': len(epochs), 'best_epoch': metadata['best_epoch'],
                      'iterations': len(batches), 'normalization_train_only_recomputed': True,
                      'class_weights_train_only_recomputed': True, 'validation_selection_replayed': True,
                      **{k: metrics[k] for k in ('accuracy', 'macro_f1', 'balanced_accuracy')}})
        all_predictions.extend({'training_seed': 42, **r} for r in predictions)
        total_epochs += len(epochs)
        total_batches += len(batches)
        print(f'Verified fold {fid}: 2 checkpoints, {len(epochs)} epochs, {len(batches)} batches', flush=True)
    assert len({r['run_id'] for r in folds}) == 5
    oof = read_csv(result / 'oof_predictions.csv')
    key = lambda r: (r['training_seed'], r['subject_id'], r['window_id'])
    assert len({key(r) for r in oof}) == len(oof) == 2136
    assert sorted(oof, key=key) == sorted(all_predictions, key=key)
    pooled = classification_metrics([r['true_label'] for r in oof], [r['predicted_label'] for r in oof])
    compare_metrics(pooled, summary['per_seed']['42']['pooled_oof'])
    close(pooled['confusion_matrix'], np.loadtxt(result / 'aggregate_confusion_matrix.csv', delimiter=',', skiprows=1))
    for recomputed, saved in ((folds, read_csv(result / 'fold_results.csv')),
                              (subjects, read_csv(result / 'subject_metrics.csv'))):
        assert len(recomputed) == len(saved)
        for row in recomputed:
            keys = ('fold_id', 'subject_id') if 'subject_id' in row else ('fold_id',)
            other, = [r for r in saved if all(r[k] == row[k] for k in keys)]
            for metric in ('accuracy', 'macro_f1', 'balanced_accuracy'):
                close(row[metric], other[metric])
        level = 'subject_level' if 'subject_id' in recomputed[0] else 'fold_level'
        for metric, values in stats(recomputed).items():
            for stat_name in ('mean', 'std', 'n'):
                close(values[stat_name], summary['per_seed']['42'][level][metric][stat_name])
    assert total_epochs == 137 and total_batches == 5554
    snapshot = read_json(group / 'source_checksums.json')
    for rel, expected in snapshot.items():
        assert file_hash(group / 'source' / rel) == expected
    return {'status': 'passed', 'verified_at': now(), 'group_id': group_id,
            'completed_folds': 5, 'epochs': total_epochs, 'iterations': total_batches,
            'oof_windows': len(oof), 'pooled_oof': pooled, 'fold_mean_std': stats(folds),
            'folds': folds, 'subjects': subjects, 'checkpoints': checkpoints, 'tensorboard': tensorboard,
            'subject_leakage': False, 'split_checksum': digest(split),
            'dataset': {'cache_path': str(cache), 'subjects': cache_records,
                        'raw_files_not_loaded_or_modified': True},
            'source_snapshot_hashes_verified': len(snapshot),
            'source_snapshot_fingerprint': digest(snapshot),
            'protected_run_artifact_hashes': protected,
            'test_selection_evidence': ['Validation selection replayed from committed logs.',
                                        'Trainer consumes Test only after loading the selected best checkpoint.',
                                        'tests/test_resume.py::test_test_not_used_for_checkpoint_selection'],
            'dependencies_recorded_at_training': read_json(group / 'system.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.group_id)
    write_json(args.output, report)
    print(f'PASS: {report["completed_folds"]} folds / {len(report["checkpoints"])} checkpoints / '
          f'{sum(len(r["event_files"]) for r in report["tensorboard"])} TensorBoard files')


if __name__ == '__main__':
    main()

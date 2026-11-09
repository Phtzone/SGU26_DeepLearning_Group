import csv
import numpy as np
import pytest
from src.data.splits import SUBJECTS
from src.utils.tracking import write_json


@pytest.fixture
def tiny_cache(tmp_path):
    """Synthetic data for logic tests only, never research results."""
    cache = tmp_path / 'synthetic_cache'
    rng = np.random.default_rng(13)
    for subject in SUBJECTS:
        folder = cache / subject
        folder.mkdir(parents=True)
        x = rng.normal(size=(6, 5, 32)).astype(np.float32)
        np.save(folder / 'windows.npy', x)
        with (folder / 'windows.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['subject_id', 'window_id', 'start_time', 'end_time', 'original_label', 'label'])
            writer.writeheader()
            writer.writerows({'subject_id': subject, 'window_id': f'{subject}_{i}', 'start_time': i, 'end_time': i + 1, 'original_label': i % 3 + 1, 'label': i % 3} for i in range(6))
        write_json(folder / 'metadata.json', {'cache_key': 'synthetic-' + subject})
    return cache


@pytest.fixture
def fake_wandb(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock
    sdk = SimpleNamespace(Settings=lambda **kwargs: kwargs, runs=[])

    def initialize(**kwargs):
        run = SimpleNamespace(id=kwargs['id'], dir=str(tmp_path / kwargs['id']), summary={},
            log=Mock(), finish=Mock(), define_metric=Mock(), log_artifact=Mock())
        sdk.runs.append((kwargs, run))
        return run

    sdk.init = Mock(side_effect=initialize)
    sdk.Table, sdk.Image, sdk.plot_table, sdk.Artifact = Mock(), Mock(), Mock(), Mock()
    return sdk


@pytest.fixture
def synthetic_history(tmp_path_factory, tiny_cache):
    """A small legacy-schema history for tool tests; never a WESAD result."""
    from src.utils.config import load_config
    from src.utils.logging import append_csv, ITERATION_FIELDS, LEGACY_EPOCH_FIELDS, LEGACY_VALIDATION_FIELDS
    from src.training.checkpoint import save_checkpoint
    from src.evaluation.metrics import classification_metrics
    # Short, isolated paths keep figure and atomic-write tests below Windows MAX_PATH.
    root = tmp_path_factory.mktemp('history')
    group = root / 'runs/test_experiment/unit_group'
    folder = group / 'seed_42/fold_02'
    config = load_config()
    config['logging']['console']['enabled'] = False
    predictions = []
    for subject in ['S8', 'S17', 'S15']:
        for i in range(3):
            predicted = (i + int(subject == 'S8')) % 3
            probabilities = [.1, .1, .1]
            probabilities[predicted] = .8
            predictions.append({'fold_id': 2, 'subject_id': subject, 'window_id': subject + '_' + str(i),
                'true_label': i, 'predicted_label': predicted, 'start_time': float(i * 15), 'end_time': float(i * 15 + 30),
                'p_baseline': probabilities[0], 'p_stress': probabilities[1], 'p_amusement': probabilities[2]})
    metrics = classification_metrics([r['true_label'] for r in predictions], [r['predicted_label'] for r in predictions])
    metadata = {'run_id': 'synthetic_test_only', 'fold_id': 2, 'training_seed': 42, 'device': 'cpu',
        'effective_config': config, 'manifest_checksum': 'unit-test', 'status': 'completed', 'best_epoch': 1,
        'completed_epochs': 1, 'best_val_macro_f1': metrics['macro_f1'], 'training_duration_seconds': 1,
        'window_counts': {'train': 9, 'validation': 9, 'test': 9}, 'trainable_parameters': 45059,
        'subjects': {'train': ['S2'], 'validation': ['S3'], 'test': ['S8', 'S17', 'S15']},
        'label_distribution': {role: [3, 3, 3] for role in ['train', 'validation', 'test']}}
    write_json(folder / 'metadata.json', metadata)
    write_json(folder / 'metrics/test.json', metrics)
    write_json(group / 'experiment_manifest.json', {'effective_config': config, 'group_id': 'unit_group', 'cache_path': str(tiny_cache), 'status': 'completed'})
    for row in predictions:
        append_csv(folder / 'metrics/predictions.csv', row)
    epoch = {key: 1 for key in LEGACY_EPOCH_FIELDS}
    epoch.update(attempt_id='unit_attempt', epoch=1, train_macro_f1=metrics['macro_f1'], val_macro_f1=metrics['macro_f1'])
    append_csv(folder / 'logs/epochs.csv', epoch, LEGACY_EPOCH_FIELDS)
    val = {key: 1 for key in LEGACY_VALIDATION_FIELDS}
    val.update(attempt_id='unit_attempt', epoch=1, val_precision_per_class=metrics['precision_per_class'],
        val_recall_per_class=metrics['recall_per_class'], val_f1_per_class=metrics['f1_per_class'], is_best_checkpoint=True)
    append_csv(folder / 'logs/validation.csv', val, LEGACY_VALIDATION_FIELDS)
    batch = {key: 1 for key in ITERATION_FIELDS}
    batch.update(run_id=metadata['run_id'], attempt_id='unit_attempt', epoch=1, batch_index=0, batch_size=9, training_seed=42, fold_id=2)
    append_csv(folder / 'logs/iterations.csv', batch, ITERATION_FIELDS)
    save_checkpoint(folder / 'checkpoints/last.pt', {'history': [epoch]})
    (folder / 'plots').mkdir()
    (folder / 'plots/original.png').write_bytes(b'original-figure-test-sentinel')
    return root, group, folder

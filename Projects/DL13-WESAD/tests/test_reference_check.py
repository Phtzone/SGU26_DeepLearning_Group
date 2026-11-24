"""Prevent official comparisons when inputs, windows or protocol differ."""
from copy import deepcopy
import csv

import pytest

from src.data.reference_check import compare_reference, require_matching_reference
from src.utils.config import load_config
from src.utils.tracking import digest, write_json


@pytest.fixture
def reference_case(tmp_path):
    split = {'original_subject_order': ['S2', 'S3']}
    config = {'data': {'source': 'common_rate', 'channels': ['synthetic']}, 'split': {'seed': 42},
              'experiment': {'id': 'common_rate64_unit', 'model': 'gru'},
              'training': {'batch_size': 32}, 'selection': {'metric': 'val_subject_macro_f1'},
              'reference_run': {'experiment': 'reference', 'group_id': 'unit'}}
    rows = {s: [{'subject_id': s, 'window_id': f'{s}_0_1920', 'label': 0,
                  'start_time': 0., 'end_time': 30.}] for s in ('S2', 'S3')}
    audit = {'subjects': [{'subject_id': s, 'cache_key': s} for s in rows],
             'dataset_fingerprint': 'current'}
    write_json(tmp_path / 'runs/reference/unit/experiment_manifest.json', {
        'effective_config': deepcopy(config), 'status': 'completed', 'manifest_checksum': digest(split),
        'cache_keys': {'S2': 'S2', 'S3': 'S3'}, 'dataset_fingerprint': 'old_metadata_bytes'})
    result = tmp_path / 'results/reference/unit'
    result.mkdir(parents=True)
    with (result / 'oof_predictions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['subject_id', 'window_id', 'true_label', 'start_time', 'end_time'])
        writer.writeheader()
        writer.writerows({'subject_id': s, 'window_id': f'{s}_0_1920', 'true_label': 0,
                         'start_time': 0, 'end_time': 30} for s in rows)
    return tmp_path, config, split, audit, rows


def test_matching_input_keys_and_windows_allow_metadata_format_difference(reference_case):
    result = require_matching_reference(*reference_case)
    assert result['passed'] and not result['fingerprint_matches']


@pytest.mark.parametrize('change', ['source_key', 'label', 'interval', 'duplicate', 'protocol', 'roster'])
def test_changed_scientific_inputs_are_blocked(reference_case, change):
    root, config, split, audit, rows = reference_case
    if change == 'source_key':
        audit['subjects'][0]['cache_key'] = 'changed-signal-or-mask'
    elif change == 'label':
        rows['S2'][0]['label'] = 1
    elif change == 'interval':
        rows['S2'][0]['end_time'] = 31
    elif change == 'duplicate':
        rows['S2'].append(dict(rows['S2'][0]))
    elif change == 'protocol':
        config['selection']['metric'] = 'val_macro_f1'
    else:
        audit['subjects'].append({'subject_id': 'S4', 'cache_key': 'S4'})
    assert not compare_reference(*reference_case)['training_allowed']
    with pytest.raises(ValueError, match='before training'):
        require_matching_reference(*reference_case)


def test_gru_14ch_matches_cnn_protocol_and_uses_own_namespace():
    cnn = load_config('experiments/common_rate64_cnn1d_grouped5fold_v1/config.yaml')
    gru = load_config('experiments/common_rate64_gru_grouped5fold_v1/config.yaml')
    for section in ('data', 'split', 'training', 'selection', 'runtime', 'tracking'):
        assert cnn[section] == gru[section]
    assert len(gru['data']['channels']) == 14
    assert gru['experiment']['model'] == 'gru'
    assert gru['experiment']['id'] != cnn['experiment']['id']


def test_runner_blocks_reference_mismatch_before_any_output_or_training(reference_case, monkeypatch):
    from unittest.mock import Mock
    import scripts.run_grouped_cv as runner
    root, config, split, audit, rows = reference_case
    audit['subjects'][0]['cache_key'] = 'changed'
    monkeypatch.setattr(runner, 'ROOT', root)
    monkeypatch.setattr(runner, 'execution_plan', lambda *a: (
        split, [1], None, {'data_source': str(root/'source'), 'missing_subjects': [], 'missing_files': []}))
    monkeypatch.setattr(runner, 'build_index', lambda *a: (audit, rows))
    train, prepare = Mock(), Mock()
    monkeypatch.setattr(runner, 'train_fold', train)
    monkeypatch.setattr(runner, 'prepare_index', prepare)
    with pytest.raises(ValueError, match='before training'):
        runner.run(config, group_id='must_not_exist')
    train.assert_not_called()
    prepare.assert_not_called()
    assert not (root / 'runs/common_rate64_unit').exists()

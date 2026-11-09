import pytest
from src.evaluation.aggregation import validate_predictions


def test_duplicate_or_incorrect_oof_rejected():
    fold = {'fold_id': 1, 'test': ['S2']}
    expected = [{'subject_id': 'S2', 'window_id': 'S2_1', 'label': 1}]
    row = {'subject_id': 'S2', 'window_id': 'S2_1', 'fold_id': 1, 'true_label': 1, 'predicted_label': 1, 'p_baseline': .1, 'p_stress': .8, 'p_amusement': .1}
    with pytest.raises(ValueError, match='Duplicate'):
        validate_predictions([row, row], fold, expected)
    with pytest.raises(ValueError, match='ground truth'):
        validate_predictions([{**row, 'true_label': 0}], fold, expected)
    with pytest.raises(ValueError, match='probabilities'):
        validate_predictions([{**row, 'p_stress': .1}], fold, expected)


def test_full_five_fold_aggregation_and_tamper_detection(tiny_cache, tmp_path):
    from src.data.splits import create_manifest
    from src.training.trainer import train_fold
    from src.evaluation.aggregation import aggregate_group
    from src.utils.config import load_config
    from src.utils.tracking import write_json, digest
    config = load_config()
    config['training'].update(max_epochs=1, batch_size=64)
    split = create_manifest()
    group = tmp_path / 'synthetic_group'
    group.mkdir()
    write_json(group / 'split_manifest.json', split)
    (group / 'split_manifest.sha256').write_text(digest(split))
    write_json(group / 'experiment_manifest.json', {'effective_config': config, 'manifest_checksum': digest(split), 'cache_path': str(tiny_cache)})
    partial = aggregate_group(group, tmp_path / 'partial', update_docs=False)
    assert partial['status'] == 'partial' and partial['per_seed'] == {}
    from src.utils.tracking import read_json
    assert read_json(tmp_path / 'partial/experiment_manifest.json')['status'] == 'partial'
    for fold in split['folds']:
        train_fold(group / 'seed_42' / f'fold_{fold["fold_id"]:02d}', tiny_cache, fold, 42, config, digest(split))
    summary = aggregate_group(group, tmp_path / 'results', update_docs=False)
    assert summary['status'] == 'completed'
    assert read_json(tmp_path / 'results/experiment_manifest.json')['status'] == 'completed'
    assert summary['per_seed']['42']['oof_windows'] == 90
    import csv
    with (tmp_path / 'results/subject_metrics.csv').open() as stream:
        assert len(list(csv.DictReader(stream))) == 15
    path = group / 'seed_42/fold_01/metrics/predictions.csv'
    with path.open('a') as stream:
        stream.write('duplicate\n')
    with pytest.raises(ValueError, match='modified'):
        aggregate_group(group, tmp_path / 'results', update_docs=False)

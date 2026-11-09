from copy import deepcopy
from unittest.mock import Mock
import pytest
from src.data.splits import ensure_manifest, SUBJECTS
from src.utils.config import load_config, same_training_config
from src.utils.tracking import write_json, read_json, digest
import scripts.run_grouped_cv as runner


@pytest.fixture
def scientific_config_manifest(tmp_path):
    """Independent protocol reference, not a saved experiment or training result."""
    reference = {
        'experiment': {'id': 'E1_CNN', 'model': 'cnn1d', 'status': 'implemented'},
        'split': {'protocol': 'grouped_subject_wise_5fold',
                  'manifest': 'splits/grouped_5fold_seed42.json', 'split_seed': 42},
        'data': {'sampling_rate': 32, 'window_seconds': 30, 'stride_seconds': 15,
                 'sensors': ['ACC', 'BVP', 'EDA'], 'labels': [1, 2, 3],
                 'max_windows_per_subject': None},
        'training': {'seeds': [42], 'max_epochs': 100, 'batch_size': 32,
                     'optimizer': 'adam', 'learning_rate': .001, 'weight_decay': .0001,
                     'loss': 'weighted_cross_entropy', 'early_stopping_patience': 15,
                     'early_stopping_min_delta': .001, 'gradient_clip_norm': 1.0},
        'selection': {'metric': 'val_macro_f1', 'mode': 'max', 'tie_breaker': 'val_loss'},
        'runtime': {'device': 'auto', 'num_workers': 0, 'deterministic': True,
                    'parallel_folds': 1, 'torch_threads': 2},
    }
    path = tmp_path / 'scientific_reference.json'
    write_json(path, {'purpose': 'unit_test_protocol_reference', 'effective_config': reference})
    return path


@pytest.fixture
def sandbox_runner(tmp_path_factory, monkeypatch):
    root = tmp_path_factory.mktemp('runner')
    raw = root / 'raw'
    for subject in SUBJECTS:
        (raw / subject).mkdir(parents=True)
        (raw / subject / (subject + '.pkl')).touch()
    ensure_manifest(root / 'splits/grouped_5fold_seed42.json')
    monkeypatch.setattr(runner, 'ROOT', root)
    monkeypatch.setattr(runner, 'raw_root', lambda: raw)
    monkeypatch.setattr(runner, 'cache_path', lambda config: root / 'cache')
    cfg = load_config()
    cfg['logging']['console']['enabled'] = False
    return root, cfg


def test_dry_run_never_preprocesses_or_trains(sandbox_runner, monkeypatch):
    root, cfg = sandbox_runner
    prepare, train = Mock(), Mock()
    monkeypatch.setattr(runner, 'prepare_dataset', prepare)
    monkeypatch.setattr(runner, 'train_fold', train)
    result = runner.run(cfg, folds=[1], dry_run=True)
    assert result['selected_folds'] == [1] and result['subjects_present'] == 15
    prepare.assert_not_called(); train.assert_not_called()
    assert not (root / 'runs').exists()


def test_single_full_resume_and_completed_skip(sandbox_runner, monkeypatch):
    root, cfg = sandbox_runner
    trained = []
    monkeypatch.setattr(runner, 'prepare_dataset', lambda *args: [{'subject_id': subject, 'cache_key': subject} for subject in SUBJECTS])
    def train(folder, cache, fold, seed, config, checksum, **kwargs):
        if (folder / 'metadata.json').exists():
            return read_json(folder / 'metadata.json')
        trained.append(fold['fold_id'])
        metadata = {'status': 'completed', 'fold_id': fold['fold_id'], 'training_seed': seed, 'completed_epochs': 1,
                    'best_epoch': 1, 'test_metrics': {'accuracy': .5, 'macro_f1': .4, 'balanced_accuracy': .3}}
        write_json(folder / 'metadata.json', metadata)
        write_json(folder / 'completion.json', {'files': {}})
        return metadata
    monkeypatch.setattr(runner, 'train_fold', train)
    def aggregate(group, update_docs):
        assert update_docs is False
        count = len(list(group.glob('seed_*/fold_*/metadata.json')))
        summary = {'status': 'completed' if count == 5 else 'partial', 'completed_folds': count}
        write_json(root / 'results/experiment_e1' / group.name / 'summary.json', summary)
        # Reproduce an aggregator copying the still-running input manifest.
        write_json(root / 'results/experiment_e1' / group.name / 'experiment_manifest.json',
                   read_json(group / 'experiment_manifest.json'))
        return summary
    monkeypatch.setattr(runner, 'aggregate_group', aggregate)
    images = Mock()
    monkeypatch.setattr(runner, 'generate_visualizations', images)
    group = runner.run(cfg, group_id='manual', folds=[1])
    assert trained == [1] and read_json(group / 'experiment_manifest.json')['status'] == 'partial'
    result_manifest = root / 'results/experiment_e1/manual/experiment_manifest.json'
    assert read_json(result_manifest) == read_json(group / 'experiment_manifest.json')
    before = {p: p.read_bytes() for p in group.rglob('*') if p.is_file()}
    runner.run(cfg, group_id='manual', resume=True)
    assert trained == [1] and before == {p: p.read_bytes() for p in group.rglob('*') if p.is_file()}
    runner.run(cfg, group_id='manual', resume=True, folds=[1, 2, 3, 4, 5])
    assert trained == [1, 2, 3, 4, 5]
    assert read_json(group / 'experiment_manifest.json')['status'] == 'completed'
    assert read_json(result_manifest) == read_json(group / 'experiment_manifest.json')
    assert images.call_count == 2
    with pytest.raises(ValueError, match='New groups must not exist'):
        runner.run(cfg, group_id='manual')


@pytest.mark.parametrize('error,status', [(KeyboardInterrupt(), 'interrupted'),
                                         (RuntimeError('synthetic failure'), 'failed')])
@pytest.mark.parametrize('stage', ['train', 'visualization'])
def test_unsuccessful_lifecycle_syncs_manifests(sandbox_runner, monkeypatch, error, status, stage):
    root, cfg = sandbox_runner
    monkeypatch.setattr(runner, 'prepare_dataset', lambda *args: [])
    group = root / 'runs/experiment_e1/lifecycle'
    result = root / 'results/experiment_e1/lifecycle'
    # A prior partial result must not remain completed/running on failure.
    result.mkdir(parents=True)
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(runner, 'train_fold', fail if stage == 'train' else Mock(return_value={}))
    monkeypatch.setattr(runner, 'aggregate_group', Mock(return_value={'status': 'completed', 'completed_folds': 5}))
    monkeypatch.setattr(runner, 'generate_visualizations', fail)
    with pytest.raises(type(error)):
        runner.run(cfg, group_id='lifecycle')
    saved = read_json(group / 'experiment_manifest.json')
    assert saved['status'] == status
    assert read_json(result / 'experiment_manifest.json') == saved
    assert not (group / '.lock').exists()


def test_legacy_group_is_read_only(sandbox_runner):
    root, cfg = sandbox_runner
    group = root / 'runs/experiment_e1/history'
    write_json(group / 'experiment_manifest.json', {'status': 'completed'})
    original = (group / 'experiment_manifest.json').read_bytes()
    with pytest.raises(ValueError, match='read-only'):
        runner.run(cfg, group_id='history', resume=True)
    assert original == (group / 'experiment_manifest.json').read_bytes()
    assert not (group / 'logs').exists()


def test_invalid_fold_selection(sandbox_runner):
    _, cfg = sandbox_runner
    for selected in ([6], [1, 1], []):
        with pytest.raises(ValueError, match='distinct IDs'):
            runner.run(cfg, folds=selected, dry_run=True)


def test_effective_scientific_configuration_unchanged(scientific_config_manifest):
    reference = read_json(scientific_config_manifest)['effective_config']
    assert same_training_config(reference, load_config('experiments/e1_cnn/config.yaml'))


@pytest.mark.parametrize('field,replacement', [('learning_rate', .002), ('seeds', [43])])
def test_resume_rejects_changed_training_config_without_writing(
        sandbox_runner, scientific_config_manifest, monkeypatch, field, replacement):
    root, cfg = sandbox_runner
    group = root / 'runs/experiment_e1/unit_resume_guard'
    write_json(group / 'experiment_manifest.json', {
        'runner_version': 'tensorboard-manual-v1', 'status': 'interrupted',
        'requested_folds': [1],
        'effective_config': read_json(scientific_config_manifest)['effective_config'],
        'manifest_checksum': digest(read_json(root / cfg['split']['manifest'])),
    })
    before = {p: p.read_bytes() for p in group.rglob('*') if p.is_file()}
    changed = deepcopy(cfg)
    changed['training'][field] = replacement
    prepare, train = Mock(), Mock()
    monkeypatch.setattr(runner, 'prepare_dataset', prepare)
    monkeypatch.setattr(runner, 'train_fold', train)
    with pytest.raises(ValueError, match='Resume effective config / split checksum mismatch'):
        runner.run(changed, group_id='unit_resume_guard', resume=True)
    prepare.assert_not_called()
    train.assert_not_called()
    assert before == {p: p.read_bytes() for p in group.rglob('*') if p.is_file()}

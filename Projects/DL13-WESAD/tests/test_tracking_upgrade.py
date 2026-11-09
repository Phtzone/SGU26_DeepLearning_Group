"""Console/local schema regression plus archived historical W&B import compatibility."""
import csv
import io
import pytest
from src.utils.config import ROOT
from src.utils.console import ConsoleLogger
from src.utils.logging import RunLogger, LEGACY_EPOCH_FIELDS, append_csv
from src.utils.history_import import import_history
from src.utils.tracking import file_hash, read_json
from src.evaluation.history import read_group


def test_console_header_summary_and_na(synthetic_history):
    _, _, folder = synthetic_history
    meta = read_json(folder / 'metadata.json')
    stream = io.StringIO()
    console = ConsoleLogger({'progress': 'never'}, stream)
    console.header(meta, 7)
    console.start_epoch(1, 100, 1)
    row = read_group('unit_group', synthetic_history[0])['folds'][0].epochs[0]
    console.epoch_summary(row, {'elapsed': 10, 'bad_epochs': 1})
    text = stream.getvalue()
    assert 'Fold' in text and '02/07' in text and 'Parameters' in text
    assert 'TRAIN' in text and 'VALIDATION' in text and 'Balanced Accuracy' in text
    assert 'N/A' in text and 'Best Epoch' in text and '\x1b' not in text
    assert console.bar is None
    console.epoch_summary(row, {})
    assert 'Total Elapsed       : N/A' in stream.getvalue()



def test_legacy_resume_sidecar_and_idempotence(tmp_path):
    path = tmp_path / 'epochs.csv'
    row = {key: 1 for key in LEGACY_EPOCH_FIELDS}
    row.update(attempt_id='old', epoch=1)
    append_csv(path, row, LEGACY_EPOCH_FIELDS)
    prefix = path.read_bytes()
    logger = RunLogger(tmp_path)
    logger.epoch(row)
    new = {**row, 'epoch': 2, 'attempt_id': 'new', 'train_accuracy': .8, 'val_accuracy': .6}
    logger.epoch(new)
    logger.epoch(new)
    assert path.read_bytes().startswith(prefix)
    with path.open() as stream:
        assert len(list(csv.DictReader(stream))) == 2
    with (tmp_path / 'epochs.v2.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 and rows[0]['train_accuracy'] == ''
    assert rows[1]['val_accuracy'] == '0.6'



def test_history_dry_run_and_duplicate_prevention(synthetic_history, fake_wandb):
    root, group, folder = synthetic_history
    original = {str(p): file_hash(p) for p in group.rglob('*') if p.is_file()}
    dry = import_history('unit_group', dry_run=True, root=root, sdk=fake_wandb)
    assert dry['folds'][0]['missing_metrics'] == ['train_accuracy', 'val_accuracy']
    fake_wandb.init.assert_not_called()
    assert not (root / 'results').exists()
    result = import_history('unit_group', root=root, sdk=fake_wandb)
    assert result['imported'] == 1
    calls = [call.args[0] for call in fake_wandb.runs[0][1].log.call_args_list]
    assert not any('train/accuracy' in row or 'val/accuracy' in row for row in calls)
    assert fake_wandb.runs[0][0]['tags'] == ['historical_import']
    assert fake_wandb.runs[0][0]['settings']['x_disable_stats'] is True
    assert import_history('unit_group', root=root, sdk=fake_wandb)['skipped'] == 1
    assert fake_wandb.init.call_count == 1
    assert original == {str(p): file_hash(p) for p in group.rglob('*') if p.is_file()}



def test_historical_regression_unchanged():
    if not (ROOT / 'runs/experiment_e1/e1_seed42_20261107').exists():
        pytest.skip('Historical E1 artifacts are not available in this checkout.')
    from scripts.verify_history import verify
    result = verify()
    assert result['status'] == 'passed' and result['protected_files'] == 184
    assert result['macro_f1'] == 0.45000014134414124
    assert result['balanced_accuracy'] == 0.4547364549300494


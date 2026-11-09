import csv
import importlib.util
import json
import random
from pathlib import Path
from unittest.mock import Mock
import numpy as np
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from src.data.splits import create_manifest
from src.training.trainer import train_fold
from src.training.checkpoint import load_checkpoint
from src.utils.config import ROOT, load_config
from src.utils.tensorboard_tracker import TensorBoardTracker
from src.utils.tracking import read_json


def config():
    result = load_config()
    result['tracking']['strict'] = True
    result['logging']['console']['enabled'] = False
    result['training'].update(max_epochs=2, batch_size=32)
    return result


def events(directory):
    return EventAccumulator(str(directory), size_guidance={'scalars': 0, 'images': 0}).Reload()


def test_actual_trainer_scalars_figures_and_steps(tiny_cache, tmp_path):
    folder = tmp_path / 'tensorboard_synthetic'
    meta = train_fold(folder, tiny_cache, create_manifest()['folds'][0], 42, config(), 'unit', total_folds=5)
    epoch_events = events(folder / 'tensorboard/epochs')
    batch_events = events(folder / 'tensorboard/batches')
    evaluation = events(folder / 'tensorboard/evaluation')
    required = {'train/online_loss', 'train/eval_loss', 'train/accuracy', 'train/macro_f1', 'train/balanced_accuracy',
        'val/loss', 'val/accuracy', 'val/macro_f1', 'val/balanced_accuracy', 'optimization/learning_rate',
        'optimization/epoch_duration_seconds', 'selection/best_val_macro_f1', 'selection/early_stopping_counter'}
    assert required <= set(epoch_events.Tags()['scalars'])
    with (folder / 'logs/epochs.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    for tag, key in [('train/eval_loss', 'train_eval_loss'), ('train/accuracy', 'train_accuracy'), ('val/accuracy', 'val_accuracy'), ('val/macro_f1', 'val_macro_f1')]:
        actual = epoch_events.Scalars(tag)
        assert [e.step for e in actual] == [1, 2]
        assert [e.value for e in actual] == pytest.approx([float(row[key]) for row in rows], abs=1e-6)
    with (folder / 'logs/iterations.csv').open() as stream:
        batches = list(csv.DictReader(stream))
    assert [e.step for e in batch_events.Scalars('batch/loss')] == list(range(1, len(batches) + 1))
    assert [e.value for e in batch_events.Scalars('batch/loss')] == pytest.approx([float(r['train_loss']) for r in batches])
    assert {'batch/loss', 'batch/learning_rate', 'batch/gradient_norm', 'batch/duration_ms'} <= set(batch_events.Tags()['scalars'])
    assert set(evaluation.Tags()['images']) == {'test/confusion_raw', 'test/confusion_normalized', 'test/per_class_f1'}
    assert evaluation.Scalars('test/macro_f1')[0].value == pytest.approx(meta['test_metrics']['macro_f1'])
    assert '\x1b' not in (folder / 'logs/training.log').read_text()


def test_resume_purges_orphan_batches_and_has_unique_visible_steps(tiny_cache, tmp_path):
    cfg = config()
    cfg['training']['max_epochs'] = 3
    folder = tmp_path / 'resume_tb'
    fold = create_manifest()['folds'][0]
    def interrupt(epoch, batch):
        if epoch == 2 and batch == 0:
            raise RuntimeError('Synthetic interruption')
    with pytest.raises(RuntimeError, match='Synthetic interruption'):
        train_fold(folder, tiny_cache, fold, 42, cfg, 'unit', interrupt=interrupt)
    assert [e.step for e in events(folder / 'tensorboard/batches').Scalars('batch/loss')] == [1, 2, 3]
    previous_files = {str(p): p.read_bytes() for p in (folder / 'tensorboard').rglob('events.out.tfevents.*')}
    train_fold(folder, tiny_cache, fold, 42, cfg, 'unit', resume=True)
    assert [e.step for e in events(folder / 'tensorboard/batches').Scalars('batch/loss')] == [1, 2, 3, 4, 5, 6]
    assert [e.step for e in events(folder / 'tensorboard/epochs').Scalars('val/macro_f1')] == [1, 2, 3]
    assert all(Path(p).read_bytes() == content for p, content in previous_files.items())


def test_local_mode_does_not_require_summarywriter(synthetic_history, tmp_path, monkeypatch):
    from src.utils.experiment_tracking import TrainingTracker
    cfg = config()
    cfg['tracking']['backend'] = 'local'
    def forbidden(*args, **kwargs):
        raise AssertionError('SummaryWriter must not be constructed in local mode')
    monkeypatch.setattr('torch.utils.tensorboard.SummaryWriter', forbidden)
    tracker = TrainingTracker(tmp_path / 'logs', cfg)
    tracker.start_run({**read_json(synthetic_history[2] / 'metadata.json'), 'effective_config': cfg}, 'local', 5)
    from src.utils.logging import ITERATION_FIELDS
    tracker.log_batch({key: 1 for key in ITERATION_FIELDS})
    tracker.finish_run('completed')
    assert (tmp_path / 'logs/iterations.csv').is_file()
    assert not list(tmp_path.rglob('events.out.tfevents.*'))


def test_backend_failure_keeps_local_records(synthetic_history, tmp_path, monkeypatch):
    from src.utils.experiment_tracking import TrainingTracker
    cfg = config()
    cfg['tracking']['strict'] = False
    def fail(**kwargs):
        raise OSError('Synthetic unavailable event directory')
    monkeypatch.setattr('torch.utils.tensorboard.SummaryWriter', fail)
    tracker = TrainingTracker(tmp_path / 'logs', cfg)
    tracker.start_run({**read_json(synthetic_history[2] / 'metadata.json'), 'effective_config': cfg}, 'failure', 5)
    from src.utils.logging import ITERATION_FIELDS
    tracker.log_batch({key: 1 for key in ITERATION_FIELDS})
    tracker.finish_run('failed')
    assert (tmp_path / 'logs/iterations.csv').is_file()
    assert 'tracking_warning' in (tmp_path / 'logs/events.jsonl').read_text()


def test_exception_closes_all_writers(tiny_cache, tmp_path, monkeypatch):
    writers = []
    def factory(**kwargs):
        writer = Mock()
        writers.append(writer)
        return writer
    monkeypatch.setattr('torch.utils.tensorboard.SummaryWriter', factory)
    def stop(epoch, batch):
        raise ValueError('Synthetic trainer exception')
    with pytest.raises(ValueError, match='Synthetic trainer exception'):
        train_fold(tmp_path / 'error', tiny_cache, create_manifest()['folds'][0], 42, config(), 'unit', interrupt=stop)
    assert len(writers) == 3
    assert all(writer.close.call_count == 1 for writer in writers)


def test_tracking_preserves_original_model_weights(tiny_cache, tmp_path, monkeypatch):
    snapshot = ROOT / 'runs/experiment_e1/e1_seed42_20261107/source/src/training/trainer.py'
    if not snapshot.exists():
        pytest.skip('Original runtime source snapshot unavailable.')
    spec = importlib.util.spec_from_file_location('original_e1_trainer', snapshot)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    cfg = config()
    cfg['training'].update(max_epochs=2, batch_size=64)
    fold = create_manifest()['folds'][0]
    original.train_fold(tmp_path / 'original', tiny_cache, fold, 42, cfg, 'unit')
    def consume(*args, **kwargs):
        random.random(); np.random.rand(); torch.rand(1)
    def factory(**kwargs):
        consume()
        return Mock(add_scalar=Mock(side_effect=consume), flush=Mock(side_effect=consume))
    monkeypatch.setattr('torch.utils.tensorboard.SummaryWriter', factory)
    train_fold(tmp_path / 'new', tiny_cache, fold, 42, cfg, 'unit')
    before, after = [load_checkpoint(tmp_path / name / 'checkpoints/last.pt') for name in ('original', 'new')]
    assert all(torch.equal(before['model'][key], after['model'][key]) for key in before['model'])
    assert [r['val_macro_f1'] for r in before['history']] == [r['val_macro_f1'] for r in after['history']]


def test_invalid_tracking_options_and_no_active_wandb():
    from src.utils.tensorboard_tracker import tracking_options
    cfg = config()
    assert 'wandb' not in cfg['tracking'] and 'wandb_enabled' not in cfg['logging']
    cfg['tracking']['backend'] = 'wandb'
    with pytest.raises(ValueError, match='retired'):
        tracking_options(cfg)
    cfg = config()
    cfg['tracking']['local']['enabled'] = False
    with pytest.raises(ValueError, match='Local CSV'):
        tracking_options(cfg)


def test_duplicate_or_backward_batch_steps_rejected(tmp_path):
    cfg = config()
    cfg['tracking']['tensorboard']['log_figures'] = False
    tracker = TensorBoardTracker(cfg, tmp_path / 'tb', writer_factory=lambda **kwargs: Mock())
    tracker.start_run({'run_id': 'unit', 'fold_id': 1, 'training_seed': 42, 'manifest_checksum': 'unit', 'effective_config': cfg}, 'unit')
    tracker.log_batch({'global_step': 1, 'train_loss': .4})
    with pytest.warns(RuntimeWarning, match='increase strictly'):
        with pytest.raises(RuntimeError, match='increase strictly'):
            tracker.log_batch({'global_step': 1, 'train_loss': .4})
    tracker.finish_run('failed')

import csv
import pytest
import torch
from src.data.splits import create_manifest
from src.training.trainer import train_fold
from src.training.checkpoint import load_checkpoint
from src.evaluation.aggregation import audit_iterations
from src.utils.config import load_config


def test_interrupted_resume_matches_uninterrupted(tiny_cache, tmp_path):
    config = load_config()
    config['training'].update(max_epochs=3, batch_size=32)
    fold = create_manifest()['folds'][0]
    reference = tmp_path / 'reference'
    interrupted = tmp_path / 'interrupted'
    train_fold(reference, tiny_cache, fold, 42, config, 'test-checksum')

    def stop(epoch, batch):
        if epoch == 2 and batch == 0:
            raise RuntimeError('Simulated mid-epoch interruption')

    with pytest.raises(RuntimeError, match='Simulated'):
        train_fold(interrupted, tiny_cache, fold, 42, config, 'test-checksum', interrupt=stop)
    assert load_checkpoint(interrupted / 'checkpoints/last.pt')['epoch'] == 1
    result = train_fold(interrupted, tiny_cache, fold, 42, config, 'test-checksum', resume=True)
    a, b = [load_checkpoint(p / 'checkpoints/last.pt') for p in [reference, interrupted]]
    for key in a['model']:
        assert torch.equal(a['model'][key], b['model'][key]), key
    assert [h['val_macro_f1'] for h in a['history']] == [h['val_macro_f1'] for h in b['history']]
    assert train_fold(interrupted, tiny_cache, fold, 42, config, 'test-checksum', resume=True) == result
    audit = audit_iterations(interrupted, b, 54, 32)
    assert audit['committed_iterations'] == 6
    assert audit['orphan_or_replayed_iterations'] == 1
    assert len({h['attempt_id'] for h in b['history']}) == 2


def test_test_not_used_for_checkpoint_selection(tiny_cache, tmp_path, monkeypatch):
    import src.training.trainer as trainer
    config = load_config()
    config['training'].update(max_epochs=2, batch_size=64)
    fold = create_manifest()['folds'][0]
    original = trainer.evaluate
    calls = []

    def observe(model, loader, device, weights):
        role = next(role for role in ['train', 'validation', 'test'] if loader.dataset.subjects == fold[role])
        calls.append(role)
        return original(model, loader, device, weights)

    monkeypatch.setattr(trainer, 'evaluate', observe)
    trainer.train_fold(tmp_path / 'fold', tiny_cache, fold, 42, config, 'test-checksum')
    assert calls == ['train', 'validation', 'train', 'validation', 'test']

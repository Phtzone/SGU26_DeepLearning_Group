from pathlib import Path
import numpy as np
import pytest
import torch
from scripts.verify_e1_baseline import safe_load_checkpoint
from src.utils.tracking import file_hash


def test_safe_checkpoint_audit_reads_numpy_rng_without_mutation(tmp_path):
    path = tmp_path / 'local.pt'
    torch.save({'model': {'weight': torch.ones(3, 5)},
                'rng': {'numpy': np.random.RandomState(42).get_state()}}, path)
    before = file_hash(path)
    allowed_before = list(torch.serialization.get_safe_globals())
    state = safe_load_checkpoint(path)
    assert torch.equal(state['model']['weight'], torch.ones(3, 5))
    assert state['rng']['numpy'][1].dtype == np.uint32
    assert file_hash(path) == before
    assert torch.serialization.get_safe_globals() == allowed_before


def test_safe_checkpoint_audit_rejects_unapproved_globals(tmp_path):
    path = tmp_path / 'unsupported.pt'
    torch.save({'unsupported_metadata': Path('local-only')}, path)
    before = file_hash(path)
    with pytest.raises(ValueError, match='Unexpected checkpoint globals'):
        safe_load_checkpoint(path)
    assert file_hash(path) == before

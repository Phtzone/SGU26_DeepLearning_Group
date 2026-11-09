import torch
from src.training.checkpoint import save_checkpoint, load_checkpoint
from src.utils.reproducibility import seed_everything, rng_state, restore_rng


def test_atomic_checkpoint_and_rng(tmp_path):
    seed_everything(42)
    state = rng_state()
    expected = torch.rand(3)
    path = tmp_path / 'last.pt'
    save_checkpoint(path, {'rng': state, 'model': {'weight': expected}})
    restored = load_checkpoint(path)
    restore_rng(restored['rng'])
    assert torch.equal(expected, torch.rand(3))
    assert not list(tmp_path.glob('*.tmp'))

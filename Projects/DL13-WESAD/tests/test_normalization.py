import numpy as np
from src.data.dataset import WindowDataset
from src.data.normalization import ChannelScaler


def test_scaler_training_only(tiny_cache):
    train = WindowDataset(tiny_cache, ['S2', 'S3'])
    scaler = ChannelScaler().fit(train)
    expected = np.concatenate([train.arrays[s] for s in train.subjects]).transpose(1, 0, 2).reshape(5, -1)
    np.testing.assert_allclose(scaler.mean, expected.mean(axis=1), atol=1e-6)
    np.testing.assert_allclose(scaler.std, expected.std(axis=1), atol=1e-6)
    before = scaler.mean.copy()
    x = np.load(tiny_cache / 'S4/windows.npy')
    np.save(tiny_cache / 'S4/windows.npy', x + 10000)
    val = WindowDataset(tiny_cache, ['S4'], scaler)
    assert val.scaler is scaler
    np.testing.assert_array_equal(before, scaler.mean)
    assert scaler.subjects == ['S2', 'S3']

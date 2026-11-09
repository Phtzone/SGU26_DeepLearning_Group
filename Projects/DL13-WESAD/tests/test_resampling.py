import numpy as np
import pytest
from src.data.resampling import resample_signal, align_labels


def test_downsample_antialias():
    t = np.arange(6400) / 64
    signal = np.sin(2 * np.pi * 25 * t)[:, None]
    result = resample_signal(signal, 64, 32, 3200)
    assert result.shape == (3200, 1)
    assert np.std(result[100:-100]) < .02


def test_eda_time_interpolation_and_identity():
    source = np.arange(8, dtype=np.float32)[:, None]
    output = resample_signal(source, 4, 32, 64)
    np.testing.assert_allclose(output[:57, 0], np.arange(57) / 8)
    np.testing.assert_array_equal(resample_signal(source, 32, 32, 8), source)


def test_duration_mismatch_fails():
    with pytest.raises(ValueError):
        resample_signal(np.ones((40, 1)), 4, 32, 64)


def test_label_intervals_preserve_transition_and_invalid():
    original = np.r_[np.ones(701), np.zeros(1), np.ones(698) * 2, np.ones(700) * 3]
    labels = align_labels(original, 32, 96)
    assert labels[31] == 0
    assert labels[32] == -1  # A sub-sample invalid interval cannot disappear.
    assert labels[33] == 1
    np.testing.assert_array_equal(labels[64:], 2)

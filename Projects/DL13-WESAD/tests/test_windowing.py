import numpy as np
import pytest
from src.data.windowing import make_windows


def test_windows_cannot_cross_states_or_invalid():
    labels = np.r_[np.zeros(64), [-1], np.ones(64), np.ones(64) * 2].astype(np.int64)
    signal = np.arange(len(labels) * 5).reshape(-1, 5)
    x, rows, audit = make_windows(signal, labels, 'S2', rate=32, window_seconds=1, stride_seconds=.5)
    assert x.shape[1:] == (5, 32)
    assert x.dtype == np.float32
    for row in rows:
        start, end = int(row['start_time'] * 32), int(row['end_time'] * 32)
        assert (labels[start:end] == row['label']).all()
    assert audit['global_grid_rejections']['invalid_region'] > 0
    assert len({r['window_id'] for r in rows}) == len(rows)


def test_short_or_invalid_regions_not_windows():
    with pytest.raises(ValueError):
        make_windows(np.ones((40, 5)), np.r_[np.zeros(20), np.ones(20)], 'S2', rate=32, window_seconds=1)

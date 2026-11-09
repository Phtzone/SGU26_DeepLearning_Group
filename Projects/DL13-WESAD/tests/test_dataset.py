import numpy as np
import pytest
from src.data.dataset import WindowDataset
from src.data.inspection import load_subject


def test_lazy_loader_identity(tiny_cache):
    dataset = WindowDataset(tiny_cache, ['S2', 'S3'])
    assert len(dataset) == 12
    assert isinstance(dataset.arrays['S2'], np.memmap)
    x, y, i = dataset[7]
    assert x.shape == (5, 32) and y == 1 and i == 7
    assert dataset.rows[i]['subject_id'] == 'S3'


def test_missing_raw_file_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_subject(tmp_path, 'S2')

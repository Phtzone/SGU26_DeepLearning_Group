"""Targeted tests for per-window median centering before Train-only Z-score."""
import csv
import numpy as np
import pytest
from torch.utils.data import DataLoader
import yaml

from src.data.common_rate import CHANNELS, INDEX_METADATA
from src.data.dataset import WindowDataset
from src.data.normalization import ChannelScaler
from src.utils.config import load_config
from src.utils.tracking import read_json, write_json


CENTER = {'enabled': True, 'method': 'median', 'channels': ['chest_Temp']}
BASE = 'experiments/common_rate64_cnn1d_grouped5fold_v1/config.yaml'
CANDIDATE = 'experiments/common_rate64_cnn1d_chest_temp_median_centering_grouped5fold_v1/config.yaml'


@pytest.fixture
def finite_timelines(tmp_path):
    cache = tmp_path / 'index'
    source = tmp_path / 'timelines'
    cache.mkdir()
    source.mkdir()
    # Deliberately present dataset channels in a different order than the stored columns.
    ordered = list(reversed(CHANNELS))
    write_json(cache / INDEX_METADATA, {'audit': {'source': str(source),
               'notebook_config': {'channels': list(CHANNELS)}, 'policy': {'channels': ordered}}})
    index = CHANNELS.index('chest_Temp')
    for j, subject in enumerate(('S2', 'S3')):
        folder = cache / subject
        folder.mkdir()
        signal = np.arange(16 * len(CHANNELS), dtype=np.float32).reshape(16, len(CHANNELS))
        signal[:, index] = np.repeat([32, 34], 8) + np.tile((np.arange(8, dtype=np.float32) - 3) / 32, 2) + 3 * j
        # This even window's center falls halfway between two representable float32 values.
        signal[3, index] = 32 + 3 * j
        signal[4, index] = np.nextafter(signal[3, index], np.float32(np.inf))
        np.save(source / f'{subject}_X.npy', signal)
        with (folder / 'windows.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(('subject_id', 'window_id', 'start_index', 'end_index', 'label'))
            writer.writerow((subject, f'{subject}_0', 0, 8, 0))
            writer.writerow((subject, f'{subject}_1', 8, 16, 1))
        write_json(folder / 'metadata.json', {'cache_key': f'fixture-{subject}'})
    return cache, source, ordered


def test_dataset_and_train_fit_apply_same_named_window_transform(finite_timelines, tmp_path):
    cache, source, ordered = finite_timelines
    ch = ordered.index('chest_Temp')
    assert ch != CHANNELS.index('chest_Temp')
    train = WindowDataset(cache, ['S2'], window_centering=CENTER)
    holdout = WindowDataset(cache, ['S3'], window_centering=CENTER)
    original = {s: (source / f'{s}_X.npy').read_bytes() for s in ('S2', 'S3')}
    scaler = ChannelScaler().fit(train)
    scaler.save(tmp_path / 'scaler.json')
    assert read_json(tmp_path / 'scaler.json')['window_centering'] == CENTER
    assert scaler.subjects == ['S2'] and scaler.count == 16
    assert scaler.mean.shape == scaler.std.shape == (14,)
    train.scaler = holdout.scaler = scaler

    for dataset, subject in ((train, 'S2'), (holdout, 'S3')):
        reference = np.load(source / f'{subject}_X.npy')[:, [CHANNELS.index(name) for name in ordered]].T
        raw = np.stack([reference[:, :8], reference[:, 8:]])
        expected = raw.copy()
        expected[:, ch] = raw[:, ch] - np.median(raw[:, ch].astype(np.float64), axis=-1, keepdims=True)
        np.testing.assert_array_equal(np.median(expected[:, ch], axis=-1), [0, 0])
        prepared = np.stack([dataset.prepared_window(subject, i) for i in range(2)])
        np.testing.assert_array_equal(prepared, expected)
        if dataset is train:
            np.testing.assert_array_equal(scaler.mean, expected.mean(axis=(0, 2), dtype=np.float64).astype(np.float32))
            np.testing.assert_array_equal(scaler.std, expected.std(axis=(0, 2), dtype=np.float64).astype(np.float32))
        x, y, offsets = next(iter(DataLoader(dataset, batch_size=2)))
        assert x.numpy().dtype == np.float32 and x.shape == (2, 14, 8)
        assert y.tolist() == offsets.tolist() == [0, 1]
        assert [row['window_id'] for row in dataset.rows] == [f'{subject}_0', f'{subject}_1']
        np.testing.assert_array_equal(x.numpy(), (expected - scaler.mean[None, :, None]) / scaler.std[None, :, None])
        assert (source / f'{subject}_X.npy').read_bytes() == original[subject]


def test_disabled_centering_preserves_existing_dataset_and_scaler(finite_timelines, tmp_path):
    cache, _, _ = finite_timelines
    raw = WindowDataset(cache, ['S2'])
    disabled = WindowDataset(cache, ['S2'], window_centering={'enabled': False})
    for i in range(len(raw)):
        np.testing.assert_array_equal(raw[i][0].numpy(), disabled[i][0].numpy())
    baseline_scaler = ChannelScaler().fit(raw)
    disabled_scaler = ChannelScaler().fit(disabled)
    np.testing.assert_array_equal(baseline_scaler.mean, disabled_scaler.mean)
    np.testing.assert_array_equal(baseline_scaler.std, disabled_scaler.std)
    disabled_scaler.save(tmp_path / 'disabled.json')
    assert 'window_centering' not in read_json(tmp_path / 'disabled.json')


def test_variant_preserves_baseline_protocol():
    baseline, candidate = load_config(BASE), load_config(CANDIDATE)
    assert 'preprocessing' not in baseline
    assert candidate.pop('preprocessing')['window_centering'] == CENTER
    assert candidate['experiment']['id'] != baseline['experiment']['id']
    candidate['experiment']['id'] = baseline['experiment']['id']
    assert candidate == baseline


@pytest.mark.parametrize('overrides', [{'method': 'mean'}, {'channels': ['missing']},
                                       {'channels': ['chest_Temp', 'chest_Temp']}, {'enabled': 'true'}])
def test_invalid_centering_policy_fails_config_validation(tmp_path, overrides):
    candidate = load_config(CANDIDATE)
    candidate['preprocessing']['window_centering'].update(overrides)
    path = tmp_path / 'invalid.yaml'
    path.write_text(yaml.safe_dump(candidate), encoding='utf-8')
    with pytest.raises(ValueError, match=r'Window( |-)?centering'):
        load_config(path)

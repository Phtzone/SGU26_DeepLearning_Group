"""Numerically stable channel statistics fitted exclusively on training windows."""
import numpy as np
from src.utils.tracking import write_json


def window_centering_channels(options, available_channels):
    """Validate an optional window-centering policy and resolve enabled channel names."""
    if options is None:
        return ()
    if not isinstance(options, dict) or set(options) - {'enabled', 'method', 'channels'}:
        raise ValueError('Invalid window_centering options.')
    if type(options.get('enabled')) is not bool or options.get('method', 'median') != 'median':
        raise ValueError('Window centering requires a boolean enabled flag and the median method.')
    channels = options.get('channels', [])
    if (not isinstance(channels, list) or not all(isinstance(channel, str) for channel in channels)
            or len(set(channels)) != len(channels)):
        raise ValueError('Window-centering channels must be a list of distinct names.')
    if not options['enabled']:
        return ()
    if not channels or not set(channels) <= set(available_channels):
        raise ValueError('Window-centering channels must be selected from the named dataset channels.')
    return tuple(channels)


class ChannelScaler:
    def fit(self, dataset):
        if dataset.scaler is not None:
            raise ValueError('Fit requires unnormalized training windows.')
        count, mean, m2 = 0, None, None
        for subject in dataset.subjects:
            for windows in dataset.prepared_blocks(subject):
                block = np.asarray(windows, dtype=np.float64).transpose(1, 0, 2).reshape(windows.shape[1], -1)
                n = block.shape[1]
                bm = block.mean(axis=1)
                b2 = ((block - bm[:, None]) ** 2).sum(axis=1)
                if mean is None:
                    mean, m2, count = bm, b2, n
                else:
                    delta = bm - mean
                    m2 += b2 + delta ** 2 * count * n / (count + n)
                    mean += delta * n / (count + n)
                    count += n
        if count == 0:
            raise ValueError('Fit requires at least one training window.')
        self.mean = mean.astype(np.float32)
        self.std = np.sqrt(m2 / count).astype(np.float32)
        self.std[self.std < 1e-8] = 1
        self.subjects, self.count = dataset.subjects, count
        self.window_centering = dataset.window_centering
        return self

    def transform(self, x):
        return ((x - self.mean[:, None]) / self.std[:, None]).astype(np.float32)

    def save(self, path):
        metadata = {'mean': self.mean.tolist(), 'std': self.std.tolist(),
                    'train_subjects': self.subjects, 'sample_count_per_channel': self.count,
                    'weighting': 'all timesteps in training windows, including overlap', 'ddof': 0}
        if self.window_centering:
            metadata['window_centering'] = self.window_centering
        write_json(path, metadata)

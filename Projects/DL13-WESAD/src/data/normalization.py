"""Numerically stable channel statistics fitted exclusively on training windows."""
import numpy as np
from src.utils.tracking import write_json


class ChannelScaler:
    def fit(self, dataset):
        if dataset.scaler is not None:
            raise ValueError('Fit requires unnormalized training windows.')
        count, mean, m2 = 0, None, None
        for subject in dataset.subjects:
            array = dataset.arrays[subject]
            for start in range(0, len(array), 64):
                block = np.asarray(array[start:start + 64], dtype=np.float64).transpose(1, 0, 2).reshape(array.shape[1], -1)
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
        self.mean = mean.astype(np.float32)
        self.std = np.sqrt(m2 / count).astype(np.float32)
        self.std[self.std < 1e-8] = 1
        self.subjects, self.count = dataset.subjects, count
        return self

    def transform(self, x):
        return ((x - self.mean[:, None]) / self.std[:, None]).astype(np.float32)

    def save(self, path):
        write_json(path, {'mean': self.mean.tolist(), 'std': self.std.tolist(),
                          'train_subjects': self.subjects, 'sample_count_per_channel': self.count,
                          'weighting': 'all timesteps in training windows, including overlap', 'ddof': 0})

"""Lazy memory-mapped windows with stable per-window identifiers."""
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from src.utils.tracking import read_json


class WindowDataset(Dataset):
    def __init__(self, cache, subjects, scaler=None):
        self.subjects = list(subjects)
        self.scaler = scaler
        self.arrays, self.rows, self.indices, self.cache_keys = {}, [], [], {}
        for subject in subjects:
            folder = cache / subject
            self.arrays[subject] = np.load(folder / 'windows.npy', mmap_mode='r')
            self.cache_keys[subject] = read_json(folder / 'metadata.json')['cache_key']
            with (folder / 'windows.csv').open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != len(self.arrays[subject]):
                raise ValueError('Cache rows and windows disagree.')
            for i, row in enumerate(rows):
                row['label'] = int(row['label'])
                self.rows.append(row)
                self.indices.append((subject, i))
        self.labels = np.array([r['label'] for r in self.rows], dtype=np.int64)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        subject, offset = self.indices[index]
        x = np.array(self.arrays[subject][offset], copy=True)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        return torch.from_numpy(x), int(self.labels[index]), index

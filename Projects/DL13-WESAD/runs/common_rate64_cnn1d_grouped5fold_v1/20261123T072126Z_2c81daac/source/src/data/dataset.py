"""Lazy memory-mapped windows with stable per-window identifiers."""
import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from src.data.common_rate import INDEX_METADATA
from src.utils.tracking import read_json


class WindowDataset(Dataset):
    def __init__(self, cache, subjects, scaler=None):
        self.subjects = list(subjects)
        self.scaler = scaler
        self.arrays, self.rows, self.indices, self.cache_keys = {}, [], [], {}
        self.subject_rows = {}
        index_path = cache / INDEX_METADATA
        self.timeline = index_path.exists()
        if self.timeline:
            audit = read_json(index_path)['audit']
            self.source = Path(audit['source'])
            all_channels = audit['notebook_config']['channels']
            channel_indices = [all_channels.index(channel) for channel in audit['policy']['channels']]
            self.channel_indices = slice(None) if channel_indices == list(range(len(all_channels))) else channel_indices
            self.input_channels = len(channel_indices)
        for subject in self.subjects:
            folder = cache / subject
            path = self.source / f'{subject}_X.npy' if self.timeline else folder / 'windows.npy'
            self.arrays[subject] = np.load(path, mmap_mode='r', allow_pickle=False)
            self.cache_keys[subject] = read_json(folder / 'metadata.json')['cache_key']
            with (folder / 'windows.csv').open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            if not self.timeline and len(rows) != len(self.arrays[subject]):
                raise ValueError('Cache rows and windows disagree.')
            self.subject_rows[subject] = rows
            for i, row in enumerate(rows):
                row['label'] = int(row['label'])
                if self.timeline:
                    row['start_index'] = int(row['start_index'])
                    row['end_index'] = int(row['end_index'])
                self.rows.append(row)
                self.indices.append((subject, i))
        if not self.timeline:
            self.input_channels = self.arrays[self.subjects[0]].shape[1]
        self.labels = np.array([r['label'] for r in self.rows], dtype=np.int64)

    def __len__(self):
        return len(self.rows)

    def raw_window(self, subject, offset):
        if self.timeline:
            row = self.subject_rows[subject][offset]
            return self.arrays[subject][row['start_index']:row['end_index'], self.channel_indices].T
        return self.arrays[subject][offset]

    def raw_blocks(self, subject, block_size=64):
        for start in range(0, len(self.subject_rows[subject]), block_size):
            stop = min(start + block_size, len(self.subject_rows[subject]))
            if self.timeline:
                yield np.stack([self.raw_window(subject, offset) for offset in range(start, stop)])
            else:
                yield self.arrays[subject][start:stop]

    def __getitem__(self, index):
        subject, offset = self.indices[index]
        x = np.array(self.raw_window(subject, offset), copy=True, order='C')
        if self.scaler is not None:
            x = self.scaler.transform(x)
        return torch.from_numpy(x), int(self.labels[index]), index

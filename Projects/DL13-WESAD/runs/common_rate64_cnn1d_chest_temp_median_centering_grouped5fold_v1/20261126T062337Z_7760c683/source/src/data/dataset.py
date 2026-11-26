"""Lazy memory-mapped windows with stable per-window identifiers."""
import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from src.data.common_rate import INDEX_METADATA
from src.data.normalization import window_centering_channels
from src.utils.tracking import read_json


class WindowDataset(Dataset):
    def __init__(self, cache, subjects, scaler=None, *, window_centering=None):
        self.subjects = list(subjects)
        self.scaler = scaler
        self.arrays, self.rows, self.indices, self.cache_keys = {}, [], [], {}
        self.subject_rows = {}
        index_path = cache / INDEX_METADATA
        self.timeline = index_path.exists()
        selected_channels = ()
        if self.timeline:
            audit = read_json(index_path)['audit']
            self.source = Path(audit['source'])
            all_channels = audit['notebook_config']['channels']
            selected_channels = audit['policy']['channels']
            channel_indices = [all_channels.index(channel) for channel in selected_channels]
            self.channel_indices = slice(None) if channel_indices == list(range(len(all_channels))) else channel_indices
            self.input_channels = len(channel_indices)
        centered_channels = window_centering_channels(window_centering, selected_channels)
        self.center_indices = [selected_channels.index(channel) for channel in centered_channels]
        self.window_centering = ({'enabled': True, 'method': 'median', 'channels': list(centered_channels)}
                                 if centered_channels else None)
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

    def _prepared_copy(self, windows):
        prepared = np.array(windows, copy=True, order='C')
        if self.center_indices:
            # Preserve the exact midpoint of the two central float32 values.
            selected = prepared[..., self.center_indices, :].astype(np.float64)
            prepared[..., self.center_indices, :] = selected - np.median(selected, axis=-1, keepdims=True)
        return prepared

    def prepared_window(self, subject, offset):
        """Copy one (channel, time) window and apply the optional pre-scaler transform."""
        return self._prepared_copy(self.raw_window(subject, offset))

    def prepared_blocks(self, subject, block_size=64):
        """Apply the same transform while streaming windows for the scaler fit."""
        for windows in self.raw_blocks(subject, block_size):
            yield self._prepared_copy(windows) if self.center_indices else windows

    def __getitem__(self, index):
        subject, offset = self.indices[index]
        x = self.prepared_window(subject, offset)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        return torch.from_numpy(x), int(self.labels[index]), index

"""Read-only validation of the main 14-channel configuration and local caches.

This does not prepare data, fit a scaler, initialize a model or train.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from src.data.splits import load_manifest
from src.utils.config import ROOT
from src.utils.tracking import read_json, file_hash, digest


def check_config(path, root=ROOT):
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    d = config['data']
    if (d['input_format'] != 'common_rate_windows_v1' or d['input_channels'] != 14
            or len(d['channels']) != 14 or len(set(d['channels'])) != 14
            or (d['sampling_rate'], d['window_seconds'], d['stride_seconds']) != (64, 30, 15)
            or d['samples_per_window'] != d['sampling_rate'] * d['window_seconds']
            or d['layout'] != 'NCT' or d['dtype'] != 'float32'
            or d['normalized'] or not d['windowed']
            or d['labels'] != [1, 2, 3] or d['label_map'] != {1: 0, 2: 1, 3: 2}
            or d['class_names'] != ['baseline', 'stress', 'amusement'] or d['num_classes'] != 3):
        raise ValueError('Invalid 14-channel data configuration.')
    if config['normalization']['fit_on'] != 'train_only_per_fold':
        raise ValueError('Normalization must be fitted on Train only in each fold.')
    split = load_manifest(root / config['split']['manifest'])
    if config['split']['protocol'] != split['protocol'] or config['split']['split_seed'] != split['split_seed']:
        raise ValueError('Split config does not match the frozen manifest.')
    continuous = root / d['continuous_cache']
    cache = root / d['window_cache']
    source = read_json(continuous / 'config.json')
    window_config = read_json(cache / 'config.json')
    if (source['channels'] != d['channels'] or source['target_fs'] != d['sampling_rate']
            or source['normalized'] or source['windowed']):
        raise ValueError('Continuous cache channel order/rate/state mismatch.')
    for key in ('channels', 'sampling_rate', 'window_seconds', 'stride_seconds', 'normalized', 'windowed'):
        if window_config[key] != d[key]:
            raise ValueError(f'Window cache mismatch: {key}')
    if source['subjects'] != split['original_subject_order'] or window_config['subjects'] != split['original_subject_order']:
        raise ValueError('Cache subjects differ from split subjects.')
    totals = np.zeros(3, dtype=np.int64)
    subjects = []
    for subject in split['original_subject_order']:
        folder = cache / subject
        meta = read_json(folder / 'metadata.json')
        for key in ('channels', 'sampling_rate', 'window_seconds', 'stride_seconds', 'normalized', 'windowed'):
            if meta['data_config'][key] != d[key]:
                raise ValueError(f'{subject}: metadata mismatch: {key}')
        for name, expected in meta['files'].items():
            if file_hash(folder / name) != expected:
                raise ValueError(f'{subject}: cached file hash mismatch: {name}')
        array = np.load(folder / 'windows.npy', mmap_mode='r', allow_pickle=False)
        with (folder / 'windows.csv').open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        if array.shape != (len(rows), 14, 1920) or array.dtype != np.float32 or list(array.shape) != meta['shape']:
            raise ValueError(f'{subject}: invalid window layout or metadata shape.')
        for row in rows:
            if row['subject_id'] != subject or int(row['original_label']) not in d['label_map'] or d['label_map'][int(row['original_label'])] != int(row['label']):
                raise ValueError(f'{subject}: invalid subject/label mapping.')
        counts = np.bincount([int(r['label']) for r in rows], minlength=3)
        if counts.tolist() != [int(meta['label_distribution'][str(i)]) for i in range(3)]:
            raise ValueError(f'{subject}: class counts disagree with metadata.')
        totals += counts
        subjects.append({'subject': subject, 'windows': len(rows), 'shape': list(array.shape)})
        del array
    return {'config': path.relative_to(root).as_posix(), 'config_sha256': file_hash(path),
            'passed': True, 'training_executed': False, 'normalization_fitted': False,
            'runner_ready': False, 'protocol': split['protocol'], 'manifest_checksum': digest(split),
            'folds': len(split['folds']), 'subjects': subjects, 'total_windows': int(totals.sum()),
            'class_counts': dict(zip(d['class_names'], totals.tolist())),
            'window_cache': d['window_cache'], 'window_shape': [14, 1920]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/common_rate_14ch.yaml'))
    args = parser.parse_args()
    print(json.dumps(check_config(args.config), indent=2))


if __name__ == '__main__':
    main()

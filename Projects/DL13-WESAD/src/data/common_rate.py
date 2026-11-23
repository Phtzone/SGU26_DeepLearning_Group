"""Validate notebook 02 exports and build a CSV index into read-only timelines."""
from collections import Counter, defaultdict
import csv
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.utils.tracking import digest, file_hash, read_json, write_json


CHANNELS = (
    'chest_ACC_X', 'chest_ACC_Y', 'chest_ACC_Z', 'chest_ECG', 'chest_EMG',
    'chest_EDA', 'chest_Temp', 'chest_Resp', 'wrist_ACC_X', 'wrist_ACC_Y',
    'wrist_ACC_Z', 'wrist_BVP', 'wrist_EDA', 'wrist_TEMP',
)
NOTEBOOK_METADATA = ('config.json', 'subjects_summary.csv', 'label_segments.csv')
INDEX_METADATA = 'common_rate_index.json'
POLICY = 'region-anchored-complete-interval-all-valid-v1'


def read_rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def build_index(source, config, subjects):
    """Return provenance and window rows without writing to source or disk."""
    source = Path(source).resolve()
    metadata = read_json(source / 'config.json')
    rate, label_rate = metadata['target_fs'], metadata['label_source_fs']
    if (metadata['strategy'], rate, label_rate, metadata['normalized'], metadata['windowed'],
            metadata['quality_guard_seconds'], metadata['resampling_method']) != (
            'common_rate', config['sampling_rate'], 700, False, False, 2.0, 'scipy.signal.resample_poly'):
        raise ValueError('Unexpected notebook 02 source metadata.')
    if tuple(metadata['channels']) != CHANNELS or sorted(metadata['subjects']) != sorted(subjects):
        raise ValueError('Notebook channel layout or subject roster differs from the protocol.')
    for suffix in ('_X.npy', '_quality_mask.npy'):
        if {p.name.removesuffix(suffix) for p in source.glob('*' + suffix)} != set(subjects):
            raise ValueError(f'Unexpected subject roster for {suffix}.')

    segments = defaultdict(list)
    for row in read_rows(source / 'label_segments.csv'):
        segments[row['Subject']].append(row)
    summary = read_rows(source / 'subjects_summary.csv')
    summaries = {row['Subject']: row for row in summary}
    if set(segments) != set(subjects) or set(summaries) != set(subjects) or len(summary) != len(subjects):
        raise ValueError('Subject rosters differ between notebook label/summary files and the split.')

    rule = {'version': POLICY, 'sampling_rate': rate, 'label_source_fs': label_rate,
            'window_seconds': config['window_seconds'], 'stride_seconds': config['stride_seconds'],
            'channels': config['channels'], 'label_mapping': {str(label): i for i, label in enumerate(config['labels'])},
            'quality_mask': 'True=valid; require every sample in the window to be True',
            'interval': 'half-open; ceil(start_700*64/700), floor(end_700*64/700)'}
    files = {name: file_hash(source / name) for name in NOTEBOOK_METADATA}
    size = rate * config['window_seconds']
    stride = rate * config['stride_seconds']
    rows_by_subject, reports = {}, []
    for subject in subjects:
        signal_path, quality_path = source / f'{subject}_X.npy', source / f'{subject}_quality_mask.npy'
        signal = np.load(signal_path, mmap_mode='r', allow_pickle=False)
        mask = np.load(quality_path, mmap_mode='r', allow_pickle=False)
        if signal.ndim != 2 or signal.shape[1] != len(CHANNELS) or signal.dtype != np.float32:
            raise ValueError(f'{subject}: expected a float32 (N, 14) timeline.')
        if mask.shape != (len(signal),) or mask.dtype != np.bool_ or not np.isfinite(signal).all():
            raise ValueError(f'{subject}: invalid boolean mask, timeline length or non-finite samples.')
        report = summaries[subject]
        invalid = int(len(mask) - np.count_nonzero(mask))
        if (len(signal), signal.shape[1], rate, invalid) != tuple(
                int(report[k]) for k in ('Output_samples', 'Channels', 'Target_Hz', 'Invalid_samples_64')):
            raise ValueError(f'{subject}: notebook summary disagrees with arrays.')

        rows = []
        previous_end, previous_label, rejected = 0, None, 0
        for segment in segments[subject]:
            label = int(segment['Label'])
            start, end = (int(segment[k]) for k in ('Start_sample_700', 'End_sample_700'))
            if start != previous_end or end <= start or label == previous_label or not all(
                    math.isclose(float(segment[k]), sample / label_rate, rel_tol=0, abs_tol=1e-8)
                    for k, sample in (('Start_sec', start), ('End_sec', end))):
                raise ValueError(f'{subject}: label segments must form the contiguous notebook timeline.')
            previous_end, previous_label = end, label
            if label not in config['labels']:
                continue
            lo = (start * rate + label_rate - 1) // label_rate
            hi = min(len(signal), end * rate // label_rate)
            # Start the grid at each independently aligned, constant-label region.
            for first in range(lo, hi - size + 1, stride):
                last = first + size
                if not mask[first:last].all():
                    rejected += 1
                    continue
                rows.append({'subject_id': subject, 'window_id': f'{subject}_{first}_{last}',
                             'start_index': first, 'end_index': last, 'start_time': first / rate,
                             'end_time': last / rate, 'original_label': label, 'label': config['labels'].index(label),
                             'quality_status': 'valid'})
        duration = previous_end / label_rate
        if not math.isclose(duration, float(report['Duration_sec']), rel_tol=0, abs_tol=1e-8) or round(duration * rate) != len(signal):
            raise ValueError(f'{subject}: label endpoint disagrees with notebook duration or array length.')
        if not rows or len({row['window_id'] for row in rows}) != len(rows):
            raise ValueError(f'{subject}: empty or duplicate window IDs.')
        subject_files = {p.name: file_hash(p) for p in (signal_path, quality_path)}
        cache_key = digest({'policy': rule, 'files': subject_files, 'rows': rows})
        counts = Counter(row['label'] for row in rows)
        reports.append({'subject_id': subject, 'cache_key': cache_key, 'shape': [len(rows), len(config['channels']), size],
                        'source_shape': list(signal.shape), 'dtype': str(signal.dtype), 'file_sha256': subject_files,
                        'invalid_samples_64': invalid, 'rejected_quality_windows': rejected,
                        'label_distribution': {str(i): counts[i] for i in range(3)}})
        rows_by_subject[subject] = rows
        files.update(subject_files)
    audit = {'source': str(source), 'notebook_config': metadata, 'policy': rule, 'source_file_sha256': files,
             'dataset_fingerprint': digest(files), 'subjects': reports,
             'windows': sum(len(rows) for rows in rows_by_subject.values()),
             'rejected_quality_windows': sum(record['rejected_quality_windows'] for record in reports)}
    return audit, rows_by_subject


def write_rows(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_index(destination, audit, rows_by_subject):
    """Publish the index once, then verify that a resume sees the same windows."""
    if destination.exists():
        saved = read_json(destination / INDEX_METADATA)
        if saved['audit'] != audit or any(file_hash(destination / rel) != sha for rel, sha in saved['index_file_sha256'].items()):
            raise ValueError('Common-rate source, policy or window index changed since the original run.')
        return
    with TemporaryDirectory(dir=destination.parent, prefix='.index-') as staging:
        root = Path(staging) / 'index'
        root.mkdir()
        write_rows(root / 'windows.csv', [r for rows in rows_by_subject.values() for r in rows])
        for record in audit['subjects']:
            subject = record['subject_id']
            folder = root / subject
            folder.mkdir()
            write_rows(folder / 'windows.csv', rows_by_subject[subject])
            write_json(folder / 'metadata.json', record)
        hashes = {p.relative_to(root).as_posix(): file_hash(p) for p in root.rglob('*') if p.is_file()}
        write_json(root / INDEX_METADATA, {'audit': audit, 'index_file_sha256': hashes})
        root.rename(destination)

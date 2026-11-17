"""Subject-at-a-time cache; raw data and fold statistics are never mixed."""
import csv
import os
import uuid
import numpy as np
from src.data.inspection import load_subject, inspect_subject, RATES
from src.data.resampling import resample_signal, align_labels
from src.data.windowing import make_windows
from src.data.splits import SUBJECTS
from src.utils.tracking import file_hash, digest, write_json, read_json

CACHE_VERSION = 'wrist-interval-labels-v1'


def prepare_subject(root, cache, subject, config):
    folder = cache / subject
    folder.mkdir(parents=True, exist_ok=True)
    source_hash = file_hash(root / subject / (subject + '.pkl'))
    key = digest({'version': CACHE_VERSION, 'data': config, 'raw_sha256': source_hash})
    manifest = folder / 'metadata.json'
    if manifest.exists():
        meta = read_json(manifest)
        if meta['cache_key'] == key and all((folder / f).exists() and file_hash(folder / f) == h for f, h in meta['files'].items()):
            print(f'[CACHE] {subject} verified', flush=True)
            return meta
    print(f'[PREPROCESS] {subject}', flush=True)
    data = load_subject(root, subject)
    inspection = inspect_subject(data, subject)
    wrist = data['signal']['wrist']
    rate = config['sampling_rate']
    length = int(round(len(data['label']) / 700 * rate))
    signals = [resample_signal(wrist[name], RATES['wrist'][name], rate, length) for name in config['sensors']]
    labels = align_labels(data['label'], rate, length)
    x, rows, counts = make_windows(np.concatenate(signals, axis=1), labels, subject, rate, config['window_seconds'], config['stride_seconds'])
    # Commit metadata last, so interrupted caches are never considered valid.
    temporary = folder / (uuid.uuid4().hex + '.npy')
    np.save(temporary, x)
    os.replace(temporary, folder / 'windows.npy')
    with (folder / 'windows.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    meta = {'subject_id': subject, 'cache_key': key, 'raw_sha256': source_hash,
            'preprocessing_version': CACHE_VERSION, 'data_config': config, 'counts': counts,
            'shape': list(x.shape), 'label_distribution': {str(k): sum(r['label'] == k for r in rows) for k in range(3)},
            'inspection': inspection, 'alignment_policy': 'Common label timeline; EDA final interval uses endpoint hold. No signal cropping.',
            'files': {name: file_hash(folder / name) for name in ('windows.npy', 'windows.csv')}}
    write_json(manifest, meta)
    return meta


def prepare_dataset(root, cache, config):
    return [prepare_subject(root, cache, subject, config['data']) for subject in SUBJECTS]

"""Build and independently verify 14-channel windows; no normalization/training.

Run from the project directory:
    python -m scripts.prepare_common_rate_windows
Existing subject caches are reused only if their inputs/configuration/files match.
"""
import argparse
import csv
import logging
from pathlib import Path
import sys

import numpy as np

from src.data.common_rate_windowing import (
    VERSION, LABEL_MAP, CLASS_NAMES, make_common_rate_windows)
from src.utils.config import ROOT
from src.utils.tracking import digest, file_hash, identifier, read_json, write_json

ROW_FIELDS = ['subject_id', 'window_id', 'start_time', 'end_time',
              'original_label', 'label', 'start_sample', 'end_sample']


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_subject(folder, signal, mask, segments, config):
    """Replay saved windows against source values and original 700-Hz boundaries."""
    arr = np.load(folder / 'windows.npy', mmap_mode='r', allow_pickle=False)
    rows = read_csv(folder / 'windows.csv')
    size = config['sampling_rate'] * config['window_seconds']
    if arr.shape != (len(rows), 14, size) or arr.dtype != np.float32:
        raise ValueError('Saved window shape/dtype mismatch.')
    seen = set()
    for i, row in enumerate(rows):
        start, end = int(row['start_sample']), int(row['end_sample'])
        if row['window_id'] in seen or not (0 <= start < end <= len(signal)) or end-start != size:
            raise ValueError('Invalid or duplicate window interval.')
        seen.add(row['window_id'])
        covering = [s for s in segments
                    if start * config['label_source_fs'] >= int(s['Start_sample_700']) * config['sampling_rate']
                    and end * config['label_source_fs'] <= int(s['End_sample_700']) * config['sampling_rate']]
        if len(covering) != 1 or int(covering[0]['Label']) not in LABEL_MAP:
            raise ValueError('Window crosses a label boundary or uses an excluded state.')
        if int(row['original_label']) != int(covering[0]['Label']) or int(row['label']) != LABEL_MAP[int(covering[0]['Label'])]:
            raise ValueError('Saved label mapping mismatch.')
        if not mask[start:end].all() or not np.array_equal(arr[i], signal[start:end].T):
            raise ValueError('Saved window differs from source or intersects invalid quality.')
    del arr
    return {'windows_checked': len(rows), 'passed': True}


def run(source, output, report, window_seconds=30, stride_seconds=15):
    report.mkdir(parents=True, exist_ok=False)
    log = logging.getLogger('common_rate_windows')
    log.setLevel(logging.INFO)
    handlers = [logging.FileHandler(report / 'windowing.log', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)]
    for h in handlers:
        h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        log.addHandler(h)
    try:
        source_config = read_json(source / 'config.json')
        if source_config['target_fs'] != 64 or source_config['label_source_fs'] != 700 or len(source_config['channels']) != 14 or source_config['normalized'] or source_config['windowed']:
            raise ValueError('Expected unnormalized, continuous common-rate 64-Hz/14-channel input.')
        config = {'version': VERSION, 'sampling_rate': 64, 'label_source_fs': source_config['label_source_fs'],
                  'window_seconds': window_seconds, 'stride_seconds': stride_seconds,
                  'channels': source_config['channels'], 'subjects': source_config['subjects'],
                  'label_map': LABEL_MAP, 'normalized': False, 'windowed': True,
                  'anchor_policy': 'continuous valid label regions; no re-anchor after quality rejection',
                  'quality_policy': 'all timepoints must have quality_mask=True',
                  'source': str(source.resolve()), 'output': str(output.resolve())}
        write_json(report / 'config.json', config)
        write_json(report / 'code_hashes.json', {
            str(p.relative_to(ROOT)): file_hash(p) for p in (
                Path(__file__), ROOT / 'src/data/common_rate_windowing.py', ROOT / 'src/data/windowing.py')})
        source_hashes = {f: file_hash(source / f) for f in ('config.json', 'label_segments.csv')}
        segments_all = read_csv(source / 'label_segments.csv')
        summaries, verification, rejections = [], {}, []
        output.mkdir(parents=True, exist_ok=True)
        log.info('Start 14-channel common-rate windowing: %s', config)
        for subject in config['subjects']:
            segments = [r for r in segments_all if r['Subject'] == subject]
            input_files = [f'{subject}_X.npy', f'{subject}_quality_mask.npy']
            hashes = {**source_hashes, **{f: file_hash(source / f) for f in input_files}}
            key = digest({'config': config, 'input_hashes': hashes})
            signal = np.load(source / input_files[0], mmap_mode='r', allow_pickle=False)
            mask = np.load(source / input_files[1], mmap_mode='r', allow_pickle=False)
            folder = output / subject
            metadata_path = folder / 'metadata.json'
            if metadata_path.exists():
                meta = read_json(metadata_path)
                if meta['cache_key'] != key or any(file_hash(folder / f) != h for f, h in meta['files'].items()):
                    raise ValueError(f'{subject}: existing cache differs; use a new --output directory.')
                log.info('%s: reusing matching cache', subject)
            else:
                folder.mkdir(parents=True, exist_ok=True)
                windows, rows, rejected, audit = make_common_rate_windows(
                    signal, mask, segments, subject, window_seconds=window_seconds, stride_seconds=stride_seconds)
                if not rows:
                    raise ValueError(f'{subject}: no retained windows; inspect quality and labels.')
                np.save(folder / 'windows.npy', windows, allow_pickle=False)
                write_csv(folder / 'windows.csv', rows, ROW_FIELDS)
                write_csv(folder / 'rejected_quality.csv', rejected, ROW_FIELDS + ['reason', 'invalid_timepoints'])
                meta = {'subject_id': subject, 'cache_key': key, 'preprocessing_version': VERSION,
                        'data_config': config, 'input_hashes': hashes, 'counts': audit,
                        'shape': list(windows.shape),
                        'label_distribution': {str(i): audit['class_counts'][name] for i, name in enumerate(CLASS_NAMES)},
                        'files': {f: file_hash(folder / f) for f in ('windows.npy', 'windows.csv', 'rejected_quality.csv')}}
                write_json(metadata_path, meta)
                del windows
            verification[subject] = verify_subject(folder, signal, mask, segments, config)
            a = meta['counts']
            summaries.append({'subject': subject, **a['class_counts'], 'label_safe_candidates': a['label_safe_candidates'],
                              'rejected_quality': a['rejected_quality'], 'kept_windows': a['kept_windows']})
            rejections.extend(read_csv(folder / 'rejected_quality.csv'))
            log.info('%s: kept=%s, rejected_quality=%s, classes=%s; verification PASS',
                     subject, a['kept_windows'], a['rejected_quality'], a['class_counts'])
            del signal, mask
        totals = {k: sum(r[k] for r in summaries) for k in summaries[0] if k != 'subject'}
        write_csv(report / 'windows_by_subject.csv', summaries, list(summaries[0]))
        write_csv(report / 'rejected_quality.csv', rejections, ROW_FIELDS + ['reason', 'invalid_timepoints'])
        write_json(report / 'verification.json', {'subjects': verification, 'all_passed': True})
        write_json(report / 'summary.json', {'config': config, 'totals': totals, 'subjects': summaries,
                   'note': 'Region-local label-safe candidates only. Transition windows are prevented, not counted as quality rejections.'})
        write_json(output / 'config.json', config)
        write_json(output / 'summary.json', {'totals': totals, 'report': str(report.resolve())})
        log.info('COMPLETE: %s; report=%s', totals, report)
        return totals
    except Exception:
        log.exception('Windowing failed; no training was performed.')
        raise
    finally:
        for h in handlers:
            log.removeHandler(h)
            h.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/processed/common_rate_64hz')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/processed/common_rate_windows_64hz_30s_15s')
    parser.add_argument('--report', type=Path, default=ROOT / 'results/common_rate_windowing' / identifier())
    args = parser.parse_args()
    run(args.source, args.output, args.report)


if __name__ == '__main__':
    main()

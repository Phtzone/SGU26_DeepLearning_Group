"""Inspect official trusted WESAD pickle files one subject at a time."""
import pickle
import numpy as np
from src.data.splits import SUBJECTS
from src.utils.tracking import file_hash, write_json, now

RATES = {'wrist': {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4},
         'chest': {k: 700 for k in ('ACC', 'ECG', 'EMG', 'EDA', 'Temp', 'Resp')}}


def load_subject(root, subject):
    path = root / subject / (subject + '.pkl')
    with path.open('rb') as stream:
        data = pickle.load(stream, encoding='latin1')
    if not {'signal', 'label', 'subject'} <= data.keys():
        raise ValueError(f'{subject}: missing WESAD keys')
    if data['subject'] != subject:
        raise ValueError(f'{subject}: pickle subject identity mismatch')
    return data


def inspect_subject(data, subject):
    labels = np.asarray(data['label'])
    if labels.ndim != 1 or not np.isfinite(labels).all() or not np.isin(labels, np.arange(8)).all():
        raise ValueError(f'{subject}: invalid labels')
    report = {'subject': subject, 'keys': sorted(data), 'labels': {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
              'label_shape': list(labels.shape), 'label_dtype': str(labels.dtype), 'label_rate': 700,
              'duration_seconds': len(labels) / 700, 'signals': {}}
    for device, signals in data['signal'].items():
        report['signals'][device] = {}
        for name, signal in signals.items():
            rate = RATES[device][name]
            expected_channels = 3 if name == 'ACC' else 1
            if signal.ndim != 2 or signal.shape[1] != expected_channels or not len(signal):
                raise ValueError(f'{subject}/{device}/{name}: invalid shape {signal.shape}')
            # Bounded temporary arrays on machines with limited RAM.
            finite = all(np.isfinite(signal[i:i + 100000]).all() for i in range(0, len(signal), 100000))
            duration = len(signal) / rate
            if not finite or abs(duration - len(labels) / 700) > 1 / rate + 1 / 700:
                raise ValueError(f'{subject}/{device}/{name}: nonfinite values or duration mismatch')
            report['signals'][device][name] = {'shape': list(signal.shape), 'dtype': str(signal.dtype),
                'rate': rate, 'duration_seconds': duration, 'finite': finite,
                'min': float(signal.min()), 'max': float(signal.max()),
                'alignment_delta_seconds': duration - len(labels) / 700}
    for name in ('ACC', 'BVP', 'EDA'):
        if name not in data['signal']['wrist']:
            raise ValueError(f'{subject}: missing wrist {name}')
    report['range_policy'] = 'Observed ranges recorded; no physiological clipping or unit conversion.'
    return report


def inspect_dataset(root, output):
    report = {'started_at': now(), 'root': str(root), 'subjects': [], 'status': 'running'}
    try:
        for subject in SUBJECTS:
            print(f'[INSPECT] {subject}', flush=True)
            data = load_subject(root, subject)
            item = inspect_subject(data, subject)
            item['raw_sha256'] = file_hash(root / subject / (subject + '.pkl'))
            report['subjects'].append(item)
            del data
            write_json(output, report)
        report['status'] = 'completed'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        report['ended_at'] = now()
        write_json(output, report)
    return report

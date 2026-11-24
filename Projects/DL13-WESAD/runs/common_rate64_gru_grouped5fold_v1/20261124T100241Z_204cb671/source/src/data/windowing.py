"""Windows anchored independently inside each uninterrupted valid label region."""
import numpy as np


def make_windows(signal, labels, subject, rate=32, window_seconds=30, stride_seconds=15):
    size, stride = int(rate * window_seconds), int(rate * stride_seconds)
    if size <= 0 or stride <= 0 or len(signal) != len(labels):
        raise ValueError('Invalid window settings or label alignment.')
    boundaries = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    windows, rows = [], []
    short_regions = 0
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        if labels[lo] < 0:
            continue
        short_regions += int(hi - lo < size)
        for start in range(int(lo), int(hi) - size + 1, stride):
            windows.append(signal[start:start + size].T)
            rows.append({'subject_id': subject, 'window_id': f'{subject}_{start}_{start + size}',
                         'start_time': start / rate, 'end_time': (start + size) / rate,
                         'original_label': int(labels[start]) + 1, 'label': int(labels[start])})
    if not windows:
        raise ValueError(f'{subject}: no valid complete windows')
    # Rejection audit for a fixed global grid (retained windows use region-local anchors).
    rejected = {'invalid_region': 0, 'transition': 0}
    for start in range(0, len(labels) - size + 1, stride):
        region = labels[start:start + size]
        if (region < 0).any():
            rejected['invalid_region'] += 1
        elif not (region == region[0]).all():
            rejected['transition'] += 1
    report = {'kept_region_anchored_windows': len(rows), 'short_valid_regions': short_regions,
              'global_grid_rejections': rejected, 'global_grid_candidates': max(0, (len(labels) - size) // stride + 1),
              'policy': 'region-local anchors; global-grid rejection counts are a separate audit'}
    return np.stack(windows).astype(np.float32), rows, report

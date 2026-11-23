"""Label-safe, quality-aware windows from the continuous common-rate export."""
import numpy as np

from src.data.windowing import make_windows

VERSION = 'common-rate-region-windows-v1'
LABEL_MAP = {1: 0, 2: 1, 3: 2}
CLASS_NAMES = ['baseline', 'stress', 'amusement']


def align_segments(segments, length, rate=64, label_rate=700):
    """Only assign 64-Hz sample intervals fully contained in a raw label segment."""
    if length <= 0 or rate <= 0 or label_rate <= 0:
        raise ValueError('Positive timeline length and rates are required.')
    labels = np.full(length, -1, dtype=np.int64)
    previous_end = 0
    for row in segments:
        start, end = int(row['Start_sample_700']), int(row['End_sample_700'])
        if start != previous_end or end <= start:
            raise ValueError('Label segments must be ordered, contiguous, and nonempty.')
        previous_end = end
        value = int(row['Label'])
        if value in LABEL_MAP:
            lo = (start * rate + label_rate - 1) // label_rate
            hi = end * rate // label_rate
            labels[lo:min(hi, length)] = LABEL_MAP[value]
    if abs(previous_end * rate - length * label_rate) > label_rate:
        raise ValueError('Label segments do not cover the signal duration.')
    return labels


def make_common_rate_windows(signal, quality_mask, segments, subject,
                             rate=64, label_rate=700, window_seconds=30,
                             stride_seconds=15):
    if signal.ndim != 2 or signal.shape[1] != 14 or signal.dtype != np.float32:
        raise ValueError('Expected float32 continuous signals of shape [time, 14].')
    if not np.isfinite(signal).all():
        raise ValueError('Nonfinite input signal; inspect preprocessing before windowing.')
    if quality_mask.shape != (len(signal),) or quality_mask.dtype != np.bool_:
        raise ValueError('Expected one boolean quality flag per timepoint.')
    if any(v <= 0 or not float(v * rate).is_integer()
           for v in (window_seconds, stride_seconds)):
        raise ValueError('Window and stride must be positive whole sample counts.')
    labels = align_segments(segments, len(signal), rate, label_rate)
    # Reuse the established region-local anchors, without altering historical code.
    windows, rows, label_audit = make_windows(
        signal, labels, subject, rate, window_seconds, stride_seconds)
    bad_prefix = np.r_[0, np.cumsum(~quality_mask, dtype=np.int64)]
    kept, rejected = [], []
    for i, row in enumerate(rows):
        start = round(row['start_time'] * rate)
        end = round(row['end_time'] * rate)
        row.update(start_sample=start, end_sample=end)
        bad = int(bad_prefix[end] - bad_prefix[start])
        if bad:
            rejected.append({**row, 'reason': 'quality_mask', 'invalid_timepoints': bad})
        else:
            kept.append(i)
    retained = [rows[i] for i in kept]
    counts = {name: sum(r['label'] == i for r in retained)
              for i, name in enumerate(CLASS_NAMES)}
    audit = {
        'label_safe_candidates': len(rows), 'rejected_quality': len(rejected),
        'kept_windows': len(retained), 'class_counts': counts,
        'flagged_input_timepoints': int((~quality_mask).sum()),
        'label_alignment': 'ceil(start*64/700), floor(end*64/700); full intervals only',
        'anchor_policy': 'stride grid restarted inside each continuous valid label region',
        'quality_policy': 'reject any candidate containing at least one False flag; do not re-anchor',
        'label_audit': label_audit,
    }
    return windows[kept], retained, rejected, audit

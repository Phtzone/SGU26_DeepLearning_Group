"""Time-based resampling with anti-aliasing for downsampling."""
import numpy as np
from math import gcd
from scipy.signal import resample_poly


def resample_signal(signal, source_rate, target_rate, target_length):
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 2 or not np.isfinite(signal).all():
        raise ValueError('Signals must be finite (time, channels) arrays.')
    duration_delta = len(signal) / source_rate - target_length / target_rate
    if abs(duration_delta) > 1 / source_rate + 1 / target_rate:
        raise ValueError('Signal duration mismatch; refusing silent truncation/extension.')
    if source_rate > target_rate:
        divisor = gcd(source_rate, target_rate)
        output = resample_poly(signal, target_rate // divisor, source_rate // divisor, axis=0)
    elif source_rate < target_rate:
        times = np.arange(target_length) / target_rate
        source_times = np.arange(len(signal)) / source_rate
        output = np.stack([np.interp(times, source_times, signal[:, c]) for c in range(signal.shape[1])], axis=1)
    else:
        output = signal
    if len(output) != target_length:
        raise ValueError(f'Resample length {len(output)} != {target_length}; inspect alignment.')
    return np.asarray(output, dtype=np.float32)


def align_labels(labels, target_rate, target_length, source_rate=700):
    """Assign only target sample intervals fully inside an original constant state."""
    labels = np.asarray(labels)
    changes = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    output = np.full(target_length, -1, dtype=np.int64)
    for start, end in zip(changes[:-1], changes[1:]):
        value = labels[start]
        if value in (1, 2, 3):
            lo = (int(start) * target_rate + source_rate - 1) // source_rate
            hi = int(end) * target_rate // source_rate
            output[lo:min(hi, target_length)] = int(value) - 1
    return output

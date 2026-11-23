"""Boundary, quality and provenance checks for the 14-channel preprocessing."""
import numpy as np
import pytest

from src.data.common_rate_windowing import align_segments, make_common_rate_windows


def segment(label, start, end):
    return {'Label': label, 'Start_sample_700': start, 'End_sample_700': end}


def signal(n):
    return np.arange(n * 14, dtype=np.float32).reshape(n, 14)


def test_fractional_boundaries_excluded_and_invalid_labels_not_mapped():
    segments = [segment(0, 0, 701), segment(1, 701, 2101), segment(2, 2101, 3500)]
    labels = align_segments(segments, 320)
    assert (labels[:65] == -1).all()
    assert (labels[65:192] == 0).all()
    assert labels[192] == -1  # one 64-Hz interval straddles the raw boundary
    assert (labels[193:] == 1).all()
    x, rows, _, _ = make_common_rate_windows(signal(320), np.ones(320, bool), segments,
                                            'S2', window_seconds=1, stride_seconds=.5)
    assert x.shape[1:] == (14, 64)
    for row in rows:
        start, end = row['start_sample'], row['end_sample']
        assert (labels[start:end] == row['label']).all()
        assert np.array_equal(x[rows.index(row)], signal(320)[start:end].T)


def test_quality_excludes_every_overlapping_window_without_reanchoring():
    mask = np.ones(192, bool)
    mask[64] = False
    x, rows, rejected, audit = make_common_rate_windows(
        signal(192), mask, [segment(2, 0, 2100)], 'S3', window_seconds=1, stride_seconds=.5)
    # [0,64) is safe; [32,96) and [64,128) must both be rejected.
    assert [r['start_sample'] for r in rows] == [0, 96, 128]
    assert [r['start_sample'] for r in rejected] == [32, 64]
    assert audit['label_safe_candidates'] == 5
    assert audit['rejected_quality'] == 2
    assert x.shape == (3, 14, 64)
    assert all(r['label'] == 1 and r['original_label'] == 2 for r in rows)


def test_no_window_can_combine_short_different_states():
    with pytest.raises(ValueError, match='no valid complete windows'):
        make_common_rate_windows(signal(64), np.ones(64, bool),
                                 [segment(1, 0, 350), segment(2, 350, 700)],
                                 'S2', window_seconds=1)


def test_all_bad_quality_keeps_empty_shape_and_audit():
    x, rows, rejected, audit = make_common_rate_windows(
        signal(64), np.zeros(64, bool), [segment(3, 0, 700)], 'S2', window_seconds=1)
    assert x.shape == (0, 14, 64) and not rows
    assert len(rejected) == audit['rejected_quality'] == 1


def test_bad_alignment_mask_and_nonfinite_inputs_fail():
    with pytest.raises(ValueError, match='contiguous'):
        align_segments([segment(1, 1, 700)], 64)
    with pytest.raises(ValueError, match='duration'):
        align_segments([segment(1, 0, 700)], 128)
    with pytest.raises(ValueError, match='quality flag'):
        make_common_rate_windows(signal(64), np.ones(63, bool), [segment(1, 0, 700)], 'S2')
    x = signal(64)
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        make_common_rate_windows(x, np.ones(64, bool), [segment(1, 0, 700)], 'S2')


def test_saved_cache_replay_reuse_and_tampering(tmp_path):
    import csv
    from scripts.prepare_common_rate_windows import run, read_csv
    from src.utils.tracking import write_json, read_json
    source, output = tmp_path / 'source', tmp_path / 'output'
    source.mkdir()
    write_json(source / 'config.json', {
        'target_fs': 64, 'label_source_fs': 700, 'channels': [f'c{i}' for i in range(14)],
        'subjects': ['S2'], 'normalized': False, 'windowed': False})
    np.save(source / 'S2_X.npy', signal(5760))
    mask = np.ones(5760, bool)
    mask[2000] = False
    np.save(source / 'S2_quality_mask.npy', mask)
    with (source / 'label_segments.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['Subject', 'Label', 'Start_sample_700', 'End_sample_700'])
        writer.writeheader()
        writer.writerows({'Subject': 'S2', **r} for r in [segment(1, 0, 31500), segment(2, 31500, 63000)])
    counts = run(source, output, tmp_path / 'report1')
    assert counts['kept_windows'] == 3 and counts['rejected_quality'] == 1
    assert read_json(tmp_path / 'report1/verification.json')['all_passed']
    assert read_csv(output / 'S2/windows.csv')[0]['subject_id'] == 'S2'
    assert run(source, output, tmp_path / 'report2') == counts
    with (output / 'S2/windows.csv').open('a') as stream:
        stream.write('tampered')
    with pytest.raises(ValueError, match='existing cache differs'):
        run(source, output, tmp_path / 'report3')

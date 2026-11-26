"""Compare a new common-rate experiment with an immutable reference run."""
import csv
import re

from src.utils.tracking import digest, read_json


def compare_reference(root, config, split, audit, rows_by_subject):
    reference = config['reference_run']
    experiment, group_id = reference['experiment'], reference['group_id']
    if not all(re.fullmatch(r'[A-Za-z0-9_-]+', x) for x in (experiment, group_id)):
        raise ValueError('Invalid reference run identifier.')
    group = root / 'runs' / experiment / group_id
    saved = read_json(group / 'experiment_manifest.json')
    expected = saved['effective_config']
    protocol = {k: config[k] == expected[k] for k in ('data', 'split', 'training', 'selection')}
    saved_keys = saved['cache_keys']
    actual_keys = {r['subject_id']: r['cache_key'] for r in audit['subjects']}
    roster_matches = set(actual_keys) == set(saved_keys) == set(split['original_subject_order'])
    subjects = [{'subject': s, 'matches': actual_keys.get(s) == saved_keys.get(s),
                 'current_cache_key': actual_keys.get(s), 'reference_cache_key': saved_keys.get(s)}
                for s in split['original_subject_order']]
    predictions = root / 'results' / experiment / group_id / 'oof_predictions.csv'
    with predictions.open(encoding='utf-8', newline='') as stream:
        prior = list(csv.DictReader(stream))
    # Only compare identity/true labels/intervals, never use Test scores for tuning.
    def identity(row, label):
        return (row['subject_id'], row['window_id'], int(row[label]),
                float(row['start_time']), float(row['end_time']))
    current = [identity(row, 'label') for rows in rows_by_subject.values() for row in rows]
    reference_rows = [identity(row, 'true_label') for row in prior]
    windows_match = (len(set(current)) == len(current) == len(reference_rows)
                     and set(current) == set(reference_rows))
    passed = (saved['status'] == 'completed' and roster_matches and all(protocol.values())
              and saved['manifest_checksum'] == digest(split)
              and all(r['matches'] for r in subjects) and windows_match)
    return {'passed': passed, 'reference_run': reference, 'reference_status': saved['status'],
            'protocol_sections_match': protocol, 'subject_roster_matches': roster_matches,
            'split_matches': saved['manifest_checksum'] == digest(split),
            'window_identity_labels_and_times_match': windows_match,
            'subjects': subjects,
            'mismatched_subjects': [r['subject'] for r in subjects if not r['matches']],
            'current_fingerprint': audit['dataset_fingerprint'],
            'reference_fingerprint': saved['dataset_fingerprint'],
            'fingerprint_matches': audit['dataset_fingerprint'] == saved['dataset_fingerprint'],
            'reference_source_audit_present': (group / 'window_index/common_rate_index.json').is_file(),
            'training_allowed': passed}


def require_matching_reference(root, config, split, audit, rows_by_subject):
    result = compare_reference(root, config, split, audit, rows_by_subject)
    if not result['passed']:
        raise ValueError('Reference comparison failed before training: '
                         f"mismatched subjects={result['mismatched_subjects']}; "
                         'run python -m scripts.preflight_common_rate_gru for details. '
                         'Obtain the reference data/audit before comparing model results.')
    return result

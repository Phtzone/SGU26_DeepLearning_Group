"""One-time local synchronization. Cleanup is separately verified and logged.

Keeps original-value byte patches so the six nearly identical local files can
be restored without retaining another full copy of the physiological arrays.
"""
import hashlib
import os
from pathlib import Path
import shutil

from src.data.common_rate import build_index
from src.data.reference_check import require_matching_reference
from src.data.splits import load_manifest
from src.utils.config import ROOT, load_config
from src.utils.tracking import file_hash, identifier, read_json, write_json


def main():
    run_id = identifier()
    report = ROOT / 'results/data_sync' / run_id
    report.mkdir(parents=True, exist_ok=False)
    received = ROOT / 'data/interim/cnn_reference_received'
    archive_report = read_json(ROOT / 'results/common_rate_gru_preflight/20261124T095357Z_ba6e804c/received_archive_check.json')
    source = ROOT / archive_report['extracted_source']
    dest = ROOT / 'data/processed/common_rate_64hz'
    ref_group = ROOT / 'runs/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac'
    expected = read_json(ref_group / 'window_index/common_rate_index.json')['audit']['source_file_sha256']
    if source.resolve().parent.parent != received.resolve() or dest.resolve() != ROOT / 'data/processed/common_rate_64hz':
        raise ValueError('Unexpected synchronization paths.')
    if len(expected) != 33 or any(file_hash(source / name) != sha for name, sha in expected.items()):
        raise ValueError('Received dataset does not match original CNN input.')
    config = load_config('experiments/common_rate64_gru_grouped5fold_v1/config.yaml')
    split = load_manifest(ROOT / config['split']['manifest'])
    old_audit, _ = build_index(dest, config['data'], split['original_subject_order'])
    write_json(report / 'before_source_audit.json', old_audit)
    changes = []
    for name, sha in expected.items():
        if file_hash(dest / name) == sha:
            continue
        if not name.endswith('_X.npy'):
            raise ValueError('Only the six previously identified signal files may differ.')
        old, new = (dest / name).read_bytes(), (source / name).read_bytes()
        if len(old) != len(new):
            raise ValueError('Unexpected file size difference; keep full backups instead.')
        patch = [{'offset': i, 'old_byte': a, 'new_byte': b}
                 for i, (a, b) in enumerate(zip(old, new)) if a != b]
        if len(patch) > 1024:
            raise ValueError('Difference exceeds the small-patch backup policy.')
        restored = bytearray(new)
        for item in patch:
            restored[item['offset']] = item['old_byte']
        old_hash = file_hash(dest / name)
        if hashlib.sha256(restored).hexdigest() != old_hash:
            raise ValueError('Exact rollback patch verification failed.')
        changes.append({'file': name, 'old_sha256': old_hash, 'new_sha256': sha,
                        'size_bytes': len(new), 'restore_patch': patch,
                        'restore_patch_verified': True})
    if {c['file'] for c in changes} != {'S4_X.npy','S6_X.npy','S7_X.npy','S8_X.npy','S11_X.npy','S15_X.npy'}:
        raise ValueError('Dataset state changed since inspection; review before synchronizing.')
    write_json(report / 'rollback_patches.json', changes)
    backup = ROOT / 'data/interim' / ('sync_backup_' + run_id)
    backup.mkdir(exist_ok=False)
    for item in changes:
        name = item['file']
        shutil.copy2(dest / name, backup / name)
        if file_hash(backup / name) != item['old_sha256']:
            raise ValueError('Temporary backup failed.')
    progress = []
    for item in changes:
        name = item['file']
        temp = dest / (run_id + '.tmp')
        with (source / name).open('rb') as inp, temp.open('xb') as out:
            shutil.copyfileobj(inp, out)
            out.flush()
            os.fsync(out.fileno())
        if file_hash(temp) != item['new_sha256']:
            raise ValueError('Staged data file hash mismatch.')
        os.replace(temp, dest / name)
        progress.append(name)
        write_json(report / 'sync_progress.json', {'replaced': progress, 'temporary_backup': str(backup)})
    if any(file_hash(dest / name) != sha for name, sha in expected.items()):
        raise ValueError('Final source hashes failed; retain backup and stop cleanup.')
    audit, rows = build_index(dest, config['data'], split['original_subject_order'])
    verification = require_matching_reference(ROOT, config, split, audit, rows)
    write_json(report / 'after_reference_check.json', verification)
    write_json(report / 'after_source_audit.json', audit)
    write_json(report / 'sync_complete.json', {'passed': True, 'files_replaced': progress,
               'all_33_hashes_match': True, 'windows': audit['windows'],
               'temporary_backup': str(backup), 'cleanup_performed': False,
               'rollback_patches': 'rollback_patches.json'})
    print(report)


if __name__ == '__main__':
    main()

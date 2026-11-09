import pytest
from src.data.splits import SUBJECTS, create_manifest, validate_manifest, ensure_manifest, load_manifest
from src.utils.tracking import digest


def test_exact_groups_and_rotation():
    manifest = validate_manifest(create_manifest())
    assert manifest['groups'] == [['S10', 'S16', 'S9'], ['S8', 'S17', 'S15'], ['S7', 'S4', 'S11'], ['S5', 'S6', 'S14'], ['S2', 'S3', 'S13']]
    assert len(manifest['folds']) == 5
    for fold in manifest['folds']:
        assert [len(fold[k]) for k in ('train', 'validation', 'test')] == [9, 3, 3]
        assert len(set(fold['train'] + fold['validation'] + fold['test'])) == 15
    for subject in SUBJECTS:
        assert [sum(subject in f[k] for f in manifest['folds']) for k in ('train', 'validation', 'test')] == [3, 1, 1]


def test_manifest_checksum_and_no_overwrite(tmp_path):
    path = tmp_path / 'split.json'
    a = ensure_manifest(path)
    original = path.read_bytes()
    assert ensure_manifest(path) == a
    assert path.read_bytes() == original
    assert path.with_suffix('.sha256').read_text().strip() == digest(a)
    with pytest.raises(ValueError):
        ensure_manifest(path, seed=43)
    path.write_text('{}')
    with pytest.raises(ValueError):
        load_manifest(path)


def test_reject_modified_split():
    manifest = create_manifest()
    manifest['folds'][0]['train'][0] = manifest['folds'][0]['test'][0]
    with pytest.raises(ValueError):
        validate_manifest(manifest)

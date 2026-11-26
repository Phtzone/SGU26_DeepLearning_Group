from codecs import BOM_UTF8
import csv
from datetime import datetime

import pytest

from scripts.research_journal import SCHEMA, reference_text, write_journal


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.reader(stream, strict=True))


def test_initialize_then_append_preserves_fields_and_bytes(tmp_path):
    journal = write_journal(tmp_path)
    assert {path.name for path in journal.iterdir()} == set(SCHEMA)
    for name, (_, header) in SCHEMA.items():
        assert read_csv(journal / name) == [list(header)]
    path = journal / 'activity_log.csv'
    original = path.read_bytes()
    timestamp = datetime(2026, 11, 26, 9, 52, 56, 506000)
    evidence = reference_text('Hiện vật, "kiểm tra"', '../../results/nhóm/log.csv')
    record = (None, timestamp, None, 'nhóm', 'dòng đầu\r\ndòng sau',
              'kết quả có "ngoặc kép", dấu phẩy', evidence, None)
    write_journal(tmp_path, {'activity_log.csv': [record, record]})
    first = path.read_bytes()
    assert first.startswith(original)
    assert first.count(BOM_UTF8) == 1
    assert read_csv(path)[1:] == [['', '2026-11-26 09:52:56.506000', '', 'nhóm', 'dòng đầu\r\ndòng sau',
                                   'kết quả có "ngoặc kép", dấu phẩy',
                                   'Hiện vật, "kiểm tra"\n../../results/nhóm/log.csv', '']]
    write_journal(tmp_path, {'activity_log.csv': [record]})
    assert path.read_bytes() == first


def test_archive_import_preserves_rows_and_refuses_conflict(tmp_path):
    first = ('', 'nhóm', 'hình', 'hình.png\n../../results/hình.png')
    later = ('', 'nhóm', 'hình bổ sung', 'hình-mới.png')
    journal = write_journal(tmp_path, {'figures.csv': [first]}, import_archive=True)
    write_journal(tmp_path, {'figures.csv': [later]})
    path = journal / 'figures.csv'
    original = path.read_bytes()
    write_journal(tmp_path, {'figures.csv': [first]}, import_archive=True)
    with pytest.raises(FileExistsError, match='differs from the workbook'):
        write_journal(tmp_path, {'figures.csv': [later]}, import_archive=True)
    assert path.read_bytes() == original
    assert read_csv(path)[1:] == [list(first), list(later)]

"""Initialize and append to an experiment's six-file research journal."""
import argparse
from codecs import BOM_UTF8
from contextlib import contextmanager
import csv
from datetime import date, datetime, time
from io import StringIO
import os
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
# The first blank column is part of the team's worksheet layout.
SCHEMA = {
    'experiments.csv': ('Thí nghiệm', ('', 'Thí Nghiệm', 'Ngày chạy', 'Người làm', 'Nội dung',
                                      'Kết quả', 'code/notebook', 'log', 'Trạng Thái')),
    'figures.csv': ('Hình', ('', 'Thí nghiệm', 'mô tả ảnh', 'Dán ảnh tại đây')),
    'activity_log.csv': ('Nhật ký', ('', 'Ngày', 'người làm', 'Thí nghiệm', 'đã làm', 'kết quả', 'Minh chứng', 'làm tiếp')),
    'debug.csv': ('Debug', ('', 'Ngày', 'Thí nghiệm', 'Lỗi', 'nguyên nhân', 'cách sửa', 'kiểm tra lại', 'Log')),
    'references.csv': ('Tham khảo', ('', 'Nghiên cứu', 'Bài toán / đầu vào', 'Cách làm', 'Đánh giá',
                                     'Kết quả tiêu biểu', 'Điểm cần lưu ý khi so với nhóm')),
    'todo.csv': ('Việc cần làm', ('', 'vịêc cần làm', 'người làm', 'vị trí', 'hoàn tất khi', 'trạng thái')),
}


def cell_text(value):
    if isinstance(value, datetime):
        return value.isoformat(sep=' ')
    if isinstance(value, (date, time)):
        return value.isoformat()
    return '' if value is None else str(value)


def reference_text(value, target):
    """Keep a hyperlink's label and target as text in the same CSV field."""
    display = cell_text(value)
    if not target or target == display:
        return display
    return display + '\n' + target if display else target


def encode_rows(rows):
    buffer = StringIO(newline='')
    csv.writer(buffer, lineterminator='\r\n').writerows(rows)
    return buffer.getvalue().encode('utf-8')


def decode_rows(data, path, header):
    if not data.startswith(BOM_UTF8):
        raise ValueError(f'Expected UTF-8 BOM in {path}')
    with StringIO(data.decode('utf-8-sig'), newline='') as buffer:
        rows = [tuple(row) for row in csv.reader(buffer, strict=True)]
    if not rows or rows[0] != header or any(len(row) != len(header) for row in rows):
        raise ValueError(f'Unexpected journal columns in {path}')
    return rows


@contextmanager
def journal_lock(experiment):
    path = experiment / '.research_journal.lock.tmp'
    try:
        lock = path.open('x', encoding='ascii')
    except FileExistsError:
        raise FileExistsError(f'Another journal write may be active: {path}') from None
    try:
        with lock:
            lock.write(f'{os.getpid()}\n')
        yield
    finally:
        path.unlink()


def write_journal(experiment_dir, records=None, *, import_archive=False):
    """Create the journal or append new rows, preserving all existing rows verbatim."""
    experiment = Path(experiment_dir).resolve(strict=True)
    if not experiment.is_dir():
        raise NotADirectoryError(experiment)
    directory = experiment / 'research_journal'
    records = records or {}
    unknown = records.keys() - SCHEMA.keys()
    if unknown:
        raise ValueError(f'Unknown journal files: {sorted(unknown)}')
    with journal_lock(experiment):
        existing = directory.exists()
        if directory.is_symlink() or (existing and directory.resolve() != directory):
            raise ValueError(f'Expected a local journal directory: {directory}')
        changes = {}
        for name, (_, header) in SCHEMA.items():
            path = directory / name
            incoming = [tuple(cell_text(value) for value in row) for row in records.get(name, ())]
            if any(len(row) != len(header) for row in incoming):
                raise ValueError(f'Expected {len(header)} fields in {path}')
            if not existing:
                desired = [header, *incoming] if import_archive else [header, *dict.fromkeys(incoming)]
                changes[name] = (None, BOM_UTF8 + encode_rows(desired), desired)
                continue
            original = path.read_bytes()
            previous = decode_rows(original, path, header)
            if import_archive:
                if previous[:len(incoming) + 1] != [header, *incoming]:
                    raise FileExistsError(f'Existing journal differs from the workbook import: {path}')
                continue
            known = set(previous[1:])
            additions = [row for row in dict.fromkeys(incoming) if row not in known]
            if additions:
                separator = b'' if original.endswith((b'\n', b'\r')) else b'\r\n'
                changes[name] = (original, original + separator + encode_rows(additions), previous + additions)
        if not changes:
            return directory
        # Stage all changes before publishing. An interrupted append can be retried without duplicates.
        with TemporaryDirectory(dir=experiment, prefix='.journal-csv-') as tmp:
            stage = Path(tmp).resolve()
            if stage.parent != experiment:
                raise ValueError(f'Unexpected staging location: {stage}')
            for name, (_, contents, expected) in changes.items():
                path = stage / name
                path.write_bytes(contents)
                if decode_rows(path.read_bytes(), path, SCHEMA[name][1]) != expected:
                    raise ValueError(f'Journal verification failed: {name}')
            if not existing:
                # A child directory can be renamed without interfering with TemporaryDirectory cleanup.
                ready = stage / 'research_journal'
                ready.mkdir()
                for name in changes:
                    (stage / name).rename(ready / name)
                if directory.exists():
                    raise FileExistsError(directory)
                ready.rename(directory)
            else:
                for name, (original, _, _) in changes.items():
                    path = directory / name
                    if path.read_bytes() != original:
                        raise RuntimeError(f'Journal changed during the write: {path}')
                    (stage / name).replace(path)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, type=Path, help='Existing directory under experiments/')
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    experiments = (ROOT / 'experiments').resolve()
    if experiment == experiments or not experiment.is_relative_to(experiments):
        parser.exit(1, f'Expected an experiment directory under {experiments}\n')
    try:
        print(write_journal(experiment))
    except (OSError, ValueError, csv.Error, RuntimeError) as error:
        parser.exit(1, f'Could not initialize Research Journal: {error}\n')


if __name__ == '__main__':
    main()

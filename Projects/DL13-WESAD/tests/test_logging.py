import csv
import json
from src.utils.logging import RunLogger, ITERATION_FIELDS, EPOCH_FIELDS


def test_iteration_live_flush_and_append(tmp_path):
    logger = RunLogger(tmp_path)
    for i in range(3):
        row = {key: i for key in ITERATION_FIELDS}
        logger.iteration(row)
        with (tmp_path / 'iterations.csv').open() as stream:
            assert len(list(csv.DictReader(stream))) == i + 1
    RunLogger(tmp_path).iteration({key: 9 for key in ITERATION_FIELDS})
    with (tmp_path / 'iterations.csv').open() as stream:
        assert len(list(csv.DictReader(stream))) == 4


def test_epoch_events_and_errors(tmp_path):
    logger = RunLogger(tmp_path)
    logger.epoch({key: 1 for key in EPOCH_FIELDS})
    logger.event('epoch_end', epoch=1)
    logger.error(ValueError('test failure'))
    assert 'test failure' in (tmp_path / 'errors.log').read_text()
    assert json.loads((tmp_path / 'events.jsonl').read_text().splitlines()[0])['event'] == 'epoch_end'

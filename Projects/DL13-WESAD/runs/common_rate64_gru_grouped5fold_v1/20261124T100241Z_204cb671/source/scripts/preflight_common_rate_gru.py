"""Audit reference data and optionally smoke-test GRU on Train/Validation only.

A smoke test on current local data is an engineering check, even when the
reference-data gate fails. It never counts as an official model experiment.
"""
import argparse
import csv
import json
import logging
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from src.data.common_rate import build_index, prepare_index
from src.data.dataset import WindowDataset
from src.data.normalization import ChannelScaler
from src.data.reference_check import compare_reference
from src.data.splits import load_manifest
from src.models import build_model
from src.training.checkpoint import save_checkpoint, load_checkpoint
from src.training.optimizer import class_weights, make_optimizer
from src.utils.config import ROOT, load_config
from src.utils.reproducibility import seed_everything
from src.utils.tracking import file_hash, identifier, read_json, write_json

DEFAULT_CONFIG = 'experiments/common_rate64_gru_grouped5fold_v1/config.yaml'


def smoke(config, split, audit, rows, report, log):
    seed_everything(42, threads=config['runtime']['torch_threads'])
    index = report / 'local_window_index'
    prepare_index(index, audit, rows)
    fold = split['folds'][0]
    # Do not construct a Test dataset or consume Test signals in the smoke test.
    train = WindowDataset(index, fold['train'])
    val = WindowDataset(index, fold['validation'])
    scaler = ChannelScaler().fit(train)
    scaler.save(report / 'normalization.json')
    train.scaler = val.scaler = scaler
    if set(scaler.subjects) & set(fold['validation'] + fold['test']):
        raise ValueError('Scaler contains held-out people.')
    model = build_model(config, train.input_channels)
    before = model.gru.weight_ih_l0.detach().clone()
    optimizer = make_optimizer(model, config['training'])
    weights = class_weights(train.labels)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    records = []
    model.train()
    for i, (x, y, _) in enumerate(DataLoader(train, batch_size=32, shuffle=True)):
        if i == 2:
            break
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if logits.shape != (len(y), 3) or not torch.isfinite(loss):
            raise ValueError('Invalid smoke output/loss.')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        record = {'batch': i, 'loss': float(loss.detach()), 'gradient_norm': float(norm),
                  'seconds': time.perf_counter()-started, 'input_shape': list(x.shape)}
        records.append(record)
        log.info('SMOKE TRAIN %s', record)
    if torch.equal(before, model.gru.weight_ih_l0):
        raise ValueError('GRU weights were not updated.')
    x, y, _ = next(iter(DataLoader(val, batch_size=32, shuffle=False)))
    model.eval()
    with torch.no_grad():
        expected = model(x)
        val_loss = float(criterion(expected, y))
    if not torch.isfinite(expected).all() or not np.isfinite(val_loss):
        raise ValueError('Nonfinite Validation smoke result.')
    path = report / 'engineering_smoke.pt'
    save_checkpoint(path, {'purpose': 'engineering_smoke_not_research', 'model': model.state_dict(),
                          'optimizer': optimizer.state_dict(), 'config': config})
    replay = build_model(config, 14)
    replay.load_state_dict(load_checkpoint(path)['model'])
    replay.eval()
    with torch.no_grad():
        torch.testing.assert_close(expected, replay(x), rtol=0, atol=0)
    # Compare Train-only statistics against saved CNN statistics; no Test scoring.
    ref = config['reference_run']
    original = read_json(ROOT / 'runs' / ref['experiment'] / ref['group_id'] / 'seed_42/fold_01/normalization.json')
    result = {'passed': True, 'purpose': 'engineering_smoke_not_research', 'fold': 1,
              'train_subjects': train.subjects, 'validation_subjects': val.subjects,
              'test_consumed': False, 'train_batches': records, 'validation_batches': 1,
              'validation_smoke_loss': val_loss, 'checkpoint_replay_exact': True,
              'trainable_parameters': sum(p.numel() for p in model.parameters()),
              'class_weights_train_only': weights.tolist(),
              'cnn_scaler_mean_max_abs_difference': float(np.max(np.abs(np.asarray(original['mean']) - scaler.mean))),
              'cnn_scaler_std_max_abs_difference': float(np.max(np.abs(np.asarray(original['std']) - scaler.std)))}
    write_json(report / 'smoke_check.json', result)
    log.info('SMOKE PASS: two Train batches, one Validation batch, exact checkpoint replay; Test untouched.')
    return result


def run(report, config_path=DEFAULT_CONFIG, run_smoke=False):
    report.mkdir(parents=True, exist_ok=False)
    log = logging.getLogger('gru_preflight')
    log.setLevel(logging.INFO)
    handler = logging.FileHandler(report / 'preflight.log', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    log.addHandler(handler)
    try:
        config = load_config(config_path)
        split = load_manifest(ROOT / config['split']['manifest'])
        audit, rows = build_index(ROOT / config['data']['processed_dir'], config['data'], split['original_subject_order'])
        result = compare_reference(ROOT, config, split, audit, rows)
        write_json(report / 'reference_check.json', result)
        write_json(report / 'current_source_audit.json', audit)
        (report / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        with (report / 'subjects.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result['subjects'][0]))
            writer.writeheader()
            writer.writerows(result['subjects'])
        write_json(report / 'reference_artifact_hashes.json', {
            str(p.relative_to(ROOT)): file_hash(p)
            for experiment in [config['reference_run']['experiment']]
            for directory in [ROOT / 'runs' / experiment / config['reference_run']['group_id'],
                              ROOT / 'results' / experiment / config['reference_run']['group_id']]
            for p in directory.rglob('*') if p.is_file()})
        log.info('REFERENCE passed=%s; mismatched subjects=%s; matching window IDs/labels/times=%s',
                 result['passed'], result['mismatched_subjects'], result['window_identity_labels_and_times_match'])
        if run_smoke:
            smoke(config, split, audit, rows, report, log)
        log.info('OFFICIAL TRAINING ALLOWED: %s', result['training_allowed'])
        return result
    except Exception:
        log.exception('Preflight failed.')
        raise
    finally:
        log.removeHandler(handler)
        handler.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/common_rate_gru_preflight' / identifier())
    args = parser.parse_args()
    result = run(args.output, args.config, args.smoke)
    print(json.dumps({'report': str(args.output), 'reference_passed': result['passed'],
                      'mismatched_subjects': result['mismatched_subjects'],
                      'official_training_allowed': result['training_allowed']}))


if __name__ == '__main__':
    main()

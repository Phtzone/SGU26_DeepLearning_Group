"""Audit actual artifacts and render the final implementation/research report."""
import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from src.data.dataset import WindowDataset
from src.data.normalization import ChannelScaler
from src.data.splits import load_manifest
from src.evaluation.aggregation import aggregate_group
from src.training.checkpoint import load_checkpoint
from src.training.optimizer import class_weights
from src.utils.config import ROOT
from src.utils.tracking import read_json, write_json, now


def audit(group_id):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', group_id):
        raise ValueError('Invalid group ID.')
    group = ROOT / 'runs/experiment_e1' / group_id
    manifest = read_json(group / 'experiment_manifest.json')
    split = load_manifest(group / 'split_manifest.json')
    if manifest['status'] != 'completed':
        raise ValueError('The final audit requires all configured folds to finish.')
    summary = aggregate_group(group)
    cache = Path(manifest['cache_path'])
    checks = []
    for seed in manifest['effective_config']['training']['seeds']:
        for fold in split['folds']:
            folder = group / f'seed_{seed}' / f'fold_{fold["fold_id"]:02d}'
            metadata = read_json(folder / 'metadata.json')
            normalization = read_json(folder / 'normalization.json')
            if normalization['train_subjects'] != fold['train']:
                raise ValueError('Normalization subjects differ from the training split.')
            if set(normalization['train_subjects']) & set(fold['validation'] + fold['test']):
                raise ValueError('Normalization leakage detected.')
            dataset = WindowDataset(cache, fold['train'])
            scaler = ChannelScaler().fit(dataset)
            np.testing.assert_allclose(normalization['mean'], scaler.mean, rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(normalization['std'], scaler.std, rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(metadata['class_weights'], class_weights(dataset.labels).numpy(), rtol=1e-6)
            if normalization['sample_count_per_channel'] != scaler.count:
                raise ValueError('Normalization sample count mismatch.')
            last = load_checkpoint(folder / 'checkpoints/last.pt')
            best = load_checkpoint(folder / 'checkpoints/best.pt')
            selected = last['history'][metadata['best_epoch'] - 1]
            if any(row['val_macro_f1'] > selected['val_macro_f1'] + 1e-12 for row in last['history']):
                raise ValueError('Selected checkpoint does not maximize Validation Macro-F1.')
            if any(abs(row['val_macro_f1'] - selected['val_macro_f1']) <= 1e-12 and row['val_loss'] < selected['val_loss'] for row in last['history']):
                raise ValueError('Selected checkpoint violates the Validation loss tie-breaker.')
            if best['epoch'] != metadata['best_epoch'] or last['epoch'] != metadata['completed_epochs']:
                raise ValueError('Checkpoint epoch metadata mismatch.')
            checks.append({'training_seed': seed, 'fold_id': fold['fold_id'], 'normalization_train_only': True,
                           'recomputed_scaler_matches': True, 'class_weights_train_only': True,
                           'validation_selection_verified': True, 'checkpoint_epochs_verified': True,
                           'normalization_timesteps_per_channel': scaler.count,
                           'errors_log_bytes': (folder / 'logs/errors.log').stat().st_size})
    output = ROOT / 'results/experiment_e1' / group_id
    write_json(output / 'artifact_audit.json', {'status': 'passed', 'audited_at': now(), 'folds': checks,
                                               'oof_and_iteration_coverage': 'verified by aggregate_group'})
    inspection = read_json(ROOT / 'data/interim/inspection.json')
    tests = ET.parse(ROOT / 'results/pytest-results.xml').getroot().find('testsuite').attrib
    if int(tests['failures']) or int(tests['errors']):
        raise ValueError('The recorded unit test suite is not passing.')
    cache_metadata = [read_json(cache / subject / 'metadata.json') for subject in split['original_subject_order']]
    total_windows = sum(item['shape'][0] for item in cache_metadata)
    window_shape = cache_metadata[0]['shape'][1:]
    files = sorted(p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()
                   and not any(part in {'data', 'runs', 'results', '__pycache__', '.pytest_cache', '.venv'} for part in p.relative_to(ROOT).parts)
                   and p.name != 'IMPLEMENTATION_REPORT.md')
    nodes = {}
    for name in files + ['IMPLEMENTATION_REPORT.md', 'results/pytest-results.xml']:
        node = nodes
        parts = name.split('/')
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = None
    for name in ['data/raw/WESAD', 'data/interim', 'data/processed', 'runs/experiment_e1/' + group_id, 'results/experiment_e1/' + group_id]:
        node = nodes
        for part in name.split('/'):
            node = node.setdefault(part, {})

    def render_tree(node, prefix=''):
        lines = []
        items = sorted(node.items(), key=lambda item: (item[1] is None, item[0]))
        for index, (name, children) in enumerate(items):
            last = index == len(items) - 1
            lines.append(prefix + ('└── ' if last else '├── ') + name + ('/' if children is not None else ''))
            if children:
                lines.extend(render_tree(children, prefix + ('    ' if last else '│   ')))
        return lines

    tree = 'DL13-WESAD/\n' + '\n'.join(render_tree(nodes)) + '\n'
    groups = '\n'.join(f'- G{i}: ' + ', '.join(subjects) for i, subjects in enumerate(split['groups'], 1))
    split_table = '| Fold | Train | Validation | Test |\n|---|---|---|---|\n' + '\n'.join(f"| {f['fold_id']} | {' '.join(f['train'])} | {' '.join(f['validation'])} | {' '.join(f['test'])} |" for f in split['folds'])
    durations = '\n'.join(f"- Seed {seed}: {result['training_duration_seconds']:.1f} giây ({result['training_duration_seconds'] / 60:.2f} phút) tổng thời lượng các folds." for seed, result in summary['per_seed'].items())
    report = f'''# DL13-WESAD — Báo cáo triển khai và E1 thực tế

Audit lúc {now()}; group `{group_id}`. Hoàn tất **{summary['completed_folds']}/{summary['expected_folds']} folds**, dùng toàn bộ WESAD windows hợp lệ.

## Cây mã nguồn và dữ liệu/artifacts

```text
{tree}```

Đây là danh sách đầy đủ source/config/docs hiện có. Raw có từng `Sx/Sx.pkl`; processed có từng subject với `windows.npy`, `windows.csv`, `metadata.json`. Cấu trúc đầy đủ bên trong từng fold được mô tả tại [README E1](experiments/e1_cnn/README.md#logging-và-checkpoints).

## Các module đã triển khai

- [Data](src/data/): inspection các tín hiệu và labels; resample anti-aliasing; alignment bảo toàn transition; region-local windowing; SHA-256 cache; memory-mapped Dataset; channel scaler Train-only; deterministic grouped splits.
- [CNN](src/models/cnn.py): kiến trúc yêu cầu, 45.059 trainable parameters, 5 channels và 960 timesteps, logits 3 lớp. GRU/CNN-GRU và ablation vẫn planned, không có kết quả giả.
- [Training](src/training/): Adam, Train-weighted CE, clipping, Validation selection/early stopping, live batch/epoch/validation logging, atomic checkpoints và epoch-boundary resume.
- [Evaluation](src/evaluation/): fixed-label metrics, subject/OOF exports, coverage/duplicate validation, committed-iteration audit và scientific figures.
- [Utilities](src/utils/): effective configuration, UTC/unique IDs, atomic JSON, provenance, RNG và system/dependency snapshots.
- [CLI](scripts/): inspect, splits, preprocessing, full grouped CV, resume, aggregation và artifact audit.

## Unit tests và dữ liệu

Pytest: **{tests['tests']} tests**, failures={tests['failures']}, errors={tests['errors']}, skipped={tests['skipped']}; {float(tests['time']):.2f} giây. [JUnit evidence](results/pytest-results.xml). Tests dùng synthetic fixtures trong thư mục tạm, không dùng làm kết quả nghiên cứu.

Inspection: **{inspection['status']}**, {len(inspection['subjects'])}/15 subjects. [Báo cáo signals/shapes/rates/ranges/durations/labels](data/interim/inspection.json). Preprocessing: {total_windows} windows `{tuple(window_shape)}`, không giới hạn mẫu, không sửa raw. Môi trường thực tế và dependency pins tại [system.json](runs/experiment_e1/{group_id}/system.json) và [requirements-runtime.txt](runs/experiment_e1/{group_id}/requirements-runtime.txt).

## Groups và split manifest

{groups}

[Manifest](splits/grouped_5fold_seed42.json) · [Checksum](splits/grouped_5fold_seed42.sha256).

SHA-256 canonical JSON: `{manifest['manifest_checksum']}`.

{split_table}

Mỗi subject Test/Validation đúng một lần, Train ba lần; không random window split và không LOSO.

## Kết quả thực tế: folds, epochs, durations và 15 subjects

{(output / 'summary.md').read_text(encoding='utf-8')}

{durations}

## Kiểm tra artifacts và leakage trên lượt chạy thật

Tất cả {len(checks)} folds đã pass audit: hashes artifacts, đúng split/OOF labels/probabilities/coverage, không trùng windows, đủ training iterations đã commit; tính lại mean/std từ đúng Train subjects khớp scaler đã lưu; class weights khớp phân bố Train; best checkpoint tối đa hóa Validation Macro-F1 và đúng tie-breaker. Test không được gọi trong selection (kiểm thử instrumentation đã pass). [Artifact audit](results/experiment_e1/{group_id}/artifact_audit.json) · [Iteration audit](results/experiment_e1/{group_id}/iteration_audit.json).

## Đường dẫn và reproduction

- [Root README](README.md) · [E1 README](experiments/e1_cnn/README.md).
- [Toàn bộ run/logs/checkpoints](runs/experiment_e1/{group_id}/) · [Source được dùng khi chạy](runs/experiment_e1/{group_id}/source/).
- [Fold CSV](results/experiment_e1/{group_id}/fold_results.csv) · [15 subject results](results/experiment_e1/{group_id}/subject_metrics.csv).
- [OOF predictions](results/experiment_e1/{group_id}/oof_predictions.csv) · [Summary JSON](results/experiment_e1/{group_id}/summary.json) · [Figures](results/experiment_e1/{group_id}/figures/).

```powershell
python -m pytest -q --junitxml=results/pytest-results.xml
python -m scripts.run_grouped_cv --experiment experiments/e1_cnn/config.yaml --resume --group-id {group_id}
python -m scripts.aggregate_results --group-id {group_id}
python -m scripts.audit_experiment --group-id {group_id}
```

Lệnh resume xác minh và skip các fold completed, không train lại. Muốn chạy mới dùng lệnh không có `--resume --group-id`; source/config/manifest và pins của group được lưu riêng. Script audit/report được bổ sung sau khi bắt đầu lượt E1, còn snapshot source của lượt chạy giữ nguyên phiên bản đã thực thi.

## Lỗi, cảnh báo và hạn chế

Số byte errors.log từng fold: {[(c['fold_id'], c['errors_log_bytes']) for c in checks]}. CPU-only; chưa kiểm chứng CUDA hay tái lập bitwise giữa nền tảng. Có biến thiên đáng kể theo subject/fold; overlap không tạo thêm subjects độc lập. Một training seed chưa đo được độ ổn định theo khởi tạo. Chưa chạy E2/E3/E4 nên không có kết luận so sánh kiến trúc. Resume đã kiểm thử ở epoch boundary và giữa epoch; mất điện làm hỏng một phần dòng CSV chưa có sửa byte tự động. Cache/runs/results/raw lớn được gitignore, cần lưu trữ riêng nếu chia sẻ tái lập.
'''
    (ROOT / 'IMPLEMENTATION_REPORT.md').write_text(report, encoding='utf-8')
    print('Artifact audit PASSED:', len(checks), 'folds; report:', ROOT / 'IMPLEMENTATION_REPORT.md')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group-id', required=True)
    audit(parser.parse_args().group_id)


if __name__ == '__main__':
    main()

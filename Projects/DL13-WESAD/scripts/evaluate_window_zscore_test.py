"""Evaluate frozen CNN/GRU checkpoints on Test; never train or select on Test."""
import argparse
import csv
from pathlib import Path
import sys


def restore_scaler(path, fold):
    import numpy as np
    from src.data.normalization import ChannelScaler
    from src.utils.tracking import read_json
    saved = read_json(path)
    if set(saved['train_subjects']) != set(fold['train']):
        raise ValueError('Scaler Train subjects differ from frozen split.')
    if set(saved['train_subjects']) & set(fold['validation'] + fold['test']):
        raise ValueError('Scaler contains held-out subjects.')
    scaler = ChannelScaler()
    scaler.mean = np.asarray(saved['mean'], dtype=np.float32)
    scaler.std = np.asarray(saved['std'], dtype=np.float32)
    if scaler.mean.shape != (14,) or scaler.std.shape != (14,) or not np.isfinite(scaler.mean).all() or not np.isfinite(scaler.std).all() or (scaler.std <= 0).any():
        raise ValueError('Invalid frozen scaler.')
    return scaler


def run(output):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from src.data.common_rate import build_index, prepare_index
    from src.data.window_preprocessing import PreprocessedWindowDataset
    from src.data.splits import load_manifest
    from src.models import build_model
    from src.training.checkpoint import load_checkpoint
    from src.training.optimizer import class_weights
    from src.evaluation.evaluator import evaluate, export_test, write_csv
    from src.evaluation.metrics import metrics_by_subject
    from src.evaluation.visualization import confusion_plot
    from src.utils.config import ROOT
    from src.utils.reproducibility import seed_everything
    from src.utils.tracking import read_json, write_json, file_hash, digest, now
    seed_everything(42, threads=2)
    sources = {
        'CNN': ROOT/'results/preprocessing_window_zscore_cnn_full/20261124_full_seed42',
        'GRU': ROOT/'results/preprocessing_window_zscore_gru_full/20261124_full_seed42',
    }
    frozen = {}
    for name, source in sources.items():
        record = read_json(source/'config.json')
        config = record.get('effective_model', record.get('effective_cnn'))
        summary = read_json(source/'summary.json')
        assert summary['status'] == 'completed' and read_json(source/'verification.json')['passed']
        assert config['experiment']['model'] == {'CNN':'cnn1d','GRU':'gru'}[name]
        frozen[name] = (source, record, config, summary)
    split = load_manifest(ROOT/frozen['CNN'][2]['split']['manifest'])
    reference = read_json(ROOT/'runs/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac/experiment_manifest.json')
    assert digest(split) == reference['manifest_checksum']
    for name, (_, record, config, _) in frozen.items():
        assert load_manifest(ROOT/config['split']['manifest']) == split
        assert config['data'] == frozen['CNN'][2]['data']
        assert record['dataset_fingerprint'] == reference['dataset_fingerprint']
    audit, rows = build_index(ROOT/frozen['CNN'][2]['data']['processed_dir'], frozen['CNN'][2]['data'], split['original_subject_order'])
    assert audit['dataset_fingerprint'] == reference['dataset_fingerprint']
    # Protect existing inputs; all new artifacts live in the output directory.
    protected = {str(p): file_hash(p) for source in sources.values() for p in source.rglob('*') if p.is_file()}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'frozen_inputs.json', {'started_at':now(), 'inputs':protected,
        'dataset_fingerprint':audit['dataset_fingerprint'], 'manifest_checksum':digest(split),
        'training_performed':False, 'selection':'existing best.pt chosen only on Validation',
        'test_usage':'post-selection exploratory assessment; historical Test results were already inspected'})
    write_json(output/'source_checksums.json', {str(p.relative_to(ROOT)):file_hash(p) for p in
        [Path(__file__), ROOT/'src/data/window_preprocessing.py', ROOT/'src/evaluation/evaluator.py',
         ROOT/'src/models/cnn.py', ROOT/'src/models/gru.py', ROOT/'src/data/normalization.py']})
    index = output/'window_index'
    prepare_index(index, audit, rows)
    all_results = {}; checks = []
    for name, (source, record, config, summary) in frozen.items():
        for mode in ['control', 'window_zscore']:
            subject_rows = []; pooled_rows = []; identities = set()
            for fold in split['folds']:
                fid = fold['fold_id']; old = source/f'fold_{fid:02d}'/mode
                for subject in split['original_subject_order']:
                    assert (source/'window_index'/subject/'windows.csv').read_bytes() == (index/subject/'windows.csv').read_bytes()
                state = load_checkpoint(old/'best.pt')
                expected = next(r for r in summary['folds'] if r['fold']==fid and r['mode']==mode)
                assert state['mode'] == mode and state['epoch'] == expected['best_epoch']
                model = build_model(config, 14); model.load_state_dict(state['model'])
                scaler = restore_scaler(old/'normalization.json', fold)
                train = PreprocessedWindowDataset(index, fold['train'], config['data']['channels'], mode)
                weights = class_weights(train.labels)
                val = PreprocessedWindowDataset(index, fold['validation'], config['data']['channels'], mode); val.scaler = scaler
                _, prob, ids = evaluate(model, DataLoader(val,batch_size=32), torch.device('cpu'), weights)
                by = metrics_by_subject(val.labels[ids], prob.argmax(1), [val.rows[i]['subject_id'] for i in ids])
                np.testing.assert_allclose(np.mean([m['macro_f1'] for m in by.values()]), expected['best_val_subject_f1'], rtol=0, atol=1e-12)
                saved = list(csv.DictReader((old/'validation_predictions.csv').open()))
                assert [(r['subject_id'], r['window_id']) for r in saved] == [(val.rows[i]['subject_id'], val.rows[i]['window_id']) for i in ids]
                np.testing.assert_allclose(prob, [[float(r[f'p_{j}']) for j in range(3)] for r in saved], rtol=0, atol=0)
                test = PreprocessedWindowDataset(index, fold['test'], config['data']['channels'], mode); test.scaler = scaler
                metrics, probabilities, indices = evaluate(model, DataLoader(test,batch_size=32), torch.device('cpu'), weights)
                folder = output/name/mode/f'fold_{fid:02d}'
                export_test(folder, test, metrics, probabilities, indices, fid)
                confusion_plot(metrics['confusion_matrix'], folder/'confusion_matrix.png')
                with (folder/'subject_metrics.csv').open() as stream: subject_rows.extend(list(csv.DictReader(stream)))
                with (folder/'predictions.csv').open() as stream: preds=list(csv.DictReader(stream)); pooled_rows.extend(preds)
                for r in preds:
                    key=(r['subject_id'],r['window_id']); assert key not in identities; identities.add(key)
                check={'model':name,'mode':mode,'fold':fid,'passed':True,'best_epoch':state['epoch'],
                       'validation_probabilities_replayed_exactly':True,'test_windows':len(test),'test_subjects':fold['test']}
                checks.append(check)
                write_json(output/'live.json',{'status':'running','completed_folds':checks,'training_performed':False})
                print(f'{name} {mode} fold={fid} Test/person F1={np.mean([float(r["macro_f1"]) for r in subject_rows[-3:]]):.4f}',flush=True)
            assert len(subject_rows)==15 and len(set(r['subject_id'] for r in subject_rows))==15 and len(identities)==2134
            aggregate={k:float(np.mean([float(r[k]) for r in subject_rows])) for k in ['macro_f1','balanced_accuracy','accuracy']}
            aggregate['std_macro_f1_across_subjects']=float(np.std([float(r['macro_f1']) for r in subject_rows],ddof=1))
            all_results[f'{name}_{mode}']=aggregate
            write_csv(output/name/mode/'subject_metrics.csv',subject_rows)
            write_csv(output/name/mode/'oof_predictions.csv',pooled_rows)
    assert all(file_hash(Path(p))==h for p,h in protected.items())
    historical = read_json(ROOT/'results/common_rate64_gru_grouped5fold_v1/20261124T100241Z_204cb671/cnn_gru_comparison.json')['models']
    write_json(output/'summary.json',{'status':'completed','primary_unit':'person','models':all_results,'historical_test':historical,
               'training_performed':False,'dataset_fingerprint':audit['dataset_fingerprint']})
    write_json(output/'verification.json',{'passed':True,'checks':checks,'all_15_test_subjects_once_per_variant':True,
        'all_2134_windows_once_per_variant':True,'source_runs_preserved':True,'training_performed':False})
    lines=['# Frozen CNN/GRU Test evaluation after window normalization','',
           'Metric: Test Macro-F1 computed per person then averaged over 15 people. No training, tuning, or checkpoint selection on Test.','',
           '| Model | Historical Test | Current control Test | Window z-score Test | Change vs current control |',
           '|---|---:|---:|---:|---:|']
    for name in sources:
        a=all_results[f'{name}_control']['macro_f1']; b=all_results[f'{name}_window_zscore']['macro_f1']
        lines.append(f'| {name} | {historical[name]["macro_f1"]:.4f} | {a:.4f} | {b:.4f} | {b-a:+.4f} |')
    lines+=['','The current control and window-z-score checkpoints were trained under the same local budget and split. Historical CNN training may differ numerically; both comparisons are shown.',
            'All 20 Validation checkpoint scores and probabilities replayed before Test scoring. Each variant covers 15 people and 2,134 Test windows exactly once. Input checkpoints and scalers are unchanged.',
            'Historical Test results were already inspected before this preprocessing exploration. Treat this as exploratory same-dataset evidence, not a fresh independent confirmation. Do not keep adjusting to these Test scores.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4)); x=np.arange(2)
    for offset,mode,label in [(-.18,'control','Current control'),(.18,'window_zscore','Window z-score')]:
        vals=[all_results[f'{name}_{mode}']['macro_f1'] for name in sources]
        bars=ax.bar(x+offset,vals,.36,label=label); ax.bar_label(bars,fmt='%.4f',padding=3)
    ax.set(xticks=x,xticklabels=list(sources),ylim=(0,1),ylabel='Test Macro-F1 (mean per person)',title='Frozen CNN/GRU checkpoints | 15 held-out people')
    ax.legend();fig.tight_layout();fig.savefig(output/'test_comparison.png',dpi=160);plt.close(fig)
    write_json(output/'live.json',{'status':'completed','models':all_results,'training_performed':False})
    print('REPORT='+str(output),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--disable-optional-pyarrow',action='store_true');args=parser.parse_args()
    if args.disable_optional_pyarrow:sys.modules['pyarrow']=None
    run(args.output)

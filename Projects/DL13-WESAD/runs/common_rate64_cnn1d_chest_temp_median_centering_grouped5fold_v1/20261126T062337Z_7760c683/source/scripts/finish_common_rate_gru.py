"""Wait for the fixed GRU run, verify/replay predictions, then compare CNN/GRU.

Never changes training, resumes a run, tunes on Test, or edits journal Excel.
"""
import argparse
import csv
import json
import logging
from pathlib import Path
import sys
import time

GRU = 'common_rate64_gru_grouped5fold_v1'
CNN = 'common_rate64_cnn1d_grouped5fold_v1'
CNN_GROUP = '20261123T072126Z_2c81daac'


def read_rows(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def paired_metrics(cnn, gru):
    import numpy as np
    from src.evaluation.metrics import metrics_by_subject
    def indexed(rows):
        result = {r['window_id']: r for r in rows}
        if len(result) != len(rows):
            raise ValueError('Duplicate OOF window ID.')
        return result
    a, b = indexed(cnn), indexed(gru)
    if set(a) != set(b) or not a:
        raise ValueError('CNN/GRU window IDs differ or are empty.')
    for key in a:
        for field in ('subject_id', 'true_label', 'fold_id', 'start_time', 'end_time'):
            x, y = a[key][field], b[key][field]
            equal = x == y if field == 'subject_id' else float(x) == float(y)
            if not equal:
                raise ValueError(f'CNN/GRU window mismatch: {key}, {field}')
    scores = {}
    for name, rows in [('CNN', cnn), ('GRU', gru)]:
        by_subject = metrics_by_subject([int(r['true_label']) for r in rows],
                                       [int(r['predicted_label']) for r in rows],
                                       [r['subject_id'] for r in rows])
        scores[name] = by_subject
    paired = [{'subject': s, 'cnn_macro_f1': scores['CNN'][s]['macro_f1'],
               'gru_macro_f1': scores['GRU'][s]['macro_f1'],
               'difference_gru_minus_cnn': scores['GRU'][s]['macro_f1']-scores['CNN'][s]['macro_f1']}
              for s in sorted(scores['CNN'], key=lambda s: int(s[1:]))]
    summary = {name: {metric: float(np.mean([m[metric] for m in by_subject.values()]))
                      for metric in ('macro_f1', 'balanced_accuracy', 'accuracy')}
               for name, by_subject in scores.items()}
    return paired, summary


def replay(group, config, log):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from src.data.dataset import WindowDataset
    from src.models import build_model
    from src.training.checkpoint import load_checkpoint
    from src.utils.tracking import read_json
    torch.set_num_threads(config['runtime']['torch_threads'])
    outcomes = []
    for fold in read_json(group/'split_manifest.json')['folds']:
        folder = group/f"seed_42/fold_{fold['fold_id']:02d}"
        norm = read_json(folder/'normalization.json')
        from src.data.normalization import ChannelScaler
        scaler = ChannelScaler()
        scaler.mean, scaler.std = np.asarray(norm['mean'], dtype=np.float32), np.asarray(norm['std'], dtype=np.float32)
        data = WindowDataset(group/'window_index', fold['test'], scaler)
        model = build_model(config, data.input_channels)
        model.load_state_dict(load_checkpoint(folder/'checkpoints/best.pt')['model'])
        model.eval()
        actual = {}
        with torch.inference_mode():
            for x, _, indices in DataLoader(data, batch_size=config['training']['batch_size'], shuffle=False):
                probability = model(x).softmax(dim=1).numpy()
                for index, p in zip(indices.tolist(), probability):
                    actual[data.rows[index]['window_id']] = p
        maximum = 0.
        saved = read_rows(folder/'metrics/predictions.csv')
        if len(saved) != len(actual):
            raise ValueError('Replay window count differs.')
        for row in saved:
            expected = np.asarray([float(row['p_'+c]) for c in ('baseline','stress','amusement')])
            got = actual.pop(row['window_id'])
            np.testing.assert_allclose(got, expected, atol=1e-7, rtol=1e-6)
            if int(got.argmax()) != int(row['predicted_label']):
                raise ValueError('Replay predicted class differs.')
            maximum = max(maximum, float(np.max(np.abs(got-expected))))
        if actual:
            raise ValueError('Unexpected replay windows.')
        outcomes.append({'fold': fold['fold_id'], 'windows':len(saved), 'max_probability_difference':maximum, 'passed':True})
        log.info('Replay fold %s PASS (%s windows)', fold['fold_id'], len(saved))
    return outcomes


def finalize(root, group_id, review, log):
    from scripts.verify_common_rate import verify
    from src.utils.tracking import read_json, write_json
    group = root/'runs'/GRU/group_id
    result = root/'results'/GRU/group_id
    verify(group_id)
    config = read_json(group/'experiment_manifest.json')['effective_config']
    replayed = replay(group, config, log)
    cnn_group = root/'runs'/CNN/CNN_GROUP
    cnn_result = root/'results'/CNN/CNN_GROUP
    prior = read_json(cnn_group/'experiment_manifest.json')
    current = read_json(group/'experiment_manifest.json')
    for key in ('dataset_fingerprint','manifest_checksum','cache_keys'):
        if prior[key] != current[key]:
            raise ValueError(f'CNN/GRU differ: {key}')
    for key in ('data','split','training','selection'):
        if prior['effective_config'][key] != config[key]:
            raise ValueError(f'CNN/GRU protocol differs: {key}')
    paired, summary = paired_metrics(read_rows(cnn_result/'oof_predictions.csv'),read_rows(result/'oof_predictions.csv'))
    write_json(result/'checkpoint_prediction_replay.json', {'passed':True,'folds':replayed})
    write_json(result/'cnn_gru_comparison.json', {'seed':42,'primary_unit':'person','models':summary,
               'subjects':paired,'cnn_group':CNN_GROUP,'gru_group':group_id})
    with (result/'cnn_gru_subject_comparison.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(paired[0])); writer.writeheader();writer.writerows(paired)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    x=np.arange(len(paired)); fig,ax=plt.subplots(figsize=(13,5),layout='constrained')
    ax.bar(x-.19,[r['cnn_macro_f1'] for r in paired],width=.38,label='CNN-1D')
    ax.bar(x+.19,[r['gru_macro_f1'] for r in paired],width=.38,label='GRU')
    ax.set(xticks=x,xticklabels=[r['subject'] for r in paired],ylim=(0,1),ylabel='Test Macro-F1 (three classes)',
           title='WESAD 14 channels | grouped 5-fold | seed 42')
    ax.legend();ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    for ext in ('png','svg'):fig.savefig(result/f'cnn_gru_subject_macro_f1.{ext}',dpi=160)
    plt.close(fig)
    text = '# CNN và GRU trên 14 kênh — seed 42\n\n'
    text += 'Cùng 2.134 cửa sổ, split 9/3/3, chuẩn hóa Train-only và tiêu chí chọn checkpoint bằng Validation Macro-F1 trung bình theo người. Đã hậu kiểm đủ 5 folds và phát lại dự đoán từ checkpoint GRU.\n\n'
    text += '| Mô hình | Test Macro-F1 trung bình 15 người | Balanced Accuracy theo người | Accuracy theo người |\n|---|---:|---:|---:|\n'
    for name,m in summary.items():text+=f"| {name} | {m['macro_f1']:.4f} | {m['balanced_accuracy']:.4f} | {m['accuracy']:.4f} |\n"
    text += '\nĐây là kết quả một seed; chưa kết luận mô hình nào luôn tốt hơn. CNN-GRU và ablation chưa có kết quả. Không dùng điểm Test để chọn lại cấu hình đã chạy.\n\n'
    text += '[Từng người](cnn_gru_subject_comparison.csv) · [Biểu đồ](cnn_gru_subject_macro_f1.png) · [Hậu kiểm](verification.json) · [Phát lại dự đoán](checkpoint_prediction_replay.json)\n'
    (result/'cnn_gru_comparison.md').write_text(text,encoding='utf-8')
    write_json(review/'status.json',{'status':'completed','group_id':group_id,'verification_passed':True,
               'prediction_replay_passed':True,'comparison_report':str(result/'cnn_gru_comparison.md')})
    log.info('COMPLETE: %s',result/'cnn_gru_comparison.md')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group-id',required=True)
    parser.add_argument('--wait',action='store_true')
    parser.add_argument('--disable-optional-pyarrow',action='store_true')
    args=parser.parse_args()
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.group_id):parser.error('Invalid group ID.')
    if args.disable_optional_pyarrow:sys.modules['pyarrow']=None
    from src.utils.config import ROOT
    from src.utils.tracking import read_json,write_json
    review=ROOT/'results/common_rate_review'/args.group_id
    review.mkdir(parents=True,exist_ok=True)
    if (review/'status.json').exists() and read_json(review/'status.json').get('status')=='completed':
        print('Review already completed; existing results preserved.');return
    log=logging.getLogger('gru_review');log.setLevel(logging.INFO)
    handler=logging.FileHandler(review/'review.log',encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'));log.addHandler(handler)
    try:
        log.info('Started; optional pyarrow disabled=%s',args.disable_optional_pyarrow)
        group=ROOT/'runs'/GRU/args.group_id
        deadline=time.monotonic()+6*3600
        while True:
            state=read_json(group/'experiment_manifest.json')
            if state['status'] in ('failed','interrupted'):raise RuntimeError('Training stopped: '+state['status'])
            if state['status']=='completed' and not (group/'.lock').exists():break
            if not args.wait:raise RuntimeError('Training is not complete; use --wait.')
            if time.monotonic()>deadline:raise TimeoutError('Timed out waiting for training.')
            write_json(review/'status.json',{'status':'waiting_for_training','group_id':args.group_id,
                       'training_status':state['status'],'comparison_created':False})
            time.sleep(10)
        finalize(ROOT,args.group_id,review,log)
    except Exception as exc:
        log.exception('Review failed.')
        write_json(review/'status.json',{'status':'failed','error':repr(exc),'group_id':args.group_id})
        raise
    finally:log.removeHandler(handler);handler.close()


if __name__=='__main__':main()

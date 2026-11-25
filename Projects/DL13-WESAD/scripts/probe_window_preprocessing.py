"""Paired CNN/GRU comparisons of window-local normalization; no Test scoring."""
import argparse
import copy
import csv
import logging
from pathlib import Path
import sys
import time


def run(output, config_path=None):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import yaml
    from src.data.common_rate import build_index,prepare_index
    from src.data.window_preprocessing import PreprocessedWindowDataset
    from src.data.normalization import ChannelScaler
    from src.data.splits import load_manifest
    from src.models import build_model
    from src.training.optimizer import make_optimizer,class_weights
    from src.training.checkpoint import save_checkpoint,load_checkpoint
    from src.training.callbacks import SelectionState
    from src.evaluation.evaluator import evaluate
    from src.evaluation.metrics import metrics_by_subject
    from src.utils.config import ROOT,load_config
    from src.utils.reproducibility import seed_everything
    from src.utils.tracking import read_json,write_json,file_hash,digest
    output.mkdir(parents=True,exist_ok=False)
    handler=logging.FileHandler(output/'run.log',encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    log=logging.getLogger('preprocessing_probe');log.setLevel(logging.INFO);log.addHandler(handler)
    path=ROOT/(config_path or 'experiments/preprocessing_window_zscore/config.yaml')
    cfg=yaml.safe_load(path.read_text(encoding='utf-8'))
    cfg_ref=load_config(cfg['experiment']['reference'])
    config=copy.deepcopy(cfg_ref)
    model_name=config['experiment']['model']
    config['training']['max_epochs']=cfg['experiment']['epochs_per_variant']
    use_early_stopping=cfg['experiment'].get('early_stopping',False)
    max_epochs=config['training']['max_epochs']
    writer=None
    split=load_manifest(ROOT/config['split']['manifest'])
    source=ROOT/config['data']['processed_dir']
    try:
        audit,rows=build_index(source,config['data'],split['original_subject_order'])
        reference=read_json(ROOT/'runs/common_rate64_cnn1d_grouped5fold_v1/20261123T072126Z_2c81daac/experiment_manifest.json')
        if audit['dataset_fingerprint']!=reference['dataset_fingerprint'] or digest(split)!=reference['manifest_checksum']:
            raise ValueError('Source/split differ from reference.')
        write_json(output/'config.json',{'probe':cfg,'effective_model':config,'model':model_name,
                   **({'effective_cnn':config} if model_name=='cnn1d' else {}),
                   'dataset_fingerprint':audit['dataset_fingerprint'],'test_evaluated':False})
        write_json(output/'source_checksums.json',{str(p.relative_to(ROOT)):file_hash(p) for p in
                   [Path(__file__),ROOT/'src/data/window_preprocessing.py',path,
                    ROOT/cfg['experiment']['reference'],ROOT/'src/models/gru.py',ROOT/'src/models/cnn.py',
                    ROOT/'src/data/normalization.py',ROOT/'src/training/callbacks.py']})
        index=output/'window_index';prepare_index(index,audit,rows)
        summary=[];verifications=[]
        for fold in split['folds']:
            if fold['fold_id'] not in cfg['experiment']['fold_ids']:continue
            for mode in cfg['experiment']['variants']:
                seed_everything(cfg['experiment']['seed'],threads=config['runtime']['torch_threads'])
                folder=output/f'fold_{fold["fold_id"]:02d}'/mode;folder.mkdir(parents=True)
                if cfg['experiment'].get('tensorboard',False):
                    from torch.utils.tensorboard import SummaryWriter
                    writer=SummaryWriter(str(folder/'tensorboard'))
                train=PreprocessedWindowDataset(index,fold['train'],config['data']['channels'],mode)
                val=PreprocessedWindowDataset(index,fold['validation'],config['data']['channels'],mode)
                # Fit on transformed Train only, never on held-out subjects.
                scaler=ChannelScaler().fit(train);scaler.save(folder/'normalization.json')
                train.scaler=val.scaler=scaler
                assert set(scaler.subjects)==set(fold['train']) and not set(scaler.subjects)&set(fold['validation']+fold['test'])
                tc=config['training'];model=build_model(config,14)
                optimizer=make_optimizer(model,tc);weights=class_weights(train.labels)
                criterion=torch.nn.CrossEntropyLoss(weight=weights)
                train_loader=DataLoader(train,batch_size=tc['batch_size'],shuffle=True)
                eval_train=DataLoader(train,batch_size=tc['batch_size'],shuffle=False)
                eval_val=DataLoader(val,batch_size=tc['batch_size'],shuffle=False)
                best=-1.;best_loss=float('inf');best_epoch=0;history=[];started=time.perf_counter()
                selection=SelectionState(tc['early_stopping_patience'],tc['early_stopping_min_delta'])
                step=0
                batch_file=(folder/'batches.csv').open('w',newline='',encoding='utf-8')
                batch_csv=csv.DictWriter(batch_file,fieldnames=['epoch','step','loss','gradient_norm'])
                batch_csv.writeheader()
                for epoch in range(1,tc['max_epochs']+1):
                    model.train();online=[]
                    for x,y,_ in train_loader:
                        optimizer.zero_grad(set_to_none=True);loss=criterion(model(x),y)
                        if not torch.isfinite(loss):raise ValueError('Nonfinite Train loss.')
                        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),tc['gradient_clip_norm'],error_if_nonfinite=True)
                        optimizer.step();online.append(float(loss.detach()))
                        step+=1
                        batch_csv.writerow({'epoch':epoch,'step':step,'loss':float(loss.detach()),'gradient_norm':float(norm)})
                        batch_file.flush()
                        if writer:writer.add_scalar('batch/loss',float(loss.detach()),step)
                    tr,_,_=evaluate(model,eval_train,torch.device('cpu'),weights)
                    vm,prob,idx=evaluate(model,eval_val,torch.device('cpu'),weights)
                    persons=metrics_by_subject(val.labels[idx],prob.argmax(1),[val.rows[i]['subject_id'] for i in idx])
                    score=float(np.mean([m['macro_f1'] for m in persons.values()]))
                    row={'epoch':epoch,'train_loss':tr['loss'],'train_macro_f1':tr['macro_f1'],
                         'val_loss':vm['loss'],'val_macro_f1':vm['macro_f1'],'val_subject_macro_f1':score}
                    history.append(row)
                    better,stop=selection.update(score,vm['loss'],epoch)
                    if better:
                        best,best_loss,best_epoch=score,vm['loss'],epoch
                        save_checkpoint(folder/'best.pt',{'model':model.state_dict(),'epoch':epoch,'mode':mode,
                                        'model_name':model_name,'purpose':cfg['experiment']['purpose']})
                    log.info('Fold %s %s %s',fold['fold_id'],mode,row)
                    print(f'fold={fold["fold_id"]} {mode} epoch={epoch}/{max_epochs} Val/person={score:.4f}',flush=True)
                    if writer:
                        for tag,key in [('train/loss','train_loss'),('train/macro_f1','train_macro_f1'),('val/loss','val_loss'),('val/macro_f1','val_macro_f1'),('val/subject_macro_f1','val_subject_macro_f1')]:
                            writer.add_scalar(tag,row[key],epoch)
                        writer.flush()
                    with (folder/'epochs.csv').open('w',newline='',encoding='utf-8') as f:
                        w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(history)
                    write_json(output/'live.json',{'status':'running','fold':fold['fold_id'],'mode':mode,'epoch':epoch,'max_epochs':max_epochs,'best_epoch':best_epoch,'best_validation_f1':best,'test_evaluated':False})
                    if use_early_stopping and stop:break
                batch_file.close()
                if writer:writer.close();writer=None
                model.load_state_dict(load_checkpoint(folder/'best.pt')['model'])
                vm,prob,idx=evaluate(model,eval_val,torch.device('cpu'),weights)
                by=metrics_by_subject(val.labels[idx],prob.argmax(1),[val.rows[i]['subject_id'] for i in idx])
                verified=float(np.mean([m['macro_f1'] for m in by.values()]))
                np.testing.assert_allclose(verified,best,rtol=0,atol=0)
                preds=[{'subject_id':val.rows[i]['subject_id'],'window_id':val.rows[i]['window_id'],
                        'true_label':int(val.labels[i]),'predicted_label':int(p.argmax()),
                        **{f'p_{j}':float(p[j]) for j in range(3)}} for i,p in zip(idx,prob)]
                with (folder/'validation_predictions.csv').open('w',newline='',encoding='utf-8') as f:
                    w=csv.DictWriter(f,fieldnames=list(preds[0]));w.writeheader();w.writerows(preds)
                summary.append({'fold':fold['fold_id'],'mode':mode,'best_epoch':best_epoch,
                                'best_val_subject_f1':best,'best_val_loss':best_loss,'epochs':len(history),
                                'seconds':time.perf_counter()-started})
                check={'fold':fold['fold_id'],'mode':mode,'passed':True,'scaler_fit_subjects':scaler.subjects,
                       'validation_subjects':val.subjects,'test_evaluated':False,'checkpoint_score_replayed':True,
                       'input_shape':list(train.raw_window(fold['train'][0],0).shape)}
                verifications.append(check);write_json(folder/'verification.json',check)
                write_json(output/'progress.json',{'completed_variants':summary,'test_evaluated':False})
        means={mode:float(np.mean([r['best_val_subject_f1'] for r in summary if r['mode']==mode])) for mode in cfg['experiment']['variants']}
        write_json(output/'summary.json',{'status':'completed','model':model_name,'folds':summary,'means':means,
                   'delta_window_zscore_minus_control':means['window_zscore']-means['control'],
                   'epochs_per_variant':max_epochs,'early_stopping':use_early_stopping,'test_evaluated':False})
        write_json(output/'verification.json',{'passed':True,'variants':verifications,'test_evaluated':False})
        text='# CNN — thăm dò chuẩn hóa theo cửa sổ\n\nCùng 14 kênh, nguồn, split 9/3/3, window 30s/stride 15s và seed 42. Hai nhánh có cùng ngân sách 5 epochs mỗi fold. Nhánh mới chuẩn hóa riêng từng cửa sổ cho 10 kênh dao động; giữ EDA và nhiệt độ ở thang gốc trước scaler Train-only. Không đánh giá Test.\n\n'
        text+='| Nhánh | Validation Macro-F1 trung bình theo người |\n|---|---:|\n'
        for mode,value in means.items():text+=f'| {mode} | {value:.4f} |\n'
        text+='\nThử ngắn 5 epochs chưa đủ kết luận hiệu quả khi huấn luyện đầy đủ hoặc trên Test. Không so điểm này trực tiếp với Test của CNN/GRU đã chạy trước.\n'
        (output/'report.md').write_text(text,encoding='utf-8')
        if use_early_stopping:
            lines=[f'# {model_name.upper()} preprocessing: full-budget Train/Validation comparison','',
                   f'Maximum {max_epochs} epochs per fold, patience {tc["early_stopping_patience"]}, min_delta {tc["early_stopping_min_delta"]}. Same subjects, seed, model and optimizer. No Test evaluation.','',
                   '| Fold | Control | Window z-score | Difference | Epochs control/new |','|---|---:|---:|---:|---:|']
            for fid in cfg['experiment']['fold_ids']:
                a,b=[next(r for r in summary if r['fold']==fid and r['mode']==m) for m in ('control','window_zscore')]
                lines.append(f'| {fid} | {a["best_val_subject_f1"]:.4f} | {b["best_val_subject_f1"]:.4f} | {b["best_val_subject_f1"]-a["best_val_subject_f1"]:+.4f} | {a["epochs"]}/{b["epochs"]} |')
            lines+=['',f'Mean per-person Validation Macro-F1: {means["control"]:.4f} -> {means["window_zscore"]:.4f}.','',
                    'Exploratory comparison with one training seed. Validation was used for selection; this is not independent Test performance. Source data and prior CNN/GRU runs are preserved.']
            (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        write_json(output/'live.json',{'status':'completed','means':means,'test_evaluated':False})
        log.info('COMPLETE %s',means);print('REPORT='+str(output),flush=True)
        return means
    except Exception:
        write_json(output/'live.json',{'status':'failed','test_evaluated':False})
        log.exception('Probe failed.');raise
    finally:
        if writer:writer.close()
        log.removeHandler(handler);handler.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path)
    p.add_argument('--config',type=Path)
    p.add_argument('--disable-optional-pyarrow',action='store_true');args=p.parse_args()
    if args.disable_optional_pyarrow:sys.modules['pyarrow']=None
    from src.utils.config import ROOT
    from src.utils.tracking import identifier
    run(args.output or ROOT/'results/preprocessing_window_zscore'/identifier(),args.config)


if __name__=='__main__':main()

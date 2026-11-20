from pathlib import Path
from collections import Counter
import csv, hashlib, io, json, math
B=Path(__file__).parent
C=json.loads((B/'corpus.json').read_text(encoding='utf-8'))
I={r['path']:r for r in json.loads((B/'inventory.json').read_text(encoding='utf-8'))}
def j(p): return json.loads(C[p])
def rows(p): return list(csv.DictReader(io.StringIO(C[p])))
def metrics(rs):
    cm=[[0]*3 for _ in range(3)]
    for r in rs: cm[int(r['true_label'])][int(r['predicted_label'])]+=1
    recall=[]; f1=[]
    for k in range(3):
        tp=cm[k][k]; support=sum(cm[k]); predicted=sum(r[k] for r in cm)
        recall.append(tp/support if support else 0)
        f1.append(2*tp/(support+predicted) if support+predicted else 0)
    return {'accuracy':sum(cm[k][k] for k in range(3))/len(rs),'macro_f1':sum(f1)/3,'balanced_accuracy':sum(recall)/3,'confusion_matrix':cm}
checks={'method':'Pure-Python text-only verification; no project imports, raw/binary loads or training', 'groups':[], 'source_snapshots':[], 'cache_metadata':{}, 'split':{}}
split=j('splits/grouped_5fold_seed42.json')
canonical=json.dumps(split,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
sha=hashlib.sha256(canonical).hexdigest()
assert sha==C['splits/grouped_5fold_seed42.sha256'].strip()
for f in split['folds']:
    a,b,c=map(set,(f['train'],f['validation'],f['test']))
    assert (len(a),len(b),len(c))==(9,3,3) and not(a&b or a&c or b&c)
checks['split']={'canonical_sha256':sha,'disjoint_roles':True}
for p in C:
    if p.endswith('/summary.json'):
        base=p.rsplit('/',1)[0]; summ=j(p); rs=rows(base+'/oof_predictions.csv')
        keys=[(r['training_seed'],r['subject_id'],r['window_id']) for r in rs]
        assert len(keys)==len(set(keys))==2136
        assert len({r['subject_id'] for r in rs})==15
        for r in rs:
            probs=[float(r['p_'+k]) for k in ['baseline','stress','amusement']]
            assert all(math.isfinite(x) and x>=0 for x in probs) and abs(sum(probs)-1)<1e-5
            assert max(range(3),key=probs.__getitem__)==int(r['predicted_label'])
            f=split['folds'][int(r['fold_id'])-1]
            assert r['subject_id'] in f['test']
        m=metrics(rs); stored=summ['per_seed']['42']
        for k in ['accuracy','macro_f1','balanced_accuracy']:
            assert abs(m[k]-stored['pooled_oof'][k])<1e-12
        sr=[metrics([r for r in rs if r['subject_id']==s]) for s in sorted({r['subject_id'] for r in rs})]
        for k in ['accuracy','macro_f1','balanced_accuracy']:
            assert abs(sum(x[k] for x in sr)/15-stored['subject_level'][k]['mean'])<1e-12
        checks['groups'].append({'path':base,'oof_windows':len(rs),'subjects':15,'pooled_oof':m,'subject_macro_f1':stored['subject_level']['macro_f1']['mean'],'text_metrics_match':True})
    if p.endswith('/source_checksums.json'):
        prefix=p.rsplit('/',1)[0]+'/source/'
        saved=j(p)
        for rel,h in saved.items(): assert I[prefix+rel]['sha256']==h, (p,rel)
        checks['source_snapshots'].append({'path':p,'files':len(saved),'verified_text_hashes':True})
total=0
for p in C:
    if p.startswith('data/processed/') and p.endswith('/metadata.json'):
        m=j(p); rs=rows(p.rsplit('/',1)[0]+'/windows.csv')
        assert len(rs)==m['shape'][0] and m['shape'][1:]==[5,960]
        assert len({r['window_id'] for r in rs})==len(rs)
        assert {str(k):sum(int(r['label'])==k for r in rs) for k in range(3)}==m['label_distribution']
        assert all(int(r['original_label'])==int(r['label'])+1 and abs(float(r['end_time'])-float(r['start_time'])-30)<1e-8 for r in rs)
        total+=len(rs)
checks['cache_metadata']={'subjects':15,'windows':total,'rows_shapes_labels_match':True,'array_contents_not_read':True}
(B/'text_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(checks,ensure_ascii=False,indent=2))

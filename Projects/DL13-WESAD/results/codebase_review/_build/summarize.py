from pathlib import Path
import csv, difflib, io, json, xml.etree.ElementTree as ET
B = Path(__file__).parent
C = json.loads((B/'corpus.json').read_text(encoding='utf-8'))
def show(p, limit=8000):
    print('\nFILE', p, '\n', C[p][:limit])
for p in C:
    if p.endswith(('summary.json','verification.json','audit.json','baseline_registry.json')) and ('source/' not in p):
        v = json.loads(C[p])
        if p.endswith('baseline_registry.json'):
            print('\nREGISTRY', list(v)); print(json.dumps({k:x for k,x in v.items() if not isinstance(x,(dict,list))},ensure_ascii=False)); continue
        show(p, 3500 if 'summary.json' in p else 6500)
print('\nXML HISTORY')
for p in C:
    if p.endswith('.xml'):
        t=ET.fromstring(C[p]); print(p,[x.attrib for x in t.iter('testsuite')]); print('SKIPS', [(x.attrib.get('name'),x.find('skipped').attrib) for x in t.iter('testcase') if x.find('skipped') is not None])
print('\nSNAPSHOT DIFFS')
for p in C:
    if '/source/' in p and p.endswith(('.py','.yaml')):
        current = p.split('/source/')[1]
        if current not in C: print('SNAPSHOT ONLY',p); continue
        if C[current] != C[p]:
            diff=''.join(difflib.unified_diff(C[p].splitlines(True),C[current].splitlines(True),n=1))
            print(p,'=> CURRENT',len(diff), 'chars')
            if len(diff)<8000: print(diff)
print('\nDATA SAMPLES')
v=json.loads(C['data/interim/inspection.json']); print('inspection keys',list(v)); print('status',v.get('status'), 'subjects',len(v.get('subjects',[]))); print(json.dumps(v['subjects'][0],ensure_ascii=False,indent=2))
show('data/processed/c369cd27d24abb09/S2/metadata.json',6000)
show('results/preprocessing_review/20261117T072651Z_efe6ab76/README.md')
for p in C:
    if p.endswith(('comparison.md','verification.json')) and 'experiment_e2/' in p: show(p)

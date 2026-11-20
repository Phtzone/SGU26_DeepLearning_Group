from pathlib import Path
from collections import Counter
import hashlib, json, re
from docx import Document

B=Path(__file__).parent; ROOT=B.parents[2]
stats=json.loads((B/'document_stats.json').read_text(encoding='utf-8'))
inventory=json.loads((B/'inventory.json').read_text(encoding='utf-8'))
symbols=json.loads((B/'symbols.json').read_text(encoding='utf-8'))
doc=Document(stats['output'])
text='\n'.join(p.text for p in doc.paragraphs)
headings=[p.text for p in doc.paragraphs if p.style.name=='Heading 1' and p.text.startswith('CHƯƠNG ')]
assert len(headings)==17
for name in ['A. Mục tiêu học tập','B. Khái niệm nền tảng','C. Liên hệ codebase','D. Phân tích hoạt động','E. Ví dụ minh họa','F. Kết luận chương','G. Câu hỏi tự kiểm tra']: assert sum(p.text==name and p.style.name=='Heading 2' for p in doc.paragraphs)==17,name
assert len(doc.tables)==stats['tables'] and len(doc.inline_shapes)==7
effective_size=doc.styles['Normal'].font.size
if effective_size is None:
    effective_pt=float(doc.styles.element.xpath('./w:docDefaults/w:rPrDefault/w:rPr/w:sz')[0].get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val'))/2
else:
    effective_pt=effective_size.pt
assert effective_pt==11
assert abs(doc.sections[0].page_width.cm-21)<.01 and abs(doc.sections[0].page_height.cm-29.7)<.01
assert 'TOC ' in doc.element.xml and 'PAGE' in doc.sections[0].footer._element.xml
for x in stats['citations']:
    assert (ROOT/x['path']).is_file(),x
    if x['symbol']:
        assert any(s['name']==x['symbol'] and s['line']==x['line'] for s in symbols[x['path']]['symbols']),x
alltext=text+'\n'+'\n'.join(cell.text for t in doc.tables for row in t.rows for cell in row.cells)
assert '\ufffd' not in alltext
assert 'C:\\Users\\' not in alltext and 'C:/Users/' not in alltext
changed=[]; text_checked=0; stat_checked=0
for r in inventory:
    path=ROOT/r['path']
    if not path.is_file(): changed.append({'path':r['path'],'reason':'missing'}); continue
    s=path.stat(); stat_checked+=1
    if (s.st_size,s.st_mtime_ns)!=(r['size'],r['mtime_ns']): changed.append({'path':r['path'],'reason':'stat_changed'})
    if 'sha256' in r:
        text_checked+=1
        if hashlib.sha256(path.read_bytes()).hexdigest()!=r['sha256']: changed.append({'path':r['path'],'reason':'text_hash_changed'})
assert not changed,changed
from lxml import etree
app=next(p for p in doc.part.package.parts if str(p.partname)=='/docProps/app.xml')
props=etree.fromstring(app.blob); pages=int(props.find('{http://schemas.openxmlformats.org/officeDocument/2006/extended-properties}Pages').text)
toc_populated=any(p.style.name.lower().startswith('toc') for p in doc.paragraphs)
assert pages==95 and toc_populated
result={'docx_valid_reopened':True,'microsoft_word_open_save_verified':True,'word_saved_page_count':pages,'TOC_populated_by_Word':toc_populated,'PDF_export_completed':False,'visual_check_limit':'Both Word PDF export attempts stalled; no complete visual page review claimed','chapters':len(headings),'tables':len(doc.tables),'figures':len(doc.inline_shapes),'citations_valid':len(stats['citations']),'A_to_G_sections_complete':True,'A4_portrait':True,'main_font_pt':11,'TOC_field':True,'page_field':True,'vietnamese_text_valid':True,'original_files_stat_checked':stat_checked,'original_text_hashes_checked':text_checked,'original_files_changed':changed,'binary_integrity_limit':'Only size and mtime compared; original binary content was not opened','new_pytest_or_training_executed':False,'zip_files_opened':False,'source_modified':False,'git_commit_or_push':False}
(B/'delivery_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))

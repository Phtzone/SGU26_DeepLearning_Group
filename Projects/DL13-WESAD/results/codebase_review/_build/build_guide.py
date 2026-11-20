from pathlib import Path
from collections import Counter
import ast, csv, io, json, math, re, textwrap
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

B=Path(__file__).parent
ROOT=B.parents[2]
OUT=B.parent/'DL13-WESAD_Complete_Codebase_Learning_Guide.docx'
C=json.loads((B/'corpus.json').read_text(encoding='utf-8'))
S=json.loads((B/'symbols.json').read_text(encoding='utf-8'))
SCOPE=json.loads((B/'scope.json').read_text(encoding='utf-8'))
CHECKS=json.loads((B/'text_checks.json').read_text(encoding='utf-8'))
E1='20261108T154303Z_474b38f9'; E2='20261117T075316Z_14889891'; CNN='20261115T101928Z_0ce52f72'; WD='20261115T101531Z_c2d7bd4b'
R1=f'results/experiment_e1/{E1}'; R2=f'results/experiment_e2/{E2}'
doc=Document(); tables=0; figures=0; chapters=0; citations=[]
sec=doc.sections[0]; sec.page_width=Cm(21); sec.page_height=Cm(29.7)
sec.top_margin=Cm(2); sec.bottom_margin=Cm(2); sec.left_margin=Cm(2.1); sec.right_margin=Cm(1.9)
sec.header_distance=Cm(.8); sec.footer_distance=Cm(.8)
for name in ['Normal','Body Text','List Bullet','List Number']:
    st=doc.styles[name]; st.font.name='Calibri'; st.font.size=Pt(11)
    st.paragraph_format.space_after=Pt(6); st.paragraph_format.line_spacing=1.12
for name,size,color in [('Title',32,'17365D'),('Subtitle',16,'52748E'),('Heading 1',20,'17365D'),('Heading 2',14,'245E82'),('Heading 3',12,'245E82')]:
    st=doc.styles[name]; st.font.name='Calibri'; st.font.size=Pt(size); st.font.color.rgb=RGBColor.from_string(color)
    st.paragraph_format.keep_with_next=True
    st.paragraph_format.space_before=Pt(12); st.paragraph_format.space_after=Pt(6)
doc.styles['Normal'].paragraph_format.widow_control=True
st=doc.styles.add_style('Mã nguồn',1); st.font.name='Consolas'; st.font.size=Pt(8.5)
st.paragraph_format.space_after=Pt(2); st.paragraph_format.line_spacing=1
st=doc.styles.add_style('Bằng chứng',1); st.font.name='Calibri'; st.font.size=Pt(9); st.font.color.rgb=RGBColor.from_string('5A6873')
st.paragraph_format.space_after=Pt(6)
st=doc.styles['Caption']; st.font.name='Calibri'; st.font.size=Pt(9); st.font.color.rgb=RGBColor.from_string('465C6C')

def field(paragraph,instruction):
    run=paragraph.add_run(); begin=OxmlElement('w:fldChar'); begin.set(qn('w:fldCharType'),'begin')
    text=OxmlElement('w:instrText'); text.set(qn('xml:space'),'preserve'); text.text=' '+instruction+' '
    sep=OxmlElement('w:fldChar'); sep.set(qn('w:fldCharType'),'separate')
    label=OxmlElement('w:t'); label.text='Cập nhật trường trong Word'
    end=OxmlElement('w:fldChar'); end.set(qn('w:fldCharType'),'end')
    for n in [begin,text,sep,label,end]: run._r.append(n)

header=sec.header.paragraphs[0]; header.add_run('DL13-WESAD  |  GIÁO TRÌNH ĐỌC HIỂU MÃ NGUỒN').font.size=Pt(8)
footer=sec.footer.paragraphs[0]; footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
footer.add_run('Rà soát ngày 18/11/2026  •  Trang ').font.size=Pt(8); field(footer,'PAGE')
settings=doc.settings.element; update=OxmlElement('w:updateFields'); update.set(qn('w:val'),'true'); settings.append(update)

def p(text='',style=None):
    para=doc.add_paragraph(style=style)
    for i,chunk in enumerate(re.split(r'(`[^`]+`)',str(text))):
        run=para.add_run(chunk[1:-1] if chunk.startswith('`') and chunk.endswith('`') else chunk)
        if chunk.startswith('`') and chunk.endswith('`'): run.font.name='Consolas'; run.font.size=Pt(9)
    return para
def h2(t): doc.add_heading(t,2)
def h3(t): doc.add_heading(t,3)
def bullets(items):
    for t in items: p(t,'List Bullet')
def ref(path,symbol=None):
    line=next((s['line'] for s in S.get(path,{}).get('symbols',[]) if s['name']==symbol),None)
    citations.append({'path':path,'symbol':symbol,'line':line})
    return path+(' — '+symbol+'()' if symbol else '')+(f' (dòng {line})' if line else '')
def source(*items): p('Bằng chứng: '+'; '.join(ref(*x) if isinstance(x,tuple) else ref(x) for x in items)+'.','Bằng chứng')
def table(title,headers,rows,widths=None):
    global tables
    tables+=1; p(f'Bảng {tables}. {title}','Caption')
    t=doc.add_table(rows=1, cols=len(headers)); t.style='Light Shading Accent 1'; t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.autofit=False
    widths=widths or [17/len(headers)]*len(headers)
    for col,w in zip(t.columns,widths): col.width=Cm(w)
    for cell,w,head in zip(t.rows[0].cells,widths,headers): cell.width=Cm(w); cell.text=str(head)
    repeat=OxmlElement('w:tblHeader'); t.rows[0]._tr.get_or_add_trPr().append(repeat)
    for row in rows:
        cells=t.add_row().cells
        for cell,w,value in zip(cells,widths,row): cell.width=Cm(w); cell.text=str(value)
    for row in t.rows:
        nosplit=OxmlElement('w:cantSplit'); row._tr.get_or_add_trPr().append(nosplit)
        for cell in row.cells:
            cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.TOP
            for para in cell.paragraphs:
                para.paragraph_format.space_after=Pt(4); para.paragraph_format.line_spacing=1.03
                for run in para.runs: run.font.size=Pt(9)
    p()
    return t
def code(text):
    for line in text.strip('\n').splitlines():
        # Soft wrap preserves code characters and prevents long console commands leaving the page.
        for part in textwrap.wrap(line,width=98,replace_whitespace=False,drop_whitespace=False,break_long_words=True,break_on_hyphens=False) or ['']:
            para=p(part,'Mã nguồn')
            shade=OxmlElement('w:shd'); shade.set(qn('w:fill'),'F1F4F7'); para._p.get_or_add_pPr().append(shade)
def excerpt(path,start,end):
    lines=C[path].splitlines(); code('\n'.join(f'{i+1:>3}  {lines[i]}' for i in range(start-1,min(end,len(lines))))); source(path)
def chapter(n,title,goal,concepts):
    global chapters
    chapters+=1; doc.add_page_break(); doc.add_heading(f'CHƯƠNG {n}. {title.upper()}',1)
    h2('A. Mục tiêu học tập'); p(goal)
    h2('B. Khái niệm nền tảng')
    for text in concepts: p(text)
def ending(example,summary,questions):
    h2('E. Ví dụ minh họa'); p(example)
    h2('F. Kết luận chương'); p(summary)
    h2('G. Câu hỏi tự kiểm tra'); bullets(questions)
def fig(name,caption):
    global figures
    figures+=1; doc.add_picture(str(B/(name+'.png')),width=Cm(16.8)); p(f'Hình {figures}. {caption}','Caption')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def diagram(name,boxes,edges,size=(12,6)):
    f,ax=plt.subplots(figsize=size); ax.set(xlim=(0,12),ylim=(0,6)); ax.axis('off')
    for key,(x,y,w,h,text) in boxes.items():
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06',fc='#EDF4FA',ec='#2F6387',lw=1.2))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
    for a,b in edges:
        x,y,w,h,_=boxes[a]; x2,y2,w2,h2,_=boxes[b]
        if abs(y-y2)<.5: start=(x+w,y+h/2); end=(x2,y2+h2/2)
        else: start=(x+w/2,y); end=(x2+w2/2,y2+h2)
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=13,color='#4D768C',lw=1.4,connectionstyle='arc3,rad=0'))
    f.savefig(B/(name+'.png'),dpi=180,bbox_inches='tight'); plt.close(f)
def graphics():
    diagram('pipeline',{
      'a':(.2,4.7,3.2,.8,'WESAD\n15 người, nhãn 700 Hz'), 'b':(4.3,4.7,3.3,.8,'Kiểm tra → lấy mẫu lại\nACC / BVP / EDA → 32 Hz'), 'c':(8.4,4.7,3.3,.8,'Căn nhãn → cửa sổ\n30 giây, bước 15 giây'),
      'd':(.2,2.7,3.2,.8,'Cache chưa chuẩn hóa\n[N, 5, 960] theo người'), 'e':(4.3,2.7,3.3,.8,'Chia người theo manifest\n9 Train / 3 Val / 3 Test'), 'f':(8.4,2.7,3.3,.8,'Fit scaler chỉ trên Train\nDataset → DataLoader'),
      'g':(.2,.7,3.2,.8,'CNN hoặc GRU\nHuấn luyện từ đầu mỗi fold'), 'h':(4.3,.7,3.3,.8,'Chọn best bằng Validation\nSau đó đánh giá Test'), 'i':(8.4,.7,3.3,.8,'OOF → tổng hợp theo người\nCSV / JSON / hình vẽ')}, [('a','b'),('b','c'),('c','d'),('d','e'),('e','f'),('f','g'),('g','h'),('h','i')])
    diagram('modules',{
      'a':(4.4,4.9,3.2,.7,'scripts.run_grouped_cv'), 'b':(.2,3.1,3.4,.9,'utils.config / tracking\nsplits / prepare_data'), 'c':(4.4,3.1,3.2,.9,'data.preprocessing\ninspection / resampling / windowing'), 'd':(8.4,3.1,3.4,.9,'training.trainer\ndataset / scaler / models'),
      'e':(.2,.8,3.4,.9,'utils.experiment_tracking\nCSV + console + TensorBoard'), 'f':(4.4,.8,3.2,.9,'evaluation.evaluator\nmetrics / predictions'), 'g':(8.4,.8,3.4,.9,'aggregation / history\nenhanced_visualization')},[('a','b'),('a','c'),('a','d'),('d','e'),('d','f'),('f','g')])
    diagram('lifecycle',{
      'a':(.2,4.7,3.2,.9,'Thiết kế + YAML + manifest\nKế hoạch chưa phải kết quả'), 'b':(4.4,4.7,3.2,.9,'GROUP_ID mới + snapshot\nTrạng thái đang chạy'), 'c':(8.5,4.7,3.2,.9,'Epoch → last.pt\nĐiểm commit để resume'),
      'd':(.2,2.6,3.2,.9,'best.pt theo Validation\nTest một lần sau selection'), 'e':(4.4,2.6,3.2,.9,'completion.json từng fold\nĐủ hash + metadata completed'), 'f':(8.5,2.6,3.2,.9,'Tổng hợp 5 folds\npartial hoặc completed'),
      'g':(3.1,.4,5.8,.9,'Hậu kiểm + registry E1 → finalized\nFinalized là hồ sơ khoa học, khác checkpoint')},[('a','b'),('b','c'),('c','d'),('d','e'),('e','f'),('f','g')])
    diagram('tensors',{
      'a':(.2,4.6,3.3,1,'Tín hiệu đồng bộ\n[T, 5], float32'), 'b':(4.4,4.6,3.2,1,'Window + chuyển trục\n[N, 5, 960]'), 'c':(8.5,4.6,3.1,1,'Batch DataLoader\n[B, 5, 960]'),
      'd':(.2,2.5,5.2,1,'Nhánh CNN: [B, 5, 960]\n→ [B, 32, 480] → [B, 64, 240]\n→ [B, 128, 240] → [B, 3]'),
      'e':(6.4,2.5,5.2,1,'Nhánh GRU: transpose\n[B, 960, 5] → hidden [1, B, 64]\n→ classifier → [B, 3]'),
      'f':(3.6,.4,4.8,.9,'Logits [B, 3], nhãn [B] int64\nCrossEntropyLoss hoặc softmax để xuất')},[('a','b'),('b','c'),('c','d'),('c','e'),('d','f'),('e','f')])
    f,ax=plt.subplots(figsize=(11,3.3)); ax.axis('off'); ax.set(xlim=(-.1,12.2),ylim=(-.6,3.3))
    for start,length,color,label in [(0,3.3,'#4B82A5','Nhãn gốc 1'),(3.3,.5,'#CC6622','0'),(3.8,8.2,'#5DA481','Nhãn gốc 2')]:
        ax.broken_barh([(start,length)],(1.7,.55),facecolors=color); ax.text(start+length/2,1.975,label,ha='center',va='center',color='white',fontsize=10)
    ax.text(0,2.9,'Ví dụ tổng hợp từ test: một đoạn nhãn không hợp lệ rất ngắn vẫn được bảo toàn',fontsize=11)
    ax.text(3.3,2.4,'701/700 giây',ha='center',fontsize=9)
    for k,i in enumerate([31,32,33]):
        x=k*3; ax.broken_barh([(x,3)],(.3,.65),facecolors='#CF7D3A' if i==32 else '#87B6CE',edgecolors='white')
        ax.text(x+1.5,.625,f'Mẫu {i} → '+('−1' if i==32 else str(0 if i==31 else 1)),ha='center',va='center',fontsize=10)
    ax.text(0,1.2,'Mẫu đích 32 chứa biên / đoạn không hợp lệ nên bị đánh dấu −1.',fontsize=10)
    ax.text(0,-.15,'Sơ đồ phóng đại vùng biên, không theo tỷ lệ; các khoảng là [bắt đầu, kết thúc).',fontsize=9)
    ax.text(0,-.5,'Đây là minh họa thuật toán, không phải tín hiệu WESAD.',fontsize=9)
    f.savefig(B/'alignment.png',dpi=180,bbox_inches='tight'); plt.close(f)
    a=json.loads(C[R1+'/summary.json'])['per_seed']['42']['pooled_oof']; cm=a['confusion_matrix']
    f,ax=plt.subplots(figsize=(7,5)); im=ax.imshow(cm,cmap='Blues')
    names=['Nền','Căng thẳng','Vui vẻ']; ax.set(xticks=range(3),yticks=range(3),xticklabels=names,yticklabels=names,xlabel='Nhãn dự đoán',ylabel='Nhãn thực',title='E1 chính thức — ma trận nhầm lẫn OOF')
    for i in range(3):
        for k in range(3): ax.text(k,i,str(cm[i][k]),ha='center',va='center',color='white' if cm[i][k]>450 else 'black')
    f.colorbar(im,ax=ax,label='Số cửa sổ'); f.tight_layout(); f.savefig(B/'confusion_e1.png',dpi=180); plt.close(f)
    rs1={r['subject_id']:float(r['macro_f1']) for r in csv.DictReader(io.StringIO(C[f'results/experiment_e1/{CNN}/subject_metrics.csv']))}
    rs2={r['subject_id']:float(r['macro_f1']) for r in csv.DictReader(io.StringIO(C[R2+'/subject_metrics.csv']))}
    names=sorted(rs1,key=lambda s:int(s[1:])); f,ax=plt.subplots(figsize=(11,4.6)); xs=list(range(15)); ax.bar([x-.2 for x in xs],[rs1[s] for s in names],width=.4,label='CNN đối chứng 15/11',color='#3179A5'); ax.bar([x+.2 for x in xs],[rs2[s] for s in names],width=.4,label='GRU 17/11',color='#CC6D36'); ax.set(xticks=xs,xticklabels=names,ylim=(0,1),ylabel='Macro-F1 theo người',xlabel='Người tham gia',title='Kết quả Test sau chọn checkpoint bằng Validation, seed 42'); ax.legend(); f.tight_layout(); f.savefig(B/'subject_comparison.png',dpi=180); plt.close(f)

def cover():
    p('DL13 / WESAD','Subtitle'); doc.add_heading('ĐỌC HIỂU\nVÀ LÀM CHỦ\nTOÀN BỘ CODEBASE',0)
    p('Giáo trình kỹ thuật bằng tiếng Việt','Subtitle')
    p('Từ dữ liệu sinh lý đến PyTorch, đánh giá theo người và quản lý bằng chứng thực nghiệm.')
    p('Dành cho sinh viên Khoa học Dữ liệu đã biết Python, Machine Learning và Deep Learning nhưng mới tiếp cận dự án.')
    p('Ngày rà soát: 18/11/2026  •  Phiên bản giáo trình: 1.0')
    p('Phạm vi: mã nguồn, cấu hình và artifacts văn bản hiện có. Bảo toàn E1 chính thức '+E1+'.')
    p('17 chương • Phụ lục hàm/lớp • Sơ đồ luồng dữ liệu • Bài tập tự kiểm chứng')
    doc.add_page_break(); doc.add_heading('Cách sử dụng giáo trình và quy ước bằng chứng',1)
    p('Đọc lần đầu theo Chương 1–3 để có bản đồ, rồi đi sâu Chương 5–10 theo luồng dữ liệu. Khi cần hiểu một tên hàm, tra phụ lục A; khi cần phân biệt trạng thái, xem Chương 14. Không cần học thuộc mọi dòng mã.')
    p('Mỗi chương có bảy phần A–G: mục tiêu, nền tảng, liên hệ mã nguồn, phân tích, ví dụ, kết luận và câu hỏi. Các ví dụ nhỏ được ghi rõ là minh họa; số liệu thực nghiệm luôn có đường dẫn nguồn. Số dòng tham chiếu thuộc mã hiện tại lúc rà soát, không thay cho version của snapshot huấn luyện.')
    table('Cách đọc mức bằng chứng',['Ký hiệu / mức','Ý nghĩa'],[
      ('Mã nguồn','Logic xác nhận bằng đọc source/config; chưa có nghĩa đã chạy thành công trên dữ liệu thật.'),('Artifacts đã lưu','Kết quả/log/XML/registry từ một thời điểm trước; chỉ có giá trị trong phạm vi mà hồ sơ ghi nhận.'),('Kiểm tra mới bằng văn bản','Trong lần biên soạn: phân tích AST 268 tệp Python, kiểm tra split/hash source và tính lại metrics từ CSV. Không chạy pytest, inference hoặc training.'),('Suy luận kỹ thuật','Lý giải thiết kế hoặc hệ quả có điều kiện; không đồng nhất với phát hiện lỗi đã tái hiện.'),('Đề xuất / chưa xác minh','Công việc tương lai hoặc nội dung không đủ bằng chứng. Không xem là chức năng đã hoàn thành.')],[4.3,12.7])
    p('Giới hạn chủ động: không mở .zip; không nạp .pkl, .npy/.npz, .pt hoặc event TensorBoard; không chạy CLI làm thay đổi dữ liệu/kết quả. Các file PDF/XLSX hiện có chỉ kiểm kê tên và kích thước, không giải nén; sổ nhật ký được phân tích qua README và generator. Thư mục notebooks hiện chỉ có tệp giữ chỗ.')
    p('Các lệnh trong giáo trình là hướng dẫn học và tái lập về sau, chưa được thực thi trong nhiệm vụ này. Đặc biệt lệnh inspect, prepare, aggregate, audit và tạo hình có thể ghi ra artifacts; đọc mô tả tác dụng trước khi tự chạy.')
    p('Phát hiện cập nhật: E2 đã có GRU và đủ năm folds; E3 vẫn dự kiến. CNN hai khối theo đề bài/proposal là mục tiêu riêng; không tìm thấy đặc tả proposal đầy đủ trong tài liệu văn bản đã đọc, nên không tự gán hyperparameters cho nó.')
    doc.add_heading('Mục lục',1); field(p(),'TOC \\o "1-3" \\h \\z \\u')

def finish():
    doc.core_properties.title='DL13-WESAD — Giáo trình đọc hiểu và làm chủ codebase'
    doc.core_properties.subject='Phân tích mã nguồn, dữ liệu, mô hình và bằng chứng thực nghiệm'
    doc.core_properties.author='Giáo trình kỹ thuật DL13-WESAD'
    doc.core_properties.keywords='WESAD, PyTorch, CNN, GRU, subject-wise, tiếng Việt'
    doc.save(OUT)
    (B/'document_stats.json').write_text(json.dumps({'chapters':chapters,'tables':tables,'figures':figures,'diagrams':5,'scientific_charts':2,'paragraphs':len(doc.paragraphs),'citations':citations,'output':str(OUT)},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(OUT),'chapters':chapters,'tables':tables,'figures':figures,'size':OUT.stat().st_size},ensure_ascii=False))

if __name__=='__main__':
    graphics(); cover()
    from chapters import write_chapters
    write_chapters(globals())
    from appendices import write_appendices
    write_appendices(globals())
    finish()

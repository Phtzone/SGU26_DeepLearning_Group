from pathlib import Path
import json, math
import fitz
from PIL import Image, ImageDraw
B=Path(__file__).parent
pdf=fitz.open(B/'guide_layout_check.pdf')
result={'pages':len(pdf),'outside_page':[],'nearly_empty_pages':[],'page_sizes':sorted(set((round(p.rect.width,1),round(p.rect.height,1)) for p in pdf))}
for i,page in enumerate(pdf):
    text=page.get_text()
    if len(text.strip())<60: result['nearly_empty_pages'].append(i+1)
    for block in page.get_text('dict')['blocks']:
        if block['type']!=0: continue
        for line in block['lines']:
            for span in line['spans']:
                x0,y0,x1,y1=span['bbox']
                if x0<-.5 or y0<-.5 or x1>page.rect.width+.5 or y1>page.rect.height+.5:
                    result['outside_page'].append({'page':i+1,'bbox':span['bbox'],'text':span['text'][:100]})
    for image in page.get_image_info():
        x0,y0,x1,y1=image['bbox']
        assert x0>=0 and y0>=0 and x1<=page.rect.width+.5 and y1<=page.rect.height+.5,(i,image)
result['chapter_pages']={}
for i,page in enumerate(pdf):
    text=page.get_text()
    if 'A. Mục tiêu học tập' in text:
        result['chapter_pages'][i+1]=text[:180]
for start in range(0,len(pdf),20):
    sheet=Image.new('RGB',(1000,5*370),'#dddddd'); draw=ImageDraw.Draw(sheet)
    for k in range(start,min(start+20,len(pdf))):
        pix=pdf[k].get_pixmap(matrix=fitz.Matrix(.38,.38),alpha=False)
        im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
        im.thumbnail((235,335))
        x=(k-start)%4*250+(250-im.width)//2; y=(k-start)//4*370+25
        sheet.paste(im,(x,y)); draw.text((x,y-17),f'Trang {k+1}',fill='black')
    sheet.save(B/f'contact_{start+1:03d}.png')
for page_number in [1,2,10,20,30,45,60,75,len(pdf)]:
    if page_number<=len(pdf): pdf[page_number-1].get_pixmap(matrix=fitz.Matrix(1.35,1.35),alpha=False).save(B/f'page_{page_number:03d}.png')
(B/'pdf_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))

from pathlib import Path
import json,hashlib,zipfile,re,xml.etree.ElementTree as ET
from pypdf import PdfReader,PdfWriter
import pypdfium2 as pdfium
from PIL import Image,ImageOps,ImageDraw
O=Path(__file__).resolve().parent;R=O.parent;OLD=R/'results_analysis_20260924'
w=PdfWriter()
for p in [OLD/'科研矢量图集.pdf',O/'qa/new_figures.pdf']:w.append(str(p))
with (O/'科研矢量图集.pdf').open('wb') as f:w.write(f)
reader=PdfReader(O/'科研矢量图集.pdf');assert len(reader.pages)==53
record={'pdf_pages':[],'svg_files':[],'originals':[]}
for i,p in enumerate(reader.pages,1):
 n=len(p.images);t=len(p.extract_text() or '');assert n==0 and t>0,(i,n,t)
 record['pdf_pages'].append({'figure':i,'images':n,'text_chars':t})
for p in sorted((O/'figures').glob('*.svg')):
 root=ET.parse(p).getroot();n=len(root.findall('.//{http://www.w3.org/2000/svg}image'));t=len(root.findall('.//{http://www.w3.org/2000/svg}text'));assert n==0 and t>0,p
 record['svg_files'].append({'name':p.name,'images':n,'text_elements':t,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
assert len(record['svg_files'])==53
archived_html=(OLD/'实验结果详细分析.html').read_text(encoding='utf-8')
archived_svgs=re.findall(r'<svg\b.*?</svg>',archived_html,flags=re.S)
assert len(archived_svgs)==20
for p,archived in zip(sorted((O/'figures').glob('*.svg'))[:20],archived_svgs):
 archived=re.sub(r'fig\d+_','',archived)
 current=re.search(r'<svg\b.*?</svg>',p.read_text(encoding='utf-8'),flags=re.S).group(0)
 assert archived==current,p.name
 record['originals'].append({'file':p.name,'svg_payload_identical_to_archived_html':True})
md=(O/'全量实验数据分析报告.md').read_text(encoding='utf-8');embeds=re.findall(r'!\[.*?\]\((figures/[^)]+)\)',md)
assert len(embeds)==53 and len(set(embeds))==53
for s in embeds:assert (O/s).exists(),s
assert '没有受控缺失率、缺失位置' not in md
record['report_figure_count']=53;record['status']='PASS'
(O/'qa/verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
# Render all newly created PDF pages; create four-per-sheet review images.
doc=pdfium.PdfDocument(str(O/'科研矢量图集.pdf'));paths=[]
for i in range(20,53):
 page=doc[i];im=page.render(scale=1350/page.get_width()).to_pil();dest=O/'qa'/f'figure_{i+1:02d}.png';im.save(dest);paths.append(dest);page.close()
doc.close()
for offset in range(0,len(paths),4):
 canvas=Image.new('RGB',(1900,1460),'#ECEFF1');draw=ImageDraw.Draw(canvas)
 for j,p in enumerate(paths[offset:offset+4]):
  im=Image.open(p).convert('RGB');im.thumbnail((940,680));x=(j%2)*950+(950-im.width)//2;y=(j//2)*730+32+(680-im.height)//2;canvas.paste(im,(x,y));draw.text(((j%2)*950+12,(j//2)*730+8),p.stem,fill='black')
 canvas.save(O/'qa'/f'contact_{offset//4+1:02d}.png')
print('PASS: 53 PDF pages, 53 vector SVGs, 53 report images; 20 original SVG payloads identical to archived HTML.')

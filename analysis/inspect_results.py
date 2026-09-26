from pathlib import Path
import json, csv, hashlib, zipfile, xml.etree.ElementTree as ET
from collections import Counter
R=Path(__file__).resolve().parents[1]; O=R/'analysis'
old=json.loads((R/'results_analysis_20260924/analysis.json').read_text(encoding='utf-8'))
diff=[]
for name,h in old['sources'].items():
 p=R/name; new=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
 diff.append(dict(path=name,previous_sha256=h,current_sha256=new,status='unchanged' if h==new else ('changed' if new else 'missing')))
(O/'source_comparison.json').write_text(json.dumps(diff,ensure_ascii=False,indent=2),encoding='utf-8')
print('OLD INPUTS',Counter(x['status'] for x in diff))
print('DIFFERENCES',*[x['path'] for x in diff if x['status']!='unchanged'],sep='\n')
with zipfile.ZipFile(R/'复杂场景下多模态情感识别的数学建模与算法设计.docx') as z:
 root=ET.fromstring(z.read('word/document.xml'))
 paras=[''.join(p.itertext()) for p in []]
 ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
 paras=[''.join(t.text or '' for t in p.findall('.//w:t',ns)) for p in root.findall('.//w:p',ns)]
 text='\n'.join(paras);(O/'题目原文.txt').write_text(text,encoding='utf-8');print('TASK',text)
for name in ['problem2_experiment2/condition_metrics.csv','problem2_experiment2/attachment3_predictions.csv','problem3_outputs/primary_modality_local_importance_samples.csv','problem3_outputs/attachment4_explanations.csv']:
 with (R/name).open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
 print('\nTABLE',name,'N',len(rows),'FIELDS',list(rows[0]))
 for k in ['model','seed','split','group','missing_modalities','requested_rate','position','duration','variant','main_modality']:
  if k in rows[0]:print(k,dict(Counter(r[k] for r in rows)))
print('P1keys',list(json.loads((R/'problem1_outputs/problem1_complete_analysis.json').read_text(encoding='utf-8'))))
conf=json.loads((R/'problem2_experiment2/experiment_config.json').read_text(encoding='utf-8'))
print('CONDITIONS',len(conf['conditions']),conf.get('config',{}))

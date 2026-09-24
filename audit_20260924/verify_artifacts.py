"""Read-only checks of supplied data and saved results; writes only audit evidence."""
import ast
import csv
import gc
import hashlib
import json
import pickle
import struct
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from problem1.data import read_samples
from problem1.core import pool_intervals

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def movie_duration(path):
    # ISO BMFF movie header; container duration, not an acoustic timestamp.
    with path.open('rb') as f:
        def boxes(end):
            while f.tell() + 8 <= end:
                start = f.tell()
                size, kind = struct.unpack('>I4s', f.read(8))
                if size == 1:
                    size = struct.unpack('>Q', f.read(8))[0]
                if size == 0:
                    size = end - start
                stop = start + size
                if kind == b'moov':
                    value = boxes(stop)
                    if value is not None:
                        return value
                elif kind == b'mvhd':
                    version = f.read(4)[0]
                    f.read(16 if version else 8)
                    scale = struct.unpack('>I', f.read(4))[0]
                    duration = struct.unpack('>Q' if version else '>I', f.read(8 if version else 4))[0]
                    return duration / scale
                if stop <= start:
                    break
                f.seek(stop)
        return boxes(path.stat().st_size)

evidence = {'scope': 'Current project directory; no training or model inference rerun.'}
evidence['data_counts'] = {d.name: len(list(d.rglob('*.*'))) for d in (ROOT/'DATA').iterdir() if d.is_dir()}
evidence['weight_files'] = [str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.pt','.pth','.safetensors')]
labels = read_samples(ROOT/'DATA')
with (ROOT/'problem1_outputs/problem1_aligned_features.pkl').open('rb') as f:
    payload = pickle.load(f)
manifest = {r['id']: r for r in read_csv(ROOT/'problem1_outputs/manifest.csv')}
samples = {s['id']: s for s in payload['samples']}
p1 = {'samples':len(samples),'summary_rows':len(read_csv(ROOT/'problem1_outputs/problem1_feature_summary.csv')), 'issues':[], 'video_hash_matches':0, 'movie_durations':[], 'alignment_modes':{}, 'numeric_or_punctuation_tokens':[], 'truncated_samples':[]}
for label in labels:
    s = samples[label['id']]
    if any(s[k] != label[v] for k,v in [('label','label'),('annotation','annotation'),('raw_text','text')]):
        p1['issues'].append([s['id'], 'label_or_transcript_mismatch'])
    p1['video_hash_matches'] += digest(label['path']) == manifest[s['id']]['sha256']
    dur = movie_duration(label['path'])
    p1['movie_durations'].append({'id':s['id'], 'container_seconds':dur, 'saved_seconds':s['duration_seconds']})
    p1['alignment_modes'][s['text_alignment']] = p1['alignment_modes'].get(s['text_alignment'],0)+1
    for m,dim in [('text',768),('audio',74),('vision',35)]:
        x,mask,cov,spans = (np.asarray(s[m]), np.asarray(s['valid_masks'][m]), np.asarray(s['coverage'][m]),np.asarray(s['source_intervals'][m]))
        if x.shape != (50,dim) or not np.isfinite(x).all() or np.any(x[~mask] != 0):
            p1['issues'].append([s['id'],m,'features'])
        _,mask2,cov2,ix = pool_intervals(np.ones((len(spans),1)),spans,s['time_edges'])
        if not np.array_equal(mask,mask2) or not np.allclose(cov,cov2,atol=1e-4):
            p1['issues'].append([s['id'],m,'coverage_reconstruction'])
        # float32 boundary rounding may add a negligible overlap; count separately.
        if ix != s['source_indices'][m]:
            p1.setdefault('source_index_rounding_differences',[]).append([s['id'],m])
for r in read_csv(ROOT/'problem1_outputs/problem1_text_truncation.csv'):
    if any('omitted' in k and float(v or 0)>0 for k,v in r.items()):
        p1['truncated_samples'].append(r)
evidence['problem1'] = p1
del payload
gc.collect()

evidence['special_data'] = {}
for attachment in ('attachment_3_missing_modality','attachment_4_explainability'):
    for variant in ('aligned','unaligned'):
        rows=[]
        for path in sorted((ROOT/'DATA'/attachment/variant).rglob('*.pkl')):
            with path.open('rb') as f: raw=pickle.load(f)
            d=raw.get('test',raw)
            r={'file':path.name,'fields':{k:list(np.asarray(v).shape) for k,v in d.items()}}
            for m in ('text','audio','vision'):
                if m in d:
                    a=np.asarray(d[m]); a=a[0] if a.ndim==3 else a
                    zero=np.all(a==0,axis=-1)
                    r[m+'_zero_rows']=np.flatnonzero(zero).tolist()
            if 'text_bert' in d:
                tb=np.asarray(d['text_bert']);tb=tb[0] if tb.ndim==3 else tb
                r['token_mask_zero']=np.flatnonzero(tb[1]==0).tolist()
                r['text_zero_but_token_visible']=[i for i in r.get('text_zero_rows',[]) if tb[1,i]>0]
            rows.append(r)
        evidence['special_data'][attachment+'/'+variant]=rows

print('Loading aligned data', flush=True)
with (ROOT/'DATA/attachment_2_standard_features/aligned_50.pkl').open('rb') as f: data=pickle.load(f)
split_ids = {k:set(map(str,d['id'])) for k,d in data.items()}
standard={}
for name,d in data.items():
    y=np.asarray(d['classification_labels'],dtype=np.int64).reshape(-1)
    reg=np.asarray(d['regression_labels']).reshape(-1)
    standard[name]={'n':len(y),'label_sign_mismatches':int(np.sum(y!=np.where(reg<0,0,np.where(reg>0,2,1)))),'class_counts':np.bincount(y,minlength=3).tolist(), 'nonfinite':{m:int(np.sum(~np.isfinite(d[m]))) for m in ('text','audio','vision')}}
standard['id_overlap']={a+'/'+b:len(split_ids[a]&split_ids[b]) for a,b in [('train','valid'),('train','test'),('valid','test')]}
standard['video_id_overlap']={a+'/'+b:len({x.split('$_$')[0] for x in split_ids[a]}&{x.split('$_$')[0] for x in split_ids[b]}) for a,b in [('train','valid'),('train','test'),('valid','test')]}
stats=read_json(ROOT/'problem2_outputs_bert_av_reproduce/normalization.json')
standard['normalization_max_abs_error']={}
for m in ('text','audio','vision'):
    x=np.asarray(data['train'][m],dtype=np.float32)
    mask=np.asarray(data['train']['text_bert'])[:,1,:].astype(bool) if m=='text' else ~np.all(np.isclose(x,0),axis=-1)
    vals=x[mask]
    mean=vals.mean(0,dtype=np.float64).astype(np.float32);std=vals.std(0,dtype=np.float64).astype(np.float32)
    std[~np.isfinite(std)|(std<1e-5)]=1;mean[~np.isfinite(mean)]=0
    standard['normalization_max_abs_error'][m]={'mean':float(np.max(np.abs(mean-np.array(stats[m]['mean'])))), 'std':float(np.max(np.abs(std-np.array(stats[m]['std']))))}
evidence['standard_data']=standard
evidence['predictions']={}
for path in sorted(ROOT.glob('problem2_outputs*/problem2_*predictions.csv')):
    rows=read_csv(path); fields=rows[0]
    p=np.array([[float(r[k]) for k in ('p_negative','p_neutral','p_positive')] for r in rows])
    pred=np.array([int(r['predicted_class']) for r in rows]); reg=np.array([float(r['regression_prediction']) for r in rows])
    out={'n':len(rows),'unique_ids':len({r['id'] for r in rows}),'prob_sum_error':float(np.max(np.abs(p.sum(1)-1))), 'argmax_mismatches':int(np.sum(p.argmax(1)!=pred)), 'score_range':[float(reg.min()),float(reg.max())]}
    if 'true_class' in fields:
        y=np.array([int(r['true_class']) for r in rows]);yr=np.array([float(r['true_regression']) for r in rows])
        cm=np.zeros((3,3),int); np.add.at(cm,(y,pred),1)
        f1=2*np.diag(cm)/np.maximum(cm.sum(0)+cm.sum(1),1)
        out.update(accuracy=float((y==pred).mean()),macro_f1=float(f1.mean()),mae=float(np.abs(yr-reg).mean()),pearson=float(np.corrcoef(yr,reg)[0,1]),confusion_matrix=cm.tolist())
        split='valid' if '_valid_' in path.name else 'test'
        truth={str(i):(int(c),float(r)) for i,c,r in zip(data[split]['id'],data[split]['classification_labels'],data[split]['regression_labels'])}
        out['truth_mismatches']=sum(r['id'] not in truth or int(r['true_class'])!=truth[r['id']][0] or abs(float(r['true_regression'])-truth[r['id']][1])>1e-5 for r in rows)
    evidence['predictions'][str(path.relative_to(ROOT))]=out

recovery=ROOT/'problem2_outputs_bert_av_reproduce/history_recovery/replay_kdviq6ne/verification.json'
verification=read_json(recovery)
target=ROOT/'problem2_outputs_bert_av_reproduce'
evidence['replay_hash_checks']={name: digest(target/name)==expected if (target/name).is_file() else 'MISSING' for name,expected in verification['target_files_sha256'].items()}
evidence['source_hash_checks']={name:digest(path)==verification['replay_sources_sha256'][name] for name,path in [('aligned_50.pkl',ROOT/'DATA/attachment_2_standard_features/aligned_50.pkl'),('problem2_bert_fusion_train.py',ROOT/'problem2_bert_fusion_train.py'),('bert_model.py',ROOT/'problem2/bert_model.py')]}
for name,path in [('problem2_bert_fusion_train.py',ROOT/'problem2_bert_fusion_train.py'),('bert_model.py',ROOT/'problem2/bert_model.py')]:
    evidence['source_hash_checks'][name]={'raw':evidence['source_hash_checks'][name], 'LF_normalized':hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==verification['replay_sources_sha256'][name]}
mod=ast.parse((ROOT/'problem2/data.py').read_text(encoding='utf-8'))
fn=next(n for n in mod.body if isinstance(n,ast.FunctionDef) and n.name=='apply_local_dropout')
choices=iter([1,10,0]*3)
stub=types.SimpleNamespace(rand=lambda shape:np.float64(0),randint=lambda *args:np.int64(next(choices)))
ns={'torch':stub,'MODALITIES':('text','audio','vision')}
exec(compile(ast.Module(body=[fn],type_ignores=[]),'problem2/data.py','exec'),ns)
batch={'x':{m:np.ones((1,50,1)) for m in ns['MODALITIES']},'mask':{m:np.arange(50)[None,:]<5 for m in ns['MODALITIES']}}
ns['apply_local_dropout'](batch,probability=1)
evidence['dropout_branch_check']={'method':'existing AST function body with deterministic randint stand-in and numpy arrays; not torch runtime test','initial_valid_rows':5,'block':[0,10],'remaining':{m:int(v.sum()) for m,v in batch['mask'].items()}}
from problem2.training_history import load_history
try:
    load_history(target)
except (ValueError,FileNotFoundError) as exc:
    evidence['current_history_loader_error']=str(exc)
(Path(__file__).parent/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
print('Audit evidence saved', flush=True)

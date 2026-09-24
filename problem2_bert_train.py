#!/usr/bin/env python3
"""Fine-tune the local BERT encoder on attachment-2 sentiment labels."""
from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from problem2.bert_model import BertSentiment


class BertDataset(Dataset):
    def __init__(self, split):
        b = np.asarray(split["text_bert"], dtype=np.int64)
        self.ids = torch.from_numpy(b[:, 0])
        self.mask = torch.from_numpy(b[:, 1])
        self.types = torch.from_numpy(b[:, 2])
        self.y = torch.from_numpy(np.asarray(split["classification_labels"], dtype=np.int64))
        self.r = torch.from_numpy(np.asarray(split["regression_labels"], dtype=np.float32))

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.ids[i], self.mask[i], self.types[i], self.y[i], self.r[i]


def evaluate(model, loader, device):
    model.eval(); ys=[]; ps=[]; rs=[]; prs=[]
    with torch.inference_mode():
        for ids, mask, types, y, r in loader:
            logits, pred_r = model(ids.to(device), mask.to(device), types.to(device))
            ys.append(y.numpy()); ps.append(logits.argmax(-1).cpu().numpy()); rs.append(r.numpy()); prs.append(pred_r.cpu().numpy())
    y=np.concatenate(ys); p=np.concatenate(ps); r=np.concatenate(rs); pr=np.concatenate(prs)
    f=[]
    for k in range(3):
        tp=((y==k)&(p==k)).sum(); pp=(p==k).sum(); yy=(y==k).sum(); precision=tp/max(pp,1); recall=tp/max(yy,1); f.append(2*precision*recall/max(precision+recall,1e-9))
    return {"accuracy":float((y==p).mean()),"macro_f1":float(np.mean(f)),"f1_by_class":f,"mae":float(np.abs(r-pr).mean()),"pearson":float(np.corrcoef(r,pr)[0,1])}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',type=Path,default=Path('D:/E_math/DATA')); ap.add_argument('--text-model',type=Path,default=Path('D:/E_math/models/bert-base-uncased')); ap.add_argument('--output',type=Path,default=Path('D:/E_math/problem2_bert_outputs')); ap.add_argument('--epochs',type=int,default=5); ap.add_argument('--batch-size',type=int,default=24); ap.add_argument('--lr',type=float,default=2e-5); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); args.output.mkdir(parents=True,exist_ok=True)
    with (args.data_root/'attachment_2_standard_features'/'aligned_50.pkl').open('rb') as f: d=pickle.load(f)
    train=BertDataset(d['train']); valid=BertDataset(d['valid']); test=BertDataset(d['test'])
    tr=DataLoader(train,batch_size=args.batch_size,shuffle=True); va=DataLoader(valid,batch_size=args.batch_size*2); te=DataLoader(test,batch_size=args.batch_size*2)
    model=BertSentiment(str(args.text_model),device); counts=np.bincount(train.y.numpy(),minlength=3).astype(np.float32); w=torch.tensor(1/np.sqrt(counts),device=device); w=w/w.mean(); ce=nn.CrossEntropyLoss(weight=w); huber=nn.SmoothL1Loss(beta=.5); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01); best=-1
    hist=[]
    for ep in range(1,args.epochs+1):
        model.train(); total=0
        for ids,mask,types,y,r in tr:
            logits,pred_r=model(ids.to(device),mask.to(device),types.to(device)); loss=ce(logits,y.to(device))+.15*huber(pred_r,r.to(device)); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); total+=float(loss)
        vm=evaluate(model,va,device); hist.append(vm); print(f'epoch {ep:02d} valid_acc={vm["accuracy"]:.4f} valid_f1={vm["macro_f1"]:.4f} valid_mae={vm["mae"]:.4f}',flush=True)
        if vm['macro_f1']>best: best=vm['macro_f1']; torch.save(model.state_dict(),args.output/'bert_sentiment.pt')
    model.load_state_dict(torch.load(args.output/'bert_sentiment.pt',map_location=device,weights_only=True)); tm=evaluate(model,te,device); (args.output/'metrics.json').write_text(json.dumps({'device':str(device),'best_valid_f1':best,'test':tm,'history':hist},indent=2),encoding='utf-8'); print(json.dumps({'device':str(device),'test':tm},indent=2))


if __name__=='__main__': main()

import argparse,json,random,csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import load_npz,subject_folds,WindowDataset,write_manifest
from .model import BAND,band_loss
from .eval import aggregate,metrics

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

def run(args):
    seed_all(args.seed)
    X,y,s,names=load_npz(args.data)
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    folds=list(subject_folds(y,s,args.folds,args.seed))
    write_manifest(dest/'fold_manifest.csv',folds,s,y,names)
    with open(dest/'config.json','w') as f:json.dump(vars(args),f,indent=2)
    fold_metrics=[];pred_rows=[]
    device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    for fold,(tr,te) in enumerate(folds):
        seed_all(args.seed+fold)
        # Train-only window/channel mean and std; no test statistics used.
        mu=X[tr].mean(axis=(0,2),keepdims=True);sd=X[tr].std(axis=(0,2),keepdims=True)+1e-5
        xtrain=(X[tr]-mu)/sd;x_test=(X[te]-mu)/sd
        local={v:i for i,v in enumerate(sorted(np.unique(s[tr])))}
        strn=np.array([local[v] for v in s[tr]],dtype='int64')
        ab=args.ablation
        model=BAND(X.shape[1],hidden=args.hidden,latent=args.latent,n_subjects=len(local),
          transition=(ab!='no_transitions'),subject_effect=(ab!='no_subject_effect'),
          adversary=(ab!='no_invariance'),probabilistic=(ab!='no_probabilistic')).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
        ds=WindowDataset(xtrain,y[tr],strn)
        loader=DataLoader(ds,batch_size=min(args.batch,len(ds)),shuffle=True,drop_last=False)
        model.train()
        for epoch in range(args.epochs):
            for xb,yb,sb in loader:
                xb=xb.to(device);yb=yb.to(device);sb=sb.to(device)
                out=model(xb,grl=1.,sample=True)
                loss,_=band_loss(out,yb,sb,adv=0 if ab=='no_invariance' else args.adv,
                  orth=0 if ab=='no_covariance' else args.orth,
                  subject_effect=(ab!='no_subject_effect'),adversary=(ab!='no_invariance'),phys=args.phys)
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
        pr=[]
        for beg in range(0,len(te),args.batch):
            bx=torch.from_numpy(x_test[beg:beg+args.batch]).to(device)
            pr.append(model.predict_new(bx,draws=args.draws).cpu().numpy())
        rawp=np.concatenate(pr);ids,yt,subp=aggregate(rawp,y[te],s[te]);score=metrics(yt,subp)
        fold_metrics.append({'fold':fold,**score,'n_subjects':len(ids)})
        reverse={i:k for k,i in names.items()}
        for sub,truth,p in zip(ids,yt,subp):pred_rows.append({'fold':fold,'subject_id':reverse[int(sub)],'y_true':int(truth),'p_control':float(p[0]),'p_pd':float(p[1])})
        print(f'Fold {fold}: BA={score["BA"]:.3f} AUROC={score["AUROC"]:.3f}; n={len(ids)}')
    keys=[k for k in fold_metrics[0] if k not in ('fold','n_subjects')]
    summary={'N_participants':len(np.unique(s)),'folds':args.folds,'ablation':args.ablation,
       'mean':{k:float(np.mean([f[k] for f in fold_metrics])) for k in keys},
       'sd':{k:float(np.std([f[k] for f in fold_metrics],ddof=1)) for k in keys}}
    (dest/'summary.json').write_text(json.dumps({'folds':fold_metrics,'summary':summary},indent=2))
    with open(dest/'subject_predictions.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=pred_rows[0].keys());writer.writeheader();writer.writerows(pred_rows)
    print('Saved to',dest);return summary

def main():
    p=argparse.ArgumentParser(description='BAND participant-disjoint training')
    p.add_argument('--data',required=True);p.add_argument('--output',default='outputs/run')
    p.add_argument('--folds',type=int,default=3);p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--hidden',type=int,default=24);p.add_argument('--latent',type=int,default=12)
    p.add_argument('--lr',type=float,default=0.001);p.add_argument('--batch',type=int,default=32)
    p.add_argument('--draws',type=int,default=8);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--adv',type=float,default=.02);p.add_argument('--orth',type=float,default=.02)
    p.add_argument('--phys',type=float,default=.05,help='Training-only beta spectral auxiliary weight')
    p.add_argument('--cpu',action='store_true');p.add_argument('--ablation',default='full',choices=['full','no_transitions','no_subject_effect','no_invariance','no_covariance','no_probabilistic'])
    run(p.parse_args())
if __name__=='__main__':main()

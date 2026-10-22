import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from .data import load_npz,subject_folds,write_manifest
from .eval import aggregate,metrics

def spectral_features(X,sfreq=250):
    f,p=welch(X,fs=sfreq,nperseg=min(X.shape[-1],round(sfreq*2)),axis=-1)
    bands=[(1,4),(4,8),(8,13),(13,30),(30,45)]
    return np.concatenate([np.log(np.maximum(p[:,:,(f>=a)&(f<b)].mean(-1),1e-18)) for a,b in bands],axis=1)

def run(npz,out,folds=3,seed=42,sfreq=250):
    X,y,s,names=load_npz(npz);feat=spectral_features(X,sfreq)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    split=list(subject_folds(y,s,folds,seed));write_manifest(out/'fold_manifest.csv',split,s,y,names)
    rows=[]; scores=[];reverse={v:k for k,v in names.items()}
    for fi,(tr,te) in enumerate(split):
        for name,mdl in [('spectral_logistic',make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,random_state=seed))),
                         ('spectral_rf',RandomForestClassifier(n_estimators=60,random_state=seed,n_jobs=1,class_weight='balanced'))]:
            # Average windows before fitting to avoid disproportionate influence of long recordings.
            subjects=np.unique(s[tr]);fs=np.stack([feat[tr][s[tr]==i].mean(0) for i in subjects]);ys=np.array([y[tr][s[tr]==i][0] for i in subjects]);
            mdl.fit(fs,ys)
            prob=mdl.predict_proba(feat[te]);ids,yt,sp=aggregate(prob,y[te],s[te]); m=metrics(yt,sp)
            scores.append({'method':name,'fold':fi,'n_subjects':len(ids),**m})
            rows.extend({'method':name,'fold':fi,'subject_id':reverse[int(i)],'y_true':int(yy),'p_pd':float(pp[1])} for i,yy,pp in zip(ids,yt,sp))
    pd.DataFrame(rows).to_csv(out/'subject_predictions.csv',index=False)
    pd.DataFrame(scores).to_csv(out/'fold_metrics.csv',index=False)
    (out/'summary.json').write_text(json.dumps({'models':sorted(set(x['method'] for x in scores)),'folds':folds},indent=2))
    return scores

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--data',required=True);parser.add_argument('--output',required=True);parser.add_argument('--folds',type=int,default=3);parser.add_argument('--seed',type=int,default=42);parser.add_argument('--sfreq',type=float,default=250)
 a=parser.parse_args();run(a.data,a.output,a.folds,a.seed,a.sfreq)

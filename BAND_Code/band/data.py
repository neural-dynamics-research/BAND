import csv
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset

class WindowDataset(Dataset):
    def __init__(self, X, y, subjects):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.subjects = np.asarray(subjects, dtype=np.int64)
        assert self.X.ndim == 3 and len(self.X) == len(self.y) == len(self.subjects)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i], self.subjects[i]

def load_npz(path):
    a = np.load(path, allow_pickle=False)
    for key in ('X', 'y', 'subject_id'):
        if key not in a: raise ValueError(f'Missing {key} in {path}')
    X = a['X'].astype('float32'); y = a['y'].astype('int64'); raw = a['subject_id']
    if X.ndim != 3 or not (len(X) == len(y) == len(raw)):
        raise ValueError('Expected X [windows, channels, samples], y [windows], subject_id [windows]')
    labels = {str(v):i for i,v in enumerate(sorted(set(map(str, raw))))}
    s = np.array([labels[str(v)] for v in raw], dtype=np.int64)
    for si in np.unique(s):
        if len(set(y[s == si])) != 1:
            raise ValueError('Primary diagnosis classification requires one class label per participant')
    if set(y.tolist()) != {0,1}: raise ValueError('Expected binary labels 0/1')
    return X,y,s,labels

def subject_folds(y,s,k=3,seed=42):
    if len(np.unique(s)) < k: raise ValueError('Not enough participants for folds')
    cv=StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    for tr,te in cv.split(np.zeros(len(y)),y,s):
        if set(s[tr]).intersection(set(s[te])): raise AssertionError('Participant leakage')
        if len(np.unique(y[te])) < 2: raise ValueError('Fold has only one test class; lower k or revise manifest')
        yield tr,te

def synthetic(out, n_subjects=16, windows_per_subject=5, channels=8, samples=128, seed=42):
    rng=np.random.default_rng(seed); X=[]; Y=[]; S=[]
    t=np.arange(samples)/128
    for s in range(n_subjects):
        label=s%2; fingerprint=rng.normal(0,.14,(channels,1))
        for j in range(windows_per_subject):
            beta=np.sin(2*np.pi*(16+label*5)*t+rng.uniform(-1,1))
            x=rng.normal(0,.5,(channels,samples))+fingerprint+(.23+label*.30)*beta[None,:]
            X.append(x); Y.append(label); S.append(f'sub-{s:03d}')
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,X=np.asarray(X,dtype='float32'),y=np.asarray(Y),subject_id=np.asarray(S))
    return out

def write_manifest(path, folds, s, y, names):
    with open(path,'w',newline='') as f:
        wr=csv.writer(f);wr.writerow(['fold','participant_id','role','class_label'])
        inv={v:k for k,v in names.items()}
        for i,(tr,te) in enumerate(folds):
            for ids,role in [(tr,'train'),(te,'test')]:
                for subj in sorted(set(s[ids])):
                    wr.writerow([i,inv[int(subj)],role,int(y[np.where(s==subj)[0][0]])])

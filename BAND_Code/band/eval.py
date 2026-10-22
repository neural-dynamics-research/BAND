import numpy as np
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,f1_score,brier_score_loss

def aggregate(probs,y,subjects):
    ids=np.unique(subjects)
    yp=np.array([int(y[subjects==s][0]) for s in ids])
    p=np.stack([probs[subjects==s].mean(axis=0) for s in ids])
    return ids,yp,p

def ece(y,p,bins=10):
    confidence=p.max(1);chosen=p.argmax(1); accuracy=(chosen==y)
    edges=np.linspace(0,1,bins+1); result=0.
    for i in range(bins):
        mask=(confidence>=edges[i]) & (confidence<=edges[i+1] if i==bins-1 else confidence<edges[i+1])
        if mask.any():result+=mask.mean()*abs(float(accuracy[mask].mean()-confidence[mask].mean()))
    return float(result)

def risk_coverage_auc(y,p):
    order=np.argsort(-p.max(axis=1)); err=(p.argmax(1)[order]!=y[order]).astype(float)
    coverage=np.arange(1,len(y)+1)/len(y); risk=np.cumsum(err)/np.arange(1,len(y)+1)
    return float(np.trapezoid(np.r_[0,risk],np.r_[0,coverage]))

def metrics(y,p):
    if len(np.unique(y))<2: raise ValueError('Both classes required')
    return {'BA':float(balanced_accuracy_score(y,p.argmax(1))),'AUROC':float(roc_auc_score(y,p[:,1])),
      'macro_F1':float(f1_score(y,p.argmax(1),average='macro')),
      'Brier':float(brier_score_loss(y,p[:,1])),'ECE':ece(y,p),'risk_coverage_AUC':risk_coverage_auc(y,p)}

import numpy as np
from .eval import ece,metrics

def remove_channels(X,fraction,seed=0):
    """Fixed channel mask shared by all windows for a specified severity."""
    X=np.asarray(X);rng=np.random.default_rng(seed);n=int(round(X.shape[1]*fraction))
    missing=rng.choice(X.shape[1],n,replace=False);out=X.copy();out[:,missing,:]=0
    return out,missing

def predict_perturbations(model,X,device,levels=(0,.1,.2,.3,.4),draws=16,batch=32,seed=0):
    import torch
    rows=[]
    for fraction in levels:
        pert,removed=remove_channels(X,fraction,seed)
        preds=[]
        for start in range(0,len(pert),batch):
            xb=torch.as_tensor(pert[start:start+batch],device=device,dtype=torch.float32)
            preds.append(model.predict_new(xb,draws=draws).detach().cpu().numpy())
        rows.append({'missing_fraction':float(fraction),'removed_channels':removed.tolist(),
                     'probabilities':np.concatenate(preds).tolist()})
    return rows

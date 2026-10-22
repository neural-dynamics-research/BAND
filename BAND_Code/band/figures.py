import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .eval import ece,risk_coverage_auc

def reliability(csv,out,bins=10):
    d=pd.read_csv(csv);y=d.y_true.to_numpy(int);p=d.p_pd.to_numpy(float)
    edges=np.linspace(0,1,bins+1);xs=[];ys=[];sizes=[]
    for j in range(bins):
        mask=(p>=edges[j])&((p<=edges[j+1]) if j==bins-1 else (p<edges[j+1]))
        if mask.any():xs.append(p[mask].mean());ys.append(y[mask].mean());sizes.append(mask.sum())
    fig,ax=plt.subplots(figsize=(5,4));ax.plot([0,1],[0,1],ls='--',color='grey')
    ax.plot(xs,ys,'o-');ax.set(xlabel='Mean predicted PD probability',ylabel='Observed PD frequency',xlim=(0,1),ylim=(0,1))
    fig.tight_layout();fig.savefig(out);plt.close(fig)
    return pd.DataFrame({'mean_probability':xs,'observed_frequency':ys,'n':sizes})

def risk_coverage(csv,out):
    d=pd.read_csv(csv);y=d.y_true.to_numpy(int);p=d.p_pd.to_numpy(float)
    conf=np.maximum(p,1-p);order=np.argsort(-conf);errors=((p>=.5).astype(int)!=y)[order]
    coverage=np.arange(1,len(y)+1)/len(y);risk=np.cumsum(errors)/np.arange(1,len(y)+1)
    fig,ax=plt.subplots(figsize=(5,4));ax.plot(coverage,risk);ax.set(xlabel='Coverage',ylabel='Selective risk',xlim=(0,1))
    fig.tight_layout();fig.savefig(out);plt.close(fig)
    return pd.DataFrame({'coverage':coverage,'risk':risk})

def main():
    p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    reliability(a.predictions,out/'reliability.pdf').to_csv(out/'reliability_source.csv',index=False)
    risk_coverage(a.predictions,out/'risk_coverage.pdf').to_csv(out/'risk_coverage_source.csv',index=False)
if __name__=='__main__':main()

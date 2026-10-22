"""Participant-level stratified label permutation; preserve one diagnosis label per subject."""
import argparse
from pathlib import Path
import numpy as np
from .data import load_npz

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
 X,y,s,names=load_npz(a.data);ids=np.unique(s);labs=np.array([y[s==i][0] for i in ids]);perm=np.random.default_rng(a.seed).permutation(labs)
 mapping=dict(zip(ids,perm));yp=np.array([mapping[i] for i in s]);reverse={v:k for k,v in names.items()}
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(out,X=X,y=yp,subject_id=np.array([reverse[int(i)] for i in s]))
if __name__=='__main__':main()

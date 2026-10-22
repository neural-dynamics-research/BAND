import argparse,json
from pathlib import Path
from types import SimpleNamespace
from .train import run

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True)
 p.add_argument('--folds',type=int,default=3);p.add_argument('--epochs',type=int,default=3)
 p.add_argument('--hidden',type=int,default=24);p.add_argument('--latent',type=int,default=12)
 p.add_argument('--lr',type=float,default=.001);p.add_argument('--batch',type=int,default=32)
 p.add_argument('--draws',type=int,default=8);p.add_argument('--seed',type=int,default=42)
 p.add_argument('--adv',type=float,default=.02);p.add_argument('--orth',type=float,default=.02)
 p.add_argument('--phys',type=float,default=.05);p.add_argument('--cpu',action='store_true')
 a=p.parse_args();summaries={}
 for ab in ['full','no_transitions','no_subject_effect','no_invariance','no_covariance','no_probabilistic']:
  kw=vars(a).copy();kw['ablation']=ab;kw['output']=str(Path(a.output)/ab)
  summaries[ab]=run(SimpleNamespace(**kw))
 (Path(a.output)/'ablation_index.json').write_text(json.dumps(summaries,indent=2))
if __name__=='__main__':main()

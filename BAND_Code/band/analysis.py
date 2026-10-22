"""CLI for physiology and clinical CSV analyses."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from .physiology import participant_features,sensor_beta_map
from .clinical import paired_medication,clinical_association

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='task',required=True)
    a=sub.add_parser('physiology');a.add_argument('--data',required=True);a.add_argument('--sfreq',type=float,default=250);a.add_argument('--output',required=True)
    a=sub.add_parser('medication');a.add_argument('--csv',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('association');a.add_argument('--csv',required=True);a.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if args.task=='physiology':
        d=np.load(args.data,allow_pickle=False);required={'X','subject_id'}
        if not required.issubset(d.files):raise ValueError('Data must contain X and subject_id')
        pd.DataFrame(participant_features(d['X'],d['subject_id'],args.sfreq)).to_csv(out/'participant_features.csv',index=False)
        np.save(out/'sensor_beta_maps.npy',sensor_beta_map(d['X'],d['subject_id'],args.sfreq))
    elif args.task=='medication':
        paired,summary=paired_medication(pd.read_csv(args.csv));paired.to_csv(out/'paired_values.csv',index=False)
        (out/'paired_stats.json').write_text(json.dumps(summary,indent=2))
    else:
        per,summary=clinical_association(pd.read_csv(args.csv));per.to_csv(out/'association_values.csv',index=False)
        (out/'association_stats.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()

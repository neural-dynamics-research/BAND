import argparse
from pathlib import Path
import numpy as np
import mne

def make_windows(path, label, seconds=4., overlap=.5, l_freq=1., h_freq=100., sfreq=250.):
    raw=mne.io.read_raw_fif(str(path),preload=True,verbose='ERROR')
    raw.pick('meg');raw.filter(l_freq,h_freq,verbose='ERROR')
    raw.notch_filter([50],verbose='ERROR')
    raw.resample(sfreq,verbose='ERROR')
    data=raw.get_data(reject_by_annotation='omit')
    size=round(seconds*sfreq);step=max(1,round(size*(1-overlap)))
    wins=np.stack([data[:,j:j+size] for j in range(0,data.shape[1]-size+1,step)])
    return wins.astype('float32'),np.full(len(wins),label)

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True,help='CSV subject_id,path,label (label 0=HC 1=PD)')
    p.add_argument('--output',required=True);a=p.parse_args()
    import csv
    X=[];y=[];subjects=[];expected_channels=None
    with open(a.manifest,newline='') as f:
        for row in csv.DictReader(f):
            w,lab=make_windows(Path(row['path']),int(row['label']))
            if expected_channels is None:expected_channels=w.shape[1]
            if w.shape[1]!=expected_channels:raise ValueError('Channel mismatch. Harmonize sensor layouts first.')
            X.append(w);y.extend(lab);subjects.extend([row['subject_id']]*len(w))
    np.savez_compressed(a.output,X=np.concatenate(X),y=np.array(y),subject_id=np.array(subjects))
if __name__=='__main__':main()

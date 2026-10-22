import numpy as np
from scipy.signal import welch,butter,sosfiltfilt,hilbert
from scipy.stats import linregress

def bandpower(windows, sfreq, band=(13,30), relative=False):
    x=np.asarray(windows,dtype=float)
    if x.ndim!=3: raise ValueError('Expected [windows, channels, samples]')
    f,p=welch(x,fs=sfreq,nperseg=min(x.shape[-1],round(2*sfreq)),axis=-1)
    sel=(f>=band[0])&(f<=band[1]); total=(f>=1)&(f<=min(100,sfreq/2))
    if sel.sum()<2:raise ValueError('Insufficient frequency resolution for beta band')
    beta=np.trapezoid(p[...,sel],f[sel],axis=-1)
    if relative: beta=beta/np.maximum(np.trapezoid(p[...,total],f[total],axis=-1),1e-20)
    return beta

def burst_durations(windows,sfreq=250,band=(13,30),threshold_percentile=75,min_duration=.10):
    """Return per-window lists of event durations in seconds; percentile is per channel/window."""
    x=np.asarray(windows,dtype=float)
    if x.ndim!=3:raise ValueError('Expected [windows,channels,samples]')
    sos=butter(4,band,fs=sfreq,btype='bandpass',output='sos')
    amp=np.abs(hilbert(sosfiltfilt(sos,x,axis=-1),axis=-1))
    thresh=np.percentile(amp,threshold_percentile,axis=-1,keepdims=True)
    events=[]
    for w in range(len(x)):
        durations=[]
        for c in range(x.shape[1]):
            mask=(amp[w,c]>thresh[w,c]).astype(int)
            edges=np.diff(np.r_[0,mask,0]); starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
            durations.extend([float((b-a)/sfreq) for a,b in zip(starts,ends) if (b-a)/sfreq>=min_duration])
        events.append(durations)
    return events

def participant_features(windows,subject_ids,sfreq=250):
    """Participant averages to avoid window-level pseudoreplication."""
    bp=bandpower(windows,sfreq,relative=True).mean(axis=1)
    bursts=burst_durations(windows,sfreq)
    rows=[]
    for sid in np.unique(subject_ids):
        mask=np.flatnonzero(np.asarray(subject_ids)==sid)
        vals=[v for i in mask for v in bursts[i]]
        rows.append(dict(subject_id=str(sid),beta_relative=float(bp[mask].mean()),
                         beta_burst_mean_seconds=float(np.mean(vals)) if vals else np.nan,
                         n_windows=int(len(mask)),n_bursts=len(vals)))
    return rows

def sensor_beta_map(windows,subject_ids,sfreq=250):
    """Rows are participant mean topographies; sensor channels must be aligned."""
    bp=bandpower(windows,sfreq,relative=True)
    return np.stack([bp[np.asarray(subject_ids)==s].mean(axis=0) for s in np.unique(subject_ids)])

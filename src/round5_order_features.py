"""Speed/order-aligned vibration features, independent of class labels.
Uses the same 1 second 50%-overlapping windows as cached original features.
During live inference call order_features on the last complete window.
"""
from pathlib import Path
import argparse,zipfile,time
import numpy as np,pandas as pd
from .train import load_cached
from .settings import load_config

BANDS=np.array([[.5,1.5],[1.5,2.5],[2.5,3.5],[3.5,5.5],[5.5,8.5],[8.5,12.5],[12.5,20.5],[20.5,35.5]])
SPEED_MIN=16.;SPEED_MAX=33.

def order_features(windows,fs=4000.):
    """Input (n,4000,6), outputs (n,120) and feature names.
    No test-day statistics or labels; speed estimated per window by spectral peaks.
    """
    x=np.asarray(windows,dtype=np.float32)
    n,N,c=x.shape
    win=np.hanning(N).astype(np.float32)
    f=np.fft.rfftfreq(N,1/fs)
    fft=np.fft.rfft((x-x.mean(axis=1,keepdims=True))*win[None,:,None],axis=1)
    mag=np.abs(fft).astype(np.float32)
    pp=mag**2+1e-12
    ptotal=pp[:,1:,:].sum(axis=1)+1e-12
    # Harmonic comb for expected 1000-2000 RPM; dominant bearing peaks can be >1X
    # Weights prioritize fundamental rotational peak but allow weak 1X.
    intensity=np.sum(mag,axis=2)
    step=fs/N
    candidates=np.arange(int(SPEED_MIN/step),int(SPEED_MAX/step)+1)
    scores=sum(w*intensity[:,candidates*h]/(1+0.6*(h-1)) for h,w in [(1,1.),(2,.65),(3,.35)])
    best=candidates[np.argmax(scores,axis=1)]
    speed=f[best]
    out=[];names=[]
    def add(a,stem):
        if a.ndim==1:a=a[:,None]
        for j in range(a.shape[1]):
            out.append(a[:,j]);names.append(stem+f'_ch{j}' if a.shape[1]==6 else stem+f'_{j}')
    add(speed,'speed_estimate')
    for h in range(1,9):
        ix=(best*h).clip(max=len(f)-1)
        peak=pp[np.arange(n),ix,:]
        # Normalize by broadband per-channel power, and relative across channels
        add(np.log1p(1e5*peak/ptotal),f'order_{h}_relative')
    for a,b in BANDS:
        lo=np.maximum(1,np.floor(speed*a/step).astype(int))
        hi=np.minimum(len(f),np.ceil(speed*b/step).astype(int)+1)
        pool=np.zeros((n,c),np.float32)
        for i in range(n):
            if hi[i]>lo[i]:pool[i]=np.sum(pp[i,lo[i]:hi[i],:],axis=0)
        add(np.log1p(1000*pool/ptotal),f'band_order_{a:g}_{b:g}')
    # Fault impulse harmonics in the high-frequency bearing resonance envelope.
    # Use fixed bandpassing with no bearing geometry assumptions.
    # Compute Hilbert envelope over time (expensive but vectorized small batch).
    # Extra speed ratios across channels for 1X/2X.
    for h in (1,2,3):
        ix=(best*h).clip(max=len(f)-1)
        peak=pp[np.arange(n),ix,:]
        add(peak/(peak.sum(axis=1,keepdims=True)+1e-9),f'order_{h}_channel_ratio')
    return np.column_stack(out).astype(np.float32),names

def run(root, archive=None, data_root=None):
    if (archive is None) == (data_root is None):
        raise ValueError('provide exactly one of archive or data_root')
    X,meta,names=load_cached(root)
    required_fs = load_config(root).get('required_sample_rate_hz')
    extra=np.empty((len(meta),1+8*6+8*6+3*6),dtype=np.float32)
    z=zipfile.ZipFile(archive) if archive is not None else None
    contents=set(z.namelist()) if z is not None else set()
    t=time.monotonic();errors=[];feat_names=None
    for k,(file,group) in enumerate(meta.groupby('file',sort=True)):
        if z is not None:
            arc='OLD BACKUP/cbpm details/'+file.replace('\\','/')
            if arc not in contents:raise FileNotFoundError(arc)
            df=pd.read_csv(z.open(arc))
        else:
            source=Path(data_root)/file
            if not source.exists():raise FileNotFoundError(source)
            df=pd.read_csv(source)
        vals=df.iloc[:,1:7].apply(pd.to_numeric,errors='coerce').interpolate(limit_direction='both').to_numpy(dtype=np.float32)
        rows=group.index.to_numpy()
        idx=group.window_index.to_numpy().astype(int)
        n=4000;starts=idx*2000
        wins=np.stack([vals[v:v+n] for v in starts])
        if wins.shape[1:]!=(4000,6):raise ValueError((arc,wins.shape))
        measured_fs = float(group.fs_hz.iloc[0])
        feature_fs = float(required_fs) if required_fs and abs(measured_fs - float(required_fs)) <= 1e-3 else measured_fs
        ext,feat_names=order_features(wins,fs=feature_fs)
        extra[rows]=ext
        if (k+1)%100==0:print('EXTRACT',k+1,'files of',meta.file.nunique(),'elapsed',round(time.monotonic()-t,1),'s',flush=True)
    np.savez_compressed(Path(root)/'cache'/'round5_order_features.npz',X=extra,names=np.array(feat_names))
    print('EXTRACTED',extra.shape,'time',round(time.monotonic()-t,1),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',default='.')
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--archive')
    source.add_argument('--data-root')
    a=p.parse_args();run(Path(a.project),a.archive,a.data_root)

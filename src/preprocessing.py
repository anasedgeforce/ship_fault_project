"""Single windowing + normalization implementation for both training and live prediction."""
from __future__ import annotations
import numpy as np
from scipy.signal import resample_poly

def estimate_fs(time):
    time=np.asarray(time,dtype=np.float64)
    d=np.diff(time)
    pos=d[np.isfinite(d)&(d>0)]
    if len(pos)<2: raise ValueError('Insufficient increasing timestamps')
    dt=float(np.median(pos)); fs=1./dt
    irregular=int(np.count_nonzero((d<=0)|(~np.isfinite(d))|(np.abs(d-dt)>max(dt*0.02,1e-7))))
    return fs, irregular, int(np.sum(d==0)), float(np.mean(np.abs(pos-dt)>max(dt*0.02,1e-7)))

def window_signal(x,fs,seconds=1.0,hop_fraction=.5):
    x=np.asarray(x,dtype=np.float32)
    if x.ndim!=2 or x.shape[1]!=6:raise ValueError('Expected array of shape (N, 6)')
    n=max(8,int(round(fs*seconds))); hop=max(1,int(round(n*hop_fraction)))
    if x.shape[0]<n:return np.empty((0,n,6),dtype=np.float32)
    # copy: ensures independently addressable windows on disk / serialization
    return np.lib.stride_tricks.sliding_window_view(x,n,axis=0)[::hop].transpose(0,2,1).copy()

class ChannelNormalizer:
    def __init__(self):self.mean=None;self.std=None
    def fit(self,windows):
        x=np.asarray(windows)
        self.mean=np.mean(x,axis=(0,1)).astype(np.float32)
        self.std=np.maximum(np.std(x,axis=(0,1)),1e-7).astype(np.float32);return self
    def transform(self,x):
        if self.mean is None:raise ValueError('Fit on training only')
        return (np.asarray(x,dtype=np.float32)-self.mean)/self.std

"""Vectorized multichannel features shared by training and serving."""
import numpy as np
from scipy import signal, stats

EPS=1e-9
TIME_FEATURE_NAME='elapsed_window_start_s'


def append_elapsed_time_feature(features, names, elapsed_starts):
    """Add the Time-column position of each window relative to recording start."""
    starts=np.asarray(elapsed_starts,dtype=np.float64)
    if starts.shape!=(len(features),) or not np.isfinite(starts).all() or np.any(starts < -1e-6):
        raise ValueError('expected one finite, nonnegative elapsed time per window')
    return np.column_stack((features,starts.astype(np.float32))), [*names,TIME_FEATURE_NAME]

def extract_features(windows, fs, speed_hz=None, bearing_geometry=None):
    """Return (float32 matrix, named feature list) for windows (batch,time,6)."""
    x=np.asarray(windows,dtype=np.float64)
    if x.ndim!=3 or x.shape[-1]!=6:raise ValueError('expected (windows,samples,6)')
    W,N,C=x.shape
    if W==0:return np.empty((0,0),dtype=np.float32), []
    av=np.abs(x); mean=np.mean(x,axis=1); rms=np.sqrt(np.mean(x*x,axis=1)+EPS)
    std=np.std(x,axis=1); peak=np.max(av,axis=1); p2p=np.ptp(x,axis=1)
    absmean=np.mean(av,axis=1)+EPS
    sqrtabs=np.mean(np.sqrt(av+EPS),axis=1)**2+EPS
    vals=[];names=[]
    def add(arr,name):
        a=np.asarray(arr)
        if a.ndim==1:a=a[:,None]
        for j in range(a.shape[1]):
            nm=f'{name}_ch{j}' if a.shape[1]==C else f'{name}_{j}'
            vals.append(a[:,j]); names.append(nm)
    add(mean,'mean');add(std,'std');add(rms,'rms');add(peak,'peak');add(p2p,'p2p')
    add(peak/(rms+EPS),'crest');add(stats.skew(x,axis=1,bias=False),'skew')
    add(stats.kurtosis(x,axis=1,bias=False),'kurtosis');add(std**2,'variance')
    add(rms/absmean,'shape');add(peak/absmean,'impulse');add(peak/sqrtabs,'clearance')
    win=np.hanning(N)[None,:,None]
    fft=np.fft.rfft(x*win,axis=1)
    freqs=np.fft.rfftfreq(N,1./fs)
    power=np.abs(fft)**2 + EPS
    mags=np.abs(fft)*2/max(np.sum(win),EPS)
    power[:,0,:]=EPS
    add(freqs[np.argmax(power,axis=1)],'dominant_hz')
    total=power.sum(axis=1)+EPS
    add(np.sum(power*freqs[None,:,None],axis=1)/total,'centroid_hz')
    spec_prob=power/total[:,None,:]
    add(-np.sum(spec_prob*np.log(spec_prob+EPS),axis=1)/np.log(max(2,len(freqs))),'spectral_entropy')
    nyquist=fs*.5
    for lo,hi in [(0,50),(50,150),(150,500),(500,1200),(1200,2000)]:
        mask=(freqs>=lo)&(freqs<min(hi,nyquist))
        add(np.sum(power[:,mask,:],axis=1)/total if np.any(mask) else np.zeros_like(rms),f'band_{lo}_{hi}')
    if speed_hz is not None and float(speed_hz)>0:
        for harmonic in (1,2,3):
            ix=int(np.argmin(np.abs(freqs-(harmonic*float(speed_hz)))))
            add(mags[:,ix,:],f'harmonic_{harmonic}')
    else:
        for harmonic in (1,2,3):add(np.zeros_like(rms),f'harmonic_{harmonic}')
    # Envelope demodulation, bandpass with Nyquist guard.
    flo=max(30.,nyquist*.13);fhi=min(nyquist*.85,1300.)
    if fhi>flo*1.5 and N>64:
        try:
            sos=signal.butter(3,[flo,fhi],btype='bandpass',fs=fs,output='sos')
            filtered=signal.sosfiltfilt(sos,x,axis=1)
            env=np.abs(signal.hilbert(filtered,axis=1))
            env=env-np.mean(env,axis=1,keepdims=True)
            ep=np.abs(np.fft.rfft(env*win,axis=1))**2
            ef=np.fft.rfftfreq(N,1/fs)
            for lo,hi in [(0,50),(50,150),(150,350),(350,650)]:
                mask=(ef>=lo)&(ef<min(hi,nyquist))
                add(ep[:,mask,:].sum(axis=1) if np.any(mask) else np.zeros_like(rms),f'env_{lo}_{hi}')
            if bearing_geometry and all(k in bearing_geometry for k in ('bpfo_hz','bpfi_hz')):
                for key in ('bpfo_hz','bpfi_hz'):
                    ix=int(np.argmin(np.abs(ef-float(bearing_geometry[key]))))
                    add(ep[:,ix,:],key)
        except ValueError:
            for lo,hi in [(0,50),(50,150),(150,350),(350,650)]:add(np.zeros_like(rms),f'env_{lo}_{hi}')
    else:
        for lo,hi in [(0,50),(50,150),(150,350),(350,650)]:add(np.zeros_like(rms),f'env_{lo}_{hi}')
    # Six channels: DE V,H,A / NDE V,H,A
    for i,j,name in [(0,3,'de_nde_v'),(1,4,'de_nde_h'),(2,5,'de_nde_a'),
                     (0,1,'de_vh'),(1,2,'de_ha'),(3,4,'nde_vh'),(4,5,'nde_ha')]:
        add((rms[:,i]/(rms[:,j]+EPS)),f'rms_ratio_{name}')
    for i,j,name in [(0,3,'v'),(1,4,'h'),(2,5,'a')]:
        # phase at dominant power of first channel
        ix=np.argmax(power[:,:,i],axis=1)
        phase=np.angle(fft[np.arange(W),ix,i]*np.conj(fft[np.arange(W),ix,j]))
        add(np.cos(phase),f'phase_cos_{name}')
        add(np.sin(phase),f'phase_sin_{name}')
    out=np.column_stack(vals)
    out=np.nan_to_num(out,nan=0.,posinf=0.,neginf=0.).astype(np.float32)
    return out,names

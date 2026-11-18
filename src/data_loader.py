"""Audit all files, create label map, and cache all window features and metadata."""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np,pandas as pd,yaml
from .labels import attributes,label_for
from .preprocessing import estimate_fs,window_signal
from .features import extract_features,append_elapsed_time_feature,TIME_FEATURE_NAME


def root_files(root):
    root=Path(root)
    return sorted(root.glob('data_day*'+'/**/*.csv'))

def read_signal(file,channels):
    df=pd.read_csv(file)
    if list(df.columns)!=['Time']+list(channels):
        raise ValueError('CSV must contain Time followed by the six velocity channels in the documented order')
    result=df.apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float64)
    if not np.isfinite(result[:,0]).all():raise ValueError('Time values must all be finite numbers')
    if not np.isfinite(result[:,1:]).all():raise ValueError('sensor readings must all be finite numbers')
    return result

def generate_cache(project,config):
    project=Path(project);root=Path(config['data_root']); cache=project/'cache';cache.mkdir(exist_ok=True)
    files=root_files(root)
    if not files:raise RuntimeError('No CSV files found; edit config/config.yaml data_root')
    configured_exclusions={str(item).replace('\\','/') for item in config.get('excluded_files',[])}
    reserved_sensor_hashes=set(config.get('excluded_sensor_sha256',[]))
    folders=sorted({f.relative_to(root).as_posix().split('/',1)[1].rsplit('/',1)[0] for f in files})
    label_map={f:label_for(f) for f in folders}
    (project/'config'/'label_map.yaml').write_text(yaml.safe_dump(label_map,sort_keys=True))
    rows=[];metadata=[]; feats=[];f_names=None;examples={};rms_rows=[];skipped=[]
    # estimate speed only from normal baseline recordings, before other feature calculation.
    normal_file=next((f for f in files if f.relative_to(root).as_posix() not in configured_exclusions and attributes(f.relative_to(root).as_posix().split('/',1)[1].rsplit('/',1)[0])['fault_type']=='normal'),None)
    speed=config.get('run_speed_hz')
    if speed is None and normal_file:
        from scipy.signal import find_peaks
        try:
            arr=read_signal(normal_file,config['expected_channels']);fs,_,_,_=estimate_fs(arr[:,0])
            ps=np.abs(np.fft.rfft(arr[:min(len(arr),32000),1]-np.mean(arr[:,1])))**2
            fr=np.fft.rfftfreq(min(len(arr),32000),1/fs)
            allowed=(fr>=5)&(fr<=100)
            speed=float(fr[allowed][np.argmax(ps[allowed])])
        except Exception:speed=None
    if speed is None:speed=25.0 # documented fallback
    state={
        'estimated_speed_hz': float(speed),
        'speed_source': 'configured_reference' if config.get('run_speed_hz') is not None else 'recording_estimate',
        'speed_estimation_file': (str(normal_file) if normal_file and config.get('run_speed_hz') is None else None),
        'time_feature': config.get('time_feature'),
    }
    (cache/'data_settings.json').write_text(json.dumps(state,indent=2))
    for idx,file in enumerate(files):
        rel=file.relative_to(root).as_posix();day,folder,filename=rel.split('/',1)[0],rel.split('/',1)[1].rsplit('/',1)[0],file.name
        label=label_map[folder];a=attributes(folder)
        entry=dict(day=day,folder=folder,label=label,file=rel)
        if rel in configured_exclusions:
            reason='Configured file exclusion'
            entry.update(status='excluded',reason=reason,windows=0)
            skipped.append(dict(file=rel,reason=reason))
            rows.append(entry)
            continue
        try:
            arr=read_signal(file,config['expected_channels'])
            if reserved_sensor_hashes:
                sensor_values=np.ascontiguousarray(arr[:,1:].astype('<f8',copy=False))
                sensor_hash=hashlib.sha256(
                    np.asarray(sensor_values.shape,dtype='<i8').tobytes()+sensor_values.tobytes()).hexdigest()
                if sensor_hash in reserved_sensor_hashes:
                    reason='Reserved unseen_test sensor signal; excluded from training'
                    entry.update(status='excluded_reserved',reason=reason,windows=0)
                    skipped.append(dict(file=rel,reason=reason))
                    rows.append(entry)
                    continue
            fs,irregular,duplicate,irregular_fraction=estimate_fs(arr[:,0]);n=len(arr);duration=float(arr[-1,0]-arr[0,0])
            if np.any(np.diff(arr[:,0]) <= 0):
                raise ValueError('timestamps must strictly increase')
            invalid=int(np.count_nonzero(~np.isfinite(arr)));frac=invalid/max(1,arr.size)
            checks={}
            for j,channel in enumerate(config['expected_channels']):
                v=arr[:,j+1];q=v[np.isfinite(v)]
                for k,vv in [('mean',np.mean(q) if len(q) else np.nan),('std',np.std(q) if len(q) else np.nan),('min',np.min(q) if len(q) else np.nan),('max',np.max(q) if len(q) else np.nan)]:checks[f'channel{j}_{k}']=float(vv)
                checks[f'channel{j}_flatline']=bool(len(q) and np.std(q)<1e-6)
                checks[f'channel{j}_saturation']=bool(len(q) and (np.mean(q==np.max(q))>0.05 or np.mean(q==np.min(q))>0.05))
            entry.update(rows=n,fs_hz=fs,duration_s=duration,nan_inf=invalid,duplicates=duplicate,irregular_timestamps=irregular,irregular_fraction=irregular_fraction,**checks)
            if frac>config['max_nan_fraction']:raise ValueError(f'NaN/inf fraction {frac:.5f}')
            if irregular_fraction>config['max_timestamp_irregular_fraction'] or duplicate>0:raise ValueError(f'timestamp irregular/duplicate: {irregular_fraction:.5f}, {duplicate}')
            if not (100 <= fs <= 50000):raise ValueError(f'Implausible sampling rate: {fs}')
            if config.get('required_sample_rate_hz') and abs(fs-float(config['required_sample_rate_hz']))>1e-3:
                raise ValueError(f"expected {config['required_sample_rate_hz']} Hz samples, got {fs:g}")
            if any(checks[f'channel{j}_flatline'] for j in range(6)):raise ValueError('flat-line channel')
            if invalid:arr[:,1:]=pd.DataFrame(arr[:,1:]).interpolate(limit_direction='both').to_numpy()
            windows=window_signal(arr[:,1:],fs,config['window_seconds'],config['hop_fraction'])
            if not len(windows):raise ValueError('Too short for a window')
            feature_fs = float(config['required_sample_rate_hz']) if config.get('required_sample_rate_hz') else fs
            X,names=extract_features(windows,feature_fs,speed,config.get('bearing_geometry'))
            if config.get('time_feature')==TIME_FEATURE_NAME:
                hop=max(1,int(round(max(8,int(round(fs*config['window_seconds'])))*config['hop_fraction'])))
                starts=np.arange(len(windows))*hop
                X,names=append_elapsed_time_feature(X,names,arr[starts,0]-arr[0,0])
            if f_names is None:f_names=names
            elif names!=f_names:raise ValueError('Inconsistent feature schemas')
            feats.append(X)
            rms=np.sqrt(np.mean(windows**2,axis=1))
            rms_rows.extend(dict(day=day,label=label,channel=c,rms=float(v)) for c in range(6) for v in rms[:,c])
            for wi in range(len(X)):
                metadata.append(dict(day=day,folder=folder,file=rel,label=label,window_index=wi,fs_hz=fs,**a))
            if label not in examples:examples[label]=(arr[:min(len(arr),32000),1:].copy(),fs,rel)
            entry.update(status='ok',reason='',windows=len(windows))
        except Exception as exc:
            entry.update(status='excluded',reason=str(exc),windows=0)
            skipped.append(dict(file=rel,reason=str(exc)))
        rows.append(entry)
        if (idx+1)%100==0:print(f'Audited {idx+1}/{len(files)} files',flush=True)
    audit=pd.DataFrame(rows);audit.to_csv(project/'reports'/'data_audit.csv',index=False)
    pd.DataFrame(skipped,columns=['file','reason']).to_csv(project/'reports'/'excluded_files.csv',index=False)
    if not feats:raise RuntimeError('No usable windows')
    matrix=np.concatenate(feats);meta=pd.DataFrame(metadata)
    np.savez_compressed(cache/'window_features.npz',X=matrix,names=np.asarray(f_names))
    meta.to_csv(cache/'window_metadata.csv',index=False)
    pd.DataFrame(rms_rows).to_csv(cache/'window_rms.csv',index=False)
    pd.DataFrame([dict(file=v[2],label=k,fs_hz=v[1]) for k,v in examples.items()]).to_csv(project/'reports'/'example_index.csv',index=False)
    _create_figures(project,examples,pd.DataFrame(rms_rows))
    _audit_md(project,audit,meta,state,skipped)
    from .deduplicate import deduplicate_cache
    deduplicate_cache(project,config)
    from .round5_order_features import run as extract_order_features
    extract_order_features(project, data_root=root)
    from .cache_utils import mark_cache
    mark_cache(project,config)
    print('WINDOW MATRIX',matrix.shape, 'GOOD FILES',sum(audit.status=='ok'),'EXCLUSIONS',len(skipped),flush=True)
    return matrix,meta,f_names

def _create_figures(project,examples,rms):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=project/'reports'/'figures';out.mkdir(exist_ok=True)
    for label,(x,fs,file) in examples.items():
        fig,axes=plt.subplots(2,1,figsize=(9,5),layout='constrained')
        n=min(len(x),int(fs*.15));t=np.arange(n)/fs
        for c in range(6):axes[0].plot(t,x[:n,c],lw=.55,label=f'Ch {c+1}')
        axes[0].set(title=f'{label} — {file}',xlabel='Time (s)',ylabel='Velocity (mm/s)')
        axes[0].legend(ncol=6,fontsize=7)
        f,p=__import__('scipy').signal.welch(x[:,0],fs=fs,nperseg=min(2048,len(x)))
        axes[1].semilogy(f,p+1e-14);axes[1].set(xlim=(0,min(fs/2,1600)),xlabel='Frequency (Hz)',ylabel='Welch PSD')
        fig.savefig(out/('example_'+label+'.png'),dpi=105);plt.close(fig)
    # Each channel, day-specific distributions of window RMS per condition
    for day,gg in rms.groupby('day'):
        labels=sorted(gg.label.unique());fig,axs=plt.subplots(3,2,figsize=(15,max(8,round(len(labels)*.35))),layout='constrained')
        for c,ax in enumerate(axs.ravel()):
            vals=[gg[(gg.label==label)&(gg.channel==c)].rms.to_numpy() for label in labels]
            ax.boxplot(vals,showfliers=False,vert=False,manage_ticks=False)
            ax.set_yticks(range(1,len(labels)+1));ax.set_yticklabels(labels,fontsize=5)
            ax.set_xlabel('Window RMS (mm/s)');ax.set_title(f'Channel {c+1}')
        fig.suptitle('Day-to-day condition check — '+day)
        fig.savefig(out/('rms_'+day+'.png'),dpi=110);plt.close(fig)

def _audit_md(project,audit,meta,state,skipped):
    counts=meta.groupby(['label','day']).agg(windows=('label','size'),files=('file','nunique')).reset_index()
    counts.to_csv(project/'reports'/'class_counts.csv',index=False)
    days=counts.groupby('label').day.nunique();rare=days[days<3].index.tolist()
    text=['# Dataset audit','',f'Files inspected: **{len(audit)}**; usable: **{sum(audit.status=="ok")}**; excluded: **{len(skipped)}**.',
          f'Windows: **{len(meta)}**. Harmonic reference: **{state["estimated_speed_hz"]:.2f} Hz** '
          f'({state["speed_source"]}; not a measured tachometer signal).',
          '','## Representation by day and class','',counts.to_markdown(index=False),'',
          '## Classes absent from one or more days','',', '.join(rare) or 'None','',
          '## Excluded/altered recordings','',pd.DataFrame(skipped,columns=['file','reason']).to_markdown(index=False) if skipped else 'None.',
          '','## Interpretation','', 'Conditions absent on a day cannot have their same-class recall measured on that day. '
          'LODO results must therefore be interpreted alongside class coverage and the group-wise scores.',
          '','Sampling rate is estimated independently for each file and irregular timestamps are checked. See `data_audit.csv` for channel statistics.']
    (project/'reports'/'data_audit.md').write_text('\n'.join(text))

"""Content-based de-duplication across files AND across recording days (leakage control)."""
from pathlib import Path
from collections import defaultdict
import hashlib
import numpy as np,pandas as pd,yaml

def content_duplicates(root, eligible=None):
    root=Path(root)
    files=sorted(root.glob('data_day*/**/*.csv'))
    observed={};repeated=[];hashes=[]
    for file in files:
        h=hashlib.sha256()
        with file.open('rb') as inp:
            for chunk in iter(lambda:inp.read(1<<20),b''):h.update(chunk)
        digest=h.hexdigest();rel=file.relative_to(root).as_posix()
        hashes.append(dict(file=rel,sha256=digest))
        if eligible is not None and rel not in eligible:continue
        if digest in observed:
            repeated.append(dict(file=rel,original_file=observed[digest],sha256=digest,
                                 across_days=rel.split('/')[0]!=observed[digest].split('/')[0]))
        else:observed[digest]=rel
    return pd.DataFrame(repeated,columns=['file','original_file','sha256','across_days']),pd.DataFrame(hashes)

def deduplicate_cache(project,config):
    project=Path(project)
    audit=pd.read_csv(project/'reports'/'data_audit.csv',keep_default_na=False)
    usable=set(audit.loc[audit.status=='ok','file'])
    dup,hashes=content_duplicates(config['data_root'],eligible=usable)
    dup.to_csv(project/'reports'/'duplicate_recordings.csv',index=False)
    hashes.to_csv(project/'reports'/'file_hashes.csv',index=False)
    paths=set(dup.file)
    if len(paths):
        matching=audit.file.isin(paths)
        original=dict(zip(dup.file,dup.original_file))
        audit.loc[matching,'status']='excluded_duplicate'
        audit.loc[matching,'reason']=audit.loc[matching,'file'].map(lambda s:'Exact content duplicate of '+original[s])
        audit.loc[matching,'windows']=0
    audit.to_csv(project/'reports'/'data_audit.csv',index=False)
    skipped=audit[audit.status!='ok'][['file','reason']]
    skipped.to_csv(project/'reports'/'excluded_files.csv',index=False)
    metapath=project/'cache'/'window_metadata.csv'
    meta=pd.read_csv(metapath,keep_default_na=False)
    keep=~meta.file.isin(paths).to_numpy()
    if not keep.all():
        old=np.load(project/'cache'/'window_features.npz')
        matrix=old['X'];names=old['names']
        if len(keep)!=len(matrix):raise RuntimeError('Feature cache and metadata misaligned')
        np.savez_compressed(project/'cache'/'window_features.npz',X=matrix[keep],names=names)
        meta=meta.loc[keep].reset_index(drop=True)
        meta.to_csv(metapath,index=False)
    counts=meta.groupby(['label','day']).agg(windows=('label','size'),files=('file','nunique')).reset_index()
    counts.to_csv(project/'reports'/'class_counts.csv',index=False)
    # Rebuild RMS-distribution samples from filtered feature cache to avoid duplicate influence.
    with np.load(project/'cache'/'window_features.npz') as npz:
        features=npz['X'];names=npz['names'].tolist()
    rms_array=features[:,[names.index(f'rms_ch{c}') for c in range(6)]]
    rms_rows=[]
    for c in range(6):
        rms_rows.append(pd.DataFrame(dict(day=meta.day.to_numpy(),label=meta.label.to_numpy(),
                                file=meta.file.to_numpy(),channel=c,rms=rms_array[:,c])))
    rms=pd.concat(rms_rows,ignore_index=True)
    rms.to_csv(project/'cache'/'window_rms.csv',index=False)
    from .data_loader import _create_figures
    _create_figures(project,{},rms)
    days=counts.groupby('label').day.nunique()
    notes=['# Data audit (duplicate-aware)','',
        f'CSV recordings audited: {len(audit)}. Unique usable files: {sum(audit.status=="ok")}.',
        f'Exact duplicate recordings excluded: {len(dup)} (including {sum(dup.across_days)} across-day duplicates).',
        f'Windows after de-duplication: {len(meta)}.',
        'The exclusion is made by raw-file SHA-256 before any split. See `duplicate_recordings.csv` and `excluded_files.csv`.',
        f'All {len(audit)} CSVs underwent sampling/statistical integrity checks before this exclusion.',
        '','## Classes per day',counts.to_markdown(index=False),'',
        '## Classes absent from one or more recording days',', '.join(days[days<meta.day.nunique()].index.tolist()),'',
        '## CSV integrity','See `data_audit.csv` for sampling, NaN, timing and per-channel descriptive statistics.',
        f'Integrity audit recorded {int(pd.to_numeric(audit.irregular_timestamps, errors="coerce").fillna(0).sum())} irregular intervals and '
        f'{int(pd.to_numeric(audit.nan_inf, errors="coerce").fillna(0).sum())} non-finite values across usable recordings.']
    (project/'reports'/'data_audit.md').write_text('\n'.join(notes))
    print('SHA-256 duplicates',len(dup),'across-day',sum(dup.across_days),'usable files',sum(audit.status=='ok'),'windows',len(meta),flush=True)
    return dup

if __name__=='__main__':deduplicate_cache('.',yaml.safe_load(open('config/config.yaml')))

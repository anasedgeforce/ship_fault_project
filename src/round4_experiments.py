"""Round 4: independent-*file* LODO on historical days (EXPLORATORY ONLY).

Additional days are needed for independent validation; all three existing days have
already been inspected and used for research. Does not access the held-out files.
Preprocessing is fitted on training folds only; no test-day normalization.
Run: python -m src.round4_experiments --project .
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
from dataclasses import dataclass
import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.svm import SVC
from .train import load_cached, make_split, cv_take
from .crossday_advanced import FeatureView,metrics

SEED=42
# These are study configurations; all folds evaluate exactly 3 windows per file
# for apples-to-apples comparisons. Train windows-per-file may differ.
SPECS = {
 'extra_baseline_3':  ('extra','relative',3,2,160,'balanced_subsample',1.0),
 'extra_rel_8_leaf2': ('extra','relative',8,2,180,'balanced_subsample',.7),
 'extra_rel_15_leaf2':('extra','relative',15,2,180,'balanced_subsample',.7),
 'extra_rel_8_leaf4': ('extra','relative',8,4,180,'balanced_subsample',.7),
 'extra_rel_8_leaf1': ('extra','relative',8,1,180,'balanced_subsample',.7),
 'extra_amp_8_leaf2': ('extra','relative_amp',8,2,180,'balanced_subsample',.7),
 'extra_amp_15_leaf3':('extra','relative_amp',15,3,180,'balanced_subsample',.7),
 'extra_rel_8_no_classweight': ('extra','relative',8,2,180,None,.7),
 'extra_rel_8_dayclassweight':('extra','relative',8,2,180,None,.7),
 'svm_amp_8': ('svm','relative_amp',8,10,.003,'balanced',0),
 'svm_rel_8': ('svm','relative',8,50,.003,'balanced',0),
 'lgbm_rel_8':('lgbm','relative',8,15,170,'balanced',0),
}

def create_model(key,names):
    kind,view,k,param,nt,clsw,feat=SPECS[key]
    steps=[('view',FeatureView(tuple(names),view)),('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('var',VarianceThreshold(1e-12))]
    if kind=='extra':
        model=ExtraTreesClassifier(n_estimators=nt,min_samples_leaf=param,max_features=feat,
            class_weight=clsw,n_jobs=2,random_state=SEED)
    elif kind=='svm':
        steps.append(('scale',StandardScaler()))
        model=SVC(C=param,gamma=nt,class_weight=clsw,random_state=SEED)
    elif kind=='lgbm':
        from lightgbm import LGBMClassifier
        model=LGBMClassifier(n_estimators=nt,num_leaves=param,learning_rate=.04,min_child_samples=10,
             reg_lambda=5.,class_weight=clsw,n_jobs=2,random_state=SEED,verbosity=-1)
    elif kind=='cat':
        from catboost import CatBoostClassifier
        model=CatBoostClassifier(iterations=nt,depth=param,learning_rate=.07,loss_function='MultiClass',
                 verbose=False,allow_writing_files=False,thread_count=2,random_seed=SEED,l2_leaf_reg=4.)
    return Pipeline(steps+[('clf',model)])

def load_study(root):
    X,meta,names=load_cached(root)
    hold,holdfiles=make_split(meta,.2,SEED)
    exp=set((root/'reports'/'holdout_files.txt').read_text().splitlines())
    if set(holdfiles)!=exp:raise ValueError('Locked holdout manifest mismatch')
    if any(meta.file.isin(exp).to_numpy()!=hold):raise ValueError('Holdout mask mismatch')
    enc=LabelEncoder().fit(meta.label)
    assert len(enc.classes_)==30
    return X,meta,names,hold,enc

def sample_dayclass_weights(y,day):
    import collections
    counts=collections.Counter(zip(day,y))
    ww=np.array([1./counts[(d,k)] for d,k in zip(day,y)],dtype=float)
    return ww/ww.mean()

def fit_score(key,train_mask,valid_ids,meta,X,y,names,num_classes,return_model=False):
    _,_,k,*_=SPECS[key]
    train_ids=cv_take(meta,train_mask,k)
    train_files=set(meta.file.iloc[train_ids]); val_files=set(meta.file.iloc[valid_ids])
    assert train_files.isdisjoint(val_files)
    assert not set(meta.file.iloc[valid_ids]).intersection(set(meta.file.iloc[train_ids]))
    model=create_model(key,names)
    fitkw={}
    if key.endswith('dayclassweight'):
        w=sample_dayclass_weights(y[train_ids],meta.day.iloc[train_ids].to_numpy())
        fitkw['clf__sample_weight']=w
    model.fit(X[train_ids],y[train_ids],**fitkw)
    pr=np.asarray(model.predict(X[valid_ids])).reshape(-1)
    results=metrics(y[valid_ids],pr,y[train_ids],num_classes)
    results['n_train_windows']=len(train_ids)
    results['n_valid_windows']=len(valid_ids)
    return (results,model) if return_model else results

def run(root,mode='screen'):
    root=Path(root);rr=root/'reports';rr.mkdir(exist_ok=True)
    X,meta,names,hold,enc=load_study(root)
    y=enc.transform(meta.label);dd=meta.day.to_numpy()
    val_ids=cv_take(meta,~hold,3);days=sorted(meta.day.unique());C=len(enc.classes_)
    rows=[];t=time.monotonic()
    for day in days:
        vid=val_ids[dd[val_ids]==day]
        trainmask=(~hold)&(dd!=day)
        for key in SPECS:
            try:
                out=fit_score(key,trainmask,vid,meta,X,y,names,C)
            except Exception as err:
                print('FAIL',key,day,repr(err),flush=True)
                out={'error':str(err)}
            rows.append({'day':day,'candidate':key,**out})
            pd.DataFrame(rows).to_csv(rr/'round4_screen_folds.csv',index=False)
            print('SCREEN',day,key,'accuracy',round(out.get('accuracy',-1),4),
                  'fixed30',round(out.get('macro_f1_fixed30',-1),4),
                  'elapsed',round(time.monotonic()-t,1),flush=True)
    res=pd.DataFrame(rows)
    summary=res.groupby('candidate')[['accuracy','macro_f1_observed','macro_f1_fixed30']].agg(['mean','std'])
    summary.to_csv(rr/'round4_screen_summary.csv')
    print('TOP ACCURACY\n',summary.sort_values(('accuracy','mean'),ascending=False).head(10),flush=True)
    # Nested fold-level model selection on a PRE-SPECIFIED set, using only other days.
    # Selection by mean INNER accuracy, while always reporting macro F1.
    candidates=['extra_baseline_3','extra_rel_8_leaf2','extra_rel_15_leaf2',
            'extra_amp_8_leaf2','extra_rel_8_dayclassweight','svm_amp_8','lgbm_rel_8']
    nested_rows=[];inner_rows=[]
    for outer in days:
        outer_val=val_ids[dd[val_ids]==outer]
        development=(~hold)&(dd!=outer)
        choice=[]
        for key in candidates:
            ins=[]
            for inner in days:
                if inner==outer:continue
                innerv=val_ids[dd[val_ids]==inner]
                innertrain=development&(dd!=inner)
                try: result=fit_score(key,innertrain,innerv,meta,X,y,names,C)
                except Exception as ex: result={'error':str(ex),'accuracy':-1.,'macro_f1_fixed30':-1.}
                inner_rows.append({'outer':outer,'inner':inner,'candidate':key,**result})
                ins.append(result['accuracy'])
                pd.DataFrame(inner_rows).to_csv(rr/'round4_nested_inner.csv',index=False)
            choice.append((np.mean(ins),key))
        score,win=sorted(choice,key=lambda z:(-z[0],z[1]))[0]
        result=fit_score(win,development,outer_val,meta,X,y,names,C)
        nested_rows.append({'outer':outer,'selected':win,'inner_accuracy':score,**result})
        pd.DataFrame(nested_rows).to_csv(rr/'round4_nested_outer.csv',index=False)
        print('NESTED',outer,'winner',win,'acc',round(result['accuracy'],4),'fixed30',round(result['macro_f1_fixed30'],4),'elapsed',round(time.monotonic()-t,1),flush=True)
    print('NESTED_MEANS',pd.DataFrame(nested_rows)[['accuracy','macro_f1_observed','macro_f1_fixed30']].mean().to_dict(),flush=True)
    config={'seed':SEED,'candidate_specifications':{k:list(v) for k,v in SPECS.items()},
       'nested_candidates':candidates,'holdout_untouched':True,'holdout_count_files':len(set(meta.file[hold])),
       'fold_validation_windows_per_file':3,
       'day_research_limitation':'Historical days reused for iterative development; no independent new-day evaluation'}
    (rr/'round4_protocol.json').write_text(json.dumps(config,indent=2))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,default=Path('.'));args=p.parse_args()
    run(args.project)

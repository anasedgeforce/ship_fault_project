"""Leakage-safe CV / nested model tuning and final training entry point."""
from __future__ import annotations

if __name__=='__main__' and not __package__:
    import runpy
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'run_train.py'),run_name='__main__')
    raise SystemExit
import json,hashlib, warnings,time
from pathlib import Path
import numpy as np,pandas as pd,yaml,joblib
from sklearn.metrics import (accuracy_score,f1_score,balanced_accuracy_score,cohen_kappa_score,
    matthews_corrcoef,classification_report,confusion_matrix,log_loss)
from sklearn.model_selection import StratifiedGroupKFold,GroupKFold,RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr
from .models import make_model
from .labels import attributes

NAMES=['logistic','random_forest','xgboost','svm_rbf']

def available_model_names():
    """Keep the classical workflow usable when optional XGBoost is unavailable."""
    names=list(NAMES)
    try:
        import xgboost  # noqa: F401
    except ImportError:
        names.remove('xgboost')
        print('Skipping optional xgboost model: package is not installed',flush=True)
    return names

def load_cached(project):
    project=Path(project)
    with np.load(project/'cache'/'window_features.npz') as f:
        X=f['X'];names=f['names'].tolist()
    meta=pd.read_csv(project/'cache'/'window_metadata.csv',keep_default_na=False)
    return X,meta,names

def make_split(meta,frac=.2,seed=42):
    files=meta[['file','label','day']].drop_duplicates()
    rng=np.random.default_rng(seed);hold=[]
    for (label,day),chunk in files.groupby(['label','day'],sort=True):
        ff=np.sort(chunk.file.to_numpy());rng.shuffle(ff)
        n=max(1,round(len(ff)*frac)) if len(ff)>=3 else (1 if len(ff)>1 else 0)
        hold.extend(ff[:n])
    hold=np.asarray(sorted(set(hold)))
    return meta.file.isin(hold).to_numpy(),hold

def cv_take(meta,mask,per_file=3):
    ids=np.flatnonzero(mask)
    subset=meta.iloc[ids]
    chosen=[]
    for _,r in subset.groupby('file',sort=True):
        take=np.linspace(0,len(r)-1,min(len(r),per_file)).round().astype(int)
        chosen.extend(r.index.to_numpy()[take])
    return np.asarray(sorted(chosen),dtype=int)

def metric_row(yt,yp):
    return dict(accuracy=accuracy_score(yt,yp),macro_f1=f1_score(yt,yp,average='macro',zero_division=0),
        weighted_f1=f1_score(yt,yp,average='weighted',zero_division=0),
        balanced_accuracy=balanced_accuracy_score(yt,yp),
        kappa=cohen_kappa_score(yt,yp),mcc=matthews_corrcoef(yt,yp))

def labels_to_heads(labels):
    a=[attributes(s.replace('__',' ')) for s in []] # deliberately unused; classes need direct dictionary
    return None

def train_and_evaluate(project,config):
    project=Path(project);X,meta,features=load_cached(project)
    holdmask,holdfiles=make_split(meta,config['holdout_fraction_per_day_class'],config['seed'])
    (project/'reports'/'holdout_files.txt').write_text('\n'.join(holdfiles)+'\n')
    trainmask=~holdmask
    all_labels=sorted(meta.label.unique());label_to_attrs={}
    label_map=yaml.safe_load(open(project/'config'/'label_map.yaml'))
    for folder,label in label_map.items():label_to_attrs[label]=attributes(folder)
    # Composite class may include condition folders with unspecified loader: record ambiguity.
    conflicts=[]
    for label in all_labels:
        matching=[attributes(folder) for folder,lbl in label_map.items() if lbl==label]
        for key in ('fault_type','severity','loader'):
            values={m[key] for m in matching}
            if len(values)>1:conflicts.append(f'{label}/{key}: {sorted(values)}')
    (project/'reports'/'attribute_conflicts.txt').write_text('\n'.join(conflicts))
    (project/'artifacts'/'label_attributes.json').write_text(json.dumps(label_to_attrs,indent=2))
    le=LabelEncoder().fit(all_labels)
    y=le.transform(meta.label)
    cv_ids=cv_take(meta,trainmask,config['cv_windows_per_file'])
    model_names=available_model_names()
    rows=[];predictions=[];folds=[]
    for day in sorted(meta.day.unique()):
        va=cv_ids[meta.day.iloc[cv_ids].to_numpy()==day]
        tr=cv_ids[meta.day.iloc[cv_ids].to_numpy()!=day]
        folds.append((f'lodo_{day}',tr,va))
    # Complementary group-stratified K-fold without group overlap.
    sgkf=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=config['seed'])
    for i,(tri,vai) in enumerate(sgkf.split(X[cv_ids],y[cv_ids],groups=meta.file.iloc[cv_ids])):
        folds.append((f'group5_{i+1}',cv_ids[tri],cv_ids[vai]))
    def record(model_name,scheme,fold,yy,pred,tr_acc=None):
        mm=metric_row(yy,pred)
        rows.append(dict(model=model_name,scheme=scheme,fold=fold,n=len(yy),train_accuracy=tr_acc,**mm))
        predictions.extend([dict(model=model_name,scheme=scheme,fold=fold,actual=str(le.inverse_transform([t])[0]),
                                predicted=str(le.inverse_transform([p])[0])) for t,p in zip(yy,pred)])
    # Leave validation day untouched within each fold; feature selection and scalers fit on fold train only.
    for fold,tr,va in folds:
        print('Classical CV fold',fold,'train',len(tr),'valid',len(va),flush=True)
        for model_name in model_names:
            est=make_model(model_name,config['seed'],config['cv_rf_trees'])
            try:
                if model_name=='xgboost':
                    local=LabelEncoder().fit(y[tr]);est.fit(X[tr],local.transform(y[tr]))
                    pred=local.inverse_transform(est.predict(X[va]));train_pred=local.inverse_transform(est.predict(X[tr]))
                else:
                    est.fit(X[tr],y[tr]);pred=est.predict(X[va]);train_pred=est.predict(X[tr])
                record(model_name,'A_multiclass',fold,y[va],pred,accuracy_score(y[tr],train_pred))
            except Exception as exc:
                rows.append(dict(model=model_name,scheme='A_multiclass',fold=fold,error=str(exc)))
                print('MODEL FAILED',model_name,fold,exc,flush=True)
        # Multi-head B: independent type/severity/loader; compare triplets in a separate metric space.
        for model_name in model_names:
            try:
                keys=('fault_type','severity','loader')
                targets={k:meta[k].astype(str).to_numpy() for k in keys}
                per_head={}; head_pred={}
                for key in keys:
                    coder=LabelEncoder().fit(targets[key][tr])
                    typ=coder.transform(targets[key][tr])
                    est=make_model(model_name,config['seed'],config['cv_rf_trees'])
                    est.fit(X[tr],typ)
                    per_head[key]=est
                    head_pred[key]=coder.inverse_transform(est.predict(X[va]))
                predicted=list(zip(*[head_pred[k] for k in keys]))
                truth=list(zip(*[targets[k][va] for k in keys]))
                correct=[int(p==t) for p,t in zip(predicted,truth)]
                f1s=[f1_score(targets[k][va],head_pred[k],average='macro',zero_division=0) for k in keys]
                rows.append(dict(model=model_name,scheme='B_multihead',fold=fold,n=len(va),
                   accuracy=float(np.mean(correct)),macro_f1=float(np.mean(f1s)),
                   fault_type_macro_f1=f1s[0],severity_macro_f1=f1s[1],loader_macro_f1=f1s[2],
                   joint_accuracy=float(np.mean(correct))))
            except Exception as exc:
                rows.append(dict(model=model_name,scheme='B_multihead',fold=fold,error=str(exc)))
                print('HEAD FAILED',model_name,fold,exc,flush=True)
    cv=pd.DataFrame(rows);cv.to_csv(project/'reports'/'cv_results.csv',index=False)
    pd.DataFrame(predictions).to_csv(project/'reports'/'cv_predictions.csv',index=False)
    summary=cv.groupby(['scheme','model','fold'],as_index=False).first()
    summary=summary.assign(protocol=lambda d:np.where(d.fold.str.startswith('lodo_'),'LODO','Grouped5'))
    result=summary.groupby(['protocol','scheme','model']).agg(
         macro_f1_mean=('macro_f1','mean'),macro_f1_std=('macro_f1','std'),
         accuracy_mean=('accuracy','mean'),accuracy_std=('accuracy','std'),folds=('macro_f1','count')).reset_index()
    result.to_csv(project/'reports'/'cv_summary.csv',index=False)
    # Select classical track on primary LODO class-level F1. Deep results appended later.
    choices=result[(result.protocol=='LODO')&(result.scheme=='A_multiclass')].dropna(subset=['macro_f1_mean'])
    if len(choices)==0:raise RuntimeError('All classical models failed')
    chosen=str(choices.sort_values(['macro_f1_mean','model'],ascending=[False,True]).iloc[0].model)
    print('CHOSEN CLASSICAL',chosen,flush=True)
    nested_tuning(project,X,y,meta,cv_ids,config)
    return finalize(project,config,chosen,X,meta,features,holdmask,le,cv,rows)

def nested_tuning(project,X,y,meta,cv_ids,config):
    """Grouped inner randomized search within outer LODO; evaluate RF and SVM."""
    rows=[]
    grid={
        'random_forest':{'model__max_depth':[8,16,None], 'model__min_samples_leaf':[1,2,4]},
        'svm_rbf':{'model__C':[.5,2.,8.], 'model__gamma':['scale',.01,.001]}}
    for day in sorted(meta.day.unique()):
        tr=cv_ids[meta.day.iloc[cv_ids].to_numpy()!=day];va=cv_ids[meta.day.iloc[cv_ids].to_numpy()==day]
        if len(np.unique(y[tr]))<2:continue
        for name,params in grid.items():
            base=make_model(name,config['seed'],55)
            inner=GroupKFold(n_splits=2)
            rs=RandomizedSearchCV(base,params,n_iter=3,cv=inner,scoring='f1_macro',
                  n_jobs=1,random_state=config['seed'],refit=True,error_score=np.nan)
            try:
                rs.fit(X[tr],y[tr],groups=meta.file.iloc[tr]);pred=rs.predict(X[va])
                score=f1_score(y[va],pred,average='macro',zero_division=0)
                rows.append(dict(model=name,outer_day=day,nested_macro_f1=score,
                                 inner_best=rs.best_score_,params=str(rs.best_params_)))
            except Exception as e: rows.append(dict(model=name,outer_day=day,error=str(e)))
    pd.DataFrame(rows).to_csv(Path(project)/'reports'/'nested_cv.csv',index=False)

def finalize(project,config,chosen,X,meta,features,holdmask,le,cv,rows):
    # IMPORTANT: model selection above used CV on non-test recordings only.
    y=le.transform(meta.label);tr=np.flatnonzero(~holdmask);te=np.flatnonzero(holdmask)
    if set(meta.file.iloc[tr])&set(meta.file.iloc[te]):raise AssertionError('Group leakage')
    est=make_model(chosen,config['seed'],config['final_rf_trees'])
    print('Training final',chosen,'on',len(tr),'windows; TEST HELD OUT',len(te),flush=True)
    est.fit(X[tr],y[tr])
    # Only one holdout pass; do not select model or thresholds from this performance.
    probs=est.predict_proba(X[te]);pred=est.classes_[probs.argmax(axis=1)]
    result=metric_row(y[te],pred)
    (project/'reports'/'final_metrics.json').write_text(json.dumps(result,indent=2))
    pd.DataFrame(classification_report(y[te],pred,labels=np.arange(len(le.classes_)),target_names=le.classes_,output_dict=True,zero_division=0)).T.to_csv(project/'reports'/'per_class_f1.csv')
    pd.DataFrame(confusion_matrix(y[te],pred,labels=np.arange(len(le.classes_))),index=le.classes_,columns=le.classes_).to_csv(project/'reports'/'confusion_raw.csv')
    cm=confusion_matrix(y[te],pred,labels=np.arange(len(le.classes_)),normalize='true')
    pd.DataFrame(cm,index=le.classes_,columns=le.classes_).to_csv(project/'reports'/'confusion_normalized.csv')
    pd.DataFrame(dict(file=meta.file.iloc[te].to_numpy(),truth=le.inverse_transform(y[te]),
        pred=le.inverse_transform(pred),confidence=probs.max(axis=1))).to_csv(project/'reports'/'holdout_predictions.csv',index=False)
    # Fit OOD on train-only features after fold-learned feature transform, cap dimensionality for robust covariance.
    transformed=est[:-1].transform(X[tr]);ood=IsolationForest(n_estimators=130,max_samples=min(1024,len(tr)),contamination='auto',n_jobs=1,random_state=config['seed'])
    ood.fit(transformed)
    thresh=float(np.quantile(ood.decision_function(transformed),1-float(config['ood_quantile'])))
    joblib.dump(est,project/'artifacts'/'model.joblib',compress=3)
    joblib.dump(ood,project/'artifacts'/'ood.joblib',compress=3)
    joblib.dump(le,project/'artifacts'/'label_encoder.joblib')
    joblib.dump(est.named_steps['scale'],project/'artifacts'/'scaler.joblib')
    (project/'artifacts'/'feature_names.json').write_text(json.dumps(features,indent=2))
    conf=dict(config);conf['estimated_speed_hz']=json.loads((project/'cache'/'data_settings.json').read_text())['estimated_speed_hz']
    conf['chosen_model']=chosen;conf['ood_threshold']=thresh;conf['class_names']=le.classes_.tolist()
    (project/'artifacts'/'config_snapshot.json').write_text(json.dumps(conf,indent=2))
    import sklearn
    version=dict(n_windows_trained=len(tr),n_test_windows=len(te),random_seed=config['seed'],sklearn_version=sklearn.__version__)
    version['hash_sha256']=hashlib.sha256((project/'artifacts'/'model.joblib').read_bytes()).hexdigest()
    (project/'artifacts'/'version.json').write_text(json.dumps(version,indent=2))
    try:
        # Permutation importances on held-out training-validation SAMPLE only (NOT test).
        vtr=cv_take(meta,~holdmask,2);rng=np.random.default_rng(config['seed']);vtr=rng.choice(vtr,min(80,len(vtr)),replace=False)
        p=permutation_importance(est,X[vtr],y[vtr],n_repeats=1,scoring='f1_macro',random_state=config['seed'],n_jobs=1)
        pd.DataFrame(dict(feature=features,importance_mean=p.importances_mean,importance_std=p.importances_std)).sort_values('importance_mean',ascending=False).to_csv(project/'reports'/'feature_importance.csv',index=False)
    except Exception as e:(project/'reports'/'feature_importance_error.txt').write_text(str(e))
    return est,le,X,meta,holdmask,result


if __name__=='__main__':
    import runpy
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'run_train.py'),run_name='__main__')

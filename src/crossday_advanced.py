"""Exploratory advanced cross-day benchmark; untouched locked holdout.

Run:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.crossday_advanced

Only the non-test 80% files are used; each file contributes three windows to
three outer day folds, with all model selection inside the two training days.
DO NOT claim an unbiased performance gain after inspecting these outer folds:
all experiments across prior sessions have reused the same three recording days.
"""
from __future__ import annotations
import json
import time
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import StandardScaler,RobustScaler,QuantileTransformer,LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score,balanced_accuracy_score
from .train import load_cached,make_split,cv_take

SEED=42

class FeatureView(BaseEstimator,TransformerMixin):
    """Deterministic transformations that use no validation-day information."""
    def __init__(self,names,view='log'):
        self.names=names;self.view=view
    def fit(self,X,y=None):
        n=list(self.names)
        self.log_idx_=np.array([i for i,s in enumerate(n) if s.startswith(('mean_','std_','rms_','peak_','p2p_','variance_','env_','harmonic_','band_'))])
        self.shape_idx_=np.array([i for i,s in enumerate(n) if s.startswith(('crest_','skew_','kurtosis_','shape_','impulse_','clearance_','dominant_hz_','centroid_hz_','spectral_entropy_','rms_ratio_','phase_'))])
        self.freq_idx_=np.array([i for i,s in enumerate(n) if s.startswith(('dominant_hz_','centroid_hz_','spectral_entropy_','band_','harmonic_','env_','rms_ratio_','phase_'))])
        self.idx_groups_={prefix:[[n.index('%s_ch%d'%(prefix,i)) for i in range(6)] for prefix in []] for prefix in []}
        self.groups_=[]
        for family in (['band_0_50','band_50_150','band_150_500','band_500_1200','band_1200_2000'],['harmonic_1','harmonic_2','harmonic_3'],['env_0_50','env_50_150','env_150_350','env_350_650']):
            self.groups_.append(np.array([[n.index('%s_ch%d'%(prefix,c)) for prefix in family] for c in range(6)]))
        self.rms_idx_=np.array([n.index('rms_ch%d'%c) for c in range(6)])
        return self
    def transform(self,X):
        X=np.asarray(X,dtype=np.float64).copy()
        if self.view=='raw': return X
        if self.view in ('log','log_freq','quantile','pca'):
            a=X[:,self.log_idx_];X[:,self.log_idx_]=np.sign(a)*np.log1p(np.abs(a))
            if self.view=='log_freq':return X[:,self.freq_idx_]
            return X
        if self.view=='shape':return X[:,self.shape_idx_]
        if self.view in ('relative','relative_amp'):
            for group in self.groups_:
                block=np.maximum(X[:,group],0)
                denominators=np.sum(block,axis=2,keepdims=True)
                norm=block/(denominators+1e-12)
                X[:,group]=np.log1p(1000*norm)
            r=np.maximum(X[:,self.rms_idx_],0)
            X[:,self.rms_idx_]=r/(r.sum(axis=1,keepdims=True)+1e-12)
            if self.view=='relative':
                names=list(self.names)
                time_idx=np.array([names.index('elapsed_window_start_s')],dtype=int) if 'elapsed_window_start_s' in names else np.array([],dtype=int)
                only=np.concatenate([self.shape_idx_,self.rms_idx_,time_idx]+[g.ravel() for g in self.groups_])
                # deduplicate indices while preserving feature order
                return X[:,np.unique(only)]
            # relative amplitude + compressed remaining original features
            other=np.setdiff1d(self.log_idx_,np.concatenate([self.rms_idx_]+[g.ravel() for g in self.groups_]))
            s=X[:,other];X[:,other]=np.sign(s)*np.log1p(np.abs(s))
            return X
        raise ValueError(self.view)

SPECS={
    'svm_log_c3_gscale':('log','standard','svm',3,'scale'),
    'svm_log_c20_gscale':('log','standard','svm',20,'scale'),
    'svm_log_c100_gscale':('log','standard','svm',100,'scale'),
    'svm_log_c20_g001':('log','standard','svm',20,.001),
    'svm_log_c20_g003':('log','standard','svm',20,.003),
    'svm_log_c20_g01':('log','standard','svm',20,.01),
    'svm_robust_c10_g003':('log','robust','svm',10,.003),
    'svm_raw_c10_g003':('raw','standard','svm',10,.003),
    'svm_shape_c10_gscale':('shape','standard','svm',10,'scale'),
    'svm_shape_c50_g001':('shape','standard','svm',50,.001),
    'svm_freq_c10_gscale':('log_freq','standard','svm',10,'scale'),
    'svm_relative_c10_gscale':('relative','standard','svm',10,'scale'),
    'svm_relative_c50_g003':('relative','standard','svm',50,.003),
    'svm_relative_amp_c10':('relative_amp','standard','svm',10,.003),
    'svm_quantile_c10':('quantile','quantile','svm',10,'scale'),
    'svm_pca35_c10':('pca','pca35','svm',10,'scale'),
    'svm_pca65_c10':('pca','pca65','svm',10,'scale'),
    'logistic_log_c01':('log','standard','logistic',.1,None),
    'logistic_log_c10':('log','standard','logistic',10,None),
    'logistic_relative_c1':('relative','standard','logistic',1,None),
    'extra_trees_raw_leaf2':('raw','none','extra_trees',2,None),
    'extra_trees_raw_leaf5':('raw','none','extra_trees',5,None),
    'extra_trees_rel_leaf2':('relative','none','extra_trees',2,None),
    'hist_gb_log_leaf15':('log','none','hist_gb',15,None),
}

def make_estimator(key,names):
    view,scale,model,c,g=SPECS[key]
    steps=[('view',FeatureView(tuple(names),view)),('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('var',VarianceThreshold(1e-12))]
    if scale=='standard':steps.append(('scale',StandardScaler()))
    elif scale=='robust':steps.append(('scale',RobustScaler(quantile_range=(10,90))))
    elif scale=='quantile':steps.append(('scale',QuantileTransformer(n_quantiles=150,output_distribution='normal',random_state=SEED)))
    elif scale.startswith('pca'):
        steps.append(('scale',StandardScaler()))
        steps.append(('pca',PCA(n_components=int(scale[3:]),random_state=SEED)))
    if model=='svm':clf=SVC(C=c,gamma=g,class_weight='balanced',cache_size=250,random_state=SEED)
    elif model=='logistic':clf=LogisticRegression(C=c,class_weight='balanced',max_iter=1600,random_state=SEED)
    elif model=='extra_trees':clf=ExtraTreesClassifier(n_estimators=160,min_samples_leaf=int(c),class_weight='balanced_subsample',random_state=SEED,n_jobs=2)
    elif model=='hist_gb':clf=HistGradientBoostingClassifier(max_iter=110,max_leaf_nodes=15,l2_regularization=3.,min_samples_leaf=int(c),random_state=SEED)
    return Pipeline(steps+[('clf',clf)])

def metrics(y,pr,trainy,nb):
    known=np.isin(y,np.unique(trainy))
    result={'macro_f1_fixed30':f1_score(y,pr,labels=np.arange(nb),average='macro',zero_division=0),
        'macro_f1_observed':f1_score(y,pr,average='macro',zero_division=0),
        'weighted_f1':f1_score(y,pr,labels=np.arange(nb),average='weighted',zero_division=0),
        'accuracy':accuracy_score(y,pr),
        'balanced_accuracy':balanced_accuracy_score(y,pr),
        'unseen_frac':float((~known).mean())}
    if known.any():result['known_label_macro_f1']=f1_score(y[known],pr[known],labels=np.unique(trainy),average='macro',zero_division=0)
    return result

def run(root=Path('.')):
    root=Path(root);rep=root/'reports';rep.mkdir(exist_ok=True)
    X,meta,names=load_cached(root)
    hold,holdfiles=make_split(meta,.2,SEED)
    expected=set((rep/'holdout_files.txt').read_text().splitlines())
    if set(holdfiles)!=expected:raise RuntimeError('Holdout manifest mismatch')
    if any(meta.file.isin(holdfiles).to_numpy()!=hold):raise RuntimeError('Incorrect holdout masking')
    ids=cv_take(meta,~hold,3)
    if np.any(hold[ids]):raise RuntimeError('Hold-out leak')
    groups=meta.file.to_numpy();day=meta.day.to_numpy()
    y=LabelEncoder().fit_transform(meta.label)
    days=sorted(meta.day.unique());classes=np.unique(y)
    inner_rows=[];nested_rows=[];diagnostic_rows=[]
    tally=time.monotonic()
    for outer in days:
        val=ids[day[ids]==outer];dev=ids[day[ids]!=outer]
        if set(groups[val]) & set(groups[dev]):raise RuntimeError('File leak')
        candidates=[]
        for spec_name in SPECS:
            inner=[]
            for d in days:
                if d==outer:continue
                innerv=dev[day[dev]==d];innertr=dev[day[dev]!=d]
                est=make_estimator(spec_name,names)
                try:
                    est.fit(X[innertr],y[innertr]);pr=est.predict(X[innerv]);res=metrics(y[innerv],pr,y[innertr],len(classes))
                except Exception as exc:
                    print('FAILED',outer,d,spec_name,str(exc),flush=True)
                    res={'macro_f1_fixed30':-1,'error':str(exc)}
                inner_rows.append(dict(outer_day=outer,inner_day=d,candidate=spec_name,**res))
                inner.append(res['macro_f1_fixed30'])
            candidates.append((np.mean(inner),spec_name))
        best_inner,selected=sorted(candidates,key=lambda o:(-o[0],o[1]))[0]
        winner=make_estimator(selected,names)
        winner.fit(X[dev],y[dev]);pred=winner.predict(X[val])
        result=metrics(y[val],pred,y[dev],len(classes))
        nested_rows.append(dict(outer_day=outer,selected=selected,inner_fixed30_score=best_inner,**result))
        print('NESTED',outer,'selected',selected,'FIXED30',round(result['macro_f1_fixed30'],4),'OBSERVED',round(result['macro_f1_observed'],4), 'elapsed_s',round(time.monotonic()-tally,1),flush=True)
        # Predefined fixed candidate benchmarks; diagnostic on previously used days, NOT validation.
        for spec_name in SPECS:
            est=make_estimator(spec_name,names)
            try:
                est.fit(X[dev],y[dev]);pr=est.predict(X[val]);res=metrics(y[val],pr,y[dev],len(classes))
                diagnostic_rows.append(dict(outer_day=outer,candidate=spec_name,**res))
            except Exception as exc:
                diagnostic_rows.append(dict(outer_day=outer,candidate=spec_name,error=str(exc)))
        pd.DataFrame(inner_rows).to_csv(rep/'advanced_inner_cv.csv',index=False)
        pd.DataFrame(nested_rows).to_csv(rep/'advanced_nested_outer.csv',index=False)
        pd.DataFrame(diagnostic_rows).to_csv(rep/'advanced_diagnostic_lodo.csv',index=False)
        print('DIAGNOSTIC done',outer,'elapsed_s',round(time.monotonic()-tally,1),flush=True)
    result=pd.DataFrame(diagnostic_rows)
    result.groupby('candidate')[['macro_f1_fixed30','macro_f1_observed','accuracy']].agg(['mean','std']).to_csv(rep/'advanced_candidate_summary.csv')
    print('NESTED MEAN',pd.DataFrame(nested_rows)[['macro_f1_fixed30','macro_f1_observed','accuracy']].mean().to_dict())
    print('DIAGNOSTIC top',result.groupby('candidate').macro_f1_fixed30.mean().sort_values(ascending=False).head(8).to_string())
    (rep/'advanced_experiment_config.json').write_text(json.dumps({'candidate_grid':{k:list(v) for k,v in SPECS.items()},'seed':SEED,'all_labels':len(classes),'holdout_files':len(holdfiles),'cv_windows':len(ids),'cv_strategy':'nested leave-one-day-out, 2 inner day folds','test_set_access':'none','note':'Outer folds reused during exploratory research. No independent validation.'},indent=2))
    return result

if __name__=='__main__':run()

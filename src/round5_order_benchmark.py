"""Additional retrospective cross-day candidates with signal-derived RPM-order features.
Cached extra features are computed from raw CSV with a shared feature function.
Never accesses the locked original holdout; all preprocessing fitted on training data.
"""
from pathlib import Path
import argparse,time,json
import numpy as np,pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.svm import SVC
from .train import cv_take
from .round4_experiments import load_study
from .crossday_advanced import FeatureView,metrics

SEED=42
# Predefined diagnostic screen; metrics are NOT independent validation.
SPECS={
 'tree_amp_order_10_leaf2':('relative_amp', 'extra',10,2,180,'sqrt'),
 'tree_amp_order_10_leaf4':('relative_amp', 'extra',10,4,180,'sqrt'),
 'tree_amp_order_15_leaf2':('relative_amp', 'extra',15,2,180,'sqrt'),
 'tree_amp_order_15_leaf4':('relative_amp', 'extra',15,4,180,'sqrt'),
 'tree_log_order_10_leaf2':('log', 'extra',10,2,180,'sqrt'),
 'tree_order_only_10_leaf2':('only', 'extra',10,2,180,'sqrt'),
 'tree_rel_order_10_leaf2':('relative', 'extra',10,2,180,'sqrt'),
 'svm_rel_order_10_c10':('relative','svm',10,10,None,'scale'),
 'svm_amp_order_10_c10':('relative_amp','svm',10,10,None,'scale'),
 'svm_amp_order_10_c100':('relative_amp','svm',10,100,None,'scale'),
}

class OrderFeatureView(BaseEstimator,TransformerMixin):
    def __init__(self,names,view='relative_amp',orig_dim=175):
        self.names=names;self.view=view;self.orig_dim=orig_dim
    def fit(self,X,y=None):
        if self.view!='only':self.orig_=FeatureView(self.names,self.view).fit(X[:,:self.orig_dim])
        return self
    def transform(self,X):
        X=np.asarray(X);extra=X[:,self.orig_dim:]
        if self.view=='only':return extra
        old=self.orig_.transform(X[:,:self.orig_dim])
        return np.column_stack([old,extra])

def make_candidate(name,names,dim):
    view,kind,k,param,n,feat=SPECS[name]
    steps=[('view',OrderFeatureView(tuple(names),view,dim)),
           ('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
           ('var',VarianceThreshold(1e-12))]
    if kind=='extra':
        model=ExtraTreesClassifier(n_estimators=n,min_samples_leaf=param,
           max_features=feat,class_weight='balanced_subsample',n_jobs=2,random_state=SEED)
    else:
        steps.append(('scale',StandardScaler()))
        model=SVC(C=param,gamma=feat,class_weight='balanced',random_state=SEED)
    return Pipeline(steps+[('clf',model)])

def run(root=Path('.')):
    root=Path(root);orig,meta,names,hold,enc=load_study(root)
    with np.load(root/'cache'/'round5_order_features.npz') as d:
        extra=d['X'];extra_names=d['names'].tolist()
    assert len(orig)==len(extra)==len(meta)
    X=np.column_stack((orig,extra));y=enc.transform(meta.label);days=meta.day.to_numpy()
    vid=cv_take(meta,~hold,3);scores=[];st=time.monotonic()
    for day in sorted(meta.day.unique()):
        va=vid[days[vid]==day];mask=(~hold)&(days!=day)
        for name in SPECS:
            view,kind,k,*_=SPECS[name];tr=cv_take(meta,mask,k)
            assert len(set(meta.file.iloc[tr])&set(meta.file.iloc[va]))==0
            est=make_candidate(name,names,len(names))
            try:
                est.fit(X[tr],y[tr]);yp=est.predict(X[va]);r=metrics(y[va],yp,y[tr],30)
            except Exception as ex:
                r={'error':repr(ex)}
            scores.append(dict(day=day,candidate=name,n_train=len(tr),n_valid=len(va),**r))
            pd.DataFrame(scores).to_csv(root/'reports'/'round5_order_folds.csv',index=False)
            print('ORDER',day,name,r.get('accuracy',None),r.get('macro_f1_fixed30',None),'secs',round(time.monotonic()-st,1),flush=True)
    df=pd.DataFrame(scores)
    s=df.groupby('candidate')[['accuracy','macro_f1_fixed30','macro_f1_observed']].agg(['mean','std'])
    s.to_csv(root/'reports'/'round5_order_summary.csv')
    print('FINAL',s.sort_values(('accuracy','mean'),ascending=False).to_string(),flush=True)
    return scores

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--project',default='.')
    run(Path(a.parse_args().project))

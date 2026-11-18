"""Production classical estimators and PyTorch raw-window baselines."""
from __future__ import annotations
import sys
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

class CorrelationFilter(BaseEstimator,TransformerMixin):
    def __init__(self,threshold=.95):self.threshold=threshold
    def fit(self,X,y=None):
        X=np.asarray(X)
        corr=np.corrcoef(X,rowvar=False)
        corr=np.nan_to_num(corr,nan=0.0)
        upper=np.triu(np.abs(corr),k=1)
        self.keep_=np.flatnonzero(~np.any(upper>self.threshold,axis=0))
        if not len(self.keep_):self.keep_=np.array([0])
        return self
    def transform(self,X):return np.asarray(X)[:,self.keep_]

def make_model(name,seed=42,trees=70):
    pre=[('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
         ('variance',VarianceThreshold(threshold=1e-12)),
         ('corr',CorrelationFilter(.95)),('scale',StandardScaler())]
    if name=='logistic': estimator=LogisticRegression(max_iter=650,class_weight='balanced',random_state=seed)
    elif name=='random_forest': estimator=RandomForestClassifier(n_estimators=trees,min_samples_leaf=2,
        max_features='sqrt',class_weight='balanced_subsample',n_jobs=1,random_state=seed)
    elif name=='svm_rbf':estimator=SVC(C=2.,kernel='rbf',probability=True,class_weight='balanced',random_state=seed)
    elif name=='xgboost':
        from xgboost import XGBClassifier
        estimator=XGBClassifier(n_estimators=95,max_depth=4,learning_rate=.08,subsample=.85,
            colsample_bytree=.8,tree_method='hist',n_jobs=3,random_state=seed,eval_metric='mlogloss')
    else:raise ValueError(name)
    return Pipeline(pre+[('model',estimator)])

try:
    import torch
    from torch import nn
    _TORCH_IMPORT_ERROR = None
except (ImportError, OSError) as exc:  # Torch is optional for classical workflows.
    # A partial/namespace-only Torch installation may leave a broken module behind.
    # Remove it so SciPy's optional array-backend detection does not mistake it for
    # a usable Torch installation.
    sys.modules.pop('torch', None)
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR = exc


def _require_torch():
    if torch is None or nn is None:
        raise ImportError(
            "PyTorch is required only for the optional CNN/CNN-LSTM baselines. "
            "Install torch>=2.0 to run deep-learning experiments."
        ) from _TORCH_IMPORT_ERROR


if nn is not None:
    class ConvNet(nn.Module):
        def __init__(self,classes,variant='cnn'):
            super().__init__();self.variant=variant
            self.conv=nn.Sequential(nn.Conv1d(6,16,11,stride=4,padding=5),nn.BatchNorm1d(16),nn.ReLU(),
                nn.MaxPool1d(4),nn.Conv1d(16,32,7,stride=2,padding=3),nn.BatchNorm1d(32),nn.ReLU(),
                nn.MaxPool1d(4),nn.Conv1d(32,48,5,stride=2,padding=2),nn.ReLU())
            if variant=='cnn_lstm':
                self.lstm=nn.LSTM(input_size=48,hidden_size=32,num_layers=1,batch_first=True)
                self.head=nn.Linear(32,classes)
            else:self.head=nn.Linear(48,classes)
        def forward(self,x):
            x=self.conv(x.transpose(1,2))
            if self.variant=='cnn_lstm':
                x,_=self.lstm(x.transpose(1,2));return self.head(x[:,-1,:])
            return self.head(x.mean(dim=2))
else:
    class ConvNet:
        def __init__(self,*args,**kwargs):
            _require_torch()

def augmentation(x):
    _require_torch()
    x=x*(1.+torch.randn((len(x),1,6),device=x.device)*.05)
    x=x+torch.randn_like(x)*.02
    # integer time shift per minibatch
    return torch.roll(x,shifts=int(torch.randint(-30,31,(1,)).item()),dims=1)

def fit_conv(model,train_X,y,val_X=None,val_y=None,epochs=4,lr=.001,batch=48,seed=42):
    _require_torch()
    torch.manual_seed(seed);device=torch.device('cpu');model.to(device)
    train_X=torch.as_tensor(train_X,dtype=torch.float32);y=torch.as_tensor(y,dtype=torch.long)
    classes=np.bincount(y.numpy(),minlength=model.head.out_features)
    weights=np.clip(len(y)/np.maximum(1,classes)/max(1,len(classes)),.2,20.)
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    lossfn=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32))
    best=float('inf');best_state=None; patience=0
    for epoch in range(epochs):
        model.train();perm=torch.randperm(len(y));total=0.
        for ids in perm.split(batch):
            opt.zero_grad();out=model(augmentation(train_X[ids]));loss=lossfn(out,y[ids]);loss.backward();opt.step()
            total+=loss.item()*len(ids)
        model.eval()
        with torch.no_grad():
            if val_X is not None and len(val_X):
                inp=torch.as_tensor(val_X,dtype=torch.float32)
                v=torch.as_tensor(val_y,dtype=torch.long)
                vloss=np.mean([lossfn(model(z),v[i:i+batch]).item() for i,z in ((i,inp[i:i+batch]) for i in range(0,len(v),batch))])
            else:vloss=total/len(y)
        if vloss<best-1e-4:
            best=vloss;best_state={k:v.detach().clone() for k,v in model.state_dict().items()};patience=0
        else:
            patience+=1
            if patience>=3:break
    if best_state:model.load_state_dict(best_state)
    return model

def cnn_probs(model,windows,batch=64):
    _require_torch()
    model.eval();outs=[]
    with torch.no_grad():
        for i in range(0,len(windows),batch):
            outs.append(torch.softmax(model(torch.as_tensor(windows[i:i+batch],dtype=torch.float32)),dim=1).numpy())
    return np.concatenate(outs) if outs else np.empty((0,model.head.out_features))

"""Extended FD001 grouped-validation study (no test labels used in selection).
python extended_study.py --zip CMAPSSData.zip --out .
"""
import argparse,io,zipfile,warnings,itertools, json, time
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error
p=argparse.ArgumentParser();p.add_argument('--zip',required=True);p.add_argument('--out',default='.');p.add_argument('--max-iter',type=int,default=100);a=p.parse_args()
out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
cols=['engine_id','cycle']+[f'operating_{i}' for i in range(1,4)]+[f'sensor_{i}' for i in range(1,22)]
with zipfile.ZipFile(a.zip) as z:
 def read(n):return pd.read_csv(io.BytesIO(z.read(n)),sep=r'\s+',header=None)
 train=read('train_FD001.txt'); test=read('test_FD001.txt'); test_rul=read('RUL_FD001.txt').iloc[:,0].to_numpy()
train.columns=cols;test.columns=cols
train['RUL']=train.groupby('engine_id').cycle.transform('max')-train.cycle
last=test.loc[test.groupby('engine_id').cycle.idxmax()].sort_values('engine_id').copy();last['RUL']=test_rul
splits={};counts=train.groupby('engine_id').size().to_dict()
for seed in [17,42,73]:
 tr,va=train_test_split(train.engine_id.unique(),test_size=.2,random_state=seed)
 ids=[];used=0
 for eid in np.random.default_rng(seed).permutation(tr):
  if used+counts[eid]<=8000:ids.append(int(eid));used+=counts[eid]
 fit=train[train.engine_id.isin(ids)].copy(); val=train[train.engine_id.isin(va)].copy()
 assert len(fit)<=8000 and not (set(fit.engine_id)&set(val.engine_id))
 fs=[c for c in cols if c not in ('engine_id','cycle') and fit[c].std()>1e-8]+['cycle']
 imp=SimpleImputer(strategy='median'); xtr=imp.fit_transform(fit[fs]);xv=imp.transform(val[fs]); xt=imp.transform(last[fs]); sc=StandardScaler();xtr=sc.fit_transform(xtr);xv=sc.transform(xv);xt=sc.transform(xt)
 splits[seed]=dict(xtr=xtr,xv=xv,xt=xt,ytr=fit.RUL.to_numpy(dtype=float),yv=val.RUL.to_numpy(dtype=float),yt=last.RUL.to_numpy(dtype=float),train_engines=len(ids),train_cycles=len(fit),val_cycles=len(val),features=fs)

def metrics(y,p):
 y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float);d=np.abs(y-p);den=np.abs(y)+np.abs(p)
 return dict(MAE=float(np.mean(d)),RMSE=float(np.sqrt(np.mean((y-p)**2))),MAPE_nonzero_pct=float(np.mean(d[y>0]/y[y>0])*100),MAPE_RUL_ge10_pct=float(np.mean(d[y>=10]/y[y>=10])*100),sMAPE_pct=float(np.mean(np.divide(2*d,den,out=np.zeros_like(d),where=den>0))*100),MAE_RUL_0_30=float(np.mean(d[y<=30])),MAE_RUL_gt80=float(np.mean(d[y>80])))

def train_mlp(seed,layers,alpha,lr,cap=None):
 d=splits[seed]; y=d['ytr'] if cap is None else np.minimum(d['ytr'],cap)
 mlp=MLPRegressor(hidden_layer_sizes=layers,alpha=alpha,learning_rate_init=lr,max_iter=a.max_iter,early_stopping=True,n_iter_no_change=12,validation_fraction=.12,random_state=seed,solver='adam')
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',ConvergenceWarning);mlp.fit(d['xtr'],y)
 return mlp

start=time.monotonic(); rows=[]
configs=list(itertools.product([(32,16),(64,32)],[0.1,1.0],[0.0005,0.001]))
for layers,alpha,lr in configs:
 for seed in splits:
  m=train_mlp(seed,layers,alpha,lr)
  d=splits[seed];r=dict(stage='grid',seed=seed,layers=str(layers),alpha=alpha,learning_rate=lr,cap='none',n_iter=m.n_iter_,train_cycles=d['train_cycles'],**metrics(d['yv'],m.predict(d['xv'])))
  rows.append(r);print('GRID',layers,alpha,lr,seed,round(r['MAE'],2),flush=True)
grid=pd.DataFrame(rows);grid.to_csv(out/'extended_grid.csv',index=False,float_format='%.6f')
mean=grid.groupby(['layers','alpha','learning_rate'],as_index=False).agg(MAE_mean=('MAE','mean'),MAE_std=('MAE','std'),low_MAE_mean=('MAE_RUL_0_30','mean'))
mean.to_csv(out/'extended_grid_summary.csv',index=False,float_format='%.6f')
winner=mean.sort_values(['MAE_mean','MAE_std']).iloc[0]; layers=tuple(int(x) for x in winner.layers.strip('()').split(','));alpha=float(winner.alpha);lr=float(winner.learning_rate)
print('WINNER',layers,alpha,lr,flush=True)
caprows=[];predictions={}
for cap in [None,100,125,150,200]:
 for seed in splits:
  m=train_mlp(seed,layers,alpha,lr,cap)
  d=splits[seed];pr=m.predict(d['xv']);r=dict(seed=seed,layers=str(layers),alpha=alpha,learning_rate=lr,cap='none' if cap is None else str(cap),n_iter=m.n_iter_,train_cycles=d['train_cycles'],**metrics(d['yv'],pr));caprows.append(r)
  print('CAP',cap,seed,round(r['MAE'],2),round(r['MAE_RUL_0_30'],2),flush=True)
  if seed==42 and cap is None:
   tp=m.predict(d['xt']); predictions['test_uncapped']=metrics(d['yt'],tp)
capdf=pd.DataFrame(caprows);capdf.to_csv(out/'extended_caps.csv',index=False,float_format='%.6f')
capmean=capdf.groupby('cap',as_index=False).agg(MAE_mean=('MAE','mean'),MAE_std=('MAE','std'),low_MAE_mean=('MAE_RUL_0_30','mean'),low_MAE_std=('MAE_RUL_0_30','std'),MAPE_ge10_mean=('MAPE_RUL_ge10_pct','mean'),sMAPE_mean=('sMAPE_pct','mean'))
capmean['order']=capmean.cap.map({'none':0,'100':1,'125':2,'150':3,'200':4});capmean=capmean.sort_values('order').drop(columns='order');capmean.to_csv(out/'extended_caps_summary.csv',index=False,float_format='%.6f')
plt.figure(figsize=(10,5));x=np.arange(len(mean));plt.bar(x,mean.MAE_mean,yerr=mean.MAE_std,capsize=3);plt.xticks(x,[f'{r.layers}\na={r.alpha}, lr={r.learning_rate}' for r in mean.itertuples()],rotation=35,ha='right',fontsize=8);plt.ylabel('Validation MAE (cycles), mean ± SD over three seeds');plt.title('Expanded MLP tuning (engine-grouped validation)');plt.tight_layout();plt.savefig(out/'extended_tuning.png',dpi=170);plt.close()
fig,ax=plt.subplots(figsize=(9,5));x=np.arange(len(capmean));ax.errorbar(x,capmean.MAE_mean,yerr=capmean.MAE_std,marker='o',capsize=4,label='Full-range MAE');ax.errorbar(x,capmean.low_MAE_mean,yerr=capmean.low_MAE_std,marker='s',capsize=4,label='MAE for true RUL ≤ 30');ax.set_xticks(x,['No cap','100','125','150','200']);ax.set_xlabel('Upper cap on TRAIN labels only (cycles)');ax.set_ylabel('MAE on original, uncapped validation targets');ax.grid(alpha=.25);ax.legend();ax.set_title('Target RUL cap: trade-off by degradation stage');fig.tight_layout();fig.savefig(out/'extended_rul_caps.png',dpi=170);plt.close(fig)
# Average validation-MAE against effective epochs for configs; logged from model stopping iteration.
fig,ax=plt.subplots(figsize=(7,4.5));ax.scatter(grid.groupby(['layers','alpha','learning_rate']).n_iter.mean(),mean.MAE_mean,s=65);ax.set(xlabel='Mean fitted epochs across seeds',ylabel='Mean validation MAE (cycles)',title='Fitting duration vs generalization: plateau diagnostic');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'extended_plateau.png',dpi=170);plt.close(fig)
metadata=dict(seeds=list(splits),train_cycles={str(k):v['train_cycles'] for k,v in splits.items()},selected=dict(layers=layers,alpha=alpha,learning_rate=lr),max_iter=a.max_iter,models_grid=len(grid),models_caps=len(capdf),test_uncapped=predictions['test_uncapped'],elapsed_seconds=round(time.monotonic()-start,1),note='Test evaluated only after selection from group-held-out validation; results are not used for configuration selection. MAPE excludes RUL=0.')
(out/'extended_study_info.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf8')
print('DONE',json.dumps(metadata,ensure_ascii=False),flush=True)

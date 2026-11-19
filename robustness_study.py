"""Multi-seed engine-grouped validation for FD001. MAPE excludes target==0.
Run: python robustness_study.py --zip CMAPSSData.zip --out .
"""
import argparse, io, zipfile, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.exceptions import ConvergenceWarning

p=argparse.ArgumentParser()
p.add_argument('--zip',required=True)
p.add_argument('--out',default='.')
p.add_argument('--seeds',nargs='+',type=int,default=[17,42,73])
p.add_argument('--max-iter',type=int,default=120)
a=p.parse_args(); out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
cols=['engine_id','cycle']+[f'operating_{i}' for i in range(1,4)]+[f'sensor_{i}' for i in range(1,22)]
with zipfile.ZipFile(a.zip) as z:
 def read(n): return pd.read_csv(io.BytesIO(z.read(n)),sep=r'\s+',header=None)
 train=read('train_FD001.txt');test=read('test_FD001.txt');test_rul=read('RUL_FD001.txt').iloc[:,0].to_numpy()
train.columns=cols;test.columns=cols
train['RUL']=(train.groupby('engine_id').cycle.transform('max')-train.cycle).astype(float)
last=test.loc[test.groupby('engine_id').cycle.idxmax()].sort_values('engine_id').copy();last['RUL']=test_rul
# Capping here affects the training labels ONLY, not test reference labels.
configs=[('MLP_32_16_alpha0.1',(32,16),0.1),('MLP_48_24_alpha1',(48,24),1.0),('MLP_64_32_alpha1',(64,32),1.0)]
records=[]; predictions=[]
ids=train.engine_id.unique(); counts=train.groupby('engine_id').size().to_dict()
for seed in a.seeds:
 train_ids,val_ids=train_test_split(ids,test_size=.2,random_state=seed)
 chosen=[];used=0
 for eid in np.random.default_rng(seed).permutation(train_ids):
  if used+counts[eid]<=8000:chosen.append(int(eid)); used+=counts[eid]
 fit=train[train.engine_id.isin(chosen)].copy(); val=train[train.engine_id.isin(val_ids)].copy()
 assert len(fit)<=8000 and not set(fit.engine_id)&set(val.engine_id)
 feats=[c for c in cols if c not in ('engine_id','cycle') and fit[c].std()>1e-8]+['cycle']
 for cap in [None,125]:
  yfit=fit.RUL.to_numpy() if cap is None else np.minimum(fit.RUL.to_numpy(),cap)
  for name,layers,alpha in configs:
   model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),MLPRegressor(hidden_layer_sizes=layers,alpha=alpha,max_iter=a.max_iter,early_stopping=True,n_iter_no_change=15,validation_fraction=.12,random_state=seed,learning_rate_init=.001))
   with warnings.catch_warnings():
    warnings.simplefilter('ignore',ConvergenceWarning)
    model.fit(fit[feats],yfit)
   pred=model.predict(val[feats]);y=val.RUL.to_numpy()
   nz=y>0
   mape=float(np.mean(np.abs((y[nz]-pred[nz])/y[nz]))*100) if nz.any() else float('nan')
   # MAPE >= 10 reduces near-zero denominator instability.
   mask=y>=10
   mape10=float(np.mean(np.abs((y[mask]-pred[mask])/y[mask]))*100)
   denom=np.abs(y)+np.abs(pred)
   smape=float(np.mean(np.divide(2*np.abs(y-pred),denom,out=np.zeros_like(y),where=denom>0))*100)
   row=dict(seed=seed,model=name,rul_cap=('none' if cap is None else cap),train_engines=len(chosen),train_cycles=len(fit),val_engines=len(val_ids),val_cycles=len(val),MAE=mean_absolute_error(y,pred),RMSE=np.sqrt(mean_squared_error(y,pred)),MAPE_nonzero_pct=mape,MAPE_RUL_ge_10_pct=mape10,sMAPE_pct=smape,MAE_RUL_0_30=mean_absolute_error(y[y<=30],pred[y<=30]),n_iter=model[-1].n_iter_)
   records.append(row)
   print(seed,name,cap,round(row['MAE'],3),flush=True)
summary=pd.DataFrame(records)
summary.to_csv(out/'robustness_results.csv',index=False,float_format='%.5f')
agg=summary.groupby(['model','rul_cap'],as_index=False).agg(MAE_mean=('MAE','mean'),MAE_std=('MAE','std'),MAPE_RUL_ge_10_mean=('MAPE_RUL_ge_10_pct','mean'),sMAPE_mean=('sMAPE_pct','mean'),MAE_low_mean=('MAE_RUL_0_30','mean'))
agg.to_csv(out/'robustness_summary.csv',index=False,float_format='%.5f')
fig,ax=plt.subplots(figsize=(10,5.5))
for name,part in agg.groupby('model'):
 part=part.set_index('rul_cap').reindex(['none',125])
 xs=np.array([0,1]);ax.errorbar(xs,part.MAE_mean,yerr=part.MAE_std,marker='o',capsize=4,label=name)
ax.set_xticks([0,1],['Uncapped target','Cap training RUL at 125']);ax.set_ylabel('Validation MAE (cycles), mean ± SD over seeds');ax.set_title('FD001: target capping and MLP tuning (grouped engine validation)');ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'robustness_comparison.png',dpi=180);plt.close(fig)
print('\nSUMMARY\n',agg.to_string(index=False))

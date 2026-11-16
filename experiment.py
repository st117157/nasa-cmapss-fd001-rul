"""Reproducible C-MAPSS FD001 RUL study. Run: python experiment.py --zip /path/to/CMAPSSData.zip"""
import argparse, io, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

P = argparse.ArgumentParser()
P.add_argument('--zip', required=True)
P.add_argument('--out', default=str(Path(__file__).parent))
a = P.parse_args()
out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
np.random.seed(42)
cols=['engine_id','cycle']+[f'operating_{i}' for i in range(1,4)]+[f'sensor_{i}' for i in range(1,22)]
with zipfile.ZipFile(a.zip) as z:
    def read(n): return pd.read_csv(io.BytesIO(z.read(n)),sep=r'\s+',header=None)
    train=read('train_FD001.txt'); test=read('test_FD001.txt'); test_rul=read('RUL_FD001.txt').iloc[:,0].to_numpy()
train.columns=cols;test.columns=cols
train['RUL']=(train.groupby('engine_id')['cycle'].transform('max')-train['cycle']).astype(float)
# The 100 engine test RUL labels are ONLY for the last observed cycle.
last=test.loc[test.groupby('engine_id').cycle.idxmax()].sort_values('engine_id').copy()
last['RUL']=test_rul
ids=train.engine_id.unique()
train_ids,val_ids=train_test_split(ids,test_size=.2,random_state=42)
# Select complete engines only, all cycle observations and <=8000 train rows.
rng=np.random.default_rng(42)
chosen=[];used=0
counts=train.groupby('engine_id').size().to_dict()
for i in rng.permutation(train_ids):
    if used+counts[i]<=8000:
        chosen.append(int(i)); used+=counts[i]
fit=train[train.engine_id.isin(chosen)].copy().reset_index(drop=True)
val=train[train.engine_id.isin(val_ids)].copy().reset_index(drop=True)
# Drop strictly constant features on the selected TRAIN engines; no test-driven selection.
features=[c for c in cols if c not in ('engine_id','cycle')]
features=[c for c in features if fit[c].std()>1e-8]
feature_sets={'Age (cycle only)':['cycle'],'Ridge (sensors + settings)':features,'MLP (sensors + settings)':features,'MLP (+ cycle)':features+['cycle']}
models={
    'Age (cycle only)': RandomForestRegressor(n_estimators=100,min_samples_leaf=25,max_features=1.0,random_state=42,n_jobs=-1),
    'Ridge (sensors + settings)': make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=100.0)),
    'MLP (sensors + settings)':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),MLPRegressor(hidden_layer_sizes=(48,24),alpha=1.0,learning_rate_init=0.001,max_iter=350,early_stopping=True,n_iter_no_change=20,random_state=42)),
    'MLP (+ cycle)':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),MLPRegressor(hidden_layer_sizes=(48,24),alpha=1.0,learning_rate_init=0.001,max_iter=350,early_stopping=True,n_iter_no_change=20,random_state=42)),
}
metrics=[]
for name,model in models.items():
    fs=feature_sets[name]
    print('training',name,flush=True)
    model.fit(fit[fs],fit.RUL)
    for d,split in [(val,'validation_all_cycles'),(val.loc[val.groupby('engine_id').cycle.idxmax()].copy(),'validation_last_cycles'),(last,'test_last_cycles')]:
        pred=model.predict(d[fs])
        rec={'model':name,'split':split,'n':len(d),'MAE':mean_absolute_error(d.RUL,pred),'RMSE':np.sqrt(mean_squared_error(d.RUL,pred))}
        for binname,mask in [('low_RUL_0_30',d.RUL<=30),('medium_RUL_31_80',(d.RUL>30)&(d.RUL<=80)),('high_RUL_gt_80',d.RUL>80)]:
            rec[f'MAE_{binname}']=mean_absolute_error(d.RUL[mask],pred[mask]) if mask.any() else np.nan
            rec[f'n_{binname}']=int(mask.sum())
        metrics.append(rec)
    val[f'pred_{list(models).index(name)}']=model.predict(val[fs]); last[f'pred_{list(models).index(name)}']=model.predict(last[fs])
    print(name,'test MAE',metrics[-1]['MAE'],flush=True)
metrics=pd.DataFrame(metrics)
metrics.to_csv(out/'metrics.csv',index=False,float_format='%.5f')
last.to_csv(out/'test_predictions.csv',index=False,float_format='%.5f')
val.to_csv(out/'validation_predictions.csv',index=False,float_format='%.5f')
# Graph 1: held-out engines, RUL changes per cycle; best sensor MLP and age baseline.
fig,axs=plt.subplots(2,2,figsize=(13,8),sharey=False)
# choose contrasting trajectories deterministically
chosen_val=sorted(val_ids,key=lambda i:counts[i]); examples=[chosen_val[0],chosen_val[len(chosen_val)//3],chosen_val[2*len(chosen_val)//3],chosen_val[-1]]
for ax,engine in zip(axs.ravel(),examples):
    d=val[val.engine_id==engine].sort_values('cycle')
    ax.plot(d.cycle,d.RUL,label='True RUL',linewidth=2,color='black')
    ax.plot(d.cycle,d.pred_0,label='Age only',alpha=.85,linewidth=1.6)
    ax.plot(d.cycle,d.pred_2,label='MLP sensors',alpha=.85,linewidth=1.6)
    ax.plot(d.cycle,d.pred_3,label='MLP sensors + cycle',alpha=.85,linewidth=1.6)
    ax.set(title=f'Validation engine {engine}',xlabel='Observed cycle',ylabel='Remaining useful life (cycles)')
    ax.grid(alpha=.2)
handles,labels=axs[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',ncol=4,bbox_to_anchor=(.5,1.02))
fig.tight_layout(rect=(0,0,1,.95));fig.savefig(out/'rul_trajectories.png',dpi=175,bbox_inches='tight');plt.close(fig)
# Graph 2: per TRUE RUL bin, final rows of 100 independent test engines.
fig,ax=plt.subplots(figsize=(11,6))
bins=['0–30','31–80','>80']
xs=np.arange(3);w=.19
for j,name in enumerate(models):
    d=metrics[(metrics.model==name)&(metrics.split=='test_last_cycles')].iloc[0]
    ys=[d['MAE_low_RUL_0_30'],d['MAE_medium_RUL_31_80'],d['MAE_high_RUL_gt_80']]
    ax.bar(xs+(j-1.5)*w,ys,width=w,label=name)
ax.set_xticks(xs,[f'{b}\n(n={int(metrics[metrics.split=="test_last_cycles"].iloc[0]["n_"+k])})' for b,k in zip(bins,['low_RUL_0_30','medium_RUL_31_80','high_RUL_gt_80'])]);ax.set(ylabel='MAE (cycles)',xlabel='True RUL interval, test last cycle',title='Prediction error versus true RUL — FD001 test (100 engines)')
ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'mae_by_rul.png',dpi=175);plt.close(fig)
# Save simple metadata.
with open(out/'run_info.txt','w') as f:
    f.write(f'Train source: 100 engines, {len(train)} cycles\nSelected complete train engines: {len(chosen)}; {len(fit)} cycles\nValidation engines: {len(val_ids)}; {len(val)} cycles\nTest: 100 engines, last row only\nSelected train ids: {sorted(chosen)}\nValidation ids: {sorted(val_ids)}\nFeatures: {features}\nNo RUL clipping, no future-length features. Seed 42.\n')
print(metrics.to_string(index=False),flush=True)

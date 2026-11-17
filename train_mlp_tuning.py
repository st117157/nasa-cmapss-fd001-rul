"""Grouped-validation MLP tuning and learning curves for C-MAPSS FD001.

Run: python train_mlp_tuning.py --zip CMAPSSData.zip --out ./tuning
No test targets are used to choose architecture, regularization, or epoch.
"""
import argparse
import io
import json
import warnings
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument('--zip', required=True)
parser.add_argument('--out', default='./tuning')
parser.add_argument('--epochs', type=int, default=160)
args = parser.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

cols = ['engine_id', 'cycle'] + [f'operating_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
with zipfile.ZipFile(args.zip) as z:
    def read(name):
        return pd.read_csv(io.BytesIO(z.read(name)), sep=r'\s+', header=None)
    train, test, test_rul = read('train_FD001.txt'), read('test_FD001.txt'), read('RUL_FD001.txt').iloc[:, 0].to_numpy()
train.columns, test.columns = cols, cols
train['RUL'] = train.groupby('engine_id')['cycle'].transform('max') - train['cycle']
ids = train.engine_id.unique()
train_ids, val_ids = train_test_split(ids, test_size=0.2, random_state=42)
counts = train.groupby('engine_id').size().to_dict()
chosen, used = [], 0
for i in np.random.default_rng(42).permutation(train_ids):
    if used + counts[i] <= 8000:
        chosen.append(int(i))
        used += counts[i]
fit = train[train.engine_id.isin(chosen)]
val = train[train.engine_id.isin(val_ids)]
last = test.loc[test.groupby('engine_id').cycle.idxmax()].sort_values('engine_id').copy()
last['RUL'] = test_rul
assert len(fit) <= 8000 and set(fit.engine_id).isdisjoint(set(val.engine_id))
features = [c for c in cols if c != 'engine_id' and c != 'cycle' and fit[c].std() > 1e-8] + ['cycle']
imp = SimpleImputer(strategy='median')
scaler = StandardScaler()
X_train = scaler.fit_transform(imp.fit_transform(fit[features]))
X_val = scaler.transform(imp.transform(val[features]))
X_test = scaler.transform(imp.transform(last[features]))
y_train, y_val, y_test = fit.RUL.to_numpy(), val.RUL.to_numpy(), last.RUL.to_numpy()

configs = [
    ('small_l2_0p1', (32, 16), 0.1),
    ('medium_l2_0p1', (48, 24), 0.1),
    ('medium_l2_1', (48, 24), 1.0),
    ('large_l2_1', (64, 32), 1.0),
]
records, summary = [], []
models = {}
for label, hidden, alpha in configs:
    model = MLPRegressor(hidden_layer_sizes=hidden, activation='relu', solver='adam',
                         alpha=alpha, learning_rate_init=0.001, batch_size=200,
                         random_state=42, shuffle=True)
    best_val = float('inf')
    best_epoch = None
    best_weights = None
    patience = 0
    for epoch in range(1, args.epochs + 1):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            model.partial_fit(X_train, y_train)
        train_mae = mean_absolute_error(y_train, model.predict(X_train))
        val_mae = mean_absolute_error(y_val, model.predict(X_val))
        records.append({'config': label, 'epoch': epoch, 'train_MAE': train_mae, 'validation_MAE': val_mae})
        if val_mae < best_val - 0.01:
            best_val, best_epoch = val_mae, epoch
            best_weights = ([v.copy() for v in model.coefs_], [v.copy() for v in model.intercepts_])
            patience = 0
        else:
            patience += 1
        if patience >= 22 and epoch >= 35:
            break
    model.coefs_, model.intercepts_ = best_weights
    models[label] = model
    summary.append({'config': label, 'hidden_layers': str(hidden), 'alpha': alpha,
                    'epochs_run': epoch, 'best_epoch': best_epoch,
                    'best_validation_MAE': best_val})
    print(label, 'best validation MAE', round(best_val, 3), 'epoch', best_epoch, flush=True)

scores = pd.DataFrame(summary).sort_values(['best_validation_MAE', 'config']).reset_index(drop=True)
best_name = str(scores.iloc[0]['config'])
best_model = models[best_name]
test_pred = best_model.predict(X_test)
test_mae = mean_absolute_error(y_test, test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
scores.to_csv(out/'tuning_results.csv', index=False, float_format='%.6f')
pd.DataFrame(records).to_csv(out/'learning_curve.csv', index=False, float_format='%.6f')
pd.DataFrame({'engine_id': last.engine_id, 'observed_cycle': last.cycle,
              'true_RUL': y_test, 'predicted_RUL': test_pred}).to_csv(out/'tuned_test_predictions.csv', index=False)
with (out/'tuning_info.json').open('w', encoding='utf-8') as f:
    json.dump({'selection_metric': 'MAE, all cycles of 20 held-out validation engines',
               'selected_config': best_name, 'test_MAE': test_mae, 'test_RMSE': test_rmse,
               'fit_engines': len(chosen), 'fit_cycles': len(fit),
               'validation_engines': len(val_ids), 'validation_cycles': len(val),
               'test_engines': len(last), 'seed': 42, 'features': features,
               'hyperparameters_tested': [{'config': t[0], 'hidden_layers': t[1], 'alpha': t[2]} for t in configs],
               'data_leakage_prevention': 'All preprocessing fit on fit engines only; test labels used once after selection'},
              f, ensure_ascii=False, indent=2)

curve = pd.DataFrame(records)
fig, ax = plt.subplots(figsize=(10, 6))
for label in scores.config:
    d = curve[curve.config == label]
    ax.plot(d.epoch, d.validation_MAE, label=label)
ax.set(title='MLP hyperparameter search: validation MAE by epoch', xlabel='Epoch', ylabel='Validation MAE (cycles)')
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(out/'tuning_comparison.png', dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 6))
d = curve[curve.config == best_name]
ax.plot(d.epoch, d.train_MAE, label='Training MAE')
ax.plot(d.epoch, d.validation_MAE, label='Validation MAE (held-out engines)')
ax.axvline(int(scores.iloc[0].best_epoch), linestyle='--', label=f'Best epoch {int(scores.iloc[0].best_epoch)}')
ax.set(title=f'Learning curves: selected MLP ({best_name})', xlabel='Epoch', ylabel='MAE (cycles)')
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(out/'learning_curves.png', dpi=180)
plt.close(fig)
print('SELECTED', best_name, 'test MAE', round(test_mae, 4), 'test RMSE', round(test_rmse, 4), flush=True)

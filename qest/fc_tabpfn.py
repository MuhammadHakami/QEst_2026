"""TabPFN (v2, open weights) as a gate-A quantile forecaster, stacked on Chronos-2.

Why: of the gap to hindsight that survives realistic activation volumes, the price share is the forecastable part
(research/gap_to_hindsight.md s7). Lipiecki & Weron (arXiv:2609.00089, 2026) find TabPFN the one foundation model that
consistently beats market-specific EPF benchmarks statistically. Here, per target, TabPFN regresses the realised value on the
same gate-A features as qest.forecast plus Chronos-2's q10 / q50 / q90 for that target, as a residual over Chronos-2's median (stacking), on the most recent
MAX_ROWS MTUs of the fit days (TabPFN's context limit), and returns q10 / q50 / q90 from its predictive distribution.
Information set unchanged: everything is D-1 09:00 legal, as in qest.forecast. The same split-conformal widening as the
LightGBM path is applied for the test split. Weights: priorlabs TabPFN-v2 regressor checkpoint, copied into models/tabpfn (self-contained).

usage: CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m qest.fc_tabpfn [--check]   (Q3 skill vs Chronos-2, via qest.evaluate.forecasts)
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from qest.data import frame
from qest.forecast import QS, TARGETS, features

MAX_ROWS = 10000
os.environ.setdefault('TABPFN_MODEL_CACHE_DIR', str(Path(__file__).resolve().parents[1] / 'models' / 'tabpfn'))  # self-contained: the v2 regressor checkpoint lives in the folder


def predict(fit_days, pred_days, chronos_fit, chronos_pred, targets=TARGETS):
    import torch
    from tabpfn import TabPFNRegressor
    f = frame(); X = features(); fm, pm = f.day.isin(fit_days), f.day.isin(pred_days)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; out = pd.DataFrame(index=f.index[pm])
    for k, col in targets.items():
        Xf = X[fm].copy(); Xp = X[pm].copy()
        for q in (10, 50, 90):  # Chronos-2's own quantiles as features; the target is the residual over its median
            c = f'{k}_q{q}'; Xf[f'c{q}'] = chronos_fit[c].reindex(Xf.index).to_numpy() if c in chronos_fit else 0.0
            Xp[f'c{q}'] = chronos_pred[c].reindex(Xp.index).to_numpy() if c in chronos_pred else 0.0
        y = f.loc[fm, col] - Xf['c50']; m = y.notna() & Xf['c50'].notna()
        Xf, y = Xf[m].tail(MAX_ROWS), y[m].tail(MAX_ROWS)  # the most recent rows: TabPFN's context
        lo, hi = y.quantile([.005, .995]); yc = y.clip(lo, hi)
        mdl = TabPFNRegressor(device=dev, n_estimators=4, ignore_pretraining_limits=True)
        mdl.fit(Xf.fillna(0).to_numpy(np.float32), yc.to_numpy(np.float32))
        q = mdl.predict(Xp.fillna(0).to_numpy(np.float32), output_type='quantiles', quantiles=list(QS))
        for qq, arr in zip(QS, q):
            out[f'{k}_q{int(qq * 100)}'] = np.asarray(arr, float) + Xp['c50'].fillna(0).to_numpy()
        if k.endswith('_s'):
            out[[f'{k}_q{int(qq * 100)}' for qq in QS]] = out[[f'{k}_q{int(qq * 100)}' for qq in QS]].clip(0, 1)
        print('tabpfn', k, 'fit rows', len(y), flush=True)
    return out


def _check():
    from qest.evaluate import forecasts
    from qest.data import days
    from qest.forecast import skill
    f = frame(); va = days(f, 'val'); T, C = forecasts('val', 'tabpfn'), forecasts('val', 'chronos')
    st, sc = skill(T, va), skill(C, va); R = pd.DataFrame({'tabpfn_mae': st.mae, 'chronos_mae': sc.mae}); R['gain_%'] = (1 - R.tabpfn_mae / R.chronos_mae) * 100
    pd.set_option('display.width', 200); print(R.round(2).to_string())
    assert R.loc['da', 'tabpfn_mae'] < 1.1 * R.loc['da', 'chronos_mae'], 'TabPFN far worse than Chronos-2 on DA'
    print('fc_tabpfn check ok')


if __name__ == '__main__':
    _check()

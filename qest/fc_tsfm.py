"""Mistral Timestral (Oct 2026, arXiv:2610.37585; 209M, #1 zero-shot on GIFT-Eval / TIME / fev-bench at launch) as a gate-A
forecaster, drop-in beside Chronos-2 (qest.chronos): the same two information groups, the same known covariates, the same
output layout, so the comparison is model against model.

For delivery day D at gate A (D-1 09:00 Tallinn):
  group D-1: DA price and the four capacity prices, known to the end of D-1 -> 1 day ahead;
  group D-2: mFRR, imbalance, aFRR prices and active shares, known to the end of D-2 -> 2 days ahead;
  known-future covariates to the end of D: D-1 weather forecasts (hybrid PV per MW, our wind park, the national fleet),
  hour of day (sin, cos) and weekday. The variates of a group are forecast jointly. Context: CONTEXT_DAYS days (Timestral
  reads up to 32,768 steps; 56 days = 5,376 MTUs matches qest.chronos; --context 120 for the long one).
Output: '<target>_q10/50/90' per MTU, as qest.forecast. Code Apache 2.0 (research/third_party/timestral), weights Modified MIT.
usage: CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m qest.fc_tsfm --split val [--model mistralai/Timestral-209M] [--context 56]
  -> skill against Chronos-2 and LightGBM on the split; cached by qest.evaluate.forecasts(split, 'timestral').
"""
import argparse
import sys
import time

import numpy as np
import pandas as pd

from qest.chronos import GROUPS, MIN_CONTEXT_DAYS
from qest.data import frame
from qest.forecast import TARGETS, features

MODEL = 'mistralai/Timestral-209M'
CONTEXT_DAYS = 56
_F = {}


def _forecaster(model):
    if model not in _F:
        import torch
        from timestral import TimestralForecaster
        _F[model] = TimestralForecaster.from_pretrained(model, device='cuda' if torch.cuda.is_available() else 'cpu')
    return _F[model]


def predict(pred_days, model=MODEL, context_days=CONTEXT_DAYS, batch=16):
    f = frame(); X = features(); fc = _forecaster(model)
    pred_days = [d for d in pred_days if d >= f.day.min() + pd.Timedelta(days=MIN_CONTEXT_DAYS)]
    cov = pd.DataFrame({'pv_d1': X.pv_d1.to_numpy(), 'wind_d1': X.wind_d1.to_numpy(), 'fleet_wind_d1': X.get('fleet_wind_d1', pd.Series(0.0, index=f.index)).to_numpy(),
                        'sinh': np.sin(2 * np.pi * X.hour.to_numpy() / 24), 'cosh': np.cos(2 * np.pi * X.hour.to_numpy() / 24), 'dow': X.dow.to_numpy()}, index=f.index).ffill().fillna(0.0)
    Y = pd.DataFrame({k: f[c].to_numpy() for k, c in TARGETS.items()}, index=f.index); day = f.day; pos = pd.Series(np.arange(len(f)), index=f.index)
    T = context_days * 96; res = pd.DataFrame(index=f.index[f.day.isin(pred_days)])
    for L, ks in GROUPS.items():
        items = []
        for d in pred_days:
            known = pos[(day <= d - pd.Timedelta(days=L)).to_numpy()]; hz = pos[((day > d - pd.Timedelta(days=L)) & (day <= d)).to_numpy()]
            if len(known) < T // 2 or len(hz) == 0:
                continue
            k0, k1 = max(int(known.max()) + 1 - T, 0), int(known.max()) + 1; h0, h1 = int(hz.min()), int(hz.max()) + 1
            items.append((d, k0, k1, h0, h1))
        by_h = {}
        for it in items:
            by_h.setdefault((it[2] - it[1], it[4] - it[3]), []).append(it)
        t0 = time.time()
        for (Tc, H), its in by_h.items():
            for i in range(0, len(its), batch):
                chunk = its[i:i + batch]
                ctx = np.stack([Y[ks].to_numpy()[k0:k1].T for _, k0, k1, _, _ in chunk]).astype(np.float32)  # (B, V, T)
                kn = np.stack([np.r_[cov.to_numpy()[k0:k1], cov.to_numpy()[h0:h1]].T for _, k0, k1, h0, h1 in chunk]).astype(np.float32)  # (B, C, T + H)
                out = fc.forecast(ctx, prediction_length=H, known_covariates=kn)
                for j, (d, k0, k1, h0, h1) in enumerate(chunk):
                    idx = f.index[h0:h1]; own = idx[day.loc[idx].to_numpy() == d]
                    sel = np.isin(idx, own)
                    for v, k in enumerate(ks):
                        for q, qq in ((.1, 10), (.5, 50), (.9, 90)):
                            res.loc[own, f"{k}_q{qq}"] = out.quantile(q)[j, v].detach().cpu().numpy()[sel]
        print(f'timestral group D-{L}: {len(items)} days in {time.time() - t0:.0f} s', flush=True)
    for k in TARGETS:
        if k.endswith('_s'):
            res[[f'{k}_q{q}' for q in (10, 50, 90)]] = res[[f'{k}_q{q}' for q in (10, 50, 90)]].clip(0, 1)
    return res


if __name__ == '__main__':
    from qest.data import days
    from qest.evaluate import forecasts
    from qest.forecast import skill
    from qest.paths import RESULTS
    ap = argparse.ArgumentParser(); ap.add_argument('--split', default='val'); ap.add_argument('--model', default=MODEL); ap.add_argument('--context', type=int, default=CONTEXT_DAYS); ap.add_argument('--tag', default='timestral')
    a = ap.parse_args(); f = frame(); pred = days(f, a.split)
    P = predict(pred, a.model, a.context); P.to_parquet(RESULTS / f'forecast_{a.tag}_{a.split}.parquet')
    S = skill(P, pred); C = skill(forecasts(a.split, 'chronos'), pred); G = skill(forecasts(a.split, 'lgbm'), pred)
    T = pd.DataFrame({'timestral_mae': S.mae, 'chronos_mae': C.mae, 'lgbm_mae': G.mae}); T['vs_chronos_%'] = (100 * (T.timestral_mae / T.chronos_mae - 1)).round(1)
    pd.set_option('display.width', 200); print(T.round(3).to_string()); T.to_csv(RESULTS / f'skill_{a.tag}_{a.split}.csv')

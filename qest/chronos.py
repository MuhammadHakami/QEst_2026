"""Chronos-2 (Amazon, 2025: a pretrained time-series foundation model with covariates) as a zero-shot gate-A forecaster,
the 'classical deep learning' baseline beside LightGBM and persistence. No training on our data: it reads the history.

For delivery day D, at gate A (D-1 09:00 Tallinn) with the same information set as qest.forecast:
  group D-1 (published by the gate up to the end of D-1): DA price and the four capacity prices -> forecast 1 day ahead;
  group D-2 (balancing, published with delay): mFRR, imbalance, aFRR prices and active shares -> forecast 2 days ahead;
  known covariates through the end of D: the D-1 weather forecasts (hybrid PV per MW, our wind park, the national wind
  fleet) and the MTU's hour / weekday. Context: the last CONTEXT_DAYS days. Multivariate targets per group.
Output: the qest.forecast layout, '<target>_q10/50/90' per MTU of the predicted days.
usage: CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m qest.chronos   (Q3 skill vs LightGBM and persistence)
"""
import numpy as np
import pandas as pd

from qest.data import frame
from qest.forecast import TARGETS, LAG_KNOWN, features

MODEL = 'amazon/chronos-2'
CONTEXT_DAYS = 56
MIN_CONTEXT_DAYS = 7  # days with less history are not traded by forecast strategies
GROUPS = {L: [k for k in TARGETS if LAG_KNOWN.get(k, 2) == L] for L in (1, 2)}


def _pipe():
    import torch
    from chronos import Chronos2Pipeline
    return Chronos2Pipeline.from_pretrained(MODEL, device_map='cuda' if torch.cuda.is_available() else 'cpu')


def predict(pred_days, batch=32, neighbours=False):
    """neighbours: add the FI / LV / LT day-ahead prices (qest.fetch_nps) as extra series of the D-1 group, so the multivariate
    model reads the coupling that sets Estonia's price; they are forecast jointly and dropped from the output ('chronosx')."""
    f = frame(); X = features(); pipe = _pipe()
    pred_days = [d for d in pred_days if d >= f.day.min() + pd.Timedelta(days=MIN_CONTEXT_DAYS)]  # 1-2 Jan: no gate-A history yet
    ts = f.index.tz_convert(None)
    cov = pd.DataFrame({'pv_d1': X.pv_d1.to_numpy(), 'wind_d1': X.wind_d1.to_numpy(),
                        'fleet_wind_d1': X.get('fleet_wind_d1', pd.Series(0.0, index=f.index)).to_numpy(),
                        'hour': X.hour.to_numpy(), 'dow': X.dow.to_numpy()}, index=ts).ffill().fillna(0.0)
    Y = pd.DataFrame({k: f[c].to_numpy() for k, c in TARGETS.items()}, index=ts); groups = dict(GROUPS)
    if neighbours:
        from qest.fetch_nps import prices
        N = prices().reindex(f.index).ffill()
        for z in ('fi', 'lv', 'lt'):
            Y[f'da_{z}'] = N[z].to_numpy()
        groups = {L: ks + ([f'da_{z}' for z in ('fi', 'lv', 'lt')] if L == 1 else []) for L, ks in GROUPS.items()}
    day = pd.Series(f.day.to_numpy(), index=ts); out = []
    for L, ks in groups.items():
        ctx, fut = [], []
        for d in pred_days:
            known_end = day.index[(day <= d - pd.Timedelta(days=L)).to_numpy()].max()  # last MTU published by the gate
            start = known_end - pd.Timedelta(days=CONTEXT_DAYS); hz = day.index[(day > d - pd.Timedelta(days=L)) & (day <= d)]
            c = pd.concat([Y.loc[start:known_end, ks], cov.loc[start:known_end]], axis=1).reset_index(names='timestamp'); c['id'] = str(d.date())
            u = cov.loc[hz].reset_index(names='timestamp'); u['id'] = str(d.date())
            ctx.append(c); fut.append(u)
        by_h = {}
        for c, u in zip(ctx, fut):  # one prediction length per call: DST days have 92 / 100 MTUs
            by_h.setdefault(len(u), []).append((c, u))
        for H, cu in by_h.items():
            for i in range(0, len(cu), batch):
                C, U = pd.concat([c for c, _ in cu[i:i + batch]]), pd.concat([u for _, u in cu[i:i + batch]])
                out.append(pipe.predict_df(C, future_df=U, id_column='id', timestamp_column='timestamp', target=ks,
                                           prediction_length=H, quantile_levels=[.1, .5, .9]))
        print(f'chronos group D-{L}: {len(pred_days)} days', flush=True)
    P = pd.concat(out)
    P = P[pd.to_datetime(P.timestamp).map(day).to_numpy() == pd.to_datetime(P.id).to_numpy()]  # each item keeps only its own delivery day
    W = P.pivot_table(index='timestamp', columns='target_name', values=['0.1', '0.5', '0.9'])
    res = pd.DataFrame(index=f.index[f.day.isin(pred_days)])
    tsr = res.index.tz_convert(None)
    for k in TARGETS:
        for q, qq in (('0.1', 10), ('0.5', 50), ('0.9', 90)):
            res[f'{k}_q{qq}'] = W[(q, k)].reindex(tsr).to_numpy()
        if k.endswith('_s'):
            res[[f'{k}_q{q}' for q in (10, 50, 90)]] = res[[f'{k}_q{q}' for q in (10, 50, 90)]].clip(0, 1)
    return res


if __name__ == '__main__':
    from qest.data import days
    from qest.evaluate import forecasts
    from qest.forecast import skill
    f = frame(); va = days(f, 'val')
    t = pd.DataFrame({'chronos2': skill(forecasts('val', 'chronos'), va).mae, 'lightgbm': skill(forecasts('val', 'lgbm'), va).mae,
                      'persistence': skill(forecasts('val', 'persist'), va).mae})
    print('Q3 MAE (zero-shot Chronos-2 vs LightGBM fitted on Q1-Q2):'); print(t.round(3).to_string())

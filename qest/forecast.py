"""Forecasts made at gate A (D-1 09:00 Tallinn) for every MTU of delivery day D: prices, activation, PV and wind.

Information set at the gate (no look-ahead):
- day-ahead prices up to D-1 (D-1's auction cleared D-2 ~14:00); capacity prices up to D-1 (auction D-2 09:00);
- balancing (mFRR, aFRR, imbalance) and Elering's measured system up to the end of D-2 (published with delay);
- weather forecasts issued D-1 00 UTC for D (Open-Meteo previous-day runs): the park's PV, our wind park, and the
  whole Estonian wind fleet (every OSM plant through the same power curves: a national wind forecast).
Targets: DA price, the four capacity prices, the mFRR price, the imbalance price, aFRR prices and active shares.
Models: LightGBM quantile regression (10/50/90 %), one per target and quantile, fitted on the training days; persistence
(same MTU last available day) as the baseline. Walk-forward: fit on train -> score val; refit on train+val -> test.
usage: .venv/bin/python -m qest.forecast   (validation skill table)
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from qest.data import frame
from qest.paths import DERIVED

TARGETS = dict(da='nps_dayahead_eur_mwh', cap_afrr_up='cap_afrr_up_eur_mw_h', cap_afrr_down='cap_afrr_down_eur_mw_h',
               cap_mfrr_up='cap_mfrr_up_eur_mw_h', cap_mfrr_down='cap_mfrr_down_eur_mw_h', mfrr='mfrr_up_eur_mwh',
               imb='imbalance_eur_mwh', afrr_up_p='afrr4s_up_price', afrr_dn_p='afrr4s_down_price',
               afrr_up_s='afrr4s_up_share', afrr_dn_s='afrr4s_down_share')
LAG_KNOWN = dict(da=1, cap_afrr_up=1, cap_afrr_down=1, cap_mfrr_up=1, cap_mfrr_down=1)  # days back the last known value is (else 2)
QS = (.1, .5, .9)


def national_wind(kind):
    from qest.power import wind_mw
    from qest.sites import wind_plants
    p = DERIVED / f'wind_fleet_{kind}_cal.parquet'
    if p.exists():
        return pd.read_parquet(p).iloc[:, 0]
    s = sum(wind_mw(name, kind) for name, v in wind_plants().items() if (v['mw'] or 0) >= 1)
    s.rename('fleet').to_frame().to_parquet(p)
    return s


def features():
    """One row per MTU with only gate-A information about its delivery day."""
    from qest.power import generation
    f = frame(); g = generation()
    X = pd.DataFrame(index=f.index)
    X['mtu'] = f.mtu; X['hour'] = f.local.dt.hour; X['dow'] = f.local.dt.dayofweek; X['month'] = f.local.dt.month
    X['doy'] = f.local.dt.dayofyear
    by = f.set_index(['day', 'mtu'])
    for k, col in TARGETS.items():
        lag = LAG_KNOWN.get(k, 2)
        piv = by[col].unstack('mtu')  # day x mtu
        for L in (lag, lag + 1, 7):
            sh = piv.shift(L)  # value L days before, same MTU
            X[f'{k}_lag{L}'] = sh.stack(future_stack=True).reindex(pd.MultiIndex.from_arrays([f.day, f.mtu])).to_numpy()
        dm = piv.mean(axis=1).shift(lag); X[f'{k}_daymean_lag'] = f.day.map(dm).to_numpy()
        dx = piv.max(axis=1).shift(lag); X[f'{k}_daymax_lag'] = f.day.map(dx).to_numpy()
    X['pv_d1'] = g.hybrid_pv_d1_per_mw; X['wind_d1'] = g.wind_d1
    try:
        X['fleet_wind_d1'] = national_wind('d1').reindex(f.index).to_numpy()
    except FileNotFoundError:
        pass
    if 'ee_real_consumption' in f:
        cons = f.groupby('day').ee_real_consumption.mean().shift(2); X['cons_lag2'] = f.day.map(cons).to_numpy()
    return X


HALF_LIFE = 45.0  # days: recent days weigh more (prices shift between quarters)


def fit_predict(fit_days, pred_days, targets=TARGETS, X=None, delta=False, half_life=0):  # level, unweighted: best on validation
    """LightGBM quantile models fitted on fit_days, predictions for pred_days: DataFrame with columns '<target>_q<q>'.
    delta: the model learns the change from the last known value of the same MTU (trees cannot extrapolate a level the
    training days never saw; the last known value carries the level, the model the shape and the spread)."""
    import lightgbm as lgb
    f = frame(); X = features() if X is None else X
    fm, pm = f.day.isin(fit_days), f.day.isin(pred_days)
    age = (pd.Timestamp(max(fit_days)) - f.day).dt.days.to_numpy(); w = .5 ** (age / half_life) if half_life else np.ones(len(f))
    out = pd.DataFrame(index=f.index[pm])
    for k, col in targets.items():
        ref = X[f'{k}_lag{LAG_KNOWN.get(k, 2)}'].fillna(X[f'{k}_daymean_lag']).fillna(0) if delta else pd.Series(0.0, index=f.index)
        y = f[col] - ref; m = fm & y.notna()
        lo, hi = y[m].quantile([.005, .995])  # winsorise spikes for fitting only (scoring keeps them)
        for q in QS:
            mdl = lgb.LGBMRegressor(objective='quantile', alpha=q, n_estimators=400, learning_rate=.04, num_leaves=31,
                                    min_child_samples=40, subsample=.8, subsample_freq=1, colsample_bytree=.8, verbose=-1)
            mdl.fit(X[m], y[m].clip(lo, hi), sample_weight=w[m.to_numpy()])
            out[f'{k}_q{int(q * 100)}'] = mdl.predict(X[pm]) + ref[pm].to_numpy()
        if k.endswith('_s'):
            out[[c for c in out if c.startswith(k)]] = out[[c for c in out if c.startswith(k)]].clip(0, 1)
    return out


def persistence(pred_days, X=None):
    f = frame(); X = features() if X is None else X; pm = f.day.isin(pred_days)
    return pd.DataFrame({f'{k}_q50': X.loc[pm, f'{k}_lag{LAG_KNOWN.get(k, 2)}'] for k in TARGETS}, index=f.index[pm])


def skill(pred, pred_days):
    f = frame(); m = f.day.isin(pred_days); rows = {}
    for k, col in TARGETS.items():
        y = f.loc[m, col]; p = pred[f'{k}_q50']; ok = y.notna() & p.notna()
        rows[k] = dict(mae=float((p[ok] - y[ok]).abs().mean()))
        if f'{k}_q10' in pred:
            rows[k]['cover80'] = float(((y >= pred[f'{k}_q10']) & (y <= pred[f'{k}_q90']))[ok].mean())
    return pd.DataFrame(rows).T


if __name__ == '__main__':
    from qest.data import days
    f = frame(); X = features(); tr, va = days(f, 'train'), days(f, 'val')
    ps = persistence(va, X=X); rows = {}
    for name, kw in dict(level=dict(delta=False, half_life=0), level_recent=dict(delta=False), delta=dict(half_life=0), delta_recent={}).items():
        rows[name] = skill(fit_predict(tr, va, X=X, **kw), va)
    t = pd.concat({k: v.mae for k, v in rows.items()}, axis=1); t['persistence'] = skill(ps, va).mae
    t['cover80_delta_recent'] = rows['delta_recent'].cover80
    print('validation (Q3) MAE, fitted on Q1-Q2:'); print(t.round(3).to_string())

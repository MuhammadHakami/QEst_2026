"""Gate-A forecaster of the TSO's activations: P(mFRR UP / DOWN activated at our bid) and the aFRR active shares per MTU.

Why: 82 % of the gap to hindsight is knowing when Elering activates (research/gap_to_hindsight.md). The plan today uses an
hour-of-day climatology for mFRR and Chronos-2's zero-shot share for aFRR, neither of which sees the drivers of the Baltic
block's imbalance. Here LightGBM learns the activations from everything legal at D-1 09:00 Tallinn:
  dataset   hour / weekday / month, D-2 balancing lags (same MTU, day mean, day max), D-1 DA price, D-1 weather for the
            hybrid PV, our park and the national wind fleet (level, ramp, day mean / max), D-2 consumption, D-2 UP / DOWN rates
  Elering   (qest.fetch_tx) NTC to / from Finland and Latvia for D (published before the auction); yesterday's scheduled
            trade on both borders and its hourly steps (D-1's schedule was fixed at D-2 13:00); D-2 realised flows and the
            schedule deviation flow - plan; D-1's production / consumption plan and D-2's plan error
  Nord Pool (qest.fetch_nps) D-1 FI - EE and LV - EE day-ahead spreads (which way Estlink and the Latvian border will lean)
Targets: m_up / m_dn = the 15-min mFRR price crosses DA +- margin (the planner's m_up_q / m_dn_q); a_up_s / a_dn_s = PICASSO
active share of the MTU (any price). Recency-weighted (HALF_LIFE days). Skill is scored against the hour-of-day climatology
of the fit days (what the plan uses) and Chronos-2's shares: log-loss, Brier, AUC, within-day Spearman. Walk-forward: refitted before each calendar month on every day whose outcome was public (before month start minus the D-2 lag). Predictions are cached
in results/actfc_<split>.parquet and consumed by strategies.milp_act (the deterministic plan with these marginals).

usage: .venv/bin/python -m qest.actfc [--split val|test] [--check]
"""
import sys
import warnings

import numpy as np
import pandas as pd

from qest.data import frame, days
from qest.paths import RESULTS

warnings.filterwarnings('ignore')
MARGIN = 5.0  # EUR/MWh over / under DA, as qest.qscen
HALF_LIFE = 60.0  # days
LAG_DAYS = 2  # publication lag of the balancing outcomes at gate A
TARGETS = ('m_up', 'm_dn', 'a_up_s', 'a_dn_s')


def _lag(f, s, L):
    """Value of series s at the same MTU L days earlier (day x mtu pivot shift), aligned to f.index."""
    piv = pd.Series(np.asarray(s, float), index=pd.MultiIndex.from_arrays([f.day, f.mtu])).unstack('mtu')
    return piv.shift(L).stack(future_stack=True).reindex(pd.MultiIndex.from_arrays([f.day, f.mtu])).to_numpy()


def _daymap(f, s, L, how='mean'):
    d = pd.Series(np.asarray(s, float), index=f.index).groupby(f.day).agg(how).shift(L); return f.day.map(d).to_numpy()


def targets(f):
    da = f.nps_dayahead_eur_mwh; m = f.mfrr_up_eur_mwh
    return pd.DataFrame({'m_up': (m.notna() & (m >= da + MARGIN)).astype(float), 'm_dn': (f.mfrr_down_eur_mwh.notna() & (f.mfrr_down_eur_mwh <= da - MARGIN)).astype(float),
                         'a_up_s': f.afrr4s_up_share.clip(0, 1), 'a_dn_s': f.afrr4s_down_share.clip(0, 1)}, index=f.index)


def features():
    from qest.forecast import features as base
    from qest.fetch_tx import tx
    from qest.fetch_nps import prices
    f = frame(); X = base().copy(); Y = targets(f)
    for k in ('m_up', 'm_dn', 'a_up_s', 'a_dn_s'):  # D-2 lags of the targets themselves
        for L in (2, 3, 7):
            X[f'{k}_lag{L}'] = _lag(f, Y[k], L)
        X[f'{k}_rate2'] = _daymap(f, Y[k], 2); X[f'{k}_rate3'] = _daymap(f, Y[k], 3)
        X[f'{k}_rate7'] = pd.Series(_daymap(f, Y[k], 2), index=f.index).groupby(f.day).first().rolling(6, min_periods=1).mean().reindex(f.day).to_numpy()  # mean rate over D-7..D-2: the regime anchor
    if 'fleet_wind_d1' in X:
        w = X.fleet_wind_d1; X['fleet_ramp4'] = w.groupby(f.day).diff(4).to_numpy(); X['fleet_ramp8'] = w.groupby(f.day).diff(8).to_numpy()
        X['fleet_daymean'] = _daymap(f, w, 0); X['fleet_daymax'] = _daymap(f, w, 0, 'max')
    X['pv_daysum'] = _daymap(f, X.pv_d1, 0, 'sum')
    T = tx().reindex(f.index)
    for b in ('fi', 'lv'):
        X[f'ntc_{b}_in'] = T[f'cap_{b}_in_ntc'].to_numpy(); X[f'ntc_{b}_out'] = T[f'cap_{b}_out_ntc'].to_numpy()  # D's NTC: published D-1 before the auction
        p = T[f'plan_ee_{b}']; fl = T[f'flow_{"finland" if b == "fi" else "latvia"}']
        X[f'plan_{b}_lag1'] = _lag(f, p, 1); X[f'planstep_{b}_lag1'] = _lag(f, p.diff(), 1)  # yesterday's schedule and its steps
        X[f'flow_{b}_lag2'] = _lag(f, fl, 2); X[f'dev_{b}_lag2'] = _lag(f, fl - p, 2)  # realised flow and schedule deviation, D-2
        X[f'plan_{b}_daymean_lag1'] = _daymap(f, p, 1)
    X['ntc_fi_in_x_plan'] = X.ntc_fi_in - X.plan_fi_lag1.abs()  # room left on Estlink if today schedules like yesterday
    for c in ('ee_plan_production', 'ee_plan_consumption', 'ee_plan_production_renewable'):
        if c in f:
            X[f'{c}_lag1'] = _lag(f, f[c], 1)
    if 'ee_plan_production' in f:
        X['plan_err_lag2'] = _lag(f, (f.ee_real_production - f.ee_real_consumption) - (f.ee_plan_production - f.ee_plan_consumption), 2)
    P = prices().reindex(f.index)
    X = X.drop(columns=[c for c in ('doy', 'month') if c in X])  # no calendar proxies: a quarter the fit never saw must not inherit another season's regime
    for z in ('fi', 'lv'):
        X[f'spread_{z}_lag1'] = _lag(f, P[z] - P.ee, 1); X[f'spread_{z}_daymean_lag1'] = _daymap(f, P[z] - P.ee, 1)
    return X


def _fit_one(X, y, w, init, kind, seed):
    import lightgbm as lgb
    # boosted from the climatology: init_score = logit(climatology); the trees learn only the deviation the drivers explain,
    # heavily regularised so a regime shift between quarters cannot make the forecast worse than the climatology by much
    kw = dict(n_estimators=300, learning_rate=.02, num_leaves=7, min_child_samples=250, subsample=.7, subsample_freq=1, colsample_bytree=.6, reg_lambda=20.0, verbose=-1, random_state=seed)
    lo = np.log(np.clip(init, 1e-3, 1 - 1e-3) / (1 - np.clip(init, 1e-3, 1 - 1e-3)))
    mdl = lgb.LGBMRegressor(objective='cross_entropy', **kw); mdl.fit(X, np.clip(y, 1e-4, 1 - 1e-4), sample_weight=w, init_score=lo)
    return mdl


def fit_predict(fit_days, pred_days, X=None, half_life=HALF_LIFE, seed=0, monthly=True):
    """Predictions for pred_days from models fitted on fit_days; monthly=True refits before each calendar month of pred_days
    on every day before it (gate-A legal: only the past), so the Q3 regime enters the Q3 forecasts as it unfolds."""
    f = frame(); X = features() if X is None else X; Y = targets(f); h = f.local.dt.hour
    pred_days = sorted(pred_days); months = sorted({pd.Timestamp(d).to_period('M') for d in pred_days}) if monthly else [None]
    out = pd.DataFrame(index=f.index[f.day.isin(pred_days)], columns=list(TARGETS), dtype=float); imp = {}
    for mo in months:
        pdays = [d for d in pred_days if mo is None or pd.Timestamp(d).to_period('M') == mo]
        cut = pd.Timestamp(pdays[0]) - pd.Timedelta(days=LAG_DAYS)  # balancing outcomes are public with a D-2 lag: the fit for day D sees targets to D-2 only
        fdays = sorted({d for d in set(fit_days) | (set(pred_days) if monthly else set()) if pd.Timestamp(d) < cut})
        if not fdays:
            continue  # nothing before the first month: no forecast (stays NaN)
        fm, pm = f.day.isin(fdays), f.day.isin(pdays)
        age = (pd.Timestamp(max(fdays)) - f.day).dt.days.to_numpy(); w = .5 ** (age / half_life) if half_life else np.ones(len(f))
        for k in TARGETS:
            y = Y[k]; m = fm & y.notna(); ww = pd.Series(w[m.to_numpy()], index=y.index[m]); clim = (y[m] * ww).groupby(h[m]).sum() / ww.groupby(h[m]).sum()  # recency-weighted hour-of-day climatology: the level follows the regime
            mdl = _fit_one(X[m], y[m].to_numpy(), w[m.to_numpy()], h[m].map(clim).to_numpy(), k, seed)
            init_p = h[pm].map(clim).to_numpy(); lo = np.log(np.clip(init_p, 1e-3, 1 - 1e-3) / (1 - np.clip(init_p, 1e-3, 1 - 1e-3)))
            out.loc[pm, k] = 1 / (1 + np.exp(-(mdl.predict(X[pm], raw_score=True) + lo)))
            imp[k] = pd.Series(mdl.feature_importances_, index=X.columns).sort_values(ascending=False)
    out.attrs['importance'] = imp
    return out


def climatology(fit_days, pred_days):
    """The plan's own reference: mean of each target by hour of day over the fit days."""
    f = frame(); Y = targets(f); fm, pm = f.day.isin(fit_days), f.day.isin(pred_days); h = f.local.dt.hour
    return pd.DataFrame({k: h[pm].map(Y[k][fm].groupby(h[fm]).mean()).to_numpy() for k in TARGETS}, index=f.index[pm])


def _spearman_day(p, y, day):
    d = pd.DataFrame({'p': p, 'y': y, 'd': day}).dropna(); r = d.groupby('d').apply(lambda g: g.p.corr(g.y, method='spearman') if g.y.nunique() > 1 and g.p.nunique() > 1 else np.nan)
    return float(r.mean())


def score(pred, split):
    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
    f = frame(); Y = targets(f).reindex(pred.index); day = f.day.reindex(pred.index); rows = []
    for k in TARGETS:
        y, p = Y[k], pred[k].clip(1e-4, 1 - 1e-4); m = y.notna() & p.notna()
        r = dict(target=k, spearman_day=_spearman_day(p[m], y[m], day[m]), brier=brier_score_loss(y[m] if k.startswith('m_') else (y[m] > .5).astype(int), p[m]) if k.startswith('m_') else float(((y[m] - p[m]) ** 2).mean()))
        if k.startswith('m_'):
            r['logloss'] = log_loss(y[m].astype(int), p[m]); r['auc'] = roc_auc_score(y[m].astype(int), p[m])
        else:
            r['mae'] = float((y[m] - p[m]).abs().mean())
        rows.append(r)
    return pd.DataFrame(rows).set_index('target')


def run(split='val', save=True):
    f = frame(); fit = [] if split == 'train' else days(f, 'train') + (days(f, 'val') if split == 'test' else []); pred = days(f, split)  # train: monthly refits on the months before only
    X = features(); P = fit_predict(fit, pred, X); C = climatology(fit or [d for d in pred if pd.Timestamp(d) < pd.Timestamp(pred[0]) + pd.Timedelta(days=31)], pred)
    pd.set_option('display.width', 220)
    S = pd.concat({'actfc': score(P, split), 'climatology': score(C, split)}, axis=1)
    ch = RESULTS / f'forecast_chronos_{split}.parquet'
    if ch.exists():
        F = pd.read_parquet(ch).reindex(P.index); Q = pd.DataFrame({'a_up_s': F.afrr_up_s_q50, 'a_dn_s': F.afrr_dn_s_q50, 'm_up': C.m_up, 'm_dn': C.m_dn}, index=P.index)
        S = pd.concat([S, pd.concat({'chronos (shares; mFRR = climatology)': score(Q, split)}, axis=1)], axis=1)
    print(f'--- {split}: fit on {len(fit)} days, scored on {len(pred)} days'); print(S.round(3).to_string())
    print('--- top features (last refit)'); print(pd.DataFrame({k: v.head(12).index for k, v in P.attrs['importance'].items()}).to_string())
    if save:
        pd.DataFrame(P).to_parquet(RESULTS / f'actfc_{split}.parquet'); S.to_csv(RESULTS / f'actfc_skill_{split}.csv')  # attrs dropped: not JSON
    return P, S


def _check():
    """On Q3 (fit through Q2, monthly refits) the driver-based forecast ranks activations better than the hour-of-day
    climatology the plan uses (mFRR UP AUC, aFRR share Brier)."""
    P, S = run('val', save=False)
    assert S[('actfc', 'auc')]['m_up'] > S[('climatology', 'auc')]['m_up'], 'no ranking skill over the climatology on mFRR UP'
    assert S[('actfc', 'brier')]['a_up_s'] < S[('climatology', 'brier')]['a_up_s'], 'no skill over the climatology on the aFRR UP share'
    print('actfc check ok')


if __name__ == '__main__':
    if '--check' in sys.argv:
        _check()
    else:
        sp = sys.argv[sys.argv.index('--split') + 1] if '--split' in sys.argv else 'val'
        run(sp)

"""Run every strategy over a split's delivery days and compare (EUR, by market and penalty).

Walk-forward: validation (Q3) uses forecasts fitted on Q1-Q2; test (Q4, 1-25 Oct) uses forecasts refitted on Q1-Q3.
Each day: gate-A forecasts -> each strategy's plan -> settlement against the realised day (qest.market.settle_day);
storage pays the grid tariff on its net monthly import.
usage: .venv/bin/python -m qest.evaluate --split val|test [--settlement official|baltic|starter] [--units battery,hybrid,solar,wind]
       [--strategies idle,naive_da,...]
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from qest import market
from qest.data import days, frame
from qest.forecast import features, fit_predict, persistence
from qest.paths import RESULTS
from qest.power import generation
from qest.strategies import STRATEGIES, DayContext, calibration


def forecasts(split, kind='lgbm'):
    """Gate-A forecasts for a split. For test, two choices are made on validation only (fit Q1-Q2, score Q3): per target,
    LightGBM or persistence, whichever had the lower validation MAE; and a split-conformal widening of the 10-90 %
    band so it covered 80 % of validation outcomes (conformalised quantile regression)."""
    p = RESULTS / f'forecast_{kind}_{split}.parquet'
    if p.exists():
        return pd.read_parquet(p)
    from qest.forecast import TARGETS, skill
    f = frame(); X = features(); fit = days(f, 'train') if split in ('train', 'val') else days(f, ('train', 'val')); pred = days(f, split)
    if kind == 'persist':
        out = persistence(pred, X=X)
    elif kind in ('chronos', 'chronosx'):  # 'chronosx': with the neighbouring zones' prices as joint series
        from qest.chronos import predict
        out = predict(pred, neighbours=(kind == 'chronosx'))
    elif kind == 'tsens':  # the two foundation models averaged, quantile by quantile (Chronos-2 + Timestral)
        a, b = forecasts(split, 'chronos'), forecasts(split, 'timestral'); out = (a + b.reindex(a.index)[a.columns]) / 2
    elif kind.startswith('timestral'):  # Mistral Timestral zero-shot (qest.fc_tsfm); 'timestral120' = 120-day context
        from qest.fc_tsfm import predict as tpredict
        out = tpredict(pred, context_days=int(kind[9:]) if kind[9:] else 56)
    elif kind == 'tabpfn':  # TabPFN stacked on Chronos-2 (qest.fc_tabpfn), conformal widening as the LightGBM path
        from qest.fc_tabpfn import predict
        cf = pd.concat([forecasts(s, 'chronos') for s in (('train',) if split in ('train', 'val') else ('train', 'val'))])
        out = predict(fit, pred, cf, forecasts(split, 'chronos'))
        if split == 'test':
            va = days(f, 'val'); tv = forecasts('val', 'tabpfn')
            for k, col in TARGETS.items():
                y = f.loc[f.day.isin(va), col]; lo, hi = tv[f'{k}_q10'], tv[f'{k}_q90']; ok = y.notna() & lo.notna()
                E = np.maximum(lo[ok] - y[ok], y[ok] - hi[ok]); Q = float(np.quantile(E, .8 * (1 + 1 / ok.sum())))
                out[f'{k}_q10'] -= Q; out[f'{k}_q90'] += Q
                if k.endswith('_s'):
                    out[[f'{k}_q{q}' for q in (10, 50, 90)]] = out[[f'{k}_q{q}' for q in (10, 50, 90)]].clip(0, 1)
    else:
        out = fit_predict(fit, pred, X=X)
        if split == 'test':
            va = days(f, 'val'); lv, pv = forecasts('val', 'lgbm'), forecasts('val', 'persist')
            ps = persistence(pred, X=X); sl, sp = skill(lv, va), skill(pv, va)
            for k, col in TARGETS.items():
                y = f.loc[f.day.isin(va), col]
                if sp.loc[k, 'mae'] < sl.loc[k, 'mae']:  # persistence wins on validation: its median, LightGBM's spread
                    shift = ps[f'{k}_q50'] - out[f'{k}_q50']
                    for q in (10, 50, 90): out[f'{k}_q{q}'] = out[f'{k}_q{q}'] + shift
                    base = pv[f'{k}_q50'] - lv[f'{k}_q50']; lo, hi = lv[f'{k}_q10'] + base, lv[f'{k}_q90'] + base
                else:
                    lo, hi = lv[f'{k}_q10'], lv[f'{k}_q90']
                ok = y.notna() & lo.notna(); E = np.maximum(lo[ok] - y[ok], y[ok] - hi[ok])
                Q = float(np.quantile(E, .8 * (1 + 1 / ok.sum())))
                out[f'{k}_q10'] -= Q; out[f'{k}_q90'] += Q
                if k.endswith('_s'):
                    out[[f'{k}_q{q}' for q in (10, 50, 90)]] = out[[f'{k}_q{q}' for q in (10, 50, 90)]].clip(0, 1)
    RESULTS.mkdir(parents=True, exist_ok=True); out.to_parquet(p)
    return out


def fc_kind(name):
    """Which forecaster a strategy plans on, and whose Optuna knobs it falls back to."""
    if '@' in name:  # '<decider>@<forecaster>' (qest.qdecide); 'actrank' = Chronos-2 prices + rank-gated activation marginals
        k = name.split('@', 1)[1]; return 'chronos' if k == 'actrank' else k
    if name in ('milp_chronos_qrc', 'milp_chronos_rff', 'milp_tabpfn', 'milp_chronosx') or name.startswith('milp_timestral'):
        return name[5:]
    if name == 'milp_chronos' or name.startswith('qday_') or name.startswith('oracle') or name.startswith('milp_c') or name.startswith('milp_rp') or name.startswith('milp_stoch') or name.startswith('milp_scen') or name == 'milp_mk' or name.startswith('milp_act'):  # oracle reads no forecast: any cached split keeps its day loop
        return os.environ.get('QEST_FC', 'chronos')  # QEST_FC=timestral: the whole Chronos-2 family of plans on another foundation model's prices
    return 'persist' if name == 'milp_persist' else 'lgbm'


def unit_gen(g, units):
    act, fc = {}, {}
    for u in units:
        if u.kind == 'solar':
            act[u.name], fc[u.name] = g.pv_actual, g.pv_d1
        elif u.kind == 'wind':
            act[u.name], fc[u.name] = g.wind_actual * u.gen_mw / 45.0, g.wind_d1 * u.gen_mw / 45.0
        elif u.kind == 'hybrid':
            act[u.name], fc[u.name] = g.hybrid_pv_actual_per_mw * u.gen_mw, g.hybrid_pv_d1_per_mw * u.gen_mw
    return act, fc


def run(split, strategies, units, cfg, margins=None, params=None, traces=None):
    """traces: a dict to fill with each (strategy, day)'s storage trace per MTU (delivered MW, SoC MWh), for qest.twin."""
    f = frame(); g = generation(); D = days(f, split)
    if os.environ.get('QEST_DAYS'):  # quick runs on a subset of delivery days (comma-separated dates)
        keep = set(os.environ['QEST_DAYS'].split(',')); D = [d for d in D if str(pd.Timestamp(d).date()) in keep]
    fit = days(f, 'train') if split in ('train', 'val') else days(f, ('train', 'val'))
    def knobs(name):  # the knobs Optuna chose on Q3 for this strategy (qest.tune), else the LightGBM MILP's, else defaults
        if margins is not None or params is not None:
            return margins, params
        fam = 'milp_chronos' if fc_kind(name).startswith(('chronos', 'tabpfn', 'timestral')) or '@' in name else 'milp'  # '@': Chronos-2's knobs for every forecaster
        for tb in (RESULTS / f'tune_best_{name}.json', RESULTS / f'tune_best_{fam}.json', RESULTS / 'tune_best.json'):
            if tb.exists():
                import json
                B = json.loads(tb.read_text()); return B['margins'], B['params']
        return None, None
    calibs = {}

    WALK = bool(os.environ.get('QEST_WALK'))  # walk-forward: each month's calibration from the months before it (D-2 lag), whatever the split
    all_days = sorted(set(days(f, 'train')) | set(days(f, 'val')) | set(days(f, 'test')))

    def calib_for(name, d=None):
        M, P = knobs(name); mkey = pd.Timestamp(d).strftime('%Y-%m') if (WALK and d is not None) else ''; key = repr((M, P, mkey))
        if key not in calibs:
            fd = fit
            if mkey:
                cut = pd.Timestamp(d).replace(day=1) - pd.Timedelta(days=2); fd = [x for x in all_days if pd.Timestamp(x) < cut]
                if not fd:  # the first month has no past: its own days (in-sample, printed once)
                    fd = [x for x in all_days if pd.Timestamp(x).strftime('%Y-%m') == mkey]; print(f'walk-forward: {mkey} calibrated on itself (no earlier data)', flush=True)
            calibs[key] = calibration(f, fd) if M is None else calibration(f, fd, M)
            calibs[key].update(P or {})  # kappa_up / kappa_dn / tol
        return calibs[key]
    from qest.data import afrr_margin_table
    atab = afrr_margin_table().reindex(f.index).fillna(0)
    fcs = {k: forecasts(split, k) for k in {fc_kind(s) for s in strategies}}
    act, fc = unit_gen(g, units)
    u_kind = {u.name: u.kind for u in units}
    rows = []
    for name in strategies:
        fn = STRATEGIES[name]; t0 = time.time()
        for d in D:
            calib = calib_for(name, d); m = (f.day == d).to_numpy(); day = f[m]; idx = day.index
            fcd = fcs[fc_kind(name)].reindex(idx)
            if fcd.isna().all().all():
                continue  # no gate-A forecast for this day (qest.chronos.MIN_CONTEXT_DAYS)
            ctx = DayContext(idx, fcd, {k: v.reindex(idx).fillna(0).to_numpy() for k, v in fc.items()}, calib,
                             real=day if name.startswith('oracle') else None,
                             gen_real={k: v.reindex(idx).fillna(0).to_numpy() for k, v in act.items()} if name.startswith('oracle') else None,
                             afrr_tab=atab.loc[idx] if name.startswith('oracle') else None)
            plan = fn(units, ctx, cfg)
            res = market.settle_day(units, plan, day, {k: v.reindex(idx).fillna(0).to_numpy() for k, v in act.items()}, cfg, afrr_tab=atab.loc[idx])
            if traces is not None:
                traces[(name, d)] = {un: dict(idx=idx, deliv=np.asarray(r['trace']['deliv'], float), soc=np.asarray(r['trace']['soc'], float))
                                     for un, r in res.items() if u_kind.get(un) in ('battery', 'hybrid')}
            for un, r in res.items():
                rows.append(dict(strategy=name, unit=un, day=d, **{k: v for k, v in r.items() if np.isscalar(v)}))
        print(f'{name}: {len(D)} days in {time.time() - t0:.0f} s', flush=True)
    R = pd.DataFrame(rows)
    # monthly grid tariff on storage's net import
    um = {u.name: u for u in units}
    R['month'] = pd.to_datetime(R.day).dt.to_period('M')
    tar = R.groupby(['strategy', 'unit', 'month']).apply(lambda x: market.tariff(x.import_mwh.sum(), x.export_mwh.sum(), um[x.name[1]]), include_groups=False)
    return R, tar


def table(R, tar):
    comp = ['cap', 'da', 'mfrr', 'afrr', 'imbalance', 'penalty', 'leftover', 'cycle', 'tariff_mtu']
    t = R.groupby('strategy')[comp + ['total', 'imb_mwh', 'undelivered_mwh', 'curtailed_mwh']].sum()
    t['tariff'] = tar.groupby('strategy').sum(); t['total'] = t.total + t.tariff
    return t.sort_values('total', ascending=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='val'); ap.add_argument('--settlement', default='official')
    ap.add_argument('--units', default='battery,hybrid,solar,wind'); ap.add_argument('--strategies', default=','.join(STRATEGIES))
    a = ap.parse_args()
    cfg = market.Config(settlement=a.settlement); units = market.portfolio(a.units.split(','), official=cfg.official)
    R, tar = run(a.split, a.strategies.split(','), units, cfg)
    RESULTS.mkdir(parents=True, exist_ok=True); R.to_parquet(RESULTS / f'days_{a.split}_{a.settlement}.parquet')
    t = table(R, tar); t.to_csv(RESULTS / f'summary_{a.split}_{a.settlement}.csv')
    pd.set_option('display.width', 250); print(t.round(0).to_string())
    print(R.groupby(['strategy', 'unit']).total.sum().unstack().round(0).to_string())

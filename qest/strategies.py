"""Trading strategies: each maps one delivery day's gate-time information to a market plan per unit.

A strategy sees a DayContext: the day's MTU index, the forecasts made at gate A (qest.forecast quantiles), the PV and
wind D-1 forecasts, and calibration tables fitted on the training days. Only 'oracle' sees the realised day.
  idle         nothing sold ahead: every MWh meets the real-time (imbalance) market: the 'too late' end of the ladder
  naive_da     PV/wind sell their D-1 forecast day-ahead; batteries idle
  rule         + batteries charge in the 8 cheapest forecast MTUs and discharge in the 8 dearest (one cycle)
  reserve_rule + batteries sell all their power as the best-priced forecast capacity product, backed (no DA)
  milp         joint DA + reserve MILP on the median forecasts (qest.optimize), newsvendor PV/wind quantities
  milp_persist the same MILP on persistence forecasts (yesterday's prices)
  milp_chronos the same MILP on Chronos-2 zero-shot forecasts (qest.chronos)
  milp_da      the MILP without reserves (day-ahead arbitrage only)
  oracle       the MILP on the realised prices, activations and production (upper bound, not tradable)
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from qest.market import DT, PRODUCTS
from qest.optimize import newsvendor, plan_storage

MARGINS = dict(afrr_up=25.0, afrr_down=25.0, mfrr_up=100.0, mfrr_down=100.0)  # EUR/MWh beyond DA where energy bids sit: chosen on Q3 (qest.tune, results/tune_margins.csv)


@dataclass
class DayContext:
    idx: pd.DatetimeIndex
    fc: pd.DataFrame          # quantile forecasts for the day (columns '<target>_q10/50/90'); may hold only _q50
    gen_fc: dict              # unit -> D-1 forecast MW per MTU (PV, wind, hybrid PV)
    calib: dict               # activation tables from the training days (calibration())
    real: pd.DataFrame = None  # realised rows (oracle only)
    gen_real: dict = None
    afrr_tab: pd.DataFrame = None  # realised PICASSO activation per margin (oracle only)
    margins: dict = None


def calibration(f, fit_days, margins=MARGINS):
    """From the training days, by hour of day: how often an energy bid at DA +- margin is activated and the mean price
    beyond DA when it is (mFRR: the 15-min price crosses; aFRR: share of 4-s ticks, qest.data.afrr_margin_table), plus
    the any-price aFRR share (to rescale the day's forecast share)."""
    from qest.data import afrr_margin_table, MARGINS as GRID
    d = f[f.day.isin(fit_days)]; h = d.local.dt.hour; T = afrr_margin_table().reindex(d.index).fillna(0); da = d.nps_dayahead_eur_mwh
    m = d.mfrr_up_eur_mwh
    up = (m >= da + margins['mfrr_up']).where(m.notna()); dn = (m <= da - margins['mfrr_down']).where(m.notna())
    ku = min(GRID, key=lambda x: abs(x - margins['afrr_up'])); kd = min(GRID, key=lambda x: abs(x - margins['afrr_down']))
    C = dict(m_up_q=up.groupby(h).mean(), m_dn_q=dn.groupby(h).mean(),
             m_up_prem=(m - da).where(up == 1).groupby(h).mean().fillna(0), m_dn_prem=(m - da).where(dn == 1).groupby(h).mean().fillna(0),
             a_up_s=T[f'up_s_{ku}'].groupby(h).mean(), a_dn_s=T[f'dn_s_{kd}'].groupby(h).mean(),
             a_up_prem=(T[f'up_p_{ku}'] - da).where(T[f'up_s_{ku}'] > 0).groupby(h).mean().fillna(0),
             a_dn_prem=(T[f'dn_p_{kd}'] - da).where(T[f'dn_s_{kd}'] > 0).groupby(h).mean().fillna(0),
             a_up_s0=T['up_s_0'].groupby(h).mean(), a_dn_s0=T['dn_s_0'].groupby(h).mean(), margins=dict(margins), ku=ku, kd=kd)
    return C


PARTS = dict(da=('da',), cap=tuple(f'cap_{k}' for k in PRODUCTS), afrr=('a_up_s', 'a_dn_s', 'a_up_p', 'a_dn_p'), mfrr=('m_up_q', 'm_dn_q', 'm_up_p', 'm_dn_p'))


def _vol_caps(ctx, real):
    """Activation volume caps for the planner: the realised ones for the oracle, else the hour-of-day mean of the caps seen so far
    (a gate-A forecast of what the TSO activates). None until qest.fetch_btd data or QEST_ACT_CAP exist."""
    from qest.market import activated_volumes
    VA = activated_volumes(ctx.idx)
    if VA is None:
        return {}
    up, dn = np.minimum(VA['mfrr_up'], VA['afrr_up'] * 0 + VA['mfrr_up']), VA['mfrr_down']  # one cap per direction (mFRR's; aFRR's own cap applies in settlement)
    if not real:  # ponytail: with a uniform QEST_ACT_CAP the forecast equals the cap; with real data this should be an hour-of-day mean of past days
        up, dn = np.full_like(up, np.nanmean(up)), np.full_like(dn, np.nanmean(dn))
    return dict(vol_up=np.nan_to_num(up, nan=1e9, posinf=1e9), vol_dn=np.nan_to_num(dn, nan=1e9, posinf=1e9))


def prices_for_milp(ctx, q='q50', real=False, parts=()):
    """Inputs of qest.optimize.plan_storage: forecasts (or the realised day for the oracle) at the strategy's margins.
    parts: partial oracle, these input groups (keys of PARTS) taken from the realised day, the rest forecast: the value
    of perfect information per source (with 'da' the activation prices keep their forecast premium over the real DA)."""
    px = _prices(ctx, q, real)
    if parts and not real:
        r = _prices(ctx, q, True)
        for g in parts:
            for k in PARTS.get(g, ()):
                px[k] = r[k]
        if 'da' in parts and 'afrr' not in parts:
            px['a_up_p'] = r['da'] + (px['a_up_p'] - px['da_fc']); px['a_dn_p'] = r['da'] + (px['a_dn_p'] - px['da_fc'])
        if 'da' in parts and 'mfrr' not in parts:
            px['m_up_p'] = r['da'] + (px['m_up_p'] - px['da_fc']); px['m_dn_p'] = r['da'] + (px['m_dn_p'] - px['da_fc'])
    px.pop('da_fc', None); px.update(_vol_caps(ctx, real))
    from qest.market import FAT  # the planner expects the ramped energy too
    for k, f_ in (('m_up_q', 'mfrr'), ('m_dn_q', 'mfrr'), ('a_up_s', 'afrr'), ('a_dn_s', 'afrr')):
        px[k] = np.asarray(px[k], float) * FAT[f_]
    return px


def _prices(ctx, q, real):
    C = ctx.calib; M = C['margins']
    if real:
        r = ctx.real; T = ctx.afrr_tab; da = r.nps_dayahead_eur_mwh.to_numpy(); m = r.mfrr_up_eur_mwh.fillna(r.nps_dayahead_eur_mwh).to_numpy()
        return dict(da=da, **{f'cap_{k}': r[f'cap_{k}_eur_mw_h'].fillna(0).to_numpy() for k in PRODUCTS},
                    a_up_s=T[f"up_s_{C['ku']}"].to_numpy(), a_dn_s=T[f"dn_s_{C['kd']}"].to_numpy(),
                    a_up_p=T[f"up_p_{C['ku']}"].to_numpy(), a_dn_p=T[f"dn_p_{C['kd']}"].to_numpy(),
                    m_up_q=(m >= da + M['mfrr_up']).astype(float), m_dn_q=(m <= da - M['mfrr_down']).astype(float), m_up_p=m, m_dn_p=m)
    fc = ctx.fc; col = lambda k: fc[f'{k}_{q}'].fillna(0).to_numpy() if f'{k}_{q}' in fc else fc[f'{k}_q50'].fillna(0).to_numpy()
    da = col('da'); hr = ctx.idx.tz_convert('Europe/Tallinn').hour; g = lambda k: C[k].reindex(hr).fillna(0).to_numpy()
    # the day's forecast any-price aFRR share rescales the hour-of-day share at our margin
    su = np.clip(g('a_up_s') * np.clip(col('afrr_up_s'), 0, 1) / np.maximum(g('a_up_s0'), 1e-3), 0, 1)
    sd = np.clip(g('a_dn_s') * np.clip(col('afrr_dn_s'), 0, 1) / np.maximum(g('a_dn_s0'), 1e-3), 0, 1)
    # haircut on the activation energy the plan leans on (tuned on Q3, qest.tune): kappa < 1 keeps the DA schedule
    # deliverable when fewer activations come than forecast (Q4: aFRR DOWN came at a third to two-thirds of plan)
    ku, kd = C.get('kappa_up', 1.0), C.get('kappa_dn', 1.0)
    return dict(da=da, da_fc=da, cap_afrr_up=col('cap_afrr_up'), cap_afrr_down=col('cap_afrr_down'), cap_mfrr_up=col('cap_mfrr_up'), cap_mfrr_down=col('cap_mfrr_down'),
                a_up_s=su * ku, a_dn_s=sd * kd, a_up_p=da + g('a_up_prem'), a_dn_p=da + g('a_dn_prem'),
                m_up_q=g('m_up_q') * ku, m_dn_q=g('m_dn_q') * kd, m_up_p=da + g('m_up_prem'), m_dn_p=da + g('m_dn_prem'))


def _renewables(units, ctx, cfg, use_real=False):
    plan = {}
    for u in units:
        if u.kind in ('solar', 'wind'):
            g = (ctx.gen_real if use_real else ctx.gen_fc)[u.name]
            if use_real:
                plan[u.name] = dict(da=np.round(np.asarray(g) * (ctx.real.nps_dayahead_eur_mwh.to_numpy() > 0), 1))
            else:
                p = ctx.fc['da_q50'].to_numpy(); s = np.asarray(g)
                plan[u.name] = dict(da=np.round(newsvendor(.8 * s, s, 1.2 * s, p, cfg.settlement, cfg.shortfall_eur, cfg.surplus_frac), 1))
    return plan


def idle(units, ctx, cfg):
    return {}


def naive_da(units, ctx, cfg):
    return {u.name: dict(da=np.round(np.asarray(ctx.gen_fc[u.name]), 1)) for u in units if u.kind in ('solar', 'wind')}


def rule(units, ctx, cfg):
    plan = naive_da(units, ctx, cfg); p = ctx.fc['da_q50'].to_numpy(); n = len(p)
    for u in units:
        if u.kind in ('battery', 'hybrid'):
            k = int(round(u.e_mwh * u.soc0 / (u.p_mw * DT)))  # MTUs to empty from the start SoC at full power
            order = np.argsort(p); da = np.zeros(n); da[order[:k]] = -u.p_mw; da[order[-k:]] = u.p_mw
            if u.kind == 'hybrid':
                da = np.clip(da + np.asarray(ctx.gen_fc[u.name]), -u.grid_mw, u.grid_mw)
            plan[u.name] = dict(da=da)
    return plan


def reserve_rule(units, ctx, cfg):
    plan = naive_da(units, ctx, cfg); fc = ctx.fc; p = fc['da_q50'].to_numpy()
    for u in units:
        if u.kind in ('battery', 'hybrid'):
            best = max(PRODUCTS, key=lambda k: fc[f'cap_{k}_q50'].mean())
            mw = np.floor(min(u.p_mw, u.e_mwh * min(u.soc0, 1 - u.soc0) / (u.reserve_s / 3600)))
            cap = {k: np.full(len(p), mw if k == best else 0.0) for k in PRODUCTS}
            plan[u.name] = dict(da=np.round(np.asarray(ctx.gen_fc.get(u.name, np.zeros(len(p)))), 1) if u.kind == 'hybrid' else np.zeros(len(p)),
                                cap=cap, margins=ctx.calib['margins'], policy='soc_aware')
    return plan


def milp(units, ctx, cfg, real=False, q='q50', products=PRODUCTS, parts=(), gate_c=None):
    plan = _renewables(units, ctx, cfg, use_real=real or 'pv' in parts); px = prices_for_milp(ctx, q, real, parts)
    if cfg.official:
        from qest.market import grid_tariff
        px['tariff'] = grid_tariff(ctx.idx)
    for u in units:
        if u.kind in ('battery', 'hybrid'):
            pv = (ctx.gen_real if (real or 'pv' in parts) else ctx.gen_fc).get(u.name) if u.kind == 'hybrid' else None
            pl = plan_storage(u, px, pv, products=products, depth=cfg.cap_depth)
            pl.update(margins=ctx.calib['margins'], soc_ref=pl['soc'], policy='static' if real else 'soc_aware', tol=ctx.calib.get('tol', .05))
            if gate_c and 'replan' in gate_c:  # qest.replan: re-solve the rest of the day at every MTU with what is public by T-25
                imb_fc = ctx.fc['imb_q50'].fillna(ctx.fc['da_q50']).fillna(0).to_numpy() if 'imb_q50' in ctx.fc else px['da']
                pl.update(policy='replan', replan=_replan_cb(u, pl, px, imb_fc, **gate_c['replan']))
            elif gate_c:  # 'lagged': gate-C bids follow the activation state L MTUs back (qest.market.settle_day)
                pl.update(policy='lagged', **gate_c)
            plan[u.name] = pl
    return plan


def oracle(units, ctx, cfg):
    return milp(units, ctx, cfg, real=True)


def _partial(*parts):  # perfect knowledge of these inputs only, Chronos-2 for the rest, gate-C policy unchanged
    return lambda units, ctx, cfg: milp(units, ctx, cfg, parts=parts)


def _stoch(lam, source='quantile', n=8, recourse='free', base='clim', sampler='exact'):
    """Two-stage stochastic day plan (qest.stoch), Mean-CVaR with weight lam, over gate-A scenarios: 'quantile' (comonotonic
    q10 / q50 / q90 + activation draws) or 'markov' (n exact joint activation patterns from qest.qscen). recourse: see
    qest.stoch. base 'act': the median scenario carries the rank-gated activation marginals (milp_act_rank's inputs)."""
    def fn(units, ctx, cfg):
        from qest.stoch import plan_storage_stoch, scenarios
        plan = _renewables(units, ctx, cfg)
        if base == 'act':
            actfc_pred(ctx.idx); um, ua = actfc_gate(ctx.idx, ctx.calib, metric='rank'); px50 = actfc_px(ctx, cfg, tuple(p for p, u in (('mfrr', um), ('afrr', ua)) if u))
        else:
            px50 = prices_for_milp(ctx)
        if cfg.official:
            from qest.market import grid_tariff
            px50['tariff'] = grid_tariff(ctx.idx)
        if 'imb_q50' in ctx.fc:
            px50['imb'] = ctx.fc['imb_q50'].fillna(px50['da'].mean()).to_numpy()
        if source == 'markov':
            from qest.qscen import day_scenarios
            S, W = day_scenarios(ctx, px50, n, sampler=sampler)
        else:
            S, W = scenarios(ctx, ctx.calib, px50)
        for p in S:
            for k in ('tariff', 'imb'):
                if k in px50:
                    p.setdefault(k, px50[k])
        for u in units:
            if u.kind in ('battery', 'hybrid'):
                pv = ctx.gen_fc.get(u.name) if u.kind == 'hybrid' else None
                if recourse == 'guard':
                    from qest.stoch import plan_storage_guard
                    pl = plan_storage_guard(u, S, W, pv=pv, depth=cfg.cap_depth, lam=lam, tol=ctx.calib.get('tol', .05))
                else:
                    pl = plan_storage_stoch(u, S, W, pv, depth=cfg.cap_depth, lam=lam, recourse=recourse, tol=ctx.calib.get('tol', .05))
                pl.update(margins=ctx.calib['margins'], soc_ref=pl['soc'], policy='soc_aware', tol=ctx.calib.get('tol', .05)); plan[u.name] = pl
        return plan
    return fn


def _markov_marginals(units, ctx, cfg):
    """The deterministic plan with the Markov model's per-MTU activation probabilities (qest.qscen, 200 exact samples)
    in place of the hour-of-day calibration table: a better mean, no scenarios."""
    from qest import qscen
    P = qscen.params(ctx.idx[0].tz_convert('Europe/Tallinn').date()); Y = qscen.sample(P, qscen.features(ctx.fc, ctx.idx), 200, seed=0)
    pu, pdn = Y[:, :, 0].mean(0), Y[:, :, 1].mean(0); plan = _renewables(units, ctx, cfg); px = prices_for_milp(ctx)
    px['m_up_q'], px['m_dn_q'] = pu * ctx.calib.get('kappa_up', 1.0), pdn * ctx.calib.get('kappa_dn', 1.0)
    if cfg.official:
        from qest.market import grid_tariff
        px['tariff'] = grid_tariff(ctx.idx)
    for u in units:
        if u.kind in ('battery', 'hybrid'):
            pv = ctx.gen_fc.get(u.name) if u.kind == 'hybrid' else None
            pl = plan_storage(u, px, pv, depth=cfg.cap_depth); pl.update(margins=ctx.calib['margins'], soc_ref=pl['soc'], policy='soc_aware', tol=ctx.calib.get('tol', .05)); plan[u.name] = pl
    return plan



_ACT = {}


def actfc_pred(idx):
    """Gate-A activation forecasts (qest.actfc) for the MTUs idx, from results/actfc_<split>.parquet; None when absent."""
    if 'df' not in _ACT:
        from qest.paths import RESULTS
        fs = sorted(RESULTS.glob('actfc_*.parquet')); _ACT['df'] = pd.concat([pd.read_parquet(f) for f in fs]).sort_index() if fs else None
        if _ACT['df'] is not None:
            _ACT['df'] = _ACT['df'][~_ACT['df'].index.duplicated()]
    A = _ACT['df']
    if A is None:
        return None
    a = A.reindex(idx); return None if a.isna().all().all() else a


def actfc_px(ctx, cfg, parts=('mfrr', 'afrr')):
    """plan_storage inputs with the driver-based activation forecasts (qest.actfc) in place of the hour-of-day climatology
    (mFRR UP / DOWN probability) and Chronos-2's zero-shot share (aFRR), for the products in `parts`."""
    A = actfc_pred(ctx.idx); px = prices_for_milp(ctx)
    if A is not None:
        from qest.market import FAT
        C = ctx.calib; hr = ctx.idx.tz_convert('Europe/Tallinn').hour; g = lambda k: C[k].reindex(hr).fillna(0).to_numpy()
        ku, kd = C.get('kappa_up', 1.0), C.get('kappa_dn', 1.0); fill = lambda k, d: A[k].fillna(pd.Series(d, index=A.index)).to_numpy()
        if 'mfrr' in parts:
            px['m_up_q'] = fill('m_up', g('m_up_q')) * ku * FAT['mfrr']; px['m_dn_q'] = fill('m_dn', g('m_dn_q')) * kd * FAT['mfrr']
        if 'afrr' in parts:
          px['a_up_s'] = np.clip(g('a_up_s') * fill('a_up_s', g('a_up_s0')) / np.maximum(g('a_up_s0'), 1e-3), 0, 1) * ku * FAT['afrr']
          px['a_dn_s'] = np.clip(g('a_dn_s') * fill('a_dn_s', g('a_dn_s0')) / np.maximum(g('a_dn_s0'), 1e-3), 0, 1) * kd * FAT['afrr']
    return px


def _actfc_marginals(units, ctx, cfg, parts=('mfrr', 'afrr')):
    """The deterministic plan on actfc_px: same MILP, better marginals."""
    plan = _renewables(units, ctx, cfg); px = actfc_px(ctx, cfg, parts)
    if cfg.official:
        from qest.market import grid_tariff
        px['tariff'] = grid_tariff(ctx.idx)
    for u in units:
        if u.kind in ('battery', 'hybrid'):
            pv = ctx.gen_fc.get(u.name) if u.kind == 'hybrid' else None
            pl = plan_storage(u, px, pv, depth=cfg.cap_depth); pl.update(margins=ctx.calib['margins'], soc_ref=pl['soc'], policy='soc_aware', tol=ctx.calib.get('tol', .05)); plan[u.name] = pl
    return plan


_GATE = {}


def actfc_gate(idx, calib, window=30, metric='brier'):
    """Walk-forward model selection, as a trading system does it: use the activation model's marginals for a product only
    when its trailing score over the last `window` days with public outcomes (D-2 lag) beats the calibration table's.
    metric 'brier' = calibration (mean squared error); 'rank' = Spearman correlation with the outcome, which is what the
    plan profits from (which hours are likelier). Returns (use_mfrr, use_afrr); past predictions and outcomes only."""
    from qest.actfc import targets
    from qest.data import frame
    day0 = idx[0].tz_convert('Europe/Tallinn').normalize().tz_localize(None); key = (str(day0.date()), id(calib), metric)
    if key in _GATE:
        return _GATE[key]
    A = _ACT.get('df'); f = frame()
    if A is None:
        return (False, False)
    if 'Y' not in _GATE:
        _GATE['Y'] = targets(f); _GATE['hr'] = f.local.dt.hour; _GATE['day'] = f.day
    Y, hr, dd = _GATE['Y'], _GATE['hr'], _GATE['day']
    lo, hi = day0 - pd.Timedelta(days=window + 2), day0 - pd.Timedelta(days=2)  # outcomes public at gate A: up to D-2
    m = (dd >= lo) & (dd < hi); m = m & A.reindex(f.index).m_up.notna().reindex(f.index).fillna(False).to_numpy()
    if m.sum() < 96 * 10:
        _GATE[key] = (False, False); return _GATE[key]
    Ai = A.reindex(f.index[m]); h = hr[m]; g = lambda k: calib[k].reindex(h).fillna(0).to_numpy()
    def better(k, ref):
        y = Y[k][m].to_numpy(); ok = np.isfinite(y); p = Ai[k].to_numpy()[ok]; y = y[ok]; r = ref[ok]
        if metric == 'rank':
            from scipy.stats import spearmanr
            return float(spearmanr(p, y).statistic) > float(spearmanr(r, y).statistic)
        return float(np.mean((p - y) ** 2)) < float(np.mean((r - y) ** 2))
    use_m = better('m_up', g('m_up_q')) and better('m_dn', g('m_dn_q'))
    use_a = better('a_up_s', g('a_up_s0')) and better('a_dn_s', g('a_dn_s0'))
    _GATE[key] = (use_m, use_a); return _GATE[key]


def _actfc_gated(units, ctx, cfg, metric='brier'):
    actfc_pred(ctx.idx); um, ua = actfc_gate(ctx.idx, ctx.calib, metric=metric)
    parts = tuple(p for p, u in (('mfrr', um), ('afrr', ua)) if u)
    return _actfc_marginals(units, ctx, cfg, parts) if parts else _actfc_marginals(units, ctx, cfg, ())

STOCH = dict(milp_act=_actfc_marginals, milp_act_gated=_actfc_gated, milp_act_rank=lambda u, c, g: _actfc_gated(u, c, g, 'rank'), milp_act_afrr=lambda u, c, g: _actfc_marginals(u, c, g, ('afrr',)), milp_act_mfrr=lambda u, c, g: _actfc_marginals(u, c, g, ('mfrr',)), milp_mk=_markov_marginals, milp_stoch=_stoch(0.0), milp_cvar=_stoch(0.3), milp_cvar6=_stoch(0.6), milp_scen=_stoch(0.0, 'markov'), milp_scen_cvar=_stoch(0.3, 'markov'),
             milp_scen16=_stoch(0.0, 'markov', 16),
             # settlement-faithful recourse (qest.stoch), scenarios around the rank-gated marginals
             milp_scenf=_stoch(0.0, 'markov', 8, 'faithful', 'act'), milp_scenf_cvar=_stoch(0.3, 'markov', 8, 'faithful', 'act'),
             milp_scenf_q=_stoch(0.0, 'quantile', 8, 'faithful', 'act'),
             milp_scenf_clim=_stoch(0.0, 'markov', 8, 'faithful', 'clim'),
             # ... and with the settlement's SoC guard inside the recourse
             milp_sceng=_stoch(0.0, 'markov', 8, 'guard', 'act'), milp_sceng_cvar=_stoch(0.3, 'markov', 8, 'guard', 'act'),
             milp_sceng16=_stoch(0.0, 'markov', 16, 'guard', 'act'), milp_sceng4=_stoch(0.0, 'markov', 4, 'guard', 'act'),
             # the same plan on scenarios from the approximate samplers (qest.qscen_q): quantum-enhanced and single-flip MCMC
             milp_sceng_qs=_stoch(0.0, 'markov', 8, 'guard', 'act', 'qmcmc'), milp_sceng_ls=_stoch(0.0, 'markov', 8, 'guard', 'act', 'lmcmc'),
             milp_scenf_qs=_stoch(0.0, 'markov', 8, 'faithful', 'act', 'qmcmc'), milp_scenf_ls=_stoch(0.0, 'markov', 8, 'faithful', 'act', 'lmcmc'),
             milp_scenf16=_stoch(0.0, 'markov', 16, 'faithful', 'act'), milp_scenf4=_stoch(0.0, 'markov', 4, 'faithful', 'act'))


def _replan_cb(u, pl, px, imb_fc, lag_act=3, lag_imb=4, allow_dev=False):
    from qest import replan
    T = replan.tables(); da_bat = np.asarray(pl['da'], float) - (np.asarray(pl['pv'], float) if 'pv' in pl else 0.0); cap = pl['cap']

    def cb(t, soc, day):  # only indices <= t - lag are read: the information public at the T-25 gate of MTU t
        ka, ki = t - lag_act, t - lag_imb
        if ka < 0:
            known = dict(state_up=None, state_dn=None, su=0., sd=0., prem=0., k_act=lag_act, k_imb=lag_imb)
        else:
            known = dict(state_up=bool(day['mfrr'][ka] >= day['da'][ka] + replan.MARGIN), state_dn=bool(day['mfrr'][ka] <= day['da'][ka] - replan.MARGIN),
                         su=float(day['up_s'][ka]), sd=float(day['dn_s'][ka]), prem=float(day['imb'][ki] - day['da'][ki]) if ki >= 0 else 0., k_act=lag_act, k_imb=lag_imb)
        return replan.solve(u, t, soc, da_bat, cap, px, imb_fc, known, T, allow_dev)
    return cb


def _rp(**kw):
    return lambda units, ctx, cfg: milp(units, ctx, cfg, gate_c=dict(replan=kw))


REPLAN = dict(milp_rp=_rp(), milp_rpdev=_rp(allow_dev=True), milp_rp1=_rp(lag_act=1, lag_imb=1), milp_rp0=_rp(lag_act=0, lag_imb=0),
              milp_rp0dev=_rp(lag_act=0, lag_imb=0, allow_dev=True))


def _lagged(lag, margin=10.0):
    return lambda units, ctx, cfg: milp(units, ctx, cfg, gate_c=dict(lag=lag, lag_margin=margin))


LAGGED = dict(milp_c1=_lagged(1), milp_c2=_lagged(2), milp_c3=_lagged(3), milp_c4=_lagged(4), milp_c2m=_lagged(2, 25.0))
ORACLE_PARTS = dict(oracle_da=('da',), oracle_cap=('cap',), oracle_afrr=('afrr',), oracle_mfrr=('mfrr',), oracle_act=('afrr', 'mfrr'),
                    oracle_pv=('pv',), oracle_prices=('da', 'cap'), oracle_all=('da', 'cap', 'afrr', 'mfrr', 'pv'))


def milp_da(units, ctx, cfg):
    """The same MILP without reserves: day-ahead arbitrage only (does taking part in the capacity auction pay?)."""
    return milp(units, ctx, cfg, products=())


def _hybrid(mode):  # qest.qday: a quantum sampler picks the day's charge / discharge pattern, the MILP completes it
    def fn(units, ctx, cfg):
        from qest import qday
        return qday.plan(mode, units, ctx, cfg)
    return fn


STRATEGIES = dict(idle=idle, naive_da=naive_da, rule=rule, reserve_rule=reserve_rule, milp_da=milp_da, milp=milp, milp_persist=milp,
                  milp_chronos=milp, oracle=oracle, milp_chronos_qrc=milp, milp_chronos_rff=milp, milp_tabpfn=milp, milp_chronosx=milp,
                  qday_qaoa=_hybrid('qaoa'), qday_random=_hybrid('random'), qday_surrogate=_hybrid('surrogate'),
                  **{k: _partial(*v) for k, v in ORACLE_PARTS.items()}, **LAGGED, **REPLAN, **STOCH)

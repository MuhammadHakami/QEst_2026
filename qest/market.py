"""Market rules, portfolio and settlement for one Tallinn delivery day (from the KratTrade kick-off slides + starter notebook).

The order ladder (Tallinn time), earliest first:
  A. D-1 09:00  reserve capacity auction (aFRR / mFRR, UP / DOWN, whole MW, >= 1 MW): won MW earn the published capacity
                price (EUR/MW/h): the money for taking part. Capacity is an obligation: every won MTU and direction needs an
                energy bid >= the won MW, backed by power and state of charge (it cannot be used in day-ahead).
  B. D-1 13:00  day-ahead (Nord Pool SDAC, one price per 15-min MTU, 0.1 MW steps): a binding position for all 96 MTUs.
  C. T-25       energy bids for the won capacity (price free, volume >= won MW): mFRR UP activates if the 15-min price >=
                the bid, DOWN if <= (one price settles both); aFRR in the PICASSO merit order every 4 s (a bid is
                activated on the ticks whose cross-border price crosses it). Bids sit at DA +- a margin; the
                'soc_aware' policy prices a direction out when the battery drifts from its planned SoC.
  R. real time  whatever was not sold in advance is settled as imbalance: the 'too late' market.
Earlier orders carry more forecast risk (the D-1 forecast vs what happened) but lock a price; energy left to real time
meets a saturated market (in sunny or windy MTUs the imbalance price for a long position collapses, often below zero).
Our own volume moves prices too (DA_SLOPE, CAP_DEPTH: the merit-order 'saturation' of the slides' bid curves).

Settlement modes:
  'official' (the organisers' deck v3, slides 10 + 12, what the jury scores): every capacity bid accepted at the
             published price (no saturation, nothing forfeited), in-the-money mFRR bids fully activated at the mFRR price,
             aFRR per 4-s PICASSO tick, deviations (metered - committed) at the imbalance price, no extra penalty,
             100 % round-trip efficiency, and EUR 30/MWh on every imported MWh Mon-Fri 07-22 Tallinn, EUR 20 otherwise.
  'baltic'   (deviations at the published single imbalance price, capacity saturation and forfeiture, Elering's tariff).
  'starter'  (the starter notebook's toy rule: surplus at 50 % of the day-ahead price, shortfall 150 EUR/MWh).
Grid tariff for 'baltic'/'starter' (Elering price list from 5 May 2026, decision 7-3/2026-025): 14.50 EUR/MWh at 330 kV,
23.67 at 110 kV, plus the 3.73 EUR/MWh balancing capacity fee; storage pays them only on its net monthly consumption
(Electricity Market Act § 71 s. 10^1, from 1 Jan 2026).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Estonian public holidays 2026 (Riigipühade ja tähtpäevade seadus): night tariff all day (slide 10)
EE_HOLIDAYS_2026 = pd.to_datetime(['2026-01-01', '2026-02-24', '2026-04-03', '2026-04-05', '2026-05-01', '2026-05-24',
                                   '2026-06-23', '2026-06-24', '2026-08-20', '2026-12-24', '2026-12-25', '2026-12-26']).date


def grid_tariff(idx):
    """Slide 10: EUR/MWh on every imported MWh per MTU: 30 Mon-Fri 07:00-22:00 Tallinn, 20 at night, weekends, holidays."""
    loc = pd.DatetimeIndex(idx).tz_convert('Europe/Tallinn')
    day = (loc.dayofweek < 5) & (loc.hour >= 7) & (loc.hour < 22) & ~np.isin(loc.date, EE_HOLIDAYS_2026)
    return np.where(day, 30.0, 20.0)

DT = .25  # h per MTU
PRODUCTS = ('afrr_up', 'afrr_down', 'mfrr_up', 'mfrr_down')
# Full-activation-time ramps (Baltic standard products: mFRR FAT 12.5 min, scheduled activation ordered 7.5 min before the MTU;
# aFRR FAT 5 min): the energy an activation delivers inside its quarter-hour. Opt-in realism (QEST_FAT=1): the organisers'
# official scoring has no ramp, so the default keeps their numbers. mFRR: ramp 60 % -> 100 % over the first 5 min then full = .933.
import os
FAT = {'mfrr': .933, 'afrr': .97} if os.environ.get('QEST_FAT') else {'mfrr': 1.0, 'afrr': 1.0}
CAP_COL = dict(afrr_up='cap_afrr_up_eur_mw_h', afrr_down='cap_afrr_down_eur_mw_h', mfrr_up='cap_mfrr_up_eur_mw_h', mfrr_down='cap_mfrr_down_eur_mw_h')


@dataclass
class Unit:
    name: str
    kind: str                 # 'battery' | 'solar' | 'wind' | 'hybrid'
    p_mw: float = 0.0         # battery power (MW)
    e_mwh: float = 0.0        # battery energy (MWh)
    gen_mw: float = 0.0       # PV / wind nameplate (MW)
    grid_mw: float = 1e9      # connection limit (MW, both directions)
    eta_c: float = .95
    eta_d: float = .95
    soc0: float = .5          # start of every day (slide 11: batteries start at 50 %)
    tariff: float = 0.0       # EUR/MWh on net monthly import
    reserve_s: float = 900.0  # reserve must be sustainable this long (s): one MTU


@dataclass
class Config:
    settlement: str = 'official'    # 'official' | 'baltic' | 'starter'
    shortfall_eur: float = 150.0    # starter: shortfall cost; baltic: extra penalty on undelivered activated reserve
    surplus_frac: float = .5        # starter: surplus value as a share of the DA price
    da_slope: float = 0.0           # EUR/MWh per MW of our net DA volume (merit-order saturation; qest.saturation)
    cap_depth: dict = field(default_factory=lambda: dict(afrr_up=110.0, afrr_down=110.0, mfrr_up=604.0, mfrr_down=691.0))
    # Baltic need in 2026: aFRR UP 96-120 MW (Elering), mFRR UP 604 / DOWN 691 MW (AST FRR dimensioning forecast 2026-2035)
    cycle_cost: float = 0.0         # EUR per MWh discharged (degradation; starter: none)
    leftover_value: str = 'da_mean'  # end-of-day battery energy valued at the day's average DA price (slide 11)

    def __post_init__(self):
        if self.settlement == 'official':  # slide 12: every capacity bid counts as accepted at the published price
            self.cap_depth = None

    @property
    def official(self):
        return self.settlement == 'official'


def portfolio(names=('battery', 'hybrid'), official=True):
    """Slide 10/11's two sites (battery 100 MW / 200 MWh; hybrid PV 20 MW + battery 20 MW / 40 MWh on one 20 MW connection)
    plus the real solar park and a real wind park. official: 100 % round-trip efficiency, per-MTU tariff (grid_tariff)
    instead of Elering's monthly net-import tariff."""
    eta, tb, th = (1.0, 0.0, 0.0) if official else (.95, 14.50 + 3.73, 23.67 + 3.73)
    U = dict(battery=Unit('battery', 'battery', p_mw=100, e_mwh=200, tariff=tb, eta_c=eta, eta_d=eta),
             hybrid=Unit('hybrid', 'hybrid', p_mw=20, e_mwh=40, gen_mw=20, grid_mw=20, tariff=th, eta_c=eta, eta_d=eta),
             solar=Unit('solar', 'solar', gen_mw=42.3),
             wind=Unit('wind', 'wind', gen_mw=45.0))
    return [U[n] for n in names]


def cap_price_eff(p, q, depth):
    """Pay-as-cleared capacity price after our q MW joins a market of `depth` MW: the clearing price falls linearly,
    to zero at 2x depth (assumption; the auction's volumes are not in the dataset)."""
    return p * np.clip(1 - q / (2 * depth), 0, 1)


def _margin_cols(table, m):
    """Columns of the aFRR margin table for the margin grid point nearest m (inf -> never activated)."""
    from qest.data import MARGINS
    if not np.isfinite(m):
        return None
    k = min(MARGINS, key=lambda x: abs(x - m)); return k


_VA = {}


def activated_volumes(idx):
    """Activated MW per MTU and direction from the Baltic Transparency Dashboard (qest.fetch_btd), aligned to idx, or None
    while that data is not in data/raw/btd (then activation is uncapped, as the dataset alone allows)."""
    import os
    if os.environ.get('QEST_ACT_CAP'):  # sensitivity runs: a uniform cap in MW on every activation (what-if, until the real volumes are in)
        c = float(os.environ['QEST_ACT_CAP']); return {k: np.full(len(idx), c) for k in ('mfrr_up', 'mfrr_down', 'afrr_up', 'afrr_down')}
    if 'tab' not in _VA:
        from qest.fetch_btd import activated_mw
        m, a = activated_mw('normal_activations_mfrr'), activated_mw('activations_afrr'); _VA['tab'] = None if m is None and a is None else (m, a)
    if _VA['tab'] is None:
        return None
    m, a = _VA['tab']; big = np.full(len(idx), np.inf); out = {}
    for k, tab in (('mfrr', m), ('afrr', a)):
        for d in ('up', 'down'):
            out[f'{k}_{d}'] = tab[d].reindex(idx).to_numpy() if tab is not None and d in tab else big
    return out


def settle_day(units, plan, day, gen, cfg=Config(), afrr_tab=None):
    """Settle one delivery day.
    day: the day's MTU rows (qest.data frame). gen: {unit: actual available MW per MTU} (PV, wind, hybrid PV).
    afrr_tab: the day's rows of qest.data.afrr_margin_table (PICASSO activation per bid margin).
    plan: {unit: dict(da=MW per MTU (+ sell / - buy), cap={product: MW per MTU},
                      margins=dict(afrr_up, afrr_down, mfrr_up, mfrr_down) EUR/MWh beyond the DA price where energy bids sit,
                      soc_ref=planned SoC per MTU (MWh), policy='soc_aware'|'static', pv=planned hybrid PV use)}.
    Gate C (T-25) for each MTU, in order: the energy bids sit at DA +- margin; the 'soc_aware' policy prices a
    direction out (bid at the market cap) when the battery's SoC has drifted from its plan by more than 5 % of its energy
    the wrong way, or cannot back a full MTU's activation: the capacity obligation is kept (the volume is bid), the price
    makes activation unlikely. Activation: mFRR when the 15-min price crosses the bid; aFRR on the 4-s ticks whose
    PICASSO price crosses it (full activation of an in-the-money bid, at the tick's price).
    Returns {unit: dict of EUR components and physical totals}."""
    n = len(day); da_p = day.nps_dayahead_eur_mwh.to_numpy(); imb = day.imbalance_eur_mwh.fillna(day.nps_dayahead_eur_mwh).to_numpy()
    mfrr = day.mfrr_up_eur_mwh.fillna(day.nps_dayahead_eur_mwh).to_numpy()
    if afrr_tab is None:
        from qest.data import afrr_margin_table
        afrr_tab = afrr_margin_table().reindex(day.index).fillna(0)
    T = {c: afrr_tab[c].to_numpy() for c in afrr_tab}; VA = activated_volumes(day.index)
    pays = (imb > 0) if cfg.settlement != 'starter' else (da_p > 0)  # unplanned surplus is worth delivering (else curtail)
    tar = grid_tariff(day.index) if cfg.official else np.zeros(n)
    out = {}
    for u in units:
        P = plan.get(u.name, {}); z = np.zeros(n)
        da = np.asarray(P.get('da', z), float); cap = {k: np.asarray(P.get('cap', {}).get(k, z), float) for k in PRODUCTS}
        M = {**dict(afrr_up=25.0, afrr_down=25.0, mfrr_up=25.0, mfrr_down=25.0), **P.get('margins', {})}
        soc_ref = P.get('soc_ref'); policy = P.get('policy', 'static')
        r = dict(cap=0.0, da=0.0, mfrr=0.0, afrr=0.0, imbalance=0.0, penalty=0.0, leftover=0.0, cycle=0.0, tariff_mtu=0.0,
                 sold_mwh=0.0, bought_mwh=0.0, imb_mwh=0.0, undelivered_mwh=0.0, curtailed_mwh=0.0, import_mwh=0.0, export_mwh=0.0,
                 act_up_mwh=0.0, act_dn_mwh=0.0, priced_out=0)
        cp = lambda k: day[CAP_COL[k]].fillna(0).to_numpy()
        cap_pay = {k: cap[k] * (cp(k) if cfg.cap_depth is None else cap_price_eff(cp(k), cap[k], cfg.cap_depth[k])) * DT for k in PRODUCTS}
        r['da'] = float((da * (da_p - cfg.da_slope * da) * DT).sum())
        r['sold_mwh'] = float(da.clip(0).sum() * DT); r['bought_mwh'] = float((-da).clip(0).sum() * DT)
        avail = np.asarray(gen.get(u.name, z), float) if u.kind != 'battery' else z
        pv_plan = np.asarray(P['pv'], float) if 'pv' in P else None
        soc = u.soc0 * u.e_mwh if u.e_mwh else 0.0
        deliv, curt, target, soc_tr = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        act_up, act_dn = np.zeros(n), np.zeros(n); e_m, e_a = 0.0, 0.0
        for t in range(n):
            mu = dict(M); rp = None
            if u.e_mwh and policy in ('soc_aware', 'lagged', 'replan'):
                ru, rd = cap['afrr_up'][t] + cap['mfrr_up'][t], cap['afrr_down'][t] + cap['mfrr_down'][t]
                drift = soc - (soc_ref[t - 1] if (soc_ref is not None and t) else u.soc0 * u.e_mwh); tol = P.get('tol', .05) * u.e_mwh
                if drift < -tol or soc - ru * DT / u.eta_d < 0:
                    mu['afrr_up'] = mu['mfrr_up'] = np.inf; r['priced_out'] += 1
                if drift > tol or soc + rd * DT * u.eta_c > u.e_mwh:
                    mu['afrr_down'] = mu['mfrr_down'] = np.inf; r['priced_out'] += 1
            if policy == 'lagged' and t >= P.get('lag', 2):  # gate C sees the activation state L MTUs back (persistence forecast):
                s_ = t - P.get('lag', 2); ka = _margin_cols(T, M['afrr_up']); kb = _margin_cols(T, M['afrr_down'])  # a direction the TSO was
                for k, on in (('mfrr_up', mfrr[s_] >= da_p[s_] + M['mfrr_up']), ('mfrr_down', mfrr[s_] <= da_p[s_] - M['mfrr_down']),  # activating stays in
                              ('afrr_up', ka is not None and T[f'up_s_{ka}'][s_] > 0), ('afrr_down', kb is not None and T[f'dn_s_{kb}'][s_] > 0)):  # the money
                    if np.isfinite(mu[k]):  # (the SoC guard above still wins); one it was not is priced out
                        mu[k] = min(mu[k], P.get('lag_margin', 10.0)) if on else np.inf
                    if not on:
                        r['priced_out'] += 1
            if policy == 'replan':  # qest.replan: the rest of the day re-solved with what is public by T-25 (the callback applies the lags)
                ka_, kb_ = _margin_cols(T, M['afrr_up']), _margin_cols(T, M['afrr_down'])
                rp = P['replan'](t, soc, dict(mfrr=mfrr, da=da_p, imb=imb, up_s=T[f'up_s_{ka_}'] if ka_ is not None else z, dn_s=T[f'dn_s_{kb_}'] if kb_ is not None else z))
                if not rp['in_up']:
                    mu['afrr_up'] = mu['mfrr_up'] = np.inf; r['priced_out'] += 1
                if not rp['in_dn']:
                    mu['afrr_down'] = mu['mfrr_down'] = np.inf; r['priced_out'] += 1
            mup = cap['mfrr_up'][t] if mfrr[t] >= da_p[t] + mu['mfrr_up'] else 0.0
            mdn = cap['mfrr_down'][t] if mfrr[t] <= da_p[t] - mu['mfrr_down'] else 0.0
            if VA is not None:  # the TSO activated only so many MW in this MTU (Baltic dashboard, qest.fetch_btd): we get at most that
                mup, mdn = min(mup, VA['mfrr_up'][t]), min(mdn, VA['mfrr_down'][t])
            ku, kd = _margin_cols(T, mu['afrr_up']), _margin_cols(T, mu['afrr_down'])
            su, pu = (T[f'up_s_{ku}'][t], T[f'up_p_{ku}'][t]) if ku is not None else (0.0, 0.0)
            sd, pdn = (T[f'dn_s_{kd}'][t], T[f'dn_p_{kd}'][t]) if kd is not None else (0.0, 0.0)
            aup, adn = cap['afrr_up'][t] * su, cap['afrr_down'][t] * sd
            if VA is not None:
                aup, adn = min(aup, VA['afrr_up'][t]), min(adn, VA['afrr_down'][t])
            mup, mdn, aup, adn = mup * FAT['mfrr'], mdn * FAT['mfrr'], aup * FAT['afrr'], adn * FAT['afrr']  # MW-equivalent delivered over the MTU
            act_up[t], act_dn[t] = mup + aup, mdn + adn
            e_m += (mup - mdn) * mfrr[t] * DT; e_a += (aup * pu - adn * pdn) * DT
            tg = da[t] + mup + aup - mdn - adn; target[t] = tg; g = avail[t]
            if u.kind in ('solar', 'wind'):
                p = min(tg, g); spill = g - max(p, 0)
                if spill > 1e-9 and pays[t] and act_dn[t] == 0:  # unsold output spills into imbalance when it pays
                    p += spill
                else:
                    curt[t] = max(spill, 0)
                deliv[t] = p
                continue
            # battery or hybrid: the battery follows its planned net power (target minus the PV the plan uses) and also
            # covers any PV shortfall; PV above the plan spills to imbalance when it pays (never against a DOWN activation)
            want = float(np.clip(tg + (rp['dev'] if rp else 0.0), -u.grid_mw, u.grid_mw)); pv_pl = min(pv_plan[t], g) if pv_plan is not None else g
            need = want - pv_pl
            pmax_d = min(u.p_mw, soc * u.eta_d / DT); pmax_c = min(u.p_mw, (u.e_mwh - soc) / (u.eta_c * DT))
            b = float(np.clip(need, -pmax_c, pmax_d))
            soc += (-b * u.eta_c * DT) if b < 0 else (-b / u.eta_d * DT)
            p = pv_pl + b; extra = g - pv_pl
            if extra > 1e-9:
                if pays[t] and act_dn[t] == 0:
                    p += extra
                else:
                    curt[t] += extra
            if p > u.grid_mw:
                curt[t] += p - u.grid_mw; p = u.grid_mw
            r['cycle'] -= cfg.cycle_cost * max(b, 0) * DT
            r['import_mwh'] += max(-p, 0) * DT; r['export_mwh'] += max(p, 0) * DT; r['tariff_mtu'] -= tar[t] * max(-p, 0) * DT
            deliv[t] = p; soc_tr[t] = soc
        dev = deliv - target  # + long, - short (MW)
        und = np.where(act_up > 0, np.minimum((-dev).clip(0), act_up), 0) + np.where(act_dn > 0, np.minimum(dev.clip(0), act_dn), 0)
        for k in (() if cfg.official else PRODUCTS):  # baltic/starter: capacity is forfeited on MTUs whose activation was not delivered
            hit = (und > 1e-6) & (cap[k] > 0) & ((act_up > 0) if k.endswith('up') else (act_dn > 0))
            cap_pay[k] = np.where(hit, 0, cap_pay[k])
        r['cap'] = float(sum(v.sum() for v in cap_pay.values())); r['mfrr'] = float(e_m); r['afrr'] = float(e_a)
        if cfg.settlement in ('baltic', 'official'):  # official (slide 12): (metered - committed) x imbalance price, nothing else
            r['imbalance'] = float((dev * imb * DT).sum())
            r['penalty'] = 0.0 if cfg.official else -float((und * cfg.shortfall_eur * DT).sum())
        else:  # starter: surplus at a share of DA, shortfall at a fixed cost
            r['imbalance'] = float((dev.clip(0) * cfg.surplus_frac * da_p * DT).sum() - ((-dev).clip(0) * cfg.shortfall_eur * DT).sum())
        r['imb_mwh'] = float(np.abs(dev).sum() * DT); r['undelivered_mwh'] = float(und.sum() * DT); r['curtailed_mwh'] = float(curt.sum() * DT)
        r['act_up_mwh'] = float(act_up.sum() * DT); r['act_dn_mwh'] = float(act_dn.sum() * DT)
        if u.e_mwh:
            r['leftover'] = float((soc - u.soc0 * u.e_mwh) * np.nanmean(da_p))  # slide 11: leftover energy at the day's mean DA price
        r['total'] = sum(r[k] for k in ('cap', 'da', 'mfrr', 'afrr', 'imbalance', 'penalty', 'leftover', 'cycle', 'tariff_mtu'))
        r['soc_end'] = soc
        r['trace'] = dict(deliv=deliv, target=target, act_up=act_up, act_dn=act_dn, soc=soc_tr)  # per MTU (one-day views)
        out[u.name] = r
    return out


def tariff(month_import, month_export, unit):
    """Monthly grid tariff: storage pays on net monthly consumption (import - export, >= 0)."""
    return -unit.tariff * max(month_import - month_export, 0.0)

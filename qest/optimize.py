"""One delivery day's plan for a battery or hybrid site as a MILP (HiGHS via scipy.optimize.milp).

Per MTU t: charge c, discharge d, PV used pv (<= its forecast), SoC s (end of MTU), reserve capacity in whole MW
ru_a, ru_m (aFRR / mFRR UP), rd_a, rd_m (DOWN), and a binary y (charge or discharge, never both: no financial cycles).
Day-ahead position = d + pv - c (sell +, buy -).
Maximise  sum_t [ (d+pv-c) p_da + sum_k r_k p_cap,k  +  expected activation margin ] dt  +  (s_n - s_0) mean(p_da)
s.t.  SoC dynamics with the expected activation energy (aFRR: the forecast active share; mFRR: the probability the
      price crosses our bid, from the training days), power headroom per direction (d - c + ru <= P, c - d + rd <= P,
      and the connection for the hybrid), energy backing at both ends of every MTU for a full MTU's activation
      (the starter's conservative test), capacity in whole MW. SoC follows the battery's net power (DA and expected
      activations netted, as the inverter does), so planning and settlement keep the same energy books.
The same model with the realised prices and activations is the perfect-foresight oracle.
"""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from qest.market import DT, PRODUCTS

Z = {'ru_a': 'z_ua', 'ru_m': 'z_um', 'rd_a': 'z_da', 'rd_m': 'z_dm'}  # saturation slack per capacity variable


def plan_storage(u, px, pv=None, products=PRODUCTS, mip_gap=1e-3, time_limit=60, depth=None, fixed=None, relax=False):
    """u: market.Unit (battery or hybrid). px: dict of arrays per MTU:
       da, cap_afrr_up, cap_afrr_down, cap_mfrr_up, cap_mfrr_down (EUR/MW/h),
       a_up_s, a_dn_s (aFRR expected active share 0-1), a_up_p, a_dn_p (aFRR price when active),
       m_up_q, m_dn_q (mFRR activation probability), m_up_p, m_dn_p (expected mFRR price when activated).
    Optional vol_up, vol_dn (MW per MTU): the TSO's activated volume caps what our offer delivers (qest.market.activated_volumes).
    pv: forecast PV MW per MTU (hybrid). fixed: {variable: array per MTU} pinned (qest.qday: the quantum sampler's
    charge / discharge pattern for y2). relax: solve the LP relaxation and return {'objective', 'grad'}, grad = d objective /
    d pinned value per fixed variable (HiGHS bound marginals: the Benders cut of qest.qbenders).
    Returns market plan dict (da, cap, bid_up, bid_down) + 'soc', 'objective'."""
    n = len(px['da']); pv = np.zeros(n) if pv is None else np.minimum(np.asarray(pv, float), u.gen_mw)
    V = ['c', 'd', 'pv', 'ru_a', 'ru_m', 'rd_a', 'rd_m', 's', 'y', 'nc', 'nd', 'y2', 'z_ua', 'z_um', 'z_da', 'z_dm', 'imp', 'g_ua', 'g_um', 'g_da', 'g_dm']; nv = len(V); idx = {v: i * n for i, v in enumerate(V)}
    caps = ('vol_up' in px) and ('vol_dn' in px)  # the TSO activates at most vol MW per MTU and direction: g_* = the MW of our offer that gets activated
    N = nv * n; ix = lambda v, t: idx[v] + t
    P, E, ec, ed, Dr = u.p_mw, u.e_mwh, u.eta_c, u.eta_d, u.reserve_s / 3600
    on = {k: (k in products) for k in PRODUCTS}
    lb, ub = np.zeros(N), np.zeros(N); integ = np.zeros(N)
    for t in range(n):
        ub[ix('c', t)] = ub[ix('d', t)] = P; ub[ix('pv', t)] = pv[t]; ub[ix('s', t)] = E; ub[ix('y', t)] = 1; integ[ix('y', t)] = 1
        ub[ix('nc', t)] = ub[ix('nd', t)] = 2 * P; ub[ix('y2', t)] = 1; integ[ix('y2', t)] = 1
        for v, k in (('ru_a', 'afrr_up'), ('ru_m', 'mfrr_up'), ('rd_a', 'afrr_down'), ('rd_m', 'mfrr_down')):
            ub[ix(v, t)] = (min(P, depth[k]) if depth else P) if on[k] else 0; integ[ix(v, t)] = 1
            ub[ix(Z[v], t)] = P
        ub[ix('imp', t)] = 2 * P
        for v in ('g_ua', 'g_um', 'g_da', 'g_dm'):
            ub[ix(v, t)] = P
    for v, val in (fixed or {}).items():
        lb[idx[v]:idx[v] + n] = ub[idx[v]:idx[v] + n] = np.asarray(val, float)
    da, mda = np.asarray(px['da'], float), float(np.nanmean(px['da']))
    a_up_s, a_dn_s = np.asarray(px['a_up_s']), np.asarray(px['a_dn_s']); m_up_q, m_dn_q = np.asarray(px['m_up_q']), np.asarray(px['m_dn_q'])
    # expected activation energy per MW of capacity (MWh) and its value
    e_ua, e_um, e_da_, e_dm = a_up_s * DT, m_up_q * DT, a_dn_s * DT, m_dn_q * DT
    cobj = np.zeros(N)
    AV = {'ru_a': 'g_ua', 'ru_m': 'g_um', 'rd_a': 'g_da', 'rd_m': 'g_dm'}; act = (lambda v: AV[v]) if caps else (lambda v: v)  # energy terms sit on the activated MW
    for t in range(n):
        cobj[ix('d', t)] = da[t] * DT; cobj[ix('pv', t)] = da[t] * DT; cobj[ix('c', t)] = -da[t] * DT
        cobj[ix('ru_a', t)] = px['cap_afrr_up'][t] * DT; cobj[ix(act('ru_a'), t)] += e_ua[t] * px['a_up_p'][t]
        cobj[ix('ru_m', t)] = px['cap_mfrr_up'][t] * DT; cobj[ix(act('ru_m'), t)] += e_um[t] * px['m_up_p'][t]
        cobj[ix('rd_a', t)] = px['cap_afrr_down'][t] * DT; cobj[ix(act('rd_a'), t)] -= e_da_[t] * px['a_dn_p'][t]
        cobj[ix('rd_m', t)] = px['cap_mfrr_down'][t] * DT; cobj[ix(act('rd_m'), t)] -= e_dm[t] * px['m_dn_p'][t]
    tar = np.asarray(px.get('tariff', np.zeros(n)), float)
    for t in range(n):
        cobj[ix('imp', t)] = -tar[t] * DT  # slide 10: every imported MWh pays the grid tariff
    cobj[ix('s', n - 1)] += mda  # leftover energy at the day's mean DA price (slide 11)
    if depth:  # capacity saturation (market.cap_price_eff): revenue p q (1 - q / 2D), as 0.75 p q up to D/2, then 0.25 p per MW to D
        for t in range(n):
            for v, z, k in (('ru_a', 'z_ua', 'afrr_up'), ('ru_m', 'z_um', 'mfrr_up'), ('rd_a', 'z_da', 'afrr_down'), ('rd_m', 'z_dm', 'mfrr_down')):
                pc = px[f'cap_{k}'][t] * DT; cobj[ix(v, t)] -= .25 * pc; cobj[ix(z, t)] = -.5 * pc
    A = lil_matrix((23 * n, N)); lo, hi = np.full(23 * n, -np.inf), np.full(23 * n, np.inf); r = 0
    s0 = u.soc0 * E
    for t in range(n):
        # the battery's net power (the inverter nets DA and expected activations): nd - nc = d - c + E[up] - E[down] (MW)
        A[r, ix('nd', t)] = 1; A[r, ix('nc', t)] = -1; A[r, ix('d', t)] = -1; A[r, ix('c', t)] = 1
        A[r, ix(act('ru_a'), t)] = -e_ua[t] / DT; A[r, ix(act('ru_m'), t)] = -e_um[t] / DT; A[r, ix(act('rd_a'), t)] = e_da_[t] / DT; A[r, ix(act('rd_m'), t)] = e_dm[t] / DT
        lo[r] = hi[r] = 0; r += 1
        if caps:  # activated MW <= offered MW, and the two products of a direction share the TSO's activated volume
            for v in ('ru_a', 'ru_m', 'rd_a', 'rd_m'):
                A[r, ix(AV[v], t)] = 1; A[r, ix(v, t)] = -1; hi[r] = 0; r += 1
            A[r, ix('g_ua', t)] = 1; A[r, ix('g_um', t)] = 1; hi[r] = px['vol_up'][t]; r += 1
            A[r, ix('g_da', t)] = 1; A[r, ix('g_dm', t)] = 1; hi[r] = px['vol_dn'][t]; r += 1
        A[r, ix('nc', t)] = 1; A[r, ix('y2', t)] = -2 * P; hi[r] = 0; r += 1
        A[r, ix('nd', t)] = 1; A[r, ix('y2', t)] = 2 * P; hi[r] = 2 * P; r += 1
        # SoC on the net flow: s_t = s_{t-1} + ec nc dt - nd dt / ed
        A[r, ix('s', t)] = 1; A[r, ix('nc', t)] = -ec * DT; A[r, ix('nd', t)] = DT / ed
        if t: A[r, ix('s', t - 1)] = -1; lo[r] = hi[r] = 0
        else: lo[r] = hi[r] = s0
        r += 1
        # the net flow must fit the battery's power (expected activations included)
        A[r, ix('nc', t)] = 1; A[r, ix('nd', t)] = 1; hi[r] = P; r += 1
        # power headroom: d - c + ru <= P ; c - d + rd <= P
        A[r, ix('d', t)] = 1; A[r, ix('c', t)] = -1; A[r, ix('ru_a', t)] = 1; A[r, ix('ru_m', t)] = 1; hi[r] = P; r += 1
        A[r, ix('c', t)] = 1; A[r, ix('d', t)] = -1; A[r, ix('rd_a', t)] = 1; A[r, ix('rd_m', t)] = 1; hi[r] = P; r += 1
        # connection (hybrid): export + up <= G ; import + down <= G
        A[r, ix('d', t)] = 1; A[r, ix('pv', t)] = 1; A[r, ix('c', t)] = -1; A[r, ix('ru_a', t)] = 1; A[r, ix('ru_m', t)] = 1; hi[r] = u.grid_mw; r += 1
        A[r, ix('c', t)] = 1; A[r, ix('d', t)] = -1; A[r, ix('pv', t)] = -1; A[r, ix('rd_a', t)] = 1; A[r, ix('rd_m', t)] = 1; hi[r] = u.grid_mw; r += 1
        # no simultaneous charge and discharge
        A[r, ix('c', t)] = 1; A[r, ix('y', t)] = -P; hi[r] = 0; r += 1
        A[r, ix('d', t)] = 1; A[r, ix('y', t)] = P; hi[r] = P; r += 1
        # grid import (battery net charge beyond PV, expected activations included): imp >= nc - nd - pv
        A[r, ix('nc', t)] = 1; A[r, ix('nd', t)] = -1; A[r, ix('pv', t)] = -1; A[r, ix('imp', t)] = -1; hi[r] = 0; r += 1
        # energy backing for a full MTU's activation at both ends of the MTU: s >= ru Dr / ed ; E - s >= rd Dr ec
        for sv, const in (('s', None), ('prev', s0)):
            if sv == 's' or t:
                col = ix('s', t) if sv == 's' else ix('s', t - 1)
                A[r, col] = 1; A[r, ix('ru_a', t)] = -Dr / ed; A[r, ix('ru_m', t)] = -Dr / ed; lo[r] = 0; r += 1
                A[r, col] = 1; A[r, ix('rd_a', t)] = Dr * ec; A[r, ix('rd_m', t)] = Dr * ec; hi[r] = E; r += 1
            else:
                A[r, ix('ru_a', t)] = Dr / ed; A[r, ix('ru_m', t)] = Dr / ed; hi[r] = const; r += 1
                A[r, ix('rd_a', t)] = Dr * ec; A[r, ix('rd_m', t)] = Dr * ec; hi[r] = E - const; r += 1
    if depth:  # z >= r - D/2
        A2 = lil_matrix((4 * n, N)); lo2, hi2 = np.full(4 * n, -np.inf), np.zeros(4 * n); r2 = 0
        for t in range(n):
            for v, z, k in (('ru_a', 'z_ua', 'afrr_up'), ('ru_m', 'z_um', 'mfrr_up'), ('rd_a', 'z_da', 'afrr_down'), ('rd_m', 'z_dm', 'mfrr_down')):
                A2[r2, ix(v, t)] = 1; A2[r2, ix(z, t)] = -1; hi2[r2] = depth[k] / 2; r2 += 1
        from scipy.sparse import vstack
        A = vstack([A[:r].tocsr(), A2.tocsr()]).tocsr(); lo, hi = np.r_[lo[:r], lo2], np.r_[hi[:r], hi2]
    else:
        A = A[:r].tocsr(); lo, hi = lo[:r], hi[:r]
    if relax:  # LP relaxation with duals: two-sided rows split into <= / ==, as linprog wants
        from scipy.optimize import linprog
        from scipy.sparse import vstack as vs
        eq = np.isfinite(lo) & np.isfinite(hi) & (lo == hi); up = np.isfinite(hi) & ~eq; dn = np.isfinite(lo) & ~eq
        r = linprog(-cobj, A_ub=vs([A[up], -A[dn]]).tocsr(), b_ub=np.r_[hi[up], -lo[dn]], A_eq=A[eq], b_eq=lo[eq],
                    bounds=np.c_[lb, ub], method='highs')
        if r.status != 0:
            raise RuntimeError(f'LP failed: {r.message}')
        m = -(r.lower.marginals + r.upper.marginals)
        return dict(objective=-r.fun, grad={v: m[idx[v]:idx[v] + n] for v in (fixed or {})})
    res = milp(-cobj, constraints=LinearConstraint(A, lo, hi), integrality=integ, bounds=Bounds(lb, ub),
               options=dict(mip_rel_gap=mip_gap, time_limit=time_limit, disp=False))
    if res.x is None:
        raise RuntimeError(f'MILP failed: {res.message}')
    x = res.x; g = lambda v: x[idx[v]:idx[v] + n]
    plan = dict(da=np.round(g('d') + g('pv') - g('c'), 1),
                cap=dict(afrr_up=np.round(g('ru_a')), mfrr_up=np.round(g('ru_m')), afrr_down=np.round(g('rd_a')), mfrr_down=np.round(g('rd_m'))),
                soc=g('s'), objective=-res.fun, pv=g('pv'))
    return plan


def newsvendor(q10, q50, q90, price, settlement='baltic', shortfall=150.0, surplus_frac=.5):
    """DA quantity for a PV/wind unit under the starter rule: sell q with P(G < q) = surplus... critical ratio.
    Marginal MWh sold ahead earns p; if it is not produced it costs `shortfall`; if kept back and produced it earns
    surplus_frac * p. Sell up to the quantile tau = (p - surplus_frac p) / (p - surplus_frac p + shortfall) of output.
    Under 'baltic' (deviation at the imbalance price, mean ~ DA) the median is the risk-neutral choice."""
    q10, q50, q90, price = map(lambda a: np.asarray(a, float), (q10, q50, q90, price))
    if settlement != 'starter':
        return np.where(price > 0, q50, 0.0)
    gain = np.maximum(price * (1 - surplus_frac), 0); tau = gain / (gain + shortfall)
    # interpolate the forecast quantiles (10/50/90 %) at tau
    out = np.where(tau <= .1, q10 * tau / .1, np.where(tau <= .5, q10 + (q50 - q10) * (tau - .1) / .4, q50 + (q90 - q50) * (tau - .5) / .4))
    return np.where(price > 0, np.maximum(out, 0), 0.0)

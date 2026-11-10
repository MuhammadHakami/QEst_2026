"""T-25 re-plan: the gate-A day plan stands (DA position, won capacity), but at every MTU the rest of the day is re-solved
with what is public by then, and the battery's gate-C choices follow the re-plan instead of a fixed rule.

What is public at the T-25 gate of MTU t (qest.research/gap_to_hindsight.md s2b): activations and prices of MTUs up to
t - L_ACT (final mFRR / aFRR results come ~15 min after an MTU ends), the imbalance price up to t - L_IMB. From the
train split we tabulate how an activation state persists k MTUs ahead (P(UP active at t+k | state at t), the aFRR share
autocorrelation, the imbalance-premium autocorrelation) and blend that with the day's gate-A forecast beyond.

Decisions per remaining MTU tau (a small MILP, HiGHS): in_up / in_dn in {0, 1}: keep the energy bid in the money or price it
out (the won MW is still offered); dev (MW, only with allow_dev): deliberately deliver more / less than committed, settled at
the expected imbalance price (official rules: (metered - committed) x imbalance price). allow_dev is a deliberate imbalance
position: legal in the scoring, frowned on by BRP rules, so it is a separate strategy, never the default.

usage: .venv/bin/python -m qest.replan --check
"""
import json
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from qest.paths import RESULTS

DT = .25; K = 16  # horizon re-planned: 16 MTUs = 4 h (beyond it the gate-A forecast alone)
L_ACT, L_IMB = 3, 4  # publication lags in MTUs (45 min, 60 min): assumptions, see the module docstring
MARGIN = 5.0  # an mFRR price this far beyond DA counts as an activation in the persistence tables


def tables(force=False):
    """Persistence of the activation state and the imbalance premium, k = 1..K MTUs ahead, from the train split. Cached."""
    p = RESULTS / 'replan_tables.json'
    if p.exists() and not force:
        return json.loads(p.read_text())
    from qest.data import frame
    f = frame(); f = f[f.split == 'train']; da = f.nps_dayahead_eur_mwh
    up = (f.mfrr_up_eur_mwh.notna() & (f.mfrr_up_eur_mwh >= da + MARGIN)).astype(float); dn = (f.mfrr_down_eur_mwh.notna() & (f.mfrr_down_eur_mwh <= da - MARGIN)).astype(float)
    su, sd = f.afrr4s_up_share.fillna(0), f.afrr4s_down_share.fillna(0); prem = (f.imbalance_eur_mwh - da).fillna(0)
    cond = lambda a, k: [float(a[(a.shift(k) == s)].mean()) for s in (0, 1)]  # P(active at t | state s at t-k)
    T = dict(p_up=[cond(up, k) for k in range(1, K + 1)], p_dn=[cond(dn, k) for k in range(1, K + 1)],
             rho_up=[float(su.autocorr(k)) for k in range(1, K + 1)], rho_dn=[float(sd.autocorr(k)) for k in range(1, K + 1)],
             rho_imb=[float(prem.autocorr(k)) for k in range(1, K + 1)], base_up=float(up.mean()), base_dn=float(dn.mean()))
    p.write_text(json.dumps(T)); return T


def expect(T, k, state_up, state_dn, su0, sd0, prem0, fc):
    """Expected activation probabilities / shares / imbalance premium k MTUs after the last known state, blended with the
    gate-A forecast fc = (m_up_q, m_dn_q, a_up_s, a_dn_s, imb_prem) for that MTU."""
    if state_up is None or k > K:
        return fc
    if k == 0:  # the state of this very MTU is known (lag 0: an information upper bound, not a public feed)
        return (float(state_up), float(state_dn), su0, sd0, prem0)
    pu, pdn = T['p_up'][k - 1][int(state_up)], T['p_dn'][k - 1][int(state_dn)]
    w_up, w_dn = max(T['rho_up'][k - 1], 0), max(T['rho_dn'][k - 1], 0); w_i = max(T['rho_imb'][k - 1], 0)
    # the conditional tables replace the forecast's marginal where they carry information (|p - base| vs the forecast's own)
    return (pu, pdn, w_up * su0 + (1 - w_up) * fc[2], w_dn * sd0 + (1 - w_dn) * fc[3], w_i * prem0 + (1 - w_i) * fc[4])


def solve(u, t, soc, da_bat, cap, px, imb_fc, known, T, allow_dev=False, horizon=K):
    """Re-plan MTUs t..t+horizon-1. da_bat: the battery's committed net MW per MTU (+ discharge); cap: {product: MW};
    px: gate-A price dict (plan_storage layout); imb_fc: forecast imbalance price per MTU; known: dict(state_up, state_dn,
    su, sd, prem, k_act, k_imb) = last public state and how many MTUs back it is. Returns in_up, in_dn, dev for MTU t."""
    n = len(da_bat); H = min(horizon, n - t); P, E, ec, ed = u.p_mw, u.e_mwh, u.eta_c, u.eta_d
    nv = 3 * H; iu, idn, idv = 0, H, 2 * H  # in_up[h], in_dn[h], dev[h]
    c = np.zeros(nv); lb, ub = np.zeros(nv), np.ones(nv); integ = np.zeros(nv); integ[:2 * H] = 1
    eu, edn, rev_u, rev_d, pimb = np.zeros(H), np.zeros(H), np.zeros(H), np.zeros(H), np.zeros(H)
    for h in range(H):
        tau = t + h; fc = (px['m_up_q'][tau], px['m_dn_q'][tau], px['a_up_s'][tau], px['a_dn_s'][tau], imb_fc[tau] - px['da'][tau])
        pu, pdn, su, sd, prem = expect(T, h + known['k_act'], known['state_up'], known['state_dn'], known['su'], known['sd'], known['prem'], fc)
        if known['k_imb'] != known['k_act']:
            prem = expect(T, h + known['k_imb'], known['state_up'], known['state_dn'], known['su'], known['sd'], known['prem'], fc)[4]
        eu[h] = (pu * cap['mfrr_up'][tau] + su * cap['afrr_up'][tau]) * DT  # MWh expected if in the money
        edn[h] = (pdn * cap['mfrr_down'][tau] + sd * cap['afrr_down'][tau]) * DT
        rev_u[h] = (pu * cap['mfrr_up'][tau] * px['m_up_p'][tau] + su * cap['afrr_up'][tau] * px['a_up_p'][tau]) * DT
        rev_d[h] = -(pdn * cap['mfrr_down'][tau] * px['m_dn_p'][tau] + sd * cap['afrr_down'][tau] * px['a_dn_p'][tau]) * DT
        pimb[h] = px['da'][tau] + prem
        c[iu + h] = -rev_u[h]; c[idn + h] = -rev_d[h]; c[idv + h] = -pimb[h] * DT  # milp minimises
        lb[idv + h], ub[idv + h] = (-P, P) if allow_dev else (0, 0)
    mda = float(np.mean(px['da'][t:]))
    # SoC after each MTU: s_h = soc - sum_{j<=h} (da_bat + dev + eu in_up - edn in_dn)_j x DT-adjusted by efficiency (discharge / ed, charge x ec)
    # linearised with one efficiency: energy out counted at 1 / ed for discharge-side terms, in at ec for charge-side terms
    A = np.zeros((2 * H, nv)); lo, hi = np.zeros(2 * H), np.zeros(2 * H); base = soc
    for h in range(H):
        tau = t + h; d0 = da_bat[tau] * DT; base -= d0 / ed if d0 > 0 else d0 * ec
        for j in range(h + 1):
            A[h, iu + j] = -eu[j] / ed; A[h, idn + j] = edn[j] * ec; A[h, idv + j] = -DT / ed  # dev > 0 = extra discharge
        # SoC must also keep backing the offered reserves (else the standing SoC guard prices them out): ru DT / ed <= s_h <= E - rd DT ec
        ru_, rd_ = cap['afrr_up'][tau] + cap['mfrr_up'][tau], cap['afrr_down'][tau] + cap['mfrr_down'][tau]
        lo[h], hi[h] = -base + ru_ * DT / ed, E - base - rd_ * DT * ec
        # power: |da_bat + dev| + offered up/down must fit P (the offered MW are fixed; only dev moves)
        A[H + h, idv + h] = 1; lo[H + h], hi[H + h] = -P + rd_ - da_bat[tau], P - ru_ - da_bat[tau]
    c[:] += 0; c_term = np.zeros(nv)  # leftover energy at the day's mean DA price: value of final SoC
    for j in range(H):
        c_term[iu + j] = -(-eu[j] / ed) * mda; c_term[idn + j] = -(edn[j] * ec) * mda; c_term[idv + j] = -(-DT / ed) * mda
    res = milp(c + c_term, constraints=LinearConstraint(A, lo, hi), integrality=integ, bounds=Bounds(lb, ub), options=dict(disp=False, time_limit=2))
    if res.x is None:  # infeasible (SoC already outside the backing band): keep only the directions the SoC can back
        ru0, rd0 = cap['afrr_up'][t] + cap['mfrr_up'][t], cap['afrr_down'][t] + cap['mfrr_down'][t]
        return dict(in_up=bool(soc >= ru0 * DT / ed), in_dn=bool(soc <= E - rd0 * DT * ec), dev=0.0)
    return dict(in_up=bool(res.x[iu] > .5), in_dn=bool(res.x[idn] > .5), dev=float(res.x[idv]))


def _check():
    """The re-plan keeps UP in the money when the TSO is activating UP at a premium and SoC is ample, prices it out when the
    battery is empty, and with allow_dev discharges into a high expected imbalance price."""
    from qest import market
    u = market.portfolio(['battery'])[0]; n = 96; T = tables()
    px = dict(da=np.full(n, 50.), m_up_q=np.full(n, .3), m_dn_q=np.full(n, .3), a_up_s=np.full(n, .1), a_dn_s=np.full(n, .1),
              m_up_p=np.full(n, 120.), m_dn_p=np.full(n, 10.), a_up_p=np.full(n, 120.), a_dn_p=np.full(n, 10.))
    cap = dict(mfrr_up=np.full(n, 50.), afrr_up=np.zeros(n), mfrr_down=np.full(n, 50.), afrr_down=np.zeros(n)); da_bat = np.zeros(n)
    known = dict(state_up=1, state_dn=0, su=0., sd=0., prem=0., k_act=L_ACT, k_imb=L_IMB)
    r = solve(u, 10, 150., da_bat, cap, px, np.full(n, 50.), known, T); assert r['in_up'] and r['dev'] == 0, r
    r = solve(u, 10, 0., da_bat, cap, px, np.full(n, 50.), known, T); assert not r['in_up'], r
    r = solve(u, 10, 150., da_bat, cap, px, np.full(n, 50.), dict(known, prem=400.), T, allow_dev=True); assert r['dev'] > 0, r
    print('replan check ok: P(UP | UP 3 MTUs ago) =', round(T['p_up'][L_ACT - 1][1], 2), 'base', round(T['base_up'], 2), '| in / out / dev as expected')


if __name__ == '__main__':
    _check()

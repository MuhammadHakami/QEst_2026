"""Two-stage stochastic day plan with a Mean-CVaR objective (after arXiv:2609.18349, IECON 2026), on top of qest.optimize.

First stage, decided at gate A for all scenarios at once: the DA position (d, c, pv), the capacity bids (ru_a, ru_m, rd_a,
rd_m) and the capacity-saturation helpers. Second stage, one copy per scenario s: the battery's net flow (nc, nd), the SoC,
the grid import and the charge / discharge indicator, driven by that scenario's activation probabilities, shares and prices.
Objective: (1 - lam) E[profit] + lam CVaR_alpha[profit], CVaR by Rockafellar-Uryasev (eta, zeta_s >= eta - profit_s).

Scenarios come from the gate-A quantile forecasts: comonotonic q10 / q50 / q90 draws of every series (weights .25 / .5 / .25)
plus two activation-only draws around the median (shares x .5 and x 1.5): five scenarios, no new data. A KDE of joint
history, as the paper does, is the next step once the Baltic activation volumes are in (qest.fetch_btd).

Recourse per scenario: the activation energy actually delivered is a variable bounded by what the scenario activates of the
offered MW, so a long run of UP activations does not force the plan to withdraw its offers; it prices out what the SoC cannot
back, as the gate-C guard does in settlement (first version without recourse lost 25 % on Q4). Capacity is paid on the
offer, energy on delivery; the offered MW are backed in the expected SoC, as the deterministic planner does.
With one scenario and lam = 0 this reproduces qest.optimize.plan_storage within 1 % (checked). usage: python -m qest.stoch --check
"""
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from qest.market import DT, PRODUCTS
from qest.optimize import Z

FIRST = ['c', 'd', 'pv', 'ru_a', 'ru_m', 'rd_a', 'rd_m', 'y', 'z_ua', 'z_um', 'z_da', 'z_dm']
SECOND = ['nc', 'nd', 's', 'y2', 'imp', 'au', 'am', 'bda', 'bdm', 'sh', 'lg', 'pu', 'pd']  # au.. = activation MW delivered; sh / lg = short / long deviation (MW); pu / pd = priced out (guard)


def plan_storage_stoch(u, scen, w, pv=None, products=PRODUCTS, depth=None, lam=0.0, alpha=0.9, mip_gap=4e-3, time_limit=90, recourse='free', tol=.05, po=None, fixed=None):
    """scen: list of px dicts (plan_storage layout); w: scenario weights. Returns the plan_storage dict (soc = expected).
    recourse 'free': delivered activation <= what the scenario activates of the offer (the plan may price a direction out at
    will: more permissive than settlement, where only the SoC guard does that). 'faithful': the whole activated MW is
    delivered, as settlement does with an in-the-money bid; what the battery cannot back becomes a deviation settled at the
    imbalance price (short pays max(imb, da) + 5, long earns min(imb, da) - 5: never an arbitrage, only a last resort).
    'guard': faithful, plus the settlement's gate-C guard itself: a direction is priced out (nothing delivered, no deviation)
    in a scenario exactly when that scenario's SoC has drifted from the expected SoC by more than tol x E the wrong way
    (binaries pu / pd per MTU and scenario, big-M on the drift). po = (pu, pd) 0/1 arrays (n, S): the flags pinned (the
    fixed-point route of plan_storage_guard, which is what the strategies use: the free binaries do not converge in minutes)."""
    S = len(scen); w = np.asarray(w, float) / np.sum(w); n = len(scen[0]['da']); faithful = recourse in ('faithful', 'guard'); guard = recourse == 'guard'
    pv = np.zeros(n) if pv is None else np.minimum(np.asarray(pv, float), u.gen_mw)
    P, E, ec, ed, Dr = u.p_mw, u.e_mwh, u.eta_c, u.eta_d, u.reserve_s / 3600; on = {k: (k in products) for k in PRODUCTS}
    idx = {}; N = 0
    for v in FIRST:
        idx[v] = N; N += n
    for s_ in range(S):
        for v in SECOND:
            idx[(v, s_)] = N; N += n
    i_eta = N; N += 1; i_zeta = N; N += S
    ix = lambda v, t, s_=None: (idx[v] if s_ is None else idx[(v, s_)]) + t
    lb, ub, integ = np.zeros(N), np.zeros(N), np.zeros(N); ub[i_eta] = np.inf; lb[i_eta] = -np.inf; ub[i_zeta:i_zeta + S] = np.inf
    for t in range(n):
        ub[ix('c', t)] = ub[ix('d', t)] = P; ub[ix('pv', t)] = pv[t]; ub[ix('y', t)] = 1; integ[ix('y', t)] = 1
        for v, k in (('ru_a', 'afrr_up'), ('ru_m', 'mfrr_up'), ('rd_a', 'afrr_down'), ('rd_m', 'mfrr_down')):
            ub[ix(v, t)] = (min(P, depth[k]) if depth else P) if on[k] else 0; integ[ix(v, t)] = 1; ub[ix(Z[v], t)] = P
        for s_ in range(S):
            ub[ix('nc', t, s_)] = ub[ix('nd', t, s_)] = 2 * P; ub[ix('s', t, s_)] = E; ub[ix('y2', t, s_)] = 1; integ[ix('y2', t, s_)] = 1; ub[ix('imp', t, s_)] = 2 * P
            for v in ('au', 'am', 'bda', 'bdm'):
                ub[ix(v, t, s_)] = P
            ub[ix('sh', t, s_)] = ub[ix('lg', t, s_)] = 2 * P if faithful else 0
            ub[ix('pu', t, s_)] = ub[ix('pd', t, s_)] = 1 if (guard and t) else 0; integ[ix('pu', t, s_)] = integ[ix('pd', t, s_)] = 1
            if po is not None and guard and t:
                lb[ix('pu', t, s_)] = ub[ix('pu', t, s_)] = float(po[0][t, s_]); lb[ix('pd', t, s_)] = ub[ix('pd', t, s_)] = float(po[1][t, s_])
    for v, val in (fixed or {}).items():  # first-stage variables pinned (the in-model value of a given plan)
        lb[idx[v]:idx[v] + n] = ub[idx[v]:idx[v] + n] = np.asarray(val, float)
    # profit_s as a linear form over the variables (first stage + that scenario's second stage)
    prof = np.zeros((S, N)); e = {}
    for s_, px in enumerate(scen):
        da = np.asarray(px['da'], float); mda = float(np.nanmean(da)); tar = np.asarray(px.get('tariff', np.zeros(n)), float)
        # a shortfall during an UP activation settles at (about) the activation price (single imbalance pricing): never cheaper than what was earned
        imb = np.asarray(px.get('imb', da), float)
        p_sh = np.maximum.reduce([imb, da, np.asarray(px['a_up_p'], float), np.asarray(px['m_up_p'], float)]) + 5
        p_lg = np.minimum.reduce([imb, da, np.asarray(px['a_dn_p'], float), np.asarray(px['m_dn_p'], float)]) - 5
        for t in range(n):
            prof[s_, ix('sh', t, s_)] = -p_sh[t] * DT; prof[s_, ix('lg', t, s_)] = p_lg[t] * DT
        e[s_] = dict(ua=np.asarray(px['a_up_s']) * DT, um=np.asarray(px['m_up_q']) * DT, da=np.asarray(px['a_dn_s']) * DT, dm=np.asarray(px['m_dn_q']) * DT)
        for t in range(n):
            prof[s_, ix('d', t)] = da[t] * DT; prof[s_, ix('pv', t)] = da[t] * DT; prof[s_, ix('c', t)] = -da[t] * DT
            prof[s_, ix('ru_a', t)] = px['cap_afrr_up'][t] * DT; prof[s_, ix('ru_m', t)] = px['cap_mfrr_up'][t] * DT  # capacity is paid on the offer
            prof[s_, ix('rd_a', t)] = px['cap_afrr_down'][t] * DT; prof[s_, ix('rd_m', t)] = px['cap_mfrr_down'][t] * DT
            prof[s_, ix('au', t, s_)] = px['a_up_p'][t] * DT; prof[s_, ix('am', t, s_)] = px['m_up_p'][t] * DT  # energy only on what is delivered
            prof[s_, ix('bda', t, s_)] = -px['a_dn_p'][t] * DT; prof[s_, ix('bdm', t, s_)] = -px['m_dn_p'][t] * DT
            prof[s_, ix('imp', t, s_)] = -tar[t] * DT
            if depth:
                for v, z, k in (('ru_a', 'z_ua', 'afrr_up'), ('ru_m', 'z_um', 'mfrr_up'), ('rd_a', 'z_da', 'afrr_down'), ('rd_m', 'z_dm', 'mfrr_down')):
                    pc = px[f'cap_{k}'][t] * DT; prof[s_, ix(v, t)] -= .25 * pc; prof[s_, ix(z, t)] -= .5 * pc
        prof[s_, ix('s', n - 1, s_)] += mda
    cobj = (1 - lam) * (w @ prof); cobj[i_eta] += lam; cobj[i_zeta:i_zeta + S] -= lam * w / (1 - alpha)  # maximise
    rows = 13 * n + 2 * n * S + 7 * n * S + 2 * n * S + 4 * n * S + 2 * n * S + S + 4 * n + 2 * n + (14 * n * S if guard else 0)
    A = lil_matrix((rows, N)); lo, hi = np.full(rows, -np.inf), np.full(rows, np.inf); r = 0; s0 = u.soc0 * E
    for t in range(n):  # first-stage rows
        A[r, ix('d', t)] = 1; A[r, ix('c', t)] = -1; A[r, ix('ru_a', t)] = 1; A[r, ix('ru_m', t)] = 1; hi[r] = P; r += 1
        A[r, ix('c', t)] = 1; A[r, ix('d', t)] = -1; A[r, ix('rd_a', t)] = 1; A[r, ix('rd_m', t)] = 1; hi[r] = P; r += 1
        A[r, ix('d', t)] = 1; A[r, ix('pv', t)] = 1; A[r, ix('c', t)] = -1; A[r, ix('ru_a', t)] = 1; A[r, ix('ru_m', t)] = 1; hi[r] = u.grid_mw; r += 1
        A[r, ix('c', t)] = 1; A[r, ix('d', t)] = -1; A[r, ix('pv', t)] = -1; A[r, ix('rd_a', t)] = 1; A[r, ix('rd_m', t)] = 1; hi[r] = u.grid_mw; r += 1
        A[r, ix('c', t)] = 1; A[r, ix('y', t)] = -P; hi[r] = 0; r += 1
        A[r, ix('d', t)] = 1; A[r, ix('y', t)] = P; hi[r] = P; r += 1
        if depth:
            for v, z, k in (('ru_a', 'z_ua', 'afrr_up'), ('ru_m', 'z_um', 'mfrr_up'), ('rd_a', 'z_da', 'afrr_down'), ('rd_m', 'z_dm', 'mfrr_down')):
                A[r, ix(v, t)] = 1; A[r, ix(z, t)] = -1; hi[r] = depth[k] / 2; r += 1
        # backing of the offered reserves in the EXPECTED SoC (the deterministic planner's rule); the tails are handled by recourse
        for sv in (() if fixed else ('s', 'prev')):  # (a pinned plan is valued without the planning-side backing rule: settlement has the guard instead)
            if sv == 's' or t:
                for s_ in range(S):
                    A[r, ix('s', t, s_) if sv == 's' else ix('s', t - 1, s_)] = w[s_]; A[r + 1, ix('s', t, s_) if sv == 's' else ix('s', t - 1, s_)] = w[s_]
                A[r, ix('ru_a', t)] = -Dr / ed; A[r, ix('ru_m', t)] = -Dr / ed; lo[r] = 0; r += 1
                A[r, ix('rd_a', t)] = Dr * ec; A[r, ix('rd_m', t)] = Dr * ec; hi[r] = E; r += 1
            else:
                A[r, ix('ru_a', t)] = Dr / ed; A[r, ix('ru_m', t)] = Dr / ed; hi[r] = s0; r += 1
                A[r, ix('rd_a', t)] = Dr * ec; A[r, ix('rd_m', t)] = Dr * ec; hi[r] = E - s0; r += 1
        for s_ in range(S):  # second-stage rows
            E_ = e[s_]
            A[r, ix('nd', t, s_)] = 1; A[r, ix('nc', t, s_)] = -1; A[r, ix('d', t)] = -1; A[r, ix('c', t)] = 1
            A[r, ix('au', t, s_)] = -1; A[r, ix('am', t, s_)] = -1; A[r, ix('bda', t, s_)] = 1; A[r, ix('bdm', t, s_)] = 1
            A[r, ix('sh', t, s_)] = 1; A[r, ix('lg', t, s_)] = -1; lo[r] = hi[r] = 0; r += 1  # battery flow = target - short + long
            if faithful and 'vol_up' in px:  # volume-capped settlement (qest.market.activated_volumes): offers beyond what the TSO activates deliver nothing,
                # so the activated offer is held within the TSO's volume per direction (conservative: forgoes capacity pay on MW above the cap)
                A[r, ix('ru_a', t)] = E_['ua'][t] / DT; A[r, ix('ru_m', t)] = E_['um'][t] / DT; hi[r] = float(px['vol_up'][t]); r += 1
                A[r, ix('rd_a', t)] = E_['da'][t] / DT; A[r, ix('rd_m', t)] = E_['dm'][t] / DT; hi[r] = float(px['vol_dn'][t]); r += 1
            if faithful:  # a deviation only arises from what the battery cannot back: short on UP / sales, long on DOWN / purchases
                A[r, ix('sh', t, s_)] = 1; A[r, ix('au', t, s_)] = -1; A[r, ix('am', t, s_)] = -1; A[r, ix('d', t)] = -1; hi[r] = 0; r += 1
                A[r, ix('lg', t, s_)] = 1; A[r, ix('bda', t, s_)] = -1; A[r, ix('bdm', t, s_)] = -1; A[r, ix('c', t)] = -1; hi[r] = 0; r += 1
            # delivered <= what this scenario activates of the offered MW (a_up_s share, m_up_q probability); faithful: equal
            for v, cv, k in (('au', 'ru_a', 'ua'), ('am', 'ru_m', 'um'), ('bda', 'rd_a', 'da'), ('bdm', 'rd_m', 'dm')):
                if guard and t:  # delivered = activated unless priced out: au <= share ru, au <= P (1 - pu), au >= share ru - P pu
                    po = 'pu' if v in ('au', 'am') else 'pd'
                    A[r, ix(v, t, s_)] = 1; A[r, ix(cv, t)] = -E_[k][t] / DT; hi[r] = 0; r += 1
                    A[r, ix(v, t, s_)] = 1; A[r, ix(po, t, s_)] = P; hi[r] = P; r += 1
                    A[r, ix(v, t, s_)] = 1; A[r, ix(cv, t)] = -E_[k][t] / DT; A[r, ix(po, t, s_)] = P; lo[r] = 0; r += 1
                else:
                    A[r, ix(v, t, s_)] = 1; A[r, ix(cv, t)] = -E_[k][t] / DT; hi[r] = 0; lo[r] = 0 if faithful else -np.inf; r += 1
            if guard and t and po is None:  # the guard: pu = 1 <=> drift <= -tol E, pd = 1 <=> drift >= tol E, drift = s_{t-1, s} - sum_s' w s_{t-1, s'}
                TE, M = tol * E, 2 * E
                for po, sign in (('pu', -1), ('pd', 1)):  # sign * drift >= TE  <=>  po = 1
                    for k_ in (0, 1):
                        for s2 in range(S):
                            A[r, ix('s', t - 1, s2)] = sign * (-w[s2] + (1 if s2 == s_ else 0))
                        if k_ == 0:  # po = 1  =>  sign drift >= TE :  sign drift - M po >= TE - M
                            A[r, ix(po, t, s_)] = -M; lo[r] = TE - M
                        else:        # po = 0  =>  sign drift <= TE :  sign drift - M po <= TE
                            A[r, ix(po, t, s_)] = -M; hi[r] = TE
                        r += 1
            A[r, ix('nc', t, s_)] = 1; A[r, ix('y2', t, s_)] = -2 * P; hi[r] = 0; r += 1
            A[r, ix('nd', t, s_)] = 1; A[r, ix('y2', t, s_)] = 2 * P; hi[r] = 2 * P; r += 1
            A[r, ix('s', t, s_)] = 1; A[r, ix('nc', t, s_)] = -ec * DT; A[r, ix('nd', t, s_)] = DT / ed
            if t:
                A[r, ix('s', t - 1, s_)] = -1; lo[r] = hi[r] = 0
            else:
                lo[r] = hi[r] = s0
            r += 1
            A[r, ix('nc', t, s_)] = 1; A[r, ix('nd', t, s_)] = 1; hi[r] = P; r += 1
            A[r, ix('nc', t, s_)] = 1; A[r, ix('nd', t, s_)] = -1; A[r, ix('pv', t)] = -1; A[r, ix('imp', t, s_)] = -1; hi[r] = 0; r += 1
    for s_ in range(S):  # zeta_s >= eta - profit_s
        A[r, :] = prof[s_]; A[r, i_zeta + s_] = 1; A[r, i_eta] = -1; lo[r] = 0; r += 1
    A = A[:r].tocsr(); lo, hi = lo[:r], hi[:r]
    res = milp(-cobj, constraints=LinearConstraint(A, lo, hi), integrality=integ, bounds=Bounds(lb, ub), options=dict(mip_rel_gap=mip_gap, time_limit=time_limit, disp=False))
    if res.x is None:
        raise RuntimeError(f'stochastic MILP failed: {res.message}')
    x = res.x; g = lambda v: x[idx[v]:idx[v] + n]; soc = sum(w[s_] * x[idx[('s', s_)]:idx[('s', s_)] + n] for s_ in range(S))
    return dict(da=np.round(g('d') + g('pv') - g('c'), 1), cap=dict(afrr_up=np.round(g('ru_a')), mfrr_up=np.round(g('ru_m')), afrr_down=np.round(g('rd_a')), mfrr_down=np.round(g('rd_m'))),
                soc=soc, objective=float(w @ (prof @ x)), cvar=float(x[i_eta] - w @ x[i_zeta:i_zeta + S] / (1 - alpha)), pv=g('pv'),
                soc_s=np.stack([x[idx[('s', s_)]:idx[('s', s_)] + n] for s_ in range(S)], 1))


def plan_storage_guard(u, scen, w, iters=3, tol=.05, **kw):
    """The 'guard' recourse by fixed point: solve the faithful plan, run settlement's gate-C guard on each scenario's SoC path
    (priced out when the drift from the expected SoC exceeds tol x E the wrong way, or a full MTU's activation cannot be
    backed), pin the flags, re-solve; stop when the flags no longer change. Each solve is the cheap faithful MILP."""
    n, S = len(scen[0]['da']), len(scen); E, Dr = u.e_mwh, u.reserve_s / 3600; w_ = np.asarray(w, float) / np.sum(w)
    po = (np.zeros((n, S)), np.zeros((n, S))); pl = None
    for _ in range(iters):
        pl = plan_storage_stoch(u, scen, w, recourse='guard', tol=tol, po=po, **kw)
        ss = pl['soc_s']; ref = ss @ w_; ru = pl['cap']['afrr_up'] + pl['cap']['mfrr_up']; rd = pl['cap']['afrr_down'] + pl['cap']['mfrr_down']
        prev = np.r_[np.full((1, S), u.soc0 * E), ss[:-1]]; drift = prev - np.r_[u.soc0 * E, ref[:-1]][:, None]
        pu = ((drift < -tol * E) | (prev - ru[:, None] * Dr / u.eta_d < 0)).astype(float); pd = ((drift > tol * E) | (prev + rd[:, None] * Dr * u.eta_c > E)).astype(float)
        pu[0] = pd[0] = 0
        if np.array_equal(pu, po[0]) and np.array_equal(pd, po[1]):
            break
        po = (pu, pd)
    pl['priced_out'] = float(po[0].mean() + po[1].mean()); return pl


def scenarios(ctx, C, px50, qs=('q10', 'q50', 'q90'), act_scale=(.5, 1.5)):
    """Five px dicts from the gate-A quantiles: comonotonic q10 / q50 / q90 and two activation-only draws around q50."""
    from qest.strategies import prices_for_milp
    S = [prices_for_milp(ctx, q) for q in qs]; W = [.25, .5, .25]
    for k in act_scale:
        p = {kk: (np.asarray(v) * k if kk in ('a_up_s', 'a_dn_s', 'm_up_q', 'm_dn_q') else v) for kk, v in px50.items()}
        for kk in ('a_up_s', 'a_dn_s', 'm_up_q', 'm_dn_q'):
            p[kk] = np.clip(p[kk], 0, 1)
        S.append(p); W.append(.15)
    W = np.array(W); W[:3] *= .7  # .175 / .35 / .175 / .15 / .15
    return S, W


def _check():
    """One scenario, lam = 0: the stochastic plan equals plan_storage's objective (within the MIP gap); with a spread of
    activation scenarios and lam > 0 the CVaR plan gives up expected profit for a better tail."""
    from qest import market
    from qest.optimize import plan_storage
    rng = np.random.default_rng(0); u = market.portfolio(['battery'])[0]; n = 96
    da = 60 + 40 * np.sin(np.arange(n) / n * 2 * np.pi) + rng.normal(0, 5, n)
    px = dict(da=da, cap_afrr_up=np.full(n, 8.), cap_afrr_down=np.full(n, 6.), cap_mfrr_up=np.full(n, 5.), cap_mfrr_down=np.full(n, 4.),
              a_up_s=np.full(n, .1), a_dn_s=np.full(n, .1), a_up_p=da + 80, a_dn_p=da - 40, m_up_q=np.full(n, .3), m_dn_q=np.full(n, .3), m_up_p=da + 60, m_dn_p=da - 30)
    depth = dict(afrr_up=60, afrr_down=60, mfrr_up=300, mfrr_down=300)
    ref = plan_storage(u, px, depth=depth)['objective']; one = plan_storage_stoch(u, [px], [1.], depth=depth)['objective']
    assert abs(one - ref) < 1e-2 * abs(ref) + 1, (one, ref)  # recourse may price out a little of what the deterministic plan backs
    lo = {k: (np.asarray(v) * .2 if k in ('a_up_s', 'a_dn_s', 'm_up_q', 'm_dn_q') else v) for k, v in px.items()}
    hi = {k: np.clip(np.asarray(v) * 2.5, 0, 1) if k in ('a_up_s', 'a_dn_s', 'm_up_q', 'm_dn_q') else v for k, v in px.items()}
    r0 = plan_storage_stoch(u, [lo, px, hi], [.3, .4, .3], depth=depth, lam=0.); r1 = plan_storage_stoch(u, [lo, px, hi], [.3, .4, .3], depth=depth, lam=.5)
    assert r1['cvar'] >= r0['cvar'] - 1 and r1['objective'] <= r0['objective'] + 1, (r0['objective'], r0['cvar'], r1['objective'], r1['cvar'])
    print(f'stoch check ok: one scenario = plan_storage ({one:.0f} vs {ref:.0f}); lam .5: E {r0["objective"]:.0f} -> {r1["objective"]:.0f}, CVaR {r0["cvar"]:.0f} -> {r1["cvar"]:.0f}')


if __name__ == '__main__':
    _check()

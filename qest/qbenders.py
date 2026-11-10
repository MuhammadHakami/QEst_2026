r"""Hybrid quantum-annealing Benders decomposition of the day plan (the method family that won DOE's 2026 grid-planning
challenge, Qtyche: QAOA + Benders; open code: HQCMCBD, Zhao et al., IEEE QCNC 2025, research/third_party/HQCMCBD-API).

Master (quantum): the day's charge / discharge pattern, one bit per hour (y2 of qest.optimize, 23-25 bits), and the
plan value eta. Subproblem (classical): the MILP's LP relaxation with the pattern pinned, solved by HiGHS; its value
V_k and the bound marginals g_k (d value / d pinned bit) give the Benders cut  eta <= V_k + g_k . (y - y_k), valid
because an LP's optimal value is concave in its right-hand side. The master  max eta  s.t. every cut  is a QUBO
(HQCMCBD's encoding): eta = lo + D sum_j 2^j e_j (L bits), each cut an equality with a binary slack (L bits),
penalised P (V_k + g_k.(y - y_k) - eta - s_k)^2 with P = 2 / D. Up to 24 + 7 + 7 K spins, solved by
  'sqa'   simulated quantum annealing (OpenJij SQASampler, path-integral Monte Carlo of a transverse-field annealer:
          the classical emulation of what a D-Wave QPU runs),
  'sa'    classical simulated annealing on the same QUBO (OpenJij SASampler), the control,
  'exact' the master as a MILP in HiGHS (classical Benders), the reference.
Loop: 8 iterations from the price-sign pattern, stop when the master repeats a pattern. The best pattern by LP value is
then completed by the full MILP (integer MW, no financial cycles) and settled on the official rules.
usage: CUDA_VISIBLE_DEVICES="" .venv/bin/python -m qest.qbenders [--split test] [--master sqa]
  -> results/days_<split>_benders_<master>.parquet, results/benders_<split>_<master>.csv (iterations, gaps)
"""
import argparse
import time

import numpy as np

from qest.optimize import plan_storage
from qest.strategies import _renewables, prices_for_milp

ITERS, L, READS, SWEEPS = 8, 7, 32, 1000


def hours(n):
    return -(-n // 4)


def lp(u, px, pv, cfg, y):
    n = len(px['da']); r = plan_storage(u, px, pv, depth=cfg.cap_depth, fixed={'y2': np.repeat(y, 4)[:n]}, relax=True)
    g = r['grad']['y2']; H = len(y); return r['objective'], np.array([g[4 * h:4 * h + 4].sum() for h in range(H)])


def master_qubo(cuts, H):
    """cuts: [(V, g, y)]. Returns QUBO dict over 'y_h', 'e_j', 's_k_j' and the decoder of eta."""
    hi = min(V + np.where(g > 0, g * (1 - y), -g * y).sum() for V, g, y in cuts)  # every cut's own maximum bounds eta
    lo = min(V for V, _, _ in cuts) - .05 * abs(hi); D = (hi - lo) / (2 ** L - 1); P = 2.0 / D
    Q = {}

    def add(a, b, w):
        k = (a, b) if a <= b else (b, a); Q[k] = Q.get(k, 0.0) + w
    for j in range(L):  # objective: - eta (scaled by 1 / D)
        add(f'e{j}', f'e{j}', -(2 ** j))
    for k, (V, g, y0) in enumerate(cuts):
        Smax = max(V + np.where(g > 0, g * (1 - y0), -g * y0).sum() - lo, D); Ds = Smax / (2 ** L - 1)
        terms = [(f'y{h}', g[h]) for h in range(H)] + [(f'e{j}', -D * 2 ** j) for j in range(L)] + [(f's{k}_{j}', -Ds * 2 ** j) for j in range(L)]
        c = V - g @ y0 - lo  # cut: c + sum_i w_i x_i = 0
        for i, (a, wa) in enumerate(terms):  # (c + sum w x)^2 = c^2 + sum (w^2 + 2 c w) x + 2 sum_{i<j} w_i w_j x_i x_j
            add(a, a, P / D * (wa * wa + 2 * c * wa))
            for b, wb in terms[i + 1:]:
                add(a, b, P / D * 2 * wa * wb)
    return Q, (lambda s: lo + D * sum(2 ** j * s[f'e{j}'] for j in range(L)))


def solve_master(cuts, H, mode, seed):
    if mode == 'exact':  # max eta s.t. eta <= V + g.(y - y0), y binary: a tiny MILP
        from scipy.optimize import Bounds, LinearConstraint, milp
        sc = max(abs(V) for V, _, _ in cuts) or 1.0  # unit-scale the cuts: raw EUR coefficients trip HiGHS ('Solve error')
        c = np.r_[np.zeros(H), -1.0]; A = np.array([np.r_[-g / sc, 1.0] for V, g, y in cuts]); hi = np.array([(V - g @ y) / sc for V, g, y in cuts])
        r = milp(c, constraints=LinearConstraint(A, -np.inf, hi), integrality=np.r_[np.ones(H), 0], bounds=Bounds(np.r_[np.zeros(H), -1e12], np.r_[np.ones(H), 1e12]))
        if r.x is None:
            raise RuntimeError(f'exact master: {r.message} | nan in cuts: {any(np.isnan(g).any() or np.isnan(V) for V, g, _ in cuts)}')
        return [np.round(r.x[:H]).astype(int)]
    import openjij as oj
    Q, _ = master_qubo(cuts, H)
    S = (oj.SQASampler() if mode == 'sqa' else oj.SASampler()).sample_qubo(Q, num_reads=READS, num_sweeps=SWEEPS, seed=seed)
    order = np.argsort(S.record.energy); vs = list(S.variables)
    return [np.array([int(S.record.sample[i][vs.index(f'y{h}')]) for h in range(H)]) for i in order]


def benders(u, px, pv, cfg, mode, seed=0):
    n = len(px['da']); H = hours(n); p = np.array([np.mean(px['da'][4 * h:4 * h + 4]) for h in range(H)])
    y = (p > p.mean()).astype(int); seen = {}; cuts = []; log = []
    for it in range(ITERS):
        t = time.time(); V, g = lp(u, px, pv, cfg, y); seen[tuple(y)] = V; cuts.append((V, g, y.copy()))
        cand = solve_master(cuts, H, mode, seed + it); new = [c for c in cand if tuple(c) not in seen]
        log.append(dict(iter=it, lp=V, best=max(seen.values()), s=time.time() - t, new=bool(new)))
        if not new:
            break
        y = new[0]
    best = max(seen, key=seen.get); return np.array(best), log


def plan(mode, units, ctx, cfg, logs=None):
    out = _renewables(units, ctx, cfg, use_real=False); px = prices_for_milp(ctx)
    if cfg.official:
        from qest.market import grid_tariff
        px['tariff'] = grid_tariff(ctx.idx)
    n = len(px['da'])
    for u in units:
        if u.kind in ('battery', 'hybrid'):
            pv = ctx.gen_fc.get(u.name) if u.kind == 'hybrid' else None
            y, log = benders(u, px, pv, cfg, mode)
            pl = plan_storage(u, px, pv, depth=cfg.cap_depth, fixed={'y2': np.repeat(y, 4)[:n]})
            pl.update(margins=ctx.calib['margins'], soc_ref=pl['soc'], policy='soc_aware', tol=ctx.calib.get('tol', .05)); out[u.name] = pl
            if logs is not None:
                full = plan_storage(u, px, pv, depth=cfg.cap_depth)  # the free MILP's expected value, for the gap
                logs.append(dict(day=str(ctx.idx[0].date()), unit=u.name, iters=len(log), lp_best=log[-1]['best'],
                                 milp_fixed=pl['objective'], milp_free=full['objective'], s=sum(l['s'] for l in log)))
    return out


def main():
    import pandas as pd
    from qest import market, strategies
    from qest.evaluate import run, table
    from qest.paths import RESULTS
    ap = argparse.ArgumentParser(); ap.add_argument('--split', default='test'); ap.add_argument('--master', default='sqa'); a = ap.parse_args()
    logs = []; name = f'benders_{a.master}'
    strategies.STRATEGIES[name] = lambda units, ctx, cfg: plan(a.master, units, ctx, cfg, logs)
    import qest.evaluate as ev
    _k = ev.fc_kind; ev.fc_kind = lambda s: 'chronos' if s.startswith('benders_') else _k(s)
    cfg = market.Config(); units = market.portfolio(['battery', 'hybrid'], official=cfg.official)
    R, tar = run(a.split, [name], units, cfg); R.to_parquet(RESULTS / f'days_{a.split}_{name}.parquet')
    G = pd.DataFrame(logs); G.to_csv(RESULTS / f'{name}_{a.split}.csv', index=False)
    t = table(R, tar); print(t[['total', 'cap', 'da', 'imbalance']].round(0).to_string())
    print('expected-value gap to the free MILP: %.3f %%' % ((1 - G.milp_fixed.sum() / G.milp_free.sum()) * 100), '| iterations', G.iters.mean().round(2), '| s', G.s.sum().round(0))


def _check():
    """Cuts are valid upper bounds (LP value concave in the pattern) and the QUBO's best eta respects every cut."""
    from qest import market
    from qest.evaluate import fc_kind, forecasts
    from qest.data import days, frame
    from qest.strategies import DayContext, calibration
    f = frame(); d = days(f, 'test')[3]; idx = f.index[(f.day == d).to_numpy()]
    ctx = DayContext(idx, forecasts('test', fc_kind('milp_chronos')).reindex(idx), {}, calibration(f, days(f, ('train', 'val'))))
    cfg = market.Config(); u = market.portfolio(['battery'], official=True)[0]; px = prices_for_milp(ctx)
    from qest.market import grid_tariff
    px['tariff'] = grid_tariff(idx); H = hours(len(idx)); rng = np.random.default_rng(0)
    y0 = rng.integers(0, 2, H); V0, g0 = lp(u, px, None, cfg, y0)
    for _ in range(4):
        y1 = rng.integers(0, 2, H); V1, _ = lp(u, px, None, cfg, y1); assert V1 <= V0 + g0 @ (y1 - y0) + 1e-3 * abs(V0), (V1, V0 + g0 @ (y1 - y0))
    Q, eta = master_qubo([(V0, g0, y0)], H)
    import openjij as oj
    S = oj.SASampler().sample_qubo(Q, num_reads=16, num_sweeps=2000, seed=0).first.sample; y = np.array([S[f'y{h}'] for h in range(H)])
    assert eta(S) <= V0 + g0 @ (y - y0) + 3 * (abs(V0) + abs(g0).sum()) / 2 ** L, (eta(S), V0 + g0 @ (y - y0))
    print('qbenders check ok: cut valid on 4 random patterns, master respects its cut')


if __name__ == '__main__':
    import sys
    _check() if '--check' in sys.argv else main()

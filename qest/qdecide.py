r"""Quantum bid decisions: the shortlisted 2026 families pick each day's charge / discharge pattern, the LP sizes the MW.

Split (research/methods.csv, 'binaries on the QPU, MW classical': O7 PCE unit commitment, K3, S4, O10): the battery's
net flow may charge (y_h = 1) or discharge (y_h = 0) in hour h (qest.optimize's y2, 23-25 bits). Every bid (DA MW,
capacity MW per product, energy bid prices) is then sized by HiGHS with the pattern pinned and settled on the official
rules, so a quantum choice reaches euros through the same plan -> settle_day path as the classical MILP.
The QUBO (one per forecaster, day and unit, shared by every solver): the plan value V(y) = LP value with y pinned is
sampled at NS patterns (uniform, and 1-8 bit flips of the price-sign pattern) and fitted by a quadratic model
V(y) ~ c + a.y + y'By (ridge): the surrogate-QUBO step of factorisation-machine annealing (FMQA, Kitai et al. 2020).
Solvers, each returns its K best distinct patterns by QUBO energy; the LP picks the best of them:
  'qaoa'    linear-ramp QAOA (LR-QAOA; C4 TVA hour-by-hour QAOA, O17 quantum-solar), X mixer from |+>^H, p = 8,
            2000 shots: CUDA-Q kernel, target 'nvidia' (cuStateVec), 23-25 qubits;
  'wsqaoa'  biased-start QAOA (O9 copula-QAOA, independent-copula case = Egger's warm start): start RY(theta_h)|0>
            from the bit marginals of the best 10 % sampled patterns, matching rotated mixer;
  'pce'     Pauli-correlation encoding (O5 / O7, Sciorilli et al.): H bits on 5 qubits as sign <P_i P_j> over
            XX / YY / ZZ pairs (30 slots), hardware-efficient ansatz, tanh-relaxed QUBO loss, COBYLA, 4 restarts,
            then single-bit-flip descent (the papers' classical post-processing): CUDA-Q observe;
  'sqa'     simulated quantum annealing (OpenJij SQASampler: path-integral emulation of a D-Wave QPU; K3 / K5 / O10);
  'sa'      classical simulated annealing on the same QUBO (OpenJij SASampler), the control for 'sqa';
  'exact'   the QUBO's K best patterns by full enumeration on the GPU (classical; the ceiling of any QUBO solver);
  'samples' the K best of the NS LP-checked samples (classical random search with the LP budget the QUBO used).
Baseline: 'milp' (the free MILP: Chronos-2 + HiGHS when the forecaster is Chronos-2).
Strategies are '<solver>@<forecaster>' (forecaster: chronos, qrc, ens, ...; qest.evaluate.fc_kind).
usage: CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m qest.qdecide [--split test] [--fc chronos] [--check]
  -> results/days_<split>_qdecide_<fc>.parquet, results/qdecide_<split>_<fc>.csv (per day/unit/solver diagnostics)
"""
import argparse
import functools
import time
import warnings

import numpy as np

from qest.optimize import plan_storage
from qest.strategies import _renewables, prices_for_milp

warnings.filterwarnings('ignore')
NS, K, P_QAOA, SHOTS, DELTA = 384, 8, 8, 2000, .7
PCE_Q, PCE_L, PCE_RESTARTS = 5, 4, 4
OJ_SCALE = 100.0  # Q3 day 11: SQA found the QUBO minimum in 0 / 5 seeds at x1-10, 2 / 5 at x30, 5 / 5 at x100-1000 (SA: 5 / 5 at all)
PART_B = 12
SOLVERS = ('qaoa', 'wsqaoa', 'pce', 'sqa', 'sa', 'exact', 'samples')
_CACHE, LOG = {}, []


def hours(n):
    return -(-n // 4)


def lp_value(u, px, pv, cfg, y):
    n = len(px['da'])
    return plan_storage(u, px, pv, depth=cfg.cap_depth, fixed={'y2': np.repeat(y, 4)[:n]}, relax=True)['objective']


def pairs(H):
    return [(i, j) for i in range(H) for j in range(i + 1, H)]


def phi(Y):
    Y = np.asarray(Y, float); P = pairs(Y.shape[1])
    return np.c_[Y, np.stack([Y[:, i] * Y[:, j] for i, j in P], 1)]


def build(u, px, pv, cfg, seed=0):
    """Sample V(y), fit the quadratic surrogate. Returns dict(a, B, c, scale, Y, V, r2)."""
    from sklearn.linear_model import Ridge
    n = len(px['da']); H = hours(n); rng = np.random.default_rng(seed)
    p = np.array([np.mean(px['da'][4 * h:4 * h + 4]) for h in range(H)])
    cand = [(p < p.mean()).astype(int), (p > p.mean()).astype(int)]
    base = max(cand, key=lambda y: lp_value(u, px, pv, cfg, y))  # which side the price sign means is the LP's call
    Y = [base] + [rng.integers(0, 2, H) for _ in range(NS // 4)]
    while len(Y) < NS:
        y = base.copy(); f = rng.choice(H, rng.integers(1, 9), replace=False); y[f] ^= 1; Y.append(y)
    Y = np.unique(np.array(Y), axis=0); V = np.array([lp_value(u, px, pv, cfg, y) for y in Y])
    mu, sd = V.mean(), V.std() + 1e-9; F = phi(Y); t = (V - mu) / sd
    tr = rng.random(len(Y)) < .8
    r2 = Ridge(alpha=1.0).fit(F[tr], t[tr]).score(F[~tr], t[~tr])
    m = Ridge(alpha=1.0).fit(F, t); a = m.coef_[:H]; B = np.zeros((H, H))
    for (i, j), w in zip(pairs(H), m.coef_[H:]):
        B[i, j] = w
    return dict(a=a, B=B, c=m.intercept_, mu=mu, sd=sd, Y=Y, V=V, r2=r2, H=H)


def energy(q, Y):
    """QUBO energy = - surrogate value (standardised): lower is better."""
    Y = np.atleast_2d(Y).astype(float)
    return -(q['c'] + Y @ q['a'] + np.einsum('bi,ij,bj->b', Y, q['B'], Y))


def ising(q):
    """x = (1 - z) / 2 (|0> -> z = +1 -> y = 0): E = const + sum h_i z_i + sum_{i<j} J_ij z_i z_j."""
    a, B = -q['a'], -q['B']; H = q['H']
    h = -a / 2 - (B.sum(1) + B.sum(0)) / 4; J = B / 4
    return h, J


def top_k(q, Y):
    Y = np.unique(np.atleast_2d(Y).astype(int), axis=0)
    return Y[np.argsort(energy(q, Y))[:K]]


# ---------------------------------------------------------------- solvers
def _cudaq():
    import cudaq
    try:
        cudaq.set_target('nvidia')
    except RuntimeError:
        cudaq.set_target('qpp-cpu')
    return cudaq


@functools.lru_cache(maxsize=None)
def qaoa_kernel():  # compiled once per process (re-defining a kernel re-JITs it)
    cudaq = _cudaq()

    @cudaq.kernel
    def k(n: int, h: list[float], J: list[float], ii: list[int], jj: list[int], gam: list[float], bet: list[float], th: list[float]):
        q = cudaq.qvector(n)
        for i in range(n):
            ry(th[i], q[i])  # pi / 2: |+> (plain QAOA); else the warm start
        for l in range(len(gam)):
            for i in range(n):
                rz(2.0 * gam[l] * h[i], q[i])
            for e in range(len(ii)):
                x.ctrl(q[ii[e]], q[jj[e]]); rz(2.0 * gam[l] * J[e], q[jj[e]]); x.ctrl(q[ii[e]], q[jj[e]])
            for i in range(n):  # exp(-i beta H_M), H_M = - (start-state axis): ground state = the start state
                ry(-th[i], q[i]); rz(-2.0 * bet[l], q[i]); ry(th[i], q[i])
        mz(q)
    return cudaq, k


def solve_qaoa(q, warm=None, seed=0, p=P_QAOA, shots=SHOTS, delta=DELTA, counts=False, noise=None):
    cudaq, k = qaoa_kernel(); H = q['H']; h, J = ising(q)
    E = [(i, j, J[i, j]) for i, j in pairs(H) if abs(J[i, j]) > 1e-9]
    s = np.sqrt(np.mean(h ** 2) + np.mean([e[2] ** 2 for e in E] or [0]))  # angles act on an O(1) energy scale
    ramp = (np.arange(p) + .5) / p; th = [np.pi / 2] * H if warm is None else list(2 * np.arcsin(np.sqrt(warm)))
    cudaq.set_random_seed(seed)
    r = cudaq.sample(k, H, list(h / s), [e[2] / s for e in E], [e[0] for e in E], [e[1] for e in E],
                     list(delta * ramp), list(delta * (1 - ramp)), th, shots_count=shots, **({} if noise is None else dict(noise_model=noise)))
    Y = np.array([[int(c) for c in b] for b in r])  # CUDA-Q bitstrings: character i is qubit i
    if counts:
        return Y, np.array([r.count(b) for b in r])
    return top_k(q, Y), Y


def warm_marginals(q, eps=.15):
    top = q['Y'][q['V'] >= np.quantile(q['V'], .9)]
    return np.clip(top.mean(0), eps, 1 - eps)


PAULI = dict(x=np.array([[0, 1], [1, 0]], complex), y=np.array([[0, -1j], [1j, 0]]), z=np.diag([1., -1.]).astype(complex))


def pce_slots(m):
    """(label, dense matrix) of the 3 C(m, 2) two-body correlators, qubit 0 = least significant bit (get_state's order)."""
    S = []
    for P in ('z', 'x', 'y'):
        for i, j in pairs(m):
            M = np.ones((1, 1), complex)
            for k in reversed(range(m)):
                M = np.kron(M, PAULI[P] if k in (i, j) else np.eye(2))
            S.append((f'{P}{i}{P}{j}', M))
    return S


@functools.lru_cache(maxsize=None)
def pce_kernel():
    cudaq = _cudaq()

    @cudaq.kernel
    def ans(th: list[float]):
        r = cudaq.qvector(5)
        for l in range(4):
            for i in range(5):
                ry(th[l * 10 + i], r[i]); rz(th[l * 10 + 5 + i], r[i])
            for i in range(4):
                z.ctrl(r[i], r[i + 1])
    return cudaq, ans


def solve_pce(q, seed=0, restarts=PCE_RESTARTS):
    from scipy.optimize import minimize
    cudaq, ans = pce_kernel(); H = q['H']; m = PCE_Q; S = np.stack([M for _, M in pce_slots(m)[:H]]); L = PCE_L
    alpha = 1.5 * m; rng = np.random.default_rng(seed); out = []
    def ev(th):  # every correlator from one CUDA-Q state (on a QPU: 3 measurement bases, shots)
        psi = np.array(cudaq.get_state(ans, list(th)))
        return np.einsum('i,kij,j->k', psi.conj(), S, psi).real

    def loss(th):  # relaxed QUBO + a small push toward decisive bits (|tanh| -> 1)
        e = np.tanh(alpha * ev(th)); y = (1 + e) / 2
        return energy(q, y[None])[0] + .1 * np.mean(1 - e ** 2)
    for r in range(restarts):
        res = minimize(loss, rng.uniform(0, 2 * np.pi, L * 2 * m), method='COBYLA', options=dict(maxiter=300))
        y = (ev(res.x) > 0).astype(int)
        while True:  # single-bit-flip descent on the QUBO
            F = np.repeat(y[None], H, 0); F[np.arange(H), np.arange(H)] ^= 1; e = energy(q, F)
            if e.min() >= energy(q, y)[0] - 1e-12:
                break
            y = F[e.argmin()]
        out.append(y)
        for i in range(H):  # neighbours of each restart's optimum keep K candidates
            z_ = y.copy(); z_[i] ^= 1; out.append(z_)
    return top_k(q, np.array(out)), np.array(out)


def solve_oj(q, kind, seed=0):
    import openjij as oj
    H = q['H']; Q = {(i, i): -OJ_SCALE * q['a'][i] for i in range(H)}  # OpenJij's default schedules want |E| ~ 1e2
    Q.update({(i, j): -OJ_SCALE * q['B'][i, j] for i, j in pairs(H)})
    S = (oj.SQASampler() if kind == 'sqa' else oj.SASampler()).sample_qubo(Q, num_reads=32, num_sweeps=1000, seed=seed)
    Y = np.array([[int(s[i]) for i in range(H)] for s in S.samples()])
    return top_k(q, Y), Y


def solve_exact(q):
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; H = q['H']; idx = torch.arange(1 << H, device=dev)
    E = torch.full((1 << H,), -float(q['c']), dtype=torch.float64, device=dev); bit = lambda i: ((idx >> i) & 1).double()
    for i in range(H):
        bi = bit(i); E -= q['a'][i] * bi
        for j in range(i + 1, H):
            if q['B'][i, j]:
                E -= q['B'][i, j] * bi * bit(j)
    z = torch.topk(-E, K).indices.cpu().numpy(); del E
    return np.array([(int(v) >> np.arange(H)) & 1 for v in z]), None


def solve(q, solver, seed=0):
    if solver == 'qaoa':
        return solve_qaoa(q, seed=seed)
    if solver == 'wsqaoa':
        return solve_qaoa(q, warm=warm_marginals(q), seed=seed)
    if solver == 'pce':
        return solve_pce(q, seed)
    if solver in ('sqa', 'sa'):
        return solve_oj(q, solver, seed)
    if solver == 'exact':
        return solve_exact(q)
    if solver.startswith('p_'):  # partitioned on PART_B qubits, frozen schedule, looped (qest.qpart; Daedalus recipe)
        from qest import qpart
        meth, _, mode = solver[2:].partition('_'); y, _ = qpart.solve(q, PART_B, max(2, PART_B // 4), qpart.schedule(PART_B), meth, mode or 'mult', device=24, seed=seed)
        return y[None], None
    if solver == 'samples':
        return q['Y'][np.argsort(-q['V'])[:K]], None
    raise ValueError(solver)


# ---------------------------------------------------------------- strategy
def plan(solver, units, ctx, cfg, fc='chronos'):
    out = _renewables(units, ctx, cfg, use_real=False)
    if fc == 'actrank':  # the deciders on the best gate-A information: rank-gated driver-based activation marginals (qest.actfc)
        from qest.strategies import actfc_gate, actfc_pred, actfc_px
        actfc_pred(ctx.idx); um, ua = actfc_gate(ctx.idx, ctx.calib, metric='rank'); px = actfc_px(ctx, cfg, tuple(p for p, u in (('mfrr', um), ('afrr', ua)) if u))
    else:
        px = prices_for_milp(ctx)
    if cfg.official:
        from qest.market import grid_tariff
        px['tariff'] = grid_tariff(ctx.idx)
    n = len(px['da']); day = str(ctx.idx[0].date())
    for u in units:
        if u.kind not in ('battery', 'hybrid'):
            continue
        pv = ctx.gen_fc.get(u.name) if u.kind == 'hybrid' else None
        key = (fc, day, u.name)
        if key not in _CACHE:
            t = time.time(); _CACHE[key] = build(u, px, pv, cfg); _CACHE[key]['t_build'] = time.time() - t
        q = _CACHE[key]; t = time.time(); C, _ = solve(q, solver); ts = time.time() - t
        V = np.array([lp_value(u, px, pv, cfg, y) for y in C]); y = C[V.argmax()]
        pl = plan_storage(u, px, pv, depth=cfg.cap_depth, fixed={'y2': np.repeat(y, 4)[:n]})
        pl.update(margins=ctx.calib['margins'], soc_ref=pl['soc'], policy='soc_aware', tol=ctx.calib.get('tol', .05)); out[u.name] = pl
        ex = q.setdefault('exact_min', energy(q, solve_exact(q)[0][:1])[0])
        LOG.append(dict(fc=fc, day=day, unit=u.name, solver=solver, H=q['H'], r2=q['r2'], n_lp=len(q['Y']) + len(C),
                        lp_best=V.max(), sample_best=q['V'].max(), qubo_best=energy(q, C[:1])[0], qubo_min=ex,
                        hit=bool(np.isclose(energy(q, C[:1])[0], ex)), milp_fixed=pl['objective'], s_solve=ts, s_build=q['t_build']))
    return out


def register(fc, extra=()):
    from qest import strategies
    for s in (*SOLVERS, *extra):
        strategies.STRATEGIES[f'{s}@{fc}'] = (lambda s_: lambda units, ctx, cfg: plan(s_, units, ctx, cfg, fc))(s)
    strategies.STRATEGIES[f'milp@{fc}'] = strategies.milp if fc != 'actrank' else strategies.STRATEGIES['milp_act_rank']  # the free MILP on the same inputs
    return [f'milp@{fc}'] + [f'{s}@{fc}' for s in (*SOLVERS, *extra)]


def main():
    import torch._dynamo  # noqa: F401  before CUDA-Q: both bundle LLVM, the other order aborts (the distilled solvers load torch mid-run)
    import pandas as pd
    from qest import market
    from qest.evaluate import run, table
    from qest.paths import RESULTS
    ap = argparse.ArgumentParser(); ap.add_argument('--split', default='test'); ap.add_argument('--fc', default='chronos')
    ap.add_argument('--solvers', default=','.join(SOLVERS)); ap.add_argument('--tag', default='_part'); a = ap.parse_args()
    sol = a.solvers.split(','); names = [n for n in register(a.fc, [x for x in sol if x not in SOLVERS]) if n.split('@')[0] in ('milp', *sol)]
    tag = a.fc + ('' if set(sol) <= set(SOLVERS) else a.tag)  # partitioned runs never overwrite the monolithic results
    cfg = market.Config(); units = market.portfolio(['battery', 'hybrid'], official=cfg.official)
    R, tar = run(a.split, names, units, cfg); R.to_parquet(RESULTS / f'days_{a.split}_qdecide_{tag}.parquet')
    G = pd.DataFrame(LOG); G.to_csv(RESULTS / f'qdecide_{a.split}_{tag}.csv', index=False)
    t = table(R, tar); pd.set_option('display.width', 250); print(t[['total', 'cap', 'da', 'mfrr', 'afrr', 'imbalance']].round(0).to_string())
    print(G.groupby('solver')[['r2', 'lp_best', 'milp_fixed', 'hit', 's_solve']].mean().round(3).to_string())


def _check():
    """On a small random QUBO: Ising conversion matches the QUBO, exact = brute force, QAOA and warm-start QAOA beat
    uniform sampling (probability mass on the optimum), PCE reaches a local optimum."""
    rng = np.random.default_rng(0); H = 8
    q = dict(H=H, a=rng.normal(0, 1, H), B=np.triu(rng.normal(0, .5, (H, H)), 1), c=0.0)
    Yall = np.array([(z >> np.arange(H)) & 1 for z in range(1 << H)]); E = energy(q, Yall)
    h, J = ising(q); Z = 1 - 2 * Yall; Ei = Z @ h + np.einsum('bi,ij,bj->b', Z, J, Z)
    assert np.allclose(E - E.mean(), Ei - Ei.mean()), 'ising'
    assert np.isclose(energy(q, solve_exact(q)[0][:1])[0], E.min()), 'exact'
    opt = Yall[E.argmin()]
    _, S = solve_qaoa(q, shots=4000); pq = np.mean((S == opt).all(1))
    q['Y'], q['V'] = Yall, -E
    _, S2 = solve_qaoa(q, warm=warm_marginals(q), shots=4000); pw = np.mean((S2 == opt).all(1))
    print(f'P(opt): uniform {1 / 2 ** H:.4f}  qaoa {pq:.4f}  ws-qaoa {pw:.4f}')
    assert pq > 3 / 2 ** H and pw > pq, (pq, pw)
    import cudaq
    cudaq_ = _cudaq(); M = dict(pce_slots(3))

    @cudaq_.kernel
    def t3(a: float, b: float):
        r = cudaq.qvector(3); ry(a, r[0]); rx(b, r[1]); h(r[2]); ry(.3, r[2])
    psi = np.array(cudaq_.get_state(t3, .7, 1.1))
    for lab, op in (('x0x1', cudaq.spin.x(0) * cudaq.spin.x(1)), ('y1y2', cudaq.spin.y(1) * cudaq.spin.y(2)), ('z0z2', cudaq.spin.z(0) * cudaq.spin.z(2))):
        assert np.isclose(psi.conj() @ M[lab] @ psi, cudaq_.observe(t3, op, .7, 1.1).expectation(), atol=1e-5), lab
    C, _ = solve_pce(q); print('pce best rank', int((E < energy(q, C[:1])[0] - 1e-12).sum()))
    print('qdecide check ok')


if __name__ == '__main__':
    import sys
    _check() if '--check' in sys.argv else main()

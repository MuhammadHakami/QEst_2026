"""The kick-off's portfolio puzzle (slide 11) on real Estonian hours, as a QUBO, with classical and quantum solvers.

Two sites, one hour: battery 100 MW / 200 MWh; hybrid = PV 20 MW + battery 20 MW / 40 MWh on one 20 MW connection.
Battery, hybrid battery and hybrid PV each pick ONE option: DA sell, DA buy, idle (P1), + mFRR UP, mFRR DOWN (P2).
Rules (slide 11): PV + battery within the 20 MW connection; batteries start at 50 % SoC and sell only what they hold;
leftover energy is worth the day's average DA price; imports pay the grid tariff; PV can't exceed its forecast; no
financial cycles. P3 adds a second hour, the batteries linked by SoC (30 qubits).
Option values use the gate-A forecasts (what a trader knows); the plans are then scored on the realised prices.
mFRR options hold the site's full power as capacity (whole MW) and carry the expected activation margin at the
strategy's bid (qest.strategies.calibration); a battery option that would need more energy than it holds is infeasible.
QUBO: x one-hot per decision; E(x) = sum_i c_i x_i + sum_{i<j} Q_ij x_i x_j with value c = -EUR, one-hot penalties
lam (sum x - 1)^2, and pairwise penalties for the connection (hybrid PV + hybrid battery export > 20 MW) and for
two-hour SoC violations. lam = 1 + sum |EUR| bounds every invalid state above the valid optimum (starter S06).
Solvers: exact enumeration, simulated annealing (dwave-samplers), QAOA p = 1, 2 (Qiskit statevector + COBYLA, 3
starts), and uniform random sampling as the floor. Metrics as the starter's S10.
usage: .venv/bin/python -m qest.qubo --split test --level P2 [--hours 24] [--qaoa]
"""
import argparse
import itertools
import time

import numpy as np
import pandas as pd

from qest.market import DT

OPTS = {'P1': ('sell', 'buy', 'idle'), 'P2': ('sell', 'buy', 'idle', 'up', 'down'), 'P3': ('sell', 'buy', 'idle', 'up', 'down')}
SITES = ('bat', 'hbat', 'hpv')
BAT = dict(bat=(100.0, 200.0), hbat=(20.0, 40.0))  # MW, MWh (slide 10)
PV_MW, GRID, ETA = 20.0, 20.0, 1.0  # slide 10: 100 % round-trip efficiency; the tariff comes per hour (market.grid_tariff)


def hour_values(px, pv_mw, mean_da, opts):
    """EUR of each (site, option) for one hour, and each battery option's energy change (MWh, + = charged).
    px: dict with the hour's mean da, cap_mfrr_up/down (EUR/MW/h), m_up_q/m_dn_q, m_up_p/m_dn_p."""
    val, dE = {}, {}
    for s, (P, E) in BAT.items():
        tar = px.get('tariff', 0.0); held = .5 * E  # 50 % SoC at the start
        for o in opts:
            if o == 'sell':
                mwh = min(P, held * ETA); val[s, o] = mwh * (px['da'] - mean_da / ETA); dE[s, o] = -mwh / ETA  # sold energy no longer worth the day's mean
            elif o == 'buy':
                val[s, o] = P * (mean_da * ETA - px['da'] - tar); dE[s, o] = P * ETA
            elif o == 'idle':
                val[s, o] = 0.0; dE[s, o] = 0.0
            elif o == 'up':  # capacity + expected activation margin (discharge at the activation price, minus stored value)
                q = px['m_up_q']; val[s, o] = P * px['cap_mfrr_up'] + q * P * (px['m_up_p'] - mean_da / ETA); dE[s, o] = -q * P / ETA
            elif o == 'down':
                q = px['m_dn_q']; val[s, o] = P * px['cap_mfrr_down'] + q * P * (mean_da * ETA - px['m_dn_p'] - tar); dE[s, o] = q * P * ETA
    for o in opts:  # hybrid PV: sell its forecast DA, sell nothing ahead (real time), hold UP headroom, or offer DOWN (curtail)
        if o == 'sell':
            val['hpv', o] = pv_mw * px['da']
        elif o in ('buy', 'idle'):
            val['hpv', o] = pv_mw * px['imb_exp'] if o == 'idle' else -1e9  # PV cannot buy: infeasible
        elif o == 'up':  # runs below its forecast to hold headroom: capacity + activation, nothing sold ahead
            val['hpv', o] = pv_mw * px['cap_mfrr_up'] + px['m_up_q'] * pv_mw * px['m_up_p']
        elif o == 'down':  # sells its forecast and offers to curtail it
            val['hpv', o] = pv_mw * px['da'] + pv_mw * px['cap_mfrr_down'] - px['m_dn_q'] * pv_mw * px['m_dn_p']
        dE['hpv', o] = 0.0
    return val, dE


def export(site, o, pv_mw):
    """MW the option puts on the hybrid connection (+ export, - import), for the 20 MW limit (capacity counts as held power)."""
    if site == 'hpv':
        return {'sell': pv_mw, 'down': pv_mw, 'up': 0.0, 'idle': pv_mw, 'buy': 0.0}[o]
    P = BAT[site][0]
    return {'sell': P, 'buy': -P, 'idle': 0.0, 'up': P, 'down': -P}[o]


def build(hours, level):
    """hours: list of (px, pv_mw, mean_da) (one for P1/P2, two for P3). Returns variables, linear, quadratic, offset, lam."""
    opts = OPTS[level]; V = [(h, s, o) for h in range(len(hours)) for s in SITES for o in opts]
    vals, dEs = zip(*[hour_values(px, pv, md, opts) for px, pv, md in hours])
    feasible_vals = [v for d in vals for v in d.values() if v > -1e8]
    lam = 1.0 + sum(abs(v) for v in feasible_vals)
    lin = {v: -vals[v[0]][v[1], v[2]] if vals[v[0]][v[1], v[2]] > -1e8 else lam for v in V}
    quad = {}; off = 0.0
    for h in range(len(hours)):
        for s in SITES:  # one-hot: lam (sum - 1)^2 = lam (1 - sum x + 2 sum_{i<j} x_i x_j)
            g = [(h, s, o) for o in opts]; off += lam
            for v in g: lin[v] -= lam
            for a, b in itertools.combinations(g, 2): quad[a, b] = quad.get((a, b), 0) + 2 * lam
        pv = hours[h][1]
        for ob in opts:  # 20 MW connection: hybrid PV + hybrid battery export/import beyond the limit
            for op in opts:
                if abs(export('hpv', op, pv) + export('hbat', ob, pv)) > GRID + 1e-6:
                    k = ((h, 'hbat', ob), (h, 'hpv', op)); quad[k] = quad.get(k, 0) + lam
    if len(hours) == 2:  # SoC link: energy over both hours must stay within 0..E from 50 %
        for s, (P, E) in BAT.items():
            for o1 in opts:
                for o2 in opts:
                    e = .5 * E + dEs[0][s, o1] + dEs[1][s, o2]
                    if e < -1e-6 or e > E + 1e-6:
                        k = ((0, s, o1), (1, s, o2)); quad[k] = quad.get(k, 0) + lam
    return V, lin, quad, off, lam


def energy(x, V, lin, quad, off):
    e = off + sum(lin[v] for v, b in zip(V, x) if b)
    ix = {v: i for i, v in enumerate(V)}
    return e + sum(q for (a, b), q in quad.items() if x[ix[a]] and x[ix[b]])


def valid(x, V, level):
    n = len(OPTS[level]); return all(sum(x[i * n:(i + 1) * n]) == 1 for i in range(len(V) // n))


def exact(V, lin, quad, off, level):
    """Enumerate the valid (one-hot) assignments only: 5^3 = 125 per hour, 5^6 = 15,625 for P3."""
    n = len(OPTS[level]); groups = len(V) // n; best = None
    for choice in itertools.product(range(n), repeat=groups):
        x = np.zeros(len(V), int)
        for gi, c in enumerate(choice): x[gi * n + c] = 1
        e = energy(x, V, lin, quad, off)
        if best is None or e < best[0]: best = (e, x)
    return best


def to_bqm(V, lin, quad, off):
    import dimod
    return dimod.BinaryQuadraticModel({v: lin[v] for v in V}, quad, off, dimod.BINARY)


def anneal(V, lin, quad, off, reads=200, seed=0):
    from dwave.samplers import SimulatedAnnealingSampler
    ss = SimulatedAnnealingSampler().sample(to_bqm(V, lin, quad, off), num_reads=reads, seed=seed)
    X = np.array([[s[v] for v in V] for s in ss.samples()]); E = ss.record.energy
    return X, E


def qaoa(V, lin, quad, off, p=1, shots=2048, starts=3, seed=0):
    """QAOA on the exact statevector (Qiskit), angles by COBYLA on the sampled mean cost, best of `starts` starts."""
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from scipy.optimize import minimize
    n = len(V); ix = {v: i for i, v in enumerate(V)}
    # Ising: x = (1 - z) / 2
    terms, const = {}, off
    for v, c in lin.items():
        i = ix[v]; const += c / 2; terms[(i,)] = terms.get((i,), 0) - c / 2
    for (a, b), q in quad.items():
        i, j = sorted((ix[a], ix[b])); const += q / 4
        terms[(i,)] = terms.get((i,), 0) - q / 4; terms[(j,)] = terms.get((j,), 0) - q / 4; terms[(i, j)] = terms.get((i, j), 0) + q / 4
    scale = max(abs(c) for c in terms.values())
    op = SparsePauliOp.from_sparse_list([('Z' * len(k), list(k), c / scale) for k, c in terms.items()], num_qubits=n)
    ans = QAOAAnsatz(op, reps=p, flatten=True)
    rng = np.random.default_rng(seed)
    zdiag = op.to_matrix(sparse=True).diagonal().real

    def cost(th):
        psi = Statevector(ans.assign_parameters(th)).probabilities()
        return float(psi @ zdiag)
    best = None; evals = 0
    for _ in range(starts):
        r = minimize(cost, rng.uniform(0, np.pi, ans.num_parameters), method='COBYLA', options=dict(maxiter=150))
        evals += r.nfev
        if best is None or r.fun < best.fun: best = r
    probs = Statevector(ans.assign_parameters(best.x)).probabilities()
    idx = rng.choice(len(probs), size=shots, p=probs / probs.sum())
    X = np.array([[(k >> i) & 1 for i in range(n)] for k in idx])  # qubit i = bit i (little-endian)
    return X, evals, ans


def realised(x, V, hours_real, level):
    """EUR of a plan on the realised hour(s): the same option values computed with the realised prices and activations."""
    opts = OPTS[level]; vals = [hour_values(px, pv, md, opts)[0] for px, pv, md in hours_real]
    return sum(vals[v[0]][v[1], v[2]] for v, b in zip(V, x) if b and vals[v[0]][v[1], v[2]] > -1e8)


def hour_inputs(f, fc, calib, gen_fc, gen_real, day, h, real):
    """One hour's option inputs from the forecasts (gate A) or the realised day."""
    from qest.strategies import DayContext, prices_for_milp
    d = f[f.day == day]; idx = d.index
    ctx = DayContext(idx, fc.reindex(idx), None, calib, real=d if real else None,
                     afrr_tab=__import__('qest.data', fromlist=['x']).afrr_margin_table().reindex(idx).fillna(0) if real else None)
    px = prices_for_milp(ctx, real=real); hr = idx.tz_convert('Europe/Tallinn').hour; m = hr == h
    out = {k: float(np.mean(v[m])) for k, v in px.items()}
    out['imb_exp'] = float(d.imbalance_eur_mwh[m].mean()) if real else out['da']  # PV left to real time: expected ~ DA
    from qest.market import grid_tariff
    out['tariff'] = float(grid_tariff(idx)[m].mean())
    pv = float(np.mean((gen_real if real else gen_fc).reindex(idx).fillna(0).to_numpy()[m])) * PV_MW
    return out, pv, float(np.mean(px['da']))


def feasible(x, V, lin, quad, off, H, level):
    """One-hot AND no pairwise penalty paid (connection limit, SoC link): the energy equals minus the plan's value."""
    if not valid(x, V, level):
        return False
    e = energy(x, V, lin, quad, off); return bool(np.isclose(e, -realised(x, V, H, level), rtol=1e-9, atol=1e-6))


def arrays(V, lin, quad):
    """QUBO as h (m,), J (m, m) upper-triangular, in V's order."""
    ix = {v: i for i, v in enumerate(V)}; h = np.array([lin[v] for v in V]); J = np.zeros((len(V), len(V)))
    for (a, b), q in quad.items():
        i, j = sorted((ix[a], ix[b])); J[i, j] += q
    return h, J


def dense(H, J, off, chunk=64):
    """Energy tables E (B, 2^m) on the GPU for the dense simulators (m <= 20)."""
    import torch
    from qest.qsota import DEV
    m = H.shape[1]; X = ((torch.arange(1 << m, device=DEV)[:, None] >> torch.arange(m, device=DEV)) & 1).double()
    out = []
    for i in range(0, len(H), chunk):
        h = torch.as_tensor(H[i:i + chunk], device=DEV); Jc = torch.as_tensor(J[i:i + chunk], device=DEV)
        out.append(torch.as_tensor(off[i:i + chunk], device=DEV)[:, None] + h @ X.T + torch.einsum('nm,bmk,nk->bn', X, Jc, X))
    return torch.cat(out)


def classical(V, lin, quad, off, level, seed=0):
    """Classical heuristics on one instance -> {solver: (x, seconds)}."""
    import dimod
    from dwave.samplers import SimulatedAnnealingSampler, TabuSampler
    bqm = to_bqm(V, lin, quad, off); out = {}
    for name, fn in (('sa', lambda: SimulatedAnnealingSampler().sample(bqm, num_reads=200, seed=seed)),
                     ('tabu', lambda: TabuSampler().sample(bqm, num_reads=8, timeout=5, seed=seed))):
        t = time.time(); ss = fn(); el = time.time() - t
        X = np.array([[r[v] for v in V] for r in ss.samples()]); E = ss.record.energy
        ok = np.array([valid(x, V, level) for x in X]); out[name] = (X[int(np.argmin(np.where(ok, E, np.inf)))] if ok.any() else X[0], el)
    return out


def random_shots(E, shots=4096, seed=0):
    """The like-for-like control: as many uniform random bitstrings as a quantum sampler's shots, best one, same polish."""
    import torch
    from qest.qsota import DEV, polish
    g = torch.Generator(device=DEV).manual_seed(seed); d = torch.randint(0, E.shape[1], (E.shape[0], shots), device=DEV, generator=g)
    return polish(E, d[torch.arange(E.shape[0], device=DEV), torch.gather(E, 1, d).argmin(1)]), {}


QUANTUM = {  # simulated exactly (statevector) on the GPU; every sampled best gets the same one-bit polish
    'random_4096': random_shots,
    'qaoa_p1': lambda E: __import__('qest.qsota', fromlist=['x']).qaoa(E, p=1),
    'qaoa_p3': lambda E: __import__('qest.qsota', fromlist=['x']).qaoa(E, p=3),
    'sqa': lambda E: (__import__('qest.qsota', fromlist=['x']).anneal(E, quantum=True), {}),
    'bfdcqo': lambda E: __import__('qest.qsota', fromlist=['x']).bfdcqo(E),
    'falqon': lambda E: __import__('qest.qsota', fromlist=['x']).falqon(E),
    'pce': lambda E: __import__('qest.qsota', fromlist=['x']).pce(E),
}


def instances(split, level, mode='retro', ndays=0, hours=24, every=1):
    """The puzzle's instances on a split's delivery days (every n-th day, first ndays of those): (day, hour, H, HR),
    H the inputs the plan is built on (realised prices in retro mode, gate-A forecasts in prospective), HR realised."""
    from qest.data import days, frame
    from qest.evaluate import forecasts
    from qest.power import generation
    from qest.strategies import calibration
    f = frame(); g = generation(); D = days(f, split)[::every][:ndays or None]
    fit = days(f, 'train') if split == 'val' else days(f, ('train', 'val')); calib = calibration(f, fit); fc = forecasts(split)
    inst = []
    for day in D:
        hrs = list(range(24))[:hours]
        for grp in ([[h] for h in hrs] if level != 'P3' else [[h, h + 1] for h in hrs[::2] if h + 1 < 24]):
            HR = [hour_inputs(f, fc, calib, g.hybrid_pv_d1_per_mw, g.hybrid_pv_actual_per_mw, day, h, True) for h in grp]
            H = HR if mode == 'retro' else [hour_inputs(f, fc, calib, g.hybrid_pv_d1_per_mw, g.hybrid_pv_actual_per_mw, day, h, False) for h in grp]
            inst.append((day, grp[0], H, HR))
    return inst


def main():
    from qest.paths import RESULTS
    ap = argparse.ArgumentParser(); ap.add_argument('--split', default='test'); ap.add_argument('--level', default='P2')
    ap.add_argument('--mode', default='retro', help="retro: prices known, gap to the classical optimum (slide 9/12); "
                    "prospective: forecast instance, plan scored on the realised hour")
    ap.add_argument('--hours', type=int, default=24); ap.add_argument('--days', type=int, default=0)
    ap.add_argument('--quantum', default='random_4096,qaoa_p1,qaoa_p3,sqa,bfdcqo,falqon,pce')
    a = ap.parse_args()
    import torch
    from qest import qsota
    inst = instances(a.split, a.level, a.mode, ndays=a.days, hours=a.hours)
    print(len(inst), 'instances', flush=True)
    rows, plans = [], {}
    for k, (day, hour, H, HR) in enumerate(inst):
        V, lin, quad, off, lam = build(H, a.level)
        t = time.time(); eopt, xopt = exact(V, lin, quad, off, a.level); t_ex = time.time() - t
        eor, _ = exact(*build(HR, a.level)[:4], a.level)
        row = dict(day=day, hour=hour, qubits=len(V), lam=lam, e_opt=eopt, hindsight_eur=-eor, exact_eur=realised(xopt, V, HR, a.level), exact_s=t_ex,
                   plan=''.join(o[0] for (h, s, o), b in zip(V, xopt) if b))
        for name, (x, el) in classical(V, lin, quad, off, a.level).items():
            row.update({f'{name}_eur': realised(x, V, HR, a.level) if feasible(x, V, lin, quad, off, H, a.level) else 0.0, f'{name}_opt': bool(np.isclose(energy(x, V, lin, quad, off), eopt, rtol=1e-9, atol=1e-6)), f'{name}_s': el})
        rng = np.random.default_rng(k); Xr = (rng.random((256, len(V))) < 1 / len(OPTS[a.level])).astype(int)  # the floor
        okr = np.array([feasible(x, V, lin, quad, off, H, a.level) for x in Xr]); row['random_valid'] = okr.mean()
        row['random_eur'] = max([realised(x, V, HR, a.level) for x in Xr[okr]], default=0.0)
        rows.append(row); plans[k] = (V, lin, quad, off)
    R = pd.DataFrame(rows)
    # batched quantum simulators (and the random + polish control) on all instances at once
    HJ = [arrays(V, lin, quad) for V, lin, quad, off in plans.values()]; Hm, Jm = np.stack([h for h, _ in HJ]), np.stack([j for _, j in HJ])
    off = np.array([p[3] for p in plans.values()]); Vs = [p[0] for p in plans.values()]
    def score(name, X, el, stats=None):
        X = np.asarray(X); e = (X * Hm).sum(1) + np.einsum('bm,bmn,bn->b', X, Jm, X) + off
        R[f'{name}_eur'] = [realised(x, V, HR, a.level) if feasible(x, V, *plans[k][1:], H, a.level) else 0.0 for k, (x, V, (_, _, H, HR)) in enumerate(zip(X, Vs, inst))]
        R[f'{name}_opt'] = np.isclose(e, R.e_opt, rtol=1e-9, atol=1e-6); R[f'{name}_s'] = el / len(R)
        if stats and 'p_opt' in stats: R[f'{name}_popt'] = stats['p_opt']
        print(name, round(R[f'{name}_eur'].sum()), R[f'{name}_opt'].mean().round(3), f'{el:.0f} s', flush=True)
    m = Hm.shape[1]; Xr = torch.as_tensor((np.random.default_rng(1).random((len(R), m)) < 1 / len(OPTS[a.level])).astype(float), device=qsota.DEV)
    t = time.time(); score('random_polish', qsota.polish_qubo(torch.as_tensor(Hm, device=qsota.DEV, dtype=torch.float32), torch.as_tensor(Jm, device=qsota.DEV, dtype=torch.float32), Xr.float()).cpu(), time.time() - t)
    if m <= 20:
        E = dense(Hm, Jm, off); bits = lambda z: ((z.cpu().numpy()[:, None] >> np.arange(m)) & 1)
        for name in a.quantum.split(','):
            t = time.time(); z, st = QUANTUM[name](E); torch.cuda.synchronize() if torch.cuda.is_available() else None
            score(name, bits(z), time.time() - t, st)
        del E
    else:  # P3: only PCE scales past the statevector (30 bits on 6 qubits)
        t = time.time(); X, st = qsota.pce(None, qubo=(Hm, Jm)); score('pce', X.cpu().numpy(), time.time() - t)
        t = time.time(); g_ = torch.Generator(device=qsota.DEV).manual_seed(0)  # control: 4096 uniform bitstrings per instance
        h_, J_ = (torch.as_tensor(x, device=qsota.DEV, dtype=torch.float32) for x in (Hm, Jm)); best = []
        for i in range(len(Hm)):
            Xs = torch.randint(0, 2, (4096, m), device=qsota.DEV, generator=g_).float()
            best.append(Xs[((Xs @ h_[i]) + ((Xs @ J_[i]) * Xs).sum(1)).argmin()])
        score('random_4096', qsota.polish_qubo(h_, J_, torch.stack(best)).cpu().numpy(), time.time() - t)
    RESULTS.mkdir(parents=True, exist_ok=True); out = RESULTS / f'qubo_{a.level}_{a.split}_{a.mode}.csv'; R.to_csv(out, index=False)
    ex = R.exact_eur.sum(); print('exact EUR', round(ex), 'hindsight', round(R.hindsight_eur.sum()))
    for c in [c for c in R if c.endswith('_eur')]:
        print(f'{c:20s} {R[c].sum():>14,.0f}  {R[c].sum() / ex * 100:6.2f} % of exact' + (f'  optimum {R[c[:-4] + "_opt"].mean():.3f}' if c[:-4] + '_opt' in R else ''))


if __name__ == '__main__':
    main()

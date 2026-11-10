r"""Partitioned, looped QAOA with one frozen, pre-trained schedule: Daedalus's recipe (ETH, "Universal latent spaces
for operator learning", daedalus.ethz.ch) carried over to a fixed qubit budget. Daedalus is classical (its 92x is a
92x larger network on one GPU); what transfers is the structure:
  partition      the hours are cut into overlapping windows of B qubits (Daedalus: k-means seeded along a
                 space-filling curve, with overlap; on a 1-D time axis that is contiguous windows);
  frame          each window's conditioned QUBO is brought to a canonical scale (h, J / rms), as Daedalus stores
                 translation / dilation / amplitude in a static frame and the encoder sees only the shape;
  frozen         one LR-QAOA schedule (delta, p) is chosen on SYNTHETIC random QUBOs of size B, then frozen and reused
                 on every window, day and site: no per-instance angle training (Daedalus: one encoder, synthetic
                 functions only, frozen);
  processor      the only coupling between windows is classical: block-coordinate descent, each window solved with the
                 rest pinned (its linear terms absorb the pinned neighbours), looped until no window improves (the
                 qbsolv / DQAOA pattern, research/methods.csv O1 / O4);
  chunked        windows run one after another on the same B qubits (or several at once on a larger device);
  overlap        Schwarz-type: 'mult' solves windows in turn against the latest solution, 'add' solves them all at once
                 (several windows per call on a larger device) and settles overlap bits by majority, the paper's
                 segment mean over the partitions covering a point;
  shift          window boundaries move by a random offset each sweep (the paper's random rotation before its
                 space-filling-curve partitioning; windowed attention's random windows), so no boundary is permanent;
  every solver   the same partitioning wraps LR-QAOA, warm-start QAOA (warm from the carried solution), PCE and SQA;
  lean           a window is re-sent only if a neighbour changed since its last solve; shots stop at the budget.
Paper: Mousavi, Lingsch, Mishra, De Lorenzis, arXiv:2611.05735 (Nov 2026).
usage: CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m qest.qpart [--check]   -> results/qpart_*.csv / .json
"""
import json
import time

import numpy as np

from qest import qdecide
from qest.paths import RESULTS

SHOTS, LOOPS = 1000, 4
VARIANTS = (('qaoa', 'mult'), ('qaoa', 'add'), ('wsqaoa', 'mult'), ('pce', 'mult'), ('sqa', 'mult'))


def windows(H, B, o):
    if B >= H:
        return [np.arange(H)]
    st = list(range(0, H - B, B - o)) + [H - B]
    return [np.arange(s, s + B) for s in sorted(set(st))]


def condition(q, S, y):
    """QUBO of the variables S with every other bit pinned at y (energy = - value)."""
    R = np.setdiff1d(np.arange(q['H']), S); Bs = q['B'] + q['B'].T
    a = q['a'][S] + Bs[np.ix_(S, R)] @ y[R]
    return dict(H=len(S), a=a, B=q['B'][np.ix_(S, S)], c=0.0)


def synthetic(B, rng):
    return dict(H=B, a=rng.normal(0, 1, B), B=np.triu(rng.normal(0, .5, (B, B)), 1), c=0.0)


def pretrain(B, n=12, seed=0):
    """Pick (delta, p) on synthetic QUBOs only: the lowest mean normalised expected energy of the samples."""
    rng = np.random.default_rng(seed); Q = [synthetic(B, rng) for _ in range(n)]
    allY = np.array([(z >> np.arange(B)) & 1 for z in range(1 << B)]); best = None
    for p in (4, 8, 12):
        for d in (.3, .5, .7, .9, 1.2):
            sc = []
            for q in Q:
                E = qdecide.energy(q, allY); Y, c = qdecide.solve_qaoa(q, p=p, delta=d, shots=SHOTS, counts=True)
                sc.append((qdecide.energy(q, Y) @ c / c.sum() - E.min()) / (E.mean() - E.min()))
            if best is None or np.mean(sc) < best[0]:
                best = (float(np.mean(sc)), p, d)
    return dict(score=best[0], p=best[1], delta=best[2], B=B)


def schedule(B):
    S = json.loads((RESULTS / 'qpart_schedules.json').read_text())
    return S[str(B)] if str(B) in S else dict(S[max(S, key=int)], B=B)  # larger windows: the largest pre-trained schedule, transferred


def window_solve(qb, method, sched, y_cur, seed):
    """Candidates for one window. The frozen schedule serves every gate method; the warm start reads the current
    solution (the carried state, Daedalus's latent rollout); PCE and SQA reuse qest.qdecide unchanged."""
    if method == 'qaoa':
        return qdecide.solve_qaoa(qb, p=sched['p'], delta=sched['delta'], shots=SHOTS, seed=seed, counts=True)[0]
    if method == 'wsqaoa':
        return qdecide.solve_qaoa(qb, warm=np.clip(y_cur, .25, .75), p=sched['p'], delta=sched['delta'], shots=SHOTS, seed=seed, counts=True)[0]
    if method == 'nqaoa':  # the same circuit under hardware-like noise (qest.qdistill.noise_model)
        from qest import qdistill
        return qdecide.solve_qaoa(qb, p=sched['p'], delta=sched['delta'], shots=SHOTS, seed=seed, counts=True, noise=qdistill.noise_model())[0]
    if method == 'nqs' or method.startswith('nqs-'):  # the distilled model stands in for the QPU (qest.qdistill)
        from qest import qdistill
        return qdistill.window_samples(qb, method.partition('-')[2] or 'aware', seed)
    if method == 'uniform':
        return np.unique(np.random.default_rng(seed).integers(0, 2, (SHOTS, qb['H'])), axis=0)
    if method == 'pce':
        return qdecide.solve_pce(qb, seed, restarts=1)[1]  # lean: the sweeps are the restarts
    if method == 'sqa':
        return qdecide.solve_oj(qb, 'sqa', seed)[1]
    raise ValueError(method)


def qubits_of(method, B):
    return int(np.ceil((1 + np.sqrt(1 + 8 * B / 3)) / 2)) if method == 'pce' else B  # PCE: 3 C(m, 2) >= B correlators


def solve(q, B, o, sched, method='qaoa', mode='mult', shift=True, device=None, y0=None, seed=0):
    """mode 'mult': windows in turn, each against the latest solution (multiplicative Schwarz); 'add': every window
    against the same pinned solution, all at once on a device of `device` qubits, overlap bits by majority over the
    windows covering them (additive Schwarz; Daedalus's segment mean), kept only if it lowers the energy, else 'mult'.
    shift: window boundaries move by a random offset every sweep (Daedalus's random rotation before partitioning)."""
    H = q['H']; y = np.zeros(H, int) if y0 is None else y0.copy(); rng = np.random.default_rng(seed)
    stats = dict(circuits=0, qpu_calls=0, shots=0, loops=0, cz=0, qubits=qubits_of(method, B)); per = max(1, (device or B) // stats['qubits'])
    dirty = None
    for it in range(LOOPS):
        off = int(rng.integers(0, max(1, B - o))) if (shift and it) else 0
        W = [np.unique(np.clip(w - off, 0, H - 1)) for w in windows(H + off, B, o)] if off else windows(H, B, o)
        dirty = [True] * len(W) if (dirty is None or len(dirty) != len(W) or off) else dirty
        e0 = qdecide.energy(q, y)[0]; todo = [w for w in range(len(W)) if dirty[w]]
        stats['qpu_calls'] += int(np.ceil(len(todo) / per)) if mode == 'add' else len(todo)
        best = {}
        for w in todo:
            S = W[w]; qb = condition(q, S, y); Y = window_solve(qb, method, sched, y[S], seed + 97 * it + w)
            stats['circuits'] += 1; stats['shots'] += SHOTS
            stats['cz'] = max(stats['cz'], 2 * sched['p'] * int((np.abs(qb['B']) > 1e-9).sum())) if 'qaoa' in method else stats['cz']
            cand = np.repeat(y[None], len(Y) + 1, 0); cand[:-1, S] = Y; e = qdecide.energy(q, cand); k = int(e.argmin())
            best[w] = cand[k][S]; dirty[w] = False
            if mode == 'mult' and k < len(Y) and e[k] < qdecide.energy(q, y)[0] - 1e-12:
                y = cand[k]
                for v, T in enumerate(W):  # lean: only neighbours of a changed window go again
                    if v != w and (abs(v - w) == 1 or np.intersect1d(T, S).size):
                        dirty[v] = True
        if mode == 'add' and best:
            votes, cnt = np.zeros(H), np.zeros(H)
            for w, ys in best.items():
                votes[W[w]] += ys; cnt[W[w]] += 1
            ya = y.copy(); m = cnt > 0; ya[m] = (votes[m] / cnt[m] >= .5).astype(int)
            if qdecide.energy(q, ya)[0] < e0 - 1e-12:
                y = ya; dirty = [True] * len(W)
        stats['loops'] = it + 1
        if qdecide.energy(q, y)[0] >= e0 - 1e-12 and not any(dirty):
            break
    return y, stats


def day_qubos(split, n=None):
    """(day, unit name, QUBO) for the delivery days of a split, from the Chronos-2 gate-A forecasts (qest.qdecide.build)."""
    from qest import market
    from qest.data import days, frame
    from qest.evaluate import forecasts, unit_gen
    from qest.market import grid_tariff
    from qest.power import generation
    from qest.strategies import DayContext, calibration, prices_for_milp
    f = frame(); B0 = json.loads((RESULTS / 'tune_best_milp_chronos.json').read_text())
    cal = calibration(f, days(f, ('train', 'val')), B0['margins']); cal.update(B0['params'])
    cfg = market.Config(); units = market.portfolio(['battery', 'hybrid'], official=True); fc = forecasts(split, 'chronos')
    _, gfc = unit_gen(generation(), units)
    for d in days(f, split)[:n]:
        idx = f.index[(f.day == d).to_numpy()]; ctx = DayContext(idx, fc.reindex(idx), {}, cal); px = prices_for_milp(ctx); px['tariff'] = grid_tariff(idx)
        for u in units:
            pv = gfc[u.name].reindex(idx).fillna(0).to_numpy() if u.kind == 'hybrid' else None
            yield d, u.name, qdecide.build(u, px, pv, cfg)


def bench_days(scheds, Bs, n=None):
    """Q4 Chronos-2 day QUBOs (qest.qdecide.build): partitioned QAOA at B qubits vs monolithic 24-25-qubit LR-QAOA,
    scored against the exact minimum."""
    import pandas as pd
    rows = []
    for d, u, q in day_qubos('test', n):
        emin = qdecide.energy(q, qdecide.solve_exact(q)[0][:1])[0]
        t = time.time(); C, _ = qdecide.solve_qaoa(q); rows.append(dict(day=str(d.date()), unit=u, method=f"mono_{q['H']}q", qubits=q['H'],
            gap=qdecide.energy(q, C[:1])[0] - emin, hit=bool(np.isclose(qdecide.energy(q, C[:1])[0], emin)), circuits=1, s=time.time() - t))
        for B in Bs:
            for meth, mode in VARIANTS:
                t = time.time(); y, st = solve(q, B, max(2, B // 4), scheds[B], meth, mode, device=24); e = qdecide.energy(q, y)[0]
                rows.append(dict(day=str(d.date()), unit=u, method=f'{meth}_{mode}_{B}', qubits=st['qubits'], gap=e - emin, hit=bool(np.isclose(e, emin)),
                                 circuits=st['circuits'], qpu_calls=st['qpu_calls'], loops=st['loops'], cz=st['cz'], s=time.time() - t))
        print(d.date(), u, flush=True)
    D = pd.DataFrame(rows); D.to_csv(RESULTS / 'qpart_days_test.csv', index=False)
    print(D.groupby('method')[['qubits', 'hit', 'gap', 'circuits', 'qpu_calls', 'loops', 'cz', 's']].mean().round(3).to_string())


def reach(B, sched, N, seed=0, method='qaoa', mode='mult'):
    """A time-coupled QUBO N times larger than the device (chain + next-neighbour couplings, like SoC over hours):
    partitioned QAOA on B qubits vs OpenJij SA on the whole problem (no exact optimum at this size)."""
    import openjij as oj
    rng = np.random.default_rng(seed); M = np.zeros((N, N))
    for k, s_ in ((1, .8), (2, .5), (4, .3)):
        i = np.arange(N - k); M[i, i + k] = rng.normal(0, s_, N - k)
    q = dict(H=N, a=rng.normal(0, 1, N), B=M, c=0.0)
    t = time.time(); y, st = solve(q, B, max(2, B // 4), sched, method, mode); tq = time.time() - t
    Q = {(i, i): -q['a'][i] for i in range(N)}; Q.update({(i, j): -M[i, j] for i, j in zip(*np.nonzero(M))})
    t = time.time(); S = oj.SASampler().sample_qubo(Q, num_reads=32, num_sweeps=2000, seed=0); ts = time.time() - t
    ysa = np.array([S.first.sample[i] for i in range(N)])
    gr = np.zeros(N, int)  # greedy single flips from zero, the floor
    while True:
        F = np.repeat(gr[None], N, 0); F[np.arange(N), np.arange(N)] ^= 1; e = qdecide.energy(q, F)
        if e.min() >= qdecide.energy(q, gr)[0] - 1e-12:
            break
        gr = F[e.argmin()]
    out = dict(N=N, B=B, ratio=N / B, method=method, mode=mode, e_part=float(qdecide.energy(q, y)[0]), e_sa=float(qdecide.energy(q, ysa)[0]), e_greedy=float(qdecide.energy(q, gr)[0]),
               circuits=st['circuits'], loops=st['loops'], s_part=tq, s_sa=ts)
    print(out); return out


def main():
    scheds = {}
    old = json.loads((RESULTS / 'qpart_schedules.json').read_text()) if (RESULTS / 'qpart_schedules.json').exists() else {}
    for B in (8, 12, 16):
        scheds[B] = old.get(str(B)) or pretrain(B); print('schedule', scheds[B], flush=True)
    (RESULTS / 'qpart_schedules.json').write_text(json.dumps(scheds))
    s24 = dict(scheds[16], B=24)  # 2^24 is too large to pre-train on: the 16-qubit schedule, frozen and transferred
    R = [reach(24, s24, 24 * 92, method=m_, mode=md) for m_, md in (('qaoa', 'mult'), ('qaoa', 'add'), ('wsqaoa', 'mult'))]
    (RESULTS / 'qpart_reach.json').write_text(json.dumps(R))
    bench_days(scheds, (8, 12))


def _check():
    """Conditioning is exact: the window QUBO's energy differs from the full energy by a constant; windows cover all."""
    rng = np.random.default_rng(1); q = synthetic(10, rng); y = rng.integers(0, 2, 10); S = np.array([3, 4, 5, 6])
    qb = condition(q, S, y); allS = np.array([(z >> np.arange(4)) & 1 for z in range(16)])
    full = np.repeat(y[None], 16, 0); full[:, S] = allS; d = qdecide.energy(q, full) - qdecide.energy(qb, allS)
    assert np.allclose(d, d[0]); assert set(np.concatenate(windows(25, 8, 2))) == set(range(25))
    print('qpart check ok')


if __name__ == '__main__':
    import sys
    _check() if '--check' in sys.argv else main()

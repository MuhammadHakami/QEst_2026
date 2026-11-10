r"""Quantum annealing on the slide-11 puzzle, beside the gate-model optimisers (qest.qnas, qest.qsota).

  'sqa_oj'   simulated quantum annealing on the raw QUBO, OpenJij SQASampler (path-integral Monte Carlo of a
             transverse-field annealer, the emulation of a D-Wave QPU): any size, so P3's 30 variables too;
  'sa_oj'    classical simulated annealing on the same QUBO (OpenJij SASampler), the control;
  'dqa_x'    digitised quantum annealing on a gate QPU: H(s) = (1 - s) H_X + s H_cost, Trotterised into 64 steps
             from |+>^n (full space, P1 / P2);
  'dqa_xy'   the same with the XY driver from the one-hot start (constraint-preserving annealing, Hen & Spedalieri
             2016; Hadfield et al. 2019): never leaves the valid plans, so P3 fits in 15,625 states.
Retrospective (slide 9), every Q4 hour; a plan counts only if it pays no QUBO penalty (qubo.feasible).
usage: CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m qest.qanneal   -> results/qanneal_test.csv
"""
import time

import numpy as np
import pandas as pd

from qest import qnas, qubo
from qest.paths import RESULTS

READS, SWEEPS, STEPS, DT = 32, 500, 64, .8


def openjij(d, kind, level, E):
    """E: one-hot energies (B, k^G); each QUBO is divided by the spread of its better half, so the annealer's transverse
    field and temperature (OpenJij defaults, O(1)) act on O(1) energy steps instead of EUR 1e5 coefficients."""
    import openjij as oj
    S = oj.SQASampler() if kind == 'sqa' else oj.SASampler(); X = []
    Es = E.sort(1).values[:, :E.shape[1] // 2].std(1).clamp(min=1e-9).cpu().numpy()
    for k, (V, lin, quad, off, _) in enumerate(d['P']):
        ix = {v: i for i, v in enumerate(V)}; Q = {(ix[a], ix[a]): w / Es[k] for a, w in lin.items()}  # OpenJij wants int labels
        Q.update({(ix[a], ix[b]): w / Es[k] for (a, b), w in quad.items()})
        r = S.sample_qubo(Q, num_reads=READS, num_sweeps=SWEEPS, seed=k); best = r.first.sample; X.append([int(best[i]) for i in range(len(V))])
    return np.array(X)


def main():
    rows = []
    for level in ('P1', 'P2', 'P3'):
        d = qnas.load('test', level); B = len(d['inst'])
        E = qnas.energies(d['h'], d['J'], d['off'], d['sub']); xo = d['sub'][E.argmin(1)].cpu().numpy().astype(int); emin = E.min(1).values.cpu().numpy()
        ex = sum(qubo.realised(x, V, HR, level) for x, (V, *_), (_, _, H, HR) in zip(xo, d['P'], d['inst']))

        def score(name, X, el, popt=np.nan):
            e = np.array([qubo.energy(x, V, lin, quad, off) for x, (V, lin, quad, off, _) in zip(X, d['P'])])
            eur = sum(qubo.realised(x, V, HR, level) if qubo.feasible(x, V, lin, quad, off, H, level) else 0.0
                      for x, (V, lin, quad, off, _), (_, _, H, HR) in zip(X, d['P'], d['inst']))
            rows.append(dict(level=level, solver=name, opt=float(np.mean(np.isclose(e, emin, rtol=1e-9, atol=1e-6))), eur=eur,
                             eur_pct_exact=eur / ex * 100, popt=popt, s_per_instance=el / B)); print(rows[-1], flush=True)
        for kind in ('sqa', 'sa'):
            t = time.time(); X = openjij(d, kind, level, E); score(f'{kind}_oj', X, time.time() - t)
        for space in (['x', 'xy'] if level != 'P3' else ['xy']):
            arch = dict(space=space, p=STEPS, angles='ramp', dg=DT, db=DT, loss='mean', graph='ring', trotter=1, init='w')
            t = time.time(); R = qnas.solve(arch, d, np.arange(B), chunk=100 if space == 'x' else 300)
            score(f'dqa_{space}', R['x'], time.time() - t, float(R['popt'].mean()))
    D = pd.DataFrame(rows); D.to_csv(RESULTS / 'qanneal_test.csv', index=False); print(D.round(4).to_string())


if __name__ == '__main__':
    main()

r"""Quantum-enhanced Markov chain Monte Carlo on the slide-11 puzzle (Layden et al., Nature 619, 282, 2023; for
combinatorial optimisation on 117 IBM qubits, Marshall, Egger et al., arXiv:2602.06171, Feb 2026).

Hybrid: the QPU proposes, the CPU accepts. From the chain's plan s, prepare |s>, evolve under
H = (1 - g) a H_cost + g H_mix for time t (random g in [.25, .6], t in [2, 20], Trotterised in 8 steps as on hardware),
measure s'; accept with Metropolis min(1, exp(-(E(s') - E(s)) / T)) (the proposal is symmetric, so this is exact).
a scales the cost to the mixer's norm. T falls geometrically 1 -> .02 (in units of the instance's energy spread) over
the chain, so the chain anneals to low energies. Spaces as qest.qnas: full 2^n with an X mixer (P1, P2) and the
one-hot subspace with the XY ring mixer (P1-P3). Control: the same chain with a local classical proposal (flip one
bit; in the subspace, re-draw one decision's option). Same steps, same temperatures. Q4, prices known (slide 9).
usage: CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m qest.qmcmc [--steps 200]   -> results/qmcmc_test.csv
"""
import argparse
import math

import numpy as np
import pandas as pd
import torch

from qest import qnas, qubo
from qest.paths import RESULTS
from qest.qsota import DEV

TROTTER = 8


def chain(E, d, space, steps, quantum, seed=0, chunk=200):
    """E (B, N) energies of the space's states. Returns the best state visited per instance and the step it was found."""
    B, N = E.shape; g = torch.Generator(device=DEV).manual_seed(seed); ar = torch.arange(B, device=DEV)
    sd = E.std(1, keepdim=True); Es = (E - E.mean(1, keepdim=True)) / sd; nq = d['m'] if space == 'x' else None
    alpha = (math.sqrt(d['m']) if space == 'x' else math.sqrt(d['G'] * 2)) / Es.std(1, keepdim=True)  # cost to mixer scale
    s = torch.randint(0, N, (B,), device=DEV, generator=g); best, bstep = s.clone(), torch.zeros(B, dtype=torch.long, device=DEV)
    for k in range(steps):
        T = 1.0 * (.02 / 1.0) ** (k / max(steps - 1, 1))
        if quantum:
            gm = .25 + .35 * torch.rand(B, 1, device=DEV, generator=g); t = 2 + 18 * torch.rand(B, 1, device=DEV, generator=g)
            sp = torch.empty_like(s)
            for i in range(0, B, chunk):
                sl = slice(i, i + chunk); b = len(s[sl]); psi = torch.zeros(b, N, dtype=torch.complex64, device=DEV); psi[torch.arange(b), s[sl]] = 1
                dc = ((1 - gm[sl]) * alpha[sl] * t[sl] / TROTTER).float(); dm = (gm[sl] * t[sl] / TROTTER)[:, 0].float()
                for _ in range(TROTTER):
                    psi = psi * torch.exp(-1j * dc * Es[sl].float())
                    if space == 'x':
                        c, sn = torch.cos(dm).to(torch.complex64)[:, None, None], (-1j * torch.sin(dm)).to(torch.complex64)[:, None, None]
                        for q in range(nq):
                            v = psi.view(b, N >> (q + 1), 2, 1 << q); a0, a1 = v[:, :, 0], v[:, :, 1]
                            psi = torch.stack([c * a0 + sn * a1, sn * a0 + c * a1], 2).reshape(b, N)
                    else:
                        U = qnas.mixer(dm, d['k'], 'ring', 1); k_, G = d['k'], d['G']
                        for gi in range(G):
                            psi = torch.einsum('bij,bajc->baic', U, psi.reshape(b, k_ ** gi, k_, k_ ** (G - gi - 1))).reshape(b, N)
                pr = psi.abs() ** 2; sp[sl] = torch.multinomial(pr / pr.sum(1, keepdim=True), 1, generator=g)[:, 0]
        elif space == 'x':
            sp = s ^ (1 << torch.randint(0, nq, (B,), device=DEV, generator=g))
        else:  # re-draw one decision's option: digits base k, decision 0 most significant
            k_, G = d['k'], d['G']; gi = torch.randint(0, G, (B,), device=DEV, generator=g); w = k_ ** (G - 1 - gi)
            cur = (s // w) % k_; new = torch.randint(0, k_, (B,), device=DEV, generator=g); sp = s + (new - cur) * w
        dE = (E[ar, sp] - E[ar, s]) / sd[:, 0]
        acc = torch.rand(B, device=DEV, generator=g) < torch.exp(-dE.clamp(min=0) / T); s = torch.where(acc, sp, s)
        imp = E[ar, s] < E[ar, best]; best = torch.where(imp, s, best); bstep = torch.where(imp, torch.full_like(bstep, k + 1), bstep)
    return best, bstep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--steps', type=int, default=200); a = ap.parse_args(); rows = []
    for level in ('P1', 'P2', 'P3'):
        d = qnas.load('test', level); B = len(d['inst'])
        Esub = qnas.energies(d['h'], d['J'], d['off'], d['sub']); emin = Esub.min(1).values
        xo = d['sub'][Esub.argmin(1)].cpu().numpy().astype(int)
        ex = sum(qubo.realised(x, V, HR, level) for x, (V, *_), (_, _, H, HR) in zip(xo, d['P'], d['inst']))
        for space in (['x', 'xy'] if level != 'P3' else ['xy']):
            E = qnas.energies(d['h'], d['J'], d['off'], d['full'] if space == 'x' else d['sub']); X = d['full'] if space == 'x' else d['sub']
            for quantum in (True, False):
                best, bstep = chain(E, d, space, a.steps, quantum); eb = E[torch.arange(B, device=DEV), best]
                hit = (eb <= emin + 1e-9 * emin.abs().clamp(min=1)).cpu().numpy(); xs = X[best].cpu().numpy().astype(int)
                eur = sum(qubo.realised(x, V, HR, level) if qubo.feasible(x, V, lin, quad, off, H, level) else 0.0
                          for x, (V, lin, quad, off, _), (_, _, H, HR) in zip(xs, d['P'], d['inst']))
                rows.append(dict(level=level, space=space, proposal='quantum' if quantum else 'local', hit=hit.mean(), eur=eur, eur_pct_exact=eur / ex * 100,
                                 median_step=float(np.median(bstep.cpu().numpy()[hit])) if hit.any() else np.nan, steps=a.steps, instances=B))
                print(rows[-1], flush=True)
    R = pd.DataFrame(rows); R.to_csv(RESULTS / 'qmcmc_test.csv', index=False); print(R.round(4).to_string())


if __name__ == '__main__':
    main()

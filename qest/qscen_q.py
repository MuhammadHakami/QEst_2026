"""Quantum-enhanced MCMC as the scenario sampler, benchmarked against the exact one (qest.qscen).

The day plan needs joint activation scenarios; qest.qscen samples them exactly because the model is a chain. Real
balancing is not a chain once aFRR, the neighbouring zones and the Finnish price couple in, and sampling a general
Boltzmann distribution is where a quantum proposal has a documented edge (Layden et al., Nature 619, 282, 2023: the
quantum proposal mixes faster than local flips at low temperature). Here the target is known exactly, so the sampler's
quality can be measured instead of assumed: a window of 8 MTUs is 16 qubits (4 states per MTU, 2 bits each), the target
p(z) = exp(-E(z)) with E = -log P(path | z_0) from the fitted transition model, and the proposals are

  quantum   prepare |z>, evolve under (1 - g) a E + g H_X for time t (random g, t; Trotterised in 8 steps, as qest.qmcmc),
            measure z'; accept with Metropolis (the proposal is symmetric: exact Boltzmann sampling at T = 1)
  local     flip one of the 16 bits (classical control, same chain length)
  exact     qest.qscen's forward sampler (ground truth)

Metrics per window: total-variation distance of the per-MTU state marginals to the exact ones, the acceptance rate and
the integrated autocorrelation of the UP count. The one-time measurement of a window could equally be distilled
(qest.qdistill) and reused all season, the IonQ amortisation.

usage: CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m qest.qscen_q [--windows 20 --steps 400] [--check]  -> results/qscen_q.csv
"""
import argparse
import math
import sys
import time

import numpy as np
import pandas as pd
import torch

from qest import qscen
from qest.paths import RESULTS

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
W, NQ, TROTTER = 8, 16, 8  # window MTUs, qubits (2 per MTU), Trotter steps


def energies(P, X, t0, z0):
    """E over the 4^W paths of the window, as a torch vector indexed by the 16-bit integer (MTU k's state in bits 2k, 2k+1)."""
    E, idx = qscen.window_energy(P, X, t0, z0, W); code = sum(idx[k] << (2 * k) for k in range(W))
    out = np.zeros(4 ** W); out[code] = E; return torch.as_tensor(out, dtype=torch.float32, device=DEV)


def _propose(s, Es, alpha, quantum, g):
    """One proposal per chain. Es: (chains, N) energies shifted to min 0 (per chain). quantum: Trotterised evolution under
    (1 - g) alpha E + g H_X from |s>, then measure; else a single-bit flip."""
    chains, N = Es.shape
    if not quantum:
        return s ^ (1 << torch.randint(0, NQ, (chains,), device=DEV, generator=g))
    gm = .25 + .35 * torch.rand(chains, 1, device=DEV, generator=g); t = 2 + 18 * torch.rand(chains, 1, device=DEV, generator=g)
    psi = torch.zeros(chains, N, dtype=torch.complex64, device=DEV); psi[torch.arange(chains, device=DEV), s] = 1
    dc = ((1 - gm) * alpha[:, None] * t / TROTTER).float(); dm = (gm * t / TROTTER)[:, 0].float()
    c, sn = torch.cos(dm).to(torch.complex64)[:, None, None], (-1j * torch.sin(dm)).to(torch.complex64)[:, None, None]
    for _ in range(TROTTER):
        psi = psi * torch.exp(-1j * dc * Es.float())
        for q in range(NQ):
            v = psi.view(chains, N >> (q + 1), 2, 1 << q); a0, a1 = v[:, :, 0], v[:, :, 1]
            psi = torch.stack([c * a0 + sn * a1, sn * a0 + c * a1], 2).reshape(chains, N)
    pr = psi.abs() ** 2; return torch.multinomial(pr / pr.sum(1, keepdim=True), 1, generator=g)[:, 0]


def mcmc(E, steps, quantum, chains=256, seed=0, T=1.0, temps=None, init=None):
    """Metropolis at temperature T with a quantum or a single-flip proposal. E: (N,) one target for every chain, or
    (chains, N) one target per chain (the day sampler: each chain conditions on its own previous state). temps: parallel
    tempering (Marshall & Egger, arXiv:2602.06171): replicas at these temperatures (first = T), adjacent swaps after every
    step; the T replica is returned. Returns (visited states (steps, chains), acceptance at T)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    E2 = E if E.dim() == 2 else E[None].expand(chains, -1); chains, N = E2.shape
    Es = E2 - E2.min(1, keepdim=True).values; scale = Es.std(1).clamp(min=1e-6); alpha = math.sqrt(NQ) / scale  # cost to mixer scale, as qest.qmcmc
    temps = [T] if temps is None else list(temps); R = len(temps); ar = torch.arange(chains, device=DEV)
    s = torch.stack([torch.randint(0, N, (chains,), device=DEV, generator=g) for _ in range(R)]) if init is None else init.clone()
    out = torch.zeros(steps, chains, dtype=torch.long, device=DEV); acc_n = 0
    for k in range(steps):
        for r in range(R):
            sp = _propose(s[r], Es, alpha, quantum, g); dE = (E2[ar, sp] - E2[ar, s[r]]) / temps[r]
            acc = torch.rand(chains, device=DEV, generator=g) < torch.exp(-dE.clamp(min=0)); s[r] = torch.where(acc, sp, s[r])
            if r == 0:
                acc_n += int(acc.sum())
        for r in range(k % 2, R - 1, 2):  # replica swaps between neighbouring temperatures
            d = (1 / temps[r] - 1 / temps[r + 1]) * (E2[ar, s[r]] - E2[ar, s[r + 1]])
            sw = torch.rand(chains, device=DEV, generator=g) < torch.exp(d.clamp(max=0)); a, b = s[r].clone(), s[r + 1].clone()
            s[r] = torch.where(sw, b, a); s[r + 1] = torch.where(sw, a, b)
        out[k] = s[0]
    return out, acc_n / (steps * chains)


TEMPS = (1.0, 1.6, 2.5, 4.0)


def sample_day(P, X, n, seed=0, quantum=True, steps=300, temps=None):
    """n joint activation patterns of the day from the quantum-enhanced (or local) chain, window by window: z_0 ~ pi0, then
    for each block of W MTUs every chain samples z_{t0+1..t0+W} | z_{t0} from its own conditional target (one 16-qubit
    window), the last window re-samples the tail. Returns (n, T, 2) ints as qest.qscen.sample does."""
    T = len(X); rng = np.random.default_rng(seed); z = np.zeros((n, T), int); z[:, 0] = rng.choice(4, n, p=P['pi0'])
    starts = list(range(0, T - 1 - W, W)); starts.append(T - 1 - W)
    for i, t0 in enumerate(starts):
        Eall = torch.stack([energies(P, X, t0, z0) for z0 in range(4)])  # (4, N)
        E2 = Eall[torch.as_tensor(z[:, t0], device=DEV)]
        S, _ = mcmc(E2, steps, quantum, chains=n, seed=seed * 1000 + i, temps=temps)
        last = S[-1].cpu().numpy()
        for k in range(W):
            z[:, t0 + 1 + k] = (last >> (2 * k)) & 3
    return qscen.STATES[z]


def bench_day(days=6, n=64, steps=300, seed=0):
    """Per-MTU marginals and run lengths of the day samplers against the exact forward sampler (2,000 draws), on held-out
    Q3 days: TV per MTU, UP run length, seconds. -> results/qscen_q_day.csv"""
    from qest.data import frame
    from qest.evaluate import forecasts
    P = qscen.params(); f = frame(); F = forecasts('val', 'chronos'); j = f.join(F, how='inner'); j = j[j.split == 'val']
    X = qscen.features(j, j.index); dd = j.day.to_numpy(); rng = np.random.default_rng(seed); rows = []
    for d in rng.choice(pd.unique(dd), days, replace=False):
        Xd = X[dd == d]; ex = qscen.sample(P, Xd, 2000, seed=1); T = len(Xd)
        code = lambda Y: Y[:, :, 0] + 2 * Y[:, :, 1]
        mex = np.stack([np.bincount(code(ex)[:, t], minlength=4) / len(ex) for t in range(T)])
        for name, q, tp in (('quantum', True, None), ('local', False, None), ('quantum+PT', True, TEMPS), ('local+PT', False, TEMPS), ('exact (n draws)', None, None)):
            t = time.time()
            Y = qscen.sample(P, Xd, n, seed=2) if q is None else sample_day(P, Xd, n, seed=seed, quantum=q, steps=steps, temps=tp)
            dt = time.time() - t; m = np.stack([np.bincount(code(Y)[:, t_], minlength=4) / n for t_ in range(T)])
            rows.append(dict(day=str(pd.Timestamp(d).date()), sampler=name, tv=float(np.abs(m - mex).sum(1).mean() / 2), up=float(Y[:, :, 0].mean()),
                             up_exact=float(ex[:, :, 0].mean()), runlen=qscen._runlen(Y[:, :, 0].ravel()), runlen_exact=qscen._runlen(ex[:, :, 0].ravel()), s=dt))
    R = pd.DataFrame(rows); R.to_csv(RESULTS / 'qscen_q_day.csv', index=False)
    print(R.groupby('sampler')[['tv', 'up', 'up_exact', 'runlen', 'runlen_exact', 's']].mean().round(3).to_string()); return R


def marginals(states, N=4 ** W):
    """Per-MTU 4-state marginals (W, 4) from integer states."""
    m = np.zeros((W, 4)); st = states.reshape(-1).cpu().numpy()
    for k in range(W):
        m[k] = np.bincount((st >> (2 * k)) & 3, minlength=4) / len(st)
    return m


def exact_marginals(E):
    p = torch.softmax(-E.double(), 0).cpu().numpy(); idx = np.arange(len(p)); m = np.zeros((W, 4))
    for k in range(W):
        m[k] = np.bincount((idx >> (2 * k)) & 3, weights=p, minlength=4)
    return m


def iact(x):
    """Integrated autocorrelation time of a 1-d series (Sokal window)."""
    x = x - x.mean(); n = len(x); v = np.dot(x, x) / n
    if v < 1e-12:
        return 1.0
    tau = 1.0
    for lag in range(1, min(n // 4, 200)):
        r = np.dot(x[:-lag], x[lag:]) / (n - lag) / v
        if r < 0.05:
            break
        tau += 2 * r
    return tau


def run(windows=20, steps=400, burn=100, seed=0, pt=False):
    from qest.data import frame
    from qest.evaluate import forecasts
    P = qscen.params(); f = frame(); F = forecasts('val', 'chronos'); j = f.join(F, how='inner'); j = j[j.split == 'val']
    X = qscen.features(j, j.index); days = j.day.to_numpy(); rng = np.random.default_rng(seed); rows = []
    for w in range(windows):
        d = rng.choice(pd.unique(days)); m = days == d; Xd = X[m]; t0 = int(rng.integers(0, 96 - W - 1)); z0 = int(rng.integers(0, 4))
        E = energies(P, Xd, t0, z0); ex = exact_marginals(E)
        for name, q, tp in (('quantum', True, None), ('local', False, None), *((('quantum+PT', True, TEMPS), ('local+PT', False, TEMPS)) if pt else ())):
            t = time.time(); S, acc = mcmc(E, steps, q, seed=seed + w, temps=tp); dt = time.time() - t
            keep = S[burn:]; tv = float(np.abs(marginals(keep) - ex).sum(1).mean() / 2)
            ups = ((keep[:, :, None] >> (2 * torch.arange(W, device=DEV))) & 1).sum(2).float().cpu().numpy()  # UP count per step, chain
            tau = float(np.mean([iact(ups[:, c]) for c in range(0, ups.shape[1], 32)]))
            rows.append(dict(window=w, day=str(pd.Timestamp(d).date()), t0=t0, sampler=name, tv=tv, acc=acc, iact=tau, s=dt))
        ex_s = qscen.sample(P, Xd, (steps - burn) * 256 // 96 + 1, seed=w)  # the exact forward sampler's own finite-sample TV, for scale
        st = ex_s[:, t0 + 1:t0 + 1 + W]; code = (st[:, :, 0] + 2 * st[:, :, 1]); mm = np.stack([np.bincount(code[:, k], minlength=4) / len(code) for k in range(W)])
        rows.append(dict(window=w, day=str(pd.Timestamp(d).date()), t0=t0, sampler='ignore z0 (bias of the unconditioned forward sampler, for scale)', tv=float(np.abs(mm - ex).sum(1).mean() / 2), acc=1.0, iact=1.0, s=0.0))
    R = pd.DataFrame(rows); R.to_csv(RESULTS / 'qscen_q.csv', index=False)
    print(R.groupby('sampler')[['tv', 'acc', 'iact', 's']].mean().round(3).to_string()); return R


def _check():
    """On one window the quantum chain's marginals converge toward the exact ones (TV falls with steps) and the energy
    table is normalised."""
    from qest.data import frame
    from qest.evaluate import forecasts
    P = qscen.params(); f = frame(); F = forecasts('val', 'chronos'); j = f.join(F, how='inner'); j = j[j.split == 'val']
    X = qscen.features(j, j.index); days = j.day.to_numpy(); Xd = X[days == days[0]]; E = energies(P, Xd, 10, 1)
    assert abs(torch.exp(-E.double()).sum().item() - 1) < 1e-4; ex = exact_marginals(E)
    tv = []
    for steps in (20, 200):
        S, acc = mcmc(E, steps, True, chains=128, seed=1); tv.append(float(np.abs(marginals(S[steps // 2:]) - ex).sum(1).mean() / 2))
    assert tv[1] < tv[0], tv
    print(f'qscen_q check ok: quantum chain TV {tv[0]:.3f} -> {tv[1]:.3f} (acceptance {acc:.2f}), target normalised')


if __name__ == '__main__':
    if '--check' in sys.argv:
        _check()
    else:
        ap = argparse.ArgumentParser(); ap.add_argument('--windows', type=int, default=20); ap.add_argument('--steps', type=int, default=400)
        ap.add_argument('--day', action='store_true', help='benchmark the day samplers (bench_day)'); ap.add_argument('--pt', action='store_true', help='window benchmark with parallel tempering too'); a = ap.parse_args()
        bench_day(steps=a.steps) if a.day else run(a.windows, a.steps, pt=a.pt)

r"""Hybrid day scheduler: a quantum sampler proposes the day's charge / discharge pattern, the MILP completes it.

The pattern used to come from inside the MILP (binary y2 per MTU: the battery's net flow may charge, or discharge).
Here it is one bit per hour, b_h = 1 'discharge side', 0 'charge side' (23-25 qubits on DST days), the split that
2026 hybrid unit-commitment work uses (quantum picks the binaries, classical dispatches: PCE unit commitment,
arXiv:2605.17145; ADMM-distributed QAOA, arXiv:2608.04159; copula-biased QAOA on 150 IBM qubits, arXiv:2603.00260).
Quantum part: the hour bits' value as an Ising surrogate on the Chronos-2 DA forecast,
  E(b) = - sum_h v_h s_h + mu (sum_h s_h)^2,  s_h = 2 b_h - 1,  v_h = P (p_h - mean p),  mu = mean |v| / 4,
sampled by linear-ramp QAOA (LR-QAOA, Montanez-Barrera & Michielsen 2024-25: fixed angles, no training, run on IBM
to 109 qubits) p = 6, 4096 shots, statevector on the GPU. The K = 8 lowest-energy distinct shots go to HiGHS with y2
pinned (every plan stays feasible) and the plan with the best expected value under the forecast is kept.
Controls: 'random' (K uniform patterns, same completion) and 'surrogate' (the K best patterns by exact enumeration of
the same surrogate, classical): the quantum sampler can at best match the latter at this size (2^24 enumerates in ms).
usage: CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m qest.qday [--split test]   -> results/summary_<split>_hybrid.csv
"""
import argparse

import numpy as np
import torch

from qest.optimize import plan_storage
from qest.strategies import _renewables, prices_for_milp

K, P_LAYERS, SHOTS, DG, DB = 8, 6, 4096, .6, .6


def energies(v, mu):
    """E over all 2^H patterns, built bit by bit on the GPU (no 2^H x H matrix)."""
    from qest.qsota import DEV
    H = len(v); idx = torch.arange(1 << H, device=DEV); E = torch.zeros(1 << H, device=DEV, dtype=torch.float64); S = torch.zeros_like(E)
    for h in range(H):
        s = (2 * ((idx >> h) & 1) - 1).double(); E -= v[h] * s; S += s
    return E + mu * S ** 2


def lr_qaoa(Es, gam, bet):
    """Plain QAOA state from |+>^H, RX mixer, given angles: only the 2^H state vector in memory."""
    N = Es.shape[0]; H = N.bit_length() - 1; psi = torch.full((N,), N ** -.5, dtype=torch.complex64, device=Es.device)
    for g_, b_ in zip(gam, bet):
        psi = psi * torch.exp(-1j * g_ * Es); c, s = np.cos(b_), -1j * np.sin(b_)
        for q in range(H):
            v = psi.view(N >> (q + 1), 2, 1 << q); a0, a1 = v[:, 0], v[:, 1]
            psi = torch.stack([c * a0 + s * a1, s * a0 + c * a1], 1).reshape(N)
    return psi


def patterns(u, px, mode, seed=0):
    from qest.qsota import DEV
    n = len(px['da']); H = -(-n // 4); p = np.array([np.mean(px['da'][4 * h:4 * h + 4]) for h in range(H)])
    v = u.p_mw * (p - p.mean()); mu = np.abs(v).mean() / 4; g = torch.Generator(device=DEV).manual_seed(seed)
    if mode == 'random':
        return [((np.int64(z) >> np.arange(H)) & 1) for z in torch.randint(0, 1 << H, (K,), device=DEV, generator=g).tolist()]
    E = energies(v, mu)
    if mode == 'surrogate':
        z = torch.topk(-E, K).indices
    else:
        Es = ((E - E.mean()) / E.std()).float(); ramp = (torch.arange(P_LAYERS) + .5) / P_LAYERS
        with torch.no_grad():
            pr = lr_qaoa(Es, (-DG * ramp).tolist(), (DB * (1 - ramp)).tolist()).abs() ** 2  # -: |+> tops the RX mixer, anneal to the cost minimum
            cdf = torch.cumsum(pr.double(), 0); u_ = torch.rand(SHOTS, device=DEV, generator=g, dtype=torch.float64) * cdf[-1]
            shots = torch.unique(torch.searchsorted(cdf, u_).clamp(max=len(cdf) - 1)); del cdf  # multinomial stops at 2^24 categories
        z = shots[torch.argsort(E[shots])[:K]]; del pr
    del E
    return [((int(x) >> np.arange(H)) & 1) for x in z.tolist()]


def plan(mode, units, ctx, cfg):
    out = _renewables(units, ctx, cfg, use_real=False); px = prices_for_milp(ctx)
    if cfg.official:
        from qest.market import grid_tariff
        px['tariff'] = grid_tariff(ctx.idx)
    n = len(px['da'])
    for u in units:
        if u.kind in ('battery', 'hybrid'):
            pv = ctx.gen_fc.get(u.name) if u.kind == 'hybrid' else None
            best = None
            for b in patterns(u, px, mode):
                pl = plan_storage(u, px, pv, depth=cfg.cap_depth, fixed={'y2': np.repeat(b, 4)[:n]})
                if best is None or pl['objective'] > best['objective']:
                    best = pl
            best.update(margins=ctx.calib['margins'], soc_ref=best['soc'], policy='soc_aware', tol=ctx.calib.get('tol', .05))
            out[u.name] = best
    return out


def main():
    from qest import market
    from qest.evaluate import run, table
    from qest.paths import RESULTS
    ap = argparse.ArgumentParser(); ap.add_argument('--split', default='test')
    ap.add_argument('--strategies', default='milp_chronos,qday_qaoa,qday_surrogate,qday_random,milp_chronos_qrc,milp_chronos_rff'); a = ap.parse_args()
    cfg = market.Config(); units = market.portfolio(['battery', 'hybrid'], official=cfg.official)
    R, tar = run(a.split, a.strategies.split(','), units, cfg)
    R.to_parquet(RESULTS / f'days_{a.split}_hybrid.parquet'); t = table(R, tar); t.to_csv(RESULTS / f'summary_{a.split}_hybrid.csv')
    print(t[['total', 'cap', 'da', 'mfrr', 'afrr', 'imbalance', 'imb_mwh']].round(0).to_string())


def _check():
    """The surrogate's minimum by enumeration equals the brute-force minimum on a small instance."""
    v = np.array([3., -1., 2., -4., .5]); mu = .7; E = energies(v, mu).cpu().numpy()
    bf = [-(v * (2 * ((z >> np.arange(5)) & 1) - 1)).sum() + mu * (2 * ((z >> np.arange(5)) & 1) - 1).sum() ** 2 for z in range(32)]
    assert np.allclose(E, bf); print('qday check ok')


if __name__ == '__main__':
    import sys
    _check() if '--check' in sys.argv else main()

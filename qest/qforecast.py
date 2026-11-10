r"""Quantum reservoir forecaster: a standalone competitor to Chronos-2 on the same gate-A information (no Chronos input).

Method (the 2026 QRC line, research/methods.csv S1 short-term load QRC arXiv:2606.12806, S2 ResQue QRC weather, GIC
2026): a FIXED quantum dynamics maps the inputs to features and only a linear readout is trained, so nothing is
optimised on the QPU. Per target and MTU, 10 qubits carry the gate-A inputs (last known value at that MTU, the day
before it, a week before, the last known day mean and max, hour sin / cos, weekend, the D-1 national wind and PV
forecasts), each scaled to [0, pi] and loaded as RY(x_i). The reservoir is a Trotterised transverse-field Ising
evolution (ResQue's): H = sum_{i<j} J_ij Z_i Z_j + sum_i h_i X_i, random J ~ U(-1, 1), h ~ U(.5, 1.5), dt = .5,
read out after 1, 2 and 3 steps: <Z_i>, <Z_i Z_j> (55 per snapshot, 165 + the 10 raw inputs), estimated from SHOTS
measurement shots per circuit as on hardware. Readout: ridge on the q50; q10 / q90 = q50 + the 10 / 90 % residual
quantiles per hour of day (in-sample).
Simulator: the circuits are CUDA-Q kernels (target 'nvidia', cuStateVec); the fixed evolution's unitary comes from
cudaq.get_unitary and is applied to the batch of product states on the GPU (exactly the kernel's state, _check), so
~30k MTUs x 11 targets x 3 snapshots run in seconds. On hardware ('iqm' / 'ibm' targets) run `kernel` with shots.
Controls, same readout: 'linear' (the 10 raw inputs), 'rff' (raw + 165 random Fourier features, the classical twin of a
random reservoir). Hybrid 'ens': Chronos-2 and QRC averaged quantile by quantile, the weight per target picked on Q3; 'ens_rff' the same
with the RFF control (is the blend's gain quantum, or just averaging two models?).
Walk-forward as qest.forecast: val = fit Q1-Q2, predict Q3; test = refit Q1-Q3, predict Q4; ridge alpha chosen on Q3.
usage: CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m qest.qforecast [--check]
  -> results/forecast_{qrc,rff,linear,ens,ens_rff}_{val,test}.parquet, results/qforecast_skill.csv
"""
import warnings

import numpy as np
import pandas as pd

from qest.data import days, frame
from qest.forecast import LAG_KNOWN, TARGETS, features
from qest.paths import RESULTS

warnings.filterwarnings('ignore')
NQ, DT, SNAPS, SHOTS = 10, .5, (1, 2, 3), 1000
RNG = np.random.default_rng(0)
J = np.triu(RNG.uniform(-1, 1, (NQ, NQ)), 1); HX = RNG.uniform(.5, 1.5, NQ)
NF = NQ + NQ * (NQ - 1) // 2
RFF_W = RNG.normal(0, 1, (NQ, NF * len(SNAPS))); RFF_B = RNG.uniform(0, 2 * np.pi, NF * len(SNAPS))
ALPHAS = (1e-2, 1e-1, 1, 10, 100)
# Partitioned reservoir (Daedalus, arXiv:2611.05735 / qest.qpart): 32 inputs cut into overlapping windows of NQ, every
# window through the SAME frozen 10-qubit reservoir (the shared pre-trained encoder), the ridge couples them (processor).
PART_O = 2
P_RFF = np.random.default_rng(1)


def _cudaq():
    import cudaq
    try:
        cudaq.set_target('nvidia')
    except RuntimeError:
        cudaq.set_target('qpp-cpu')
    return cudaq


def kernels():
    """CUDA-Q kernels: `evolve` (the fixed reservoir, steps Trotter steps) and `kernel` (encoding + reservoir, the
    circuit a QPU runs with shots)."""
    cudaq = _cudaq()

    @cudaq.kernel
    def evolve(J: list[float], hx: list[float], steps: int, dt: float):
        q = cudaq.qvector(10)
        for s in range(steps):
            for i in range(10):
                for j in range(i + 1, 10):
                    x.ctrl(q[i], q[j]); rz(2.0 * dt * J[i * 10 + j], q[j]); x.ctrl(q[i], q[j])
            for i in range(10):
                rx(2.0 * dt * hx[i], q[i])

    @cudaq.kernel
    def kernel(a: list[float], J: list[float], hx: list[float], steps: int, dt: float):
        q = cudaq.qvector(10)
        for i in range(10):
            ry(a[i], q[i])
        for s in range(steps):
            for i in range(10):
                for j in range(i + 1, 10):
                    x.ctrl(q[i], q[j]); rz(2.0 * dt * J[i * 10 + j], q[j]); x.ctrl(q[i], q[j])
            for i in range(10):
                rx(2.0 * dt * hx[i], q[i])
    return cudaq, evolve, kernel


def _zbits(msb=False):
    """Z eigenvalues per basis index. CUDA-Q's get_state puts qubit 0 in the least significant bit, get_unitary in the
    most significant one (tested), so the batched path (unitary) uses msb=True."""
    idx = np.arange(1 << NQ); sh = NQ - 1 - np.arange(NQ) if msb else np.arange(NQ)
    return 1 - 2 * ((idx[:, None] >> sh) & 1)


def reservoir(A, shots=SHOTS, seed=0):
    """A (B, 10) angles -> (B, 165) features. Product states from the RY encoding, the fixed unitaries from CUDA-Q."""
    import torch
    cudaq, evolve, _ = kernels()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    U = [torch.as_tensor(np.array(cudaq.get_unitary(evolve, J.ravel().tolist(), HX.tolist(), s, DT)), device=dev) for s in SNAPS]
    z = torch.as_tensor(_zbits(msb=True), dtype=torch.float64, device=dev)
    pairs = [(i, j) for i in range(NQ) for j in range(i + 1, NQ)]
    zz = torch.stack([z[:, i] * z[:, j] for i, j in pairs], 1)
    O = torch.cat([z, zz], 1)  # (2^n, 55) diagonal observables
    gen = torch.Generator(device=dev).manual_seed(seed); out = []
    for b in range(0, len(A), 4096):
        a = torch.as_tensor(A[b:b + 4096], dtype=torch.float64, device=dev)
        c, s = torch.cos(a / 2), torch.sin(a / 2); psi = torch.ones(len(a), 1, dtype=torch.float64, device=dev)
        for i in range(NQ):  # kron with qubit 0 first: the most significant bit, as get_unitary
            psi = torch.einsum('bi,bj->bij', psi, torch.stack([c[:, i], s[:, i]], 1)).reshape(len(a), -1)
        feats = []
        for Ut in U:
            p = (psi.to(torch.complex128) @ Ut.T).abs() ** 2
            if shots:
                p = torch.distributions.Multinomial(shots, probs=p.clamp(min=0)).sample() / shots if dev == 'cpu' else \
                    torch.zeros_like(p).scatter_add_(1, torch.multinomial(p, shots, replacement=True, generator=gen), torch.ones(len(p), shots, dtype=p.dtype, device=dev)) / shots
            feats.append(p @ O)
        out.append(torch.cat(feats, 1).cpu().numpy())
    return np.concatenate(out)


def inputs(k, X):
    L = LAG_KNOWN.get(k, 2); h = X.hour
    return np.c_[X[f'{k}_lag{L}'], X[f'{k}_lag{L + 1}'], X[f'{k}_lag7'], X[f'{k}_daymean_lag'], X[f'{k}_daymax_lag'],
                 np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), (X.dow >= 5).astype(float),
                 X.fleet_wind_d1 if 'fleet_wind_d1' in X else 0 * h, X.pv_d1]


def inputs_wide(k, X):
    """The 10 own inputs (window 0 = the plain reservoir) + every other target's last known value and week lag + the
    D-1 national wind forecast and consumption: 32 inputs, 4 windows of 10 qubits."""
    oth = [t for t in TARGETS if t != k]
    return np.c_[inputs(k, X), np.column_stack([X[f'{t}_lag{LAG_KNOWN.get(t, 2)}'] for t in oth] + [X[f'{t}_lag7'] for t in oth]),
                 X.wind_d1, X.cons_lag2]


def design(Z, lo, hi, kind, cache):
    A = np.pi * np.clip((np.nan_to_num(Z, nan=0.0) - lo) / (hi - lo + 1e-12), 0, 1)
    if kind == 'linear':
        return A
    if kind == 'rff':
        return np.c_[A, np.cos(A @ RFF_W + RFF_B)]
    if kind in ('pqrc', 'prff'):
        from qest.qpart import windows
        W = windows(A.shape[1], NQ, PART_O); nf = len(W) * NF * len(SNAPS)
        if kind == 'prff':  # classical twin: as many random Fourier features over the same 32 inputs
            if ('prff', A.shape[1]) not in cache:
                cache[('prff', A.shape[1])] = (P_RFF.normal(0, 1, (A.shape[1], nf)), P_RFF.uniform(0, 2 * np.pi, nf))
            w, b = cache[('prff', A.shape[1])]; return np.c_[A, np.cos(A @ w + b)]
        feats = []
        for S in W:
            a = np.ascontiguousarray(A[:, S]); key = a.tobytes().__hash__()
            if key not in cache:
                cache[key] = reservoir(a)
            feats.append(cache[key])
        return np.concatenate([A] + feats, 1)
    key = A.tobytes().__hash__()
    if key not in cache:
        cache[key] = reservoir(A)
    return np.c_[A, cache[key]]


def fit_predict(k, X, y, fit_m, pred_m, kind, alpha, cache):
    from sklearn.linear_model import Ridge
    Z = (inputs_wide if kind in ('pqrc', 'prff') else inputs)(k, X); ok = fit_m & np.isfinite(y)
    lo, hi = np.nanpercentile(Z[ok], 1, 0), np.nanpercentile(Z[ok], 99, 0)
    D = design(Z[ok | pred_m], lo, hi, kind, cache); sub = np.flatnonzero(ok | pred_m)
    Dfit, Dpred = D[np.isin(sub, np.flatnonzero(ok))], D[np.isin(sub, np.flatnonzero(pred_m))]
    mu, sd = Dfit.mean(0), Dfit.std(0) + 1e-9; ylo, yhi = np.quantile(y[ok], [.005, .995])
    m = Ridge(alpha=alpha).fit((Dfit - mu) / sd, np.clip(y[ok], ylo, yhi))
    q50 = m.predict((Dpred - mu) / sd); res = y[ok] - m.predict((Dfit - mu) / sd)
    hr = X.hour.to_numpy(); qs = {q: pd.Series(res).groupby(hr[ok]).quantile(q / 100) for q in (10, 90)}
    out = {f'{k}_q50': q50}
    for q in (10, 90):
        out[f'{k}_q{q}'] = q50 + qs[q].reindex(hr[pred_m]).to_numpy()
    if k.endswith('_s'):
        out = {c: np.clip(v, 0, 1) for c, v in out.items()}
    return out


def main(kinds=('qrc', 'rff', 'linear'), tag=''):
    f = frame(); X = features(); tr, va, te = days(f, 'train'), days(f, 'val'), days(f, ('train', 'val'))
    m_tr, m_va, m_te = f.day.isin(tr).to_numpy(), f.day.isin(va).to_numpy(), f.day.isin(days(f, 'test')).to_numpy()
    out = {kind: ({}, {}) for kind in kinds}; rows = []; cache = {}
    for k, col in TARGETS.items():
        y = f[col].to_numpy(); r = dict(target=k)
        for kind in out:
            best = None
            for a in ALPHAS:  # fit Q1-Q2, score Q3
                p = fit_predict(k, X, y, m_tr, m_va, kind, a, cache); e = np.nanmean(np.abs(p[f'{k}_q50'] - y[m_va]))
                if best is None or e < best[1]:
                    best = (a, e, p)
            out[kind][0].update(best[2]); pt = fit_predict(k, X, y, m_tr | m_va, m_te, kind, best[0], cache); out[kind][1].update(pt)
            r[f'{kind}_q3'] = best[1]; r[f'{kind}_q4'] = np.nanmean(np.abs(pt[f'{k}_q50'] - y[m_te])); r[f'{kind}_alpha'] = best[0]
        rows.append(r); print({a: round(b, 3) if isinstance(b, float) else b for a, b in r.items()}, flush=True)
    for kind, (v, t) in out.items():
        pd.DataFrame(v, index=f.index[m_va]).to_parquet(RESULTS / f'forecast_{kind}_val.parquet')
        pd.DataFrame(t, index=f.index[m_te]).to_parquet(RESULTS / f'forecast_{kind}_test.parquet')
    for kind in kinds[:2]:
        ensemble(kind, rows)
    S = pd.DataFrame(rows); S.to_csv(RESULTS / f'qforecast_skill{tag}.csv', index=False)
    pd.set_option('display.width', 250); print(S[['target'] + [c for c in S if c.endswith('_q4')] + [c for c in S if c.endswith('_w_chronos')]].round(3).to_string())


def ensemble(partner, rows):
    """Chronos-2 alongside a partner forecaster, quantile by quantile, w = Chronos-2's share per target picked on Q3.
    partner 'qrc' -> forecast_ens, 'rff' -> forecast_ens_rff (its classical control)."""
    from qest.evaluate import forecasts
    f = frame(); tag = 'ens' if partner == 'qrc' else f'ens_{partner}'
    cv, ct, qv, qt = forecasts('val', 'chronos'), forecasts('test', 'chronos'), forecasts('val', partner), forecasts('test', partner)
    ev, et = pd.DataFrame(index=cv.index), pd.DataFrame(index=ct.index)
    for r, (k, col) in zip(rows, TARGETS.items()):
        yv = f[col].reindex(cv.index); qa = qv[f'{k}_q50'].reindex(cv.index)
        w = min(np.linspace(0, 1, 11), key=lambda w: np.nanmean(np.abs(w * cv[f'{k}_q50'] + (1 - w) * qa - yv)))
        for q in (10, 50, 90):
            c = f'{k}_q{q}'; ev[c] = w * cv[c] + (1 - w) * qv[c].reindex(cv.index); et[c] = w * ct[c] + (1 - w) * qt[c].reindex(ct.index)
        yt = f[col].reindex(ct.index)
        r.update({'chronos_q3': np.nanmean(np.abs(cv[f'{k}_q50'] - yv)), 'chronos_q4': np.nanmean(np.abs(ct[f'{k}_q50'] - yt)),
                  f'{tag}_w_chronos': w, f'{tag}_q4': np.nanmean(np.abs(et[f'{k}_q50'] - yt))})
    ev.to_parquet(RESULTS / f'forecast_{tag}_val.parquet'); et.to_parquet(RESULTS / f'forecast_{tag}_test.parquet')


def _check():
    """The batched GPU path equals the CUDA-Q kernel's own state (exact, no shots), and features lie in [-1, 1]."""
    cudaq, _, kernel = kernels()
    A = np.random.default_rng(1).random((3, NQ)) * np.pi; F = reservoir(A, shots=0); z = _zbits()
    for b in range(3):
        for t, s in enumerate(SNAPS):
            p = np.abs(np.array(cudaq.get_state(kernel, A[b].tolist(), J.ravel().tolist(), HX.tolist(), s, DT))) ** 2
            assert np.allclose(p @ z, F[b, t * NF:t * NF + NQ], atol=1e-6), (p @ z, F[b, t * NF:t * NF + NQ])
    assert np.abs(reservoir(A)).max() <= 1 + 1e-9
    print('qforecast check ok: batched reservoir = CUDA-Q kernel state')


if __name__ == '__main__':
    import sys
    if '--check' in sys.argv:
        _check()
    elif '--ens' in sys.argv:  # blends only, from the cached forecasts
        R = [dict(target=k) for k in TARGETS]; ensemble('qrc', R); ensemble('rff', R); S = pd.DataFrame(R)
        S.to_csv(RESULTS / 'qforecast_ens.csv', index=False); print(S.round(3).to_string())
    elif '--part' in sys.argv:  # partitioned reservoir + its classical twin, own output files
        main(('pqrc', 'prff'), '_part')
    else:
        main()

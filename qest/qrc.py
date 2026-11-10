r"""Hybrid forecaster: Chronos-2 (the classical baseline) + a quantum reservoir that corrects its median.

Quantum reservoir computing (QRC; for short-term load, Od, Pathak, Innan & Shafique, arXiv:2606.12806, Jun 2026):
a FIXED random circuit maps each MTU's gate-A inputs to high-dimensional features, and only a classical Elastic Net
readout is trained, so nothing is optimised on the QPU (hardware: one Z-basis readout per input, shots -> <Z>, <ZZ>).
Per target and MTU, 8 qubits carry: Chronos-2's q50 and its q10-q90 width, q50 one MTU before and after, the last
published value at that MTU (lag), the hour (sin, cos), the D-1 national wind forecast. Circuit, twice (re-uploading):
RY(x_i) on qubit i, then 3 fixed layers of random RX RZ + a CZ ring. Features: <Z_i>, <Z_i Z_j> (8 + 28).
The readout learns the residual y - q50 (q10 and q90 move with it). Controls, same readout: the 8 raw inputs alone
(linear) and raw + 36 random Fourier features (RFF, the classical twin of a random reservoir).
Fit on Q3 (validation, Chronos-2 run on it zero-shot); the variant per target is chosen on Q3's last third; Q4 once.
usage: .venv/bin/python -m qest.qrc   -> results/forecast_chronos_{qrc,rff}_{val,test}.parquet, results/qrc_skill.csv
"""
import math

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import ElasticNet

from qest.data import days, frame
from qest.evaluate import forecasts
from qest.forecast import TARGETS, LAG_KNOWN, features
from qest.paths import RESULTS

NQ, LAYERS, REUP = 8, 3, 2
G = torch.Generator().manual_seed(0)
ANG = torch.rand(REUP, LAYERS, NQ, 2, generator=G, dtype=torch.float64) * 2 * math.pi  # the fixed reservoir
RFF_W = torch.randn(NQ, NQ * (NQ + 1) // 2, generator=G, dtype=torch.float64); RFF_B = torch.rand(RFF_W.shape[1], generator=G, dtype=torch.float64) * 2 * math.pi


def _rot(psi, q, U):  # U (2, 2) or (B, 2, 2) on qubit q of psi (B, 2^n)
    B = psi.shape[0]; v = psi.view(B, 1 << (NQ - q - 1), 2, 1 << q)
    U = U.expand(B, 2, 2) if U.ndim == 2 else U
    return torch.einsum('bij,bajc->baic', U, v).reshape(B, -1)


def _ry(t):
    c, s = torch.cos(t / 2), torch.sin(t / 2); return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2).to(torch.complex128)


def _rx(t):
    c, s = torch.cos(t / 2), -1j * torch.sin(t / 2); return torch.stack([torch.stack([c + 0j, s], -1), torch.stack([s, c + 0j], -1)], -2)


def _rz(t):
    e = torch.exp(-.5j * t); z = torch.zeros_like(e); return torch.stack([torch.stack([e, z], -1), torch.stack([z, e.conj()], -1)], -2)


def reservoir(X):
    """X (B, 8) angles in [0, pi] -> (B, 36): <Z_i>, <Z_i Z_j> of the fixed circuit."""
    X = torch.as_tensor(X, dtype=torch.float64); B = len(X); N = 1 << NQ
    idx = torch.arange(N); bits = ((idx[:, None] >> torch.arange(NQ)) & 1); z = (1 - 2 * bits).double()
    cz = torch.ones(N, dtype=torch.complex128)
    for q in range(NQ):
        cz = cz * torch.where((bits[:, q] & bits[:, (q + 1) % NQ]).bool(), -1, 1)
    psi = torch.zeros(B, N, dtype=torch.complex128); psi[:, 0] = 1
    for r in range(REUP):
        for q in range(NQ):
            psi = _rot(psi, q, _ry(X[:, q]))
        for l in range(LAYERS):
            for q in range(NQ):
                psi = _rot(psi, q, _rz(ANG[r, l, q, 1]) @ _rx(ANG[r, l, q, 0]))
            psi = psi * cz
    p = psi.abs() ** 2; ez = p @ z
    ezz = torch.stack([p @ (z[:, i] * z[:, j]) for i in range(NQ) for j in range(i + 1, NQ)], 1)
    return torch.cat([ez, ezz], 1).numpy()


def rff(X):
    return np.cos(np.asarray(X) @ RFF_W.numpy() + RFF_B.numpy())


def inputs(k, fc, f, X):
    """The 8 gate-A inputs of target k per MTU (raw, unscaled)."""
    q50 = fc[f'{k}_q50']; d = f.day.reindex(fc.index)
    prev = q50.groupby(d).shift(1).fillna(q50); nxt = q50.groupby(d).shift(-1).fillna(q50)
    lag = X[f'{k}_lag{LAG_KNOWN.get(k, 2)}'].reindex(fc.index).fillna(q50)
    h = X.hour.reindex(fc.index) if 'hour' in X else pd.Series(fc.index.hour, index=fc.index)
    return np.c_[q50, fc[f'{k}_q90'] - fc[f'{k}_q10'], prev, nxt, lag, np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                 X.fleet_wind_d1.reindex(fc.index).fillna(0) if 'fleet_wind_d1' in X else 0 * q50]


def design(Z, lo, hi, kind):
    A = np.clip((Z - lo) / (hi - lo + 1e-12), 0, 1) * np.pi
    if kind == 'linear':
        return A
    return np.c_[A, reservoir(A) if kind == 'qrc' else rff(A)]


def fit(D, y, alpha):
    return ElasticNet(alpha=alpha, l1_ratio=.5, max_iter=5000).fit(D, y)


def main():
    f = frame(); X = features(); fv, ft = forecasts('val', 'chronos'), forecasts('test', 'chronos')
    vd = days(f, 'val'); cut = vd[int(len(vd) * 2 / 3)]; rows = []; out = {kind: (fv.copy(), ft.copy()) for kind in ('qrc', 'rff')}
    for k, col in TARGETS.items():
        Zv, Zt = inputs(k, fv, f, X), inputs(k, ft, f, X)
        yv = (f[col].reindex(fv.index) - fv[f'{k}_q50']).to_numpy(); yt = (f[col].reindex(ft.index) - ft[f'{k}_q50']).to_numpy()
        ok = np.isfinite(yv) & np.isfinite(Zv).all(1); okt = np.isfinite(yt)
        inner = (f.day.reindex(fv.index) < cut).to_numpy()
        lo, hi = np.nanpercentile(Zv[ok & inner], 1, 0), np.nanpercentile(Zv[ok & inner], 99, 0)
        sc = np.nanstd(yv[ok]); r = dict(target=k, chronos_q3_last=np.abs(yv[ok & ~inner]).mean(), chronos_q4=np.abs(yt[okt]).mean())
        for kind in ('linear', 'rff', 'qrc'):
            Dv, Dt = design(Zv, lo, hi, kind), design(Zt, lo, hi, kind)
            best = min(((a, np.abs(yv[ok & ~inner] - sc * fit(Dv[ok & inner], yv[ok & inner] / sc, a).predict(Dv[ok & ~inner])).mean())
                        for a in (1e-3, 1e-2, 1e-1)), key=lambda x: x[1])
            m = fit(Dv[ok], yv[ok] / sc, best[0]); cv, ct = sc * m.predict(Dv), sc * m.predict(Dt)
            r[f'{kind}_q3_last'] = best[1]; r[f'{kind}_q4'] = np.abs(yt[okt] - ct[okt]).mean()
            use = best[1] < r['chronos_q3_last']  # guard: correct only where it helped on Q3's held-out third
            r[f'{kind}_used'] = bool(use)
            if kind in out and use:
                for fc_, c_ in ((out[kind][0], cv), (out[kind][1], ct)):
                    for q in (10, 50, 90):
                        fc_[f'{k}_q{q}'] = fc_[f'{k}_q{q}'] + c_
                    if k.endswith('_s'):
                        fc_[[f'{k}_q{q}' for q in (10, 50, 90)]] = fc_[[f'{k}_q{q}' for q in (10, 50, 90)]].clip(0, 1)
        rows.append(r); print({a: (round(b, 3) if isinstance(b, float) else b) for a, b in r.items()}, flush=True)
    for kind, (v, t) in out.items():
        v.to_parquet(RESULTS / f'forecast_chronos_{kind}_val.parquet'); t.to_parquet(RESULTS / f'forecast_chronos_{kind}_test.parquet')
    S = pd.DataFrame(rows); S.to_csv(RESULTS / 'qrc_skill.csv', index=False); print(S.round(3).to_string())


def _check():
    """The reservoir is a valid state map: <Z> and <ZZ> in [-1, 1], deterministic, and input-dependent."""
    A = np.random.default_rng(0).random((4, NQ)) * np.pi; F1, F2 = reservoir(A), reservoir(A)
    assert F1.shape == (4, 36) and np.abs(F1).max() <= 1 + 1e-9 and np.allclose(F1, F2) and np.ptp(F1, 0).max() > 1e-3
    print('qrc check ok')


if __name__ == '__main__':
    import sys
    _check() if '--check' in sys.argv else main()

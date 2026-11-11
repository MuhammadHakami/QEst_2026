"""A conditional quantum circuit Born machine as the day's activation-scenario generator (quantum generative model).

Why: the plan's euros come from scenarios, not from the mean (research/quantum_gain_2026.md s1); every classical point
forecast here carries 30-60 % of the realised spread (s7). The classical generator of record is a logistic Markov chain
(qest.qscen). This is its quantum counterpart: a parameterised circuit on 16 qubits whose measurement distribution
p(z | c) = |<z|U(theta, c)|0>|^2 is the joint law of a window of W = 8 MTUs x (up, down) bits, conditioned on c = the previous
state (one-hot 4), the window's hour block (one-hot 12) and the driver summary (mean actfc logits of p_up / p_dn in the
window, the lagged activation rates). Trained by exact maximum likelihood on the statevector (65,536 amplitudes, batched):
on a QPU the same circuit is trained from shots (MMD / NLL estimates) and sampled by measurement. Walk-forward: refitted
before each month on windows before it (D-2 lag), as the chain. Sampling a day: z_0 ~ pi0, then window by window,
each conditioned on the last state of the previous window, the last window re-samples the tail (as qscen_q.sample_day).

Controls: the chain's held-out log-likelihood per MTU (qest.qscen) on the same windows; UP rate and UP run length against
the data; euros through the scenario plan (`milp_scenf_qg`, strategies) against the exact chain scenarios.
usage: CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m qest.qgen [--bench] -> results/qgen_bench.csv; params in results/qgen_params/
"""
import argparse
import math
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch

from qest import qscen
from qest.data import frame
from qest.paths import RESULTS

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
W, NQ, LAYERS = 8, 16, 4
COND = 4 + 12 + 4  # previous state one-hot, hour block one-hot, drivers (mean logit p_up, p_dn, lagged up / dn rates)
_P = {}


class Born(torch.nn.Module):
    """U(theta, c) = prod_l [ CZ ring . prod_q RY(a_lq(c)) RZ(b_lq) RY(w_lq) ] on |0>; a(c) = linear map of the condition."""
    def __init__(self, nc=COND, nq=NQ, layers=LAYERS):
        super().__init__(); self.nq, self.L = nq, layers
        self.enc = torch.nn.Linear(nc, layers * nq); self.w = torch.nn.Parameter(torch.randn(layers, nq, 2) * .1)
        idx = torch.arange(1 << nq); self.register_buffer('bits', ((idx[:, None] >> torch.arange(nq)) & 1).float())
        cz = torch.ones(1 << nq)
        for q in range(nq):
            cz = cz * torch.where((self.bits[:, q] * self.bits[:, (q + 1) % nq]).bool(), -1.0, 1.0)
        self.register_buffer('cz', cz)

    def _ry(self, psi, q, th):
        B = psi.shape[0]; N = 1 << self.nq; c, s = torch.cos(th / 2)[:, None, None], torch.sin(th / 2)[:, None, None]
        v = psi.view(B, N >> (q + 1), 2, 1 << q); a0, a1 = v[:, :, 0], v[:, :, 1]
        return torch.stack([c * a0 - s * a1, s * a0 + c * a1], 2).reshape(B, N)

    def _rz(self, psi, q, th):
        ph = torch.exp(-.5j * th)[:, None]; return torch.where(self.bits[None, :, q].bool(), psi * ph.conj(), psi * ph)

    def probs(self, c):
        """(B, 2^16) measurement distribution for conditions c (B, nc)."""
        B = c.shape[0]; N = 1 << self.nq; ang = torch.tanh(self.enc(c)).view(B, self.L, self.nq) * math.pi
        psi = torch.zeros(B, N, dtype=torch.complex64, device=c.device); psi[:, 0] = 1
        for l in range(self.L):
            for q in range(self.nq):
                psi = self._ry(psi, q, ang[:, l, q]); psi = self._rz(psi, q, self.w[l, q, 0].expand(B)); psi = self._ry(psi, q, self.w[l, q, 1].expand(B))
            psi = psi * self.cz
        return psi.abs() ** 2


def _code(z):
    """window states z (.., W) in 0..3 -> 16-bit integer (MTU k's state in bits 2k, 2k+1), as qest.qscen_q."""
    return sum(np.asarray(z)[..., k].astype(np.int64) << (2 * k) for k in range(W))


def _windows(j, X):
    """Training windows of a joined frame: (conditions (n, COND), codes (n,), day) for every block of W MTUs with a previous MTU."""
    z = qscen._states(j); days = j.day.to_numpy(); hr = j.local.dt.hour.to_numpy()
    pu = np.nan_to_num(X.pu.to_numpy(float)); pdn = np.nan_to_num(X.pd.to_numpy(float)); up2, dn2 = X.up2.to_numpy(float), X.dn2.to_numpy(float)
    C, Z, D = [], [], []
    first = np.r_[True, days[1:] != days[:-1]]; starts = np.flatnonzero(first)
    for s0, s1 in zip(starts, np.r_[starts[1:], len(days)]):
        T = s1 - s0
        for t0 in range(0, T - 1 - W + 1, W):
            a, b = s0 + t0, s0 + t0 + 1
            c = np.zeros(COND); c[z[a]] = 1; c[4 + hr[b] // 2] = 1; c[16] = pu[b:b + W].mean(); c[17] = pdn[b:b + W].mean(); c[18] = up2[b]; c[19] = dn2[b]
            C.append(c); Z.append(_code(z[b:b + W])); D.append(days[a])
    return np.array(C, np.float32), np.array(Z), np.array(D)


def _cond(z0, hour, pu, pdn, up2, dn2):
    c = np.zeros((len(z0), COND), np.float32); c[np.arange(len(z0)), z0] = 1; c[np.arange(len(z0)), 4 + hour // 2] = 1
    c[:, 16], c[:, 17], c[:, 18], c[:, 19] = pu, pdn, up2, dn2; return c


def fit(before=None, epochs=40, lr=3e-3, bs=128, seed=0, half_life=60.0, splits=('train', 'val', 'test')):
    from qest.evaluate import forecasts
    f = frame(); F = pd.concat([forecasts(s, os.environ.get('QEST_FC', 'chronos')) for s in splits]); j = f.join(F, how='inner'); j = j[j.split.isin(splits)]
    if before is not None:
        j = j[j.day < pd.Timestamp(before) - pd.Timedelta(days=2)]
    X = qscen.features(j, j.index); C, Z, D = _windows(j, X)
    if len(Z) < 200:
        raise ValueError('too few windows')
    age = (D.max() - D) / np.timedelta64(1, 'D'); w = (.5 ** (age / half_life)).astype(np.float32)
    torch.manual_seed(seed); m = Born().to(DEV); opt = torch.optim.Adam(m.parameters(), lr=lr)
    Ct, Zt, Wt = (torch.as_tensor(a, device=DEV) for a in (C, Z, w)); n = len(Z); g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g); tot = 0.0
        for i in range(0, n, bs):
            b = perm[i:i + bs]; p = m.probs(Ct[b]); nll = -torch.log(p[torch.arange(len(b)), Zt[b]].clamp(min=1e-12))
            loss = (nll * Wt[b]).sum() / Wt[b].sum(); opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(b)
    sd = {k: v.detach().cpu() for k, v in m.state_dict().items()}
    mu = qscen.params(None if before is None else before)['pi0'] if False else None
    z = qscen._states(j); days = j.day.to_numpy(); first = np.r_[True, days[1:] != days[:-1]]; pi0 = np.bincount(z[first], minlength=4) / first.sum()
    return dict(state=sd, pi0=pi0, n_windows=int(n), nll_train=tot / n)


def params(day=None):
    key = ('all' if day is None else pd.Timestamp(day).strftime('%Y-%m')) + ('' if os.environ.get('QEST_FC', 'chronos') == 'chronos' else '_' + os.environ['QEST_FC'])
    if key not in _P:
        d = RESULTS / 'qgen_params'; d.mkdir(exist_ok=True); p = d / f'{key}.pkl'
        if p.exists():
            _P[key] = pickle.loads(p.read_bytes())
        else:
            if day is None:
                _P[key] = fit(None)
            else:
                m0 = pd.Timestamp(day).replace(day=1)
                for k in range(12):
                    try:
                        _P[key] = fit(m0 + pd.DateOffset(months=k)); break
                    except ValueError:
                        continue
                if k:
                    print(f'qgen: {key} fitted on data to {(m0 + pd.DateOffset(months=k)).date()} (no earlier history)', flush=True)
            p.write_bytes(pickle.dumps(_P[key]))
        m = Born().to(DEV); m.load_state_dict(_P[key]['state']); _P[key]['model'] = m.eval()
    return _P[key]


def sample_day(P, X, n, seed=0):
    """n day patterns (n, T, 2) from the Born machine, window by window."""
    T = len(X); rng = np.random.default_rng(seed); g = torch.Generator(device=DEV).manual_seed(seed); z = np.zeros((n, T), int); z[:, 0] = rng.choice(4, n, p=P['pi0'])
    hr = X.index.tz_convert('Europe/Tallinn').hour.to_numpy(); pu = np.nan_to_num(X.pu.to_numpy(float)); pdn = np.nan_to_num(X.pd.to_numpy(float)); up2, dn2 = X.up2.to_numpy(float), X.dn2.to_numpy(float)
    starts = list(range(0, T - 1 - W, W)); starts.append(T - 1 - W); m = P['model']
    with torch.no_grad():
        for t0 in starts:
            b = t0 + 1; c = _cond(z[:, t0], np.full(n, hr[b]), np.full(n, pu[b:b + W].mean()), np.full(n, pdn[b:b + W].mean()), np.full(n, up2[b]), np.full(n, dn2[b]))
            p = m.probs(torch.as_tensor(c, device=DEV)); code = torch.multinomial(p, 1, generator=g)[:, 0].cpu().numpy()
            for k in range(W):
                z[:, b + k] = (code >> (2 * k)) & 3
    return qscen.STATES[z]


def bench(split='val', n=64, seed=0):
    """Held-out windows of the split: log-likelihood per MTU of the Born machine against the chain (same windows, both
    walk-forward), and day statistics against the data. -> results/qgen_bench.csv"""
    from qest.evaluate import forecasts
    f = frame(); F = forecasts(split, os.environ.get('QEST_FC', 'chronos')); j = f.join(F, how='inner'); j = j[j.split == split]
    X = qscen.features(j, j.index); rows = []
    for mo, jm in j.groupby(j.day.dt.to_period('M')):
        Xm = X.loc[jm.index]; P = params(jm.day.iloc[0]); Pc = qscen.params(jm.day.iloc[0]); C, Z, D = _windows(jm, Xm)
        with torch.no_grad():
            ll = []
            for i in range(0, len(Z), 256):
                p = P['model'].probs(torch.as_tensor(C[i:i + 256], device=DEV)); ll.append(torch.log(p[torch.arange(len(p)), torch.as_tensor(Z[i:i + 256], device=DEV)].clamp(min=1e-12)).cpu().numpy())
            ll = np.concatenate(ll) / W
        # the chain on the same windows: sum of log transition probabilities inside each window given its z0
        z = qscen._states(jm); M = np.log(np.clip(qscen.transitions(Pc, Xm), 1e-12, 1)); days = jm.day.to_numpy(); first = np.r_[True, days[1:] != days[:-1]]; starts = np.flatnonzero(first); llc = []
        for s0, s1 in zip(starts, np.r_[starts[1:], len(days)]):
            for t0 in range(0, s1 - s0 - 1 - W + 1, W):
                a = s0 + t0; llc.append(sum(M[a + 1 + k][z[a + k], z[a + 1 + k]] for k in range(W)) / W)
        ex = qscen.sample(Pc, Xm[jm.day == jm.day.iloc[0]], n, seed=1); Y = sample_day(P, Xm[jm.day == jm.day.iloc[0]], n, seed)
        up = (z % 2); rows.append(dict(month=str(mo), windows=len(Z), ll_born=float(ll.mean()), ll_chain=float(np.mean(llc)), up_data=float(up.mean()),
                                        up_born=float(Y[:, :, 0].mean()), up_chain=float(ex[:, :, 0].mean()), run_data=qscen._runlen(up), run_born=qscen._runlen(Y[:, :, 0].ravel()), run_chain=qscen._runlen(ex[:, :, 0].ravel())))
        print(rows[-1], flush=True)
    R = pd.DataFrame(rows); R.to_csv(RESULTS / f'qgen_bench_{split}.csv', index=False); print(R.round(3).to_string()); return R


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--split', default='val'); ap.add_argument('--bench', action='store_true'); a = ap.parse_args()
    if a.bench:
        bench(a.split)
    else:
        P = params(None); print('fitted', P['n_windows'], 'windows, train NLL per window', round(P['nll_train'], 3))

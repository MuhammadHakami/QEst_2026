"""A quantum fine-tuning head for the activation targets (after Kim, Baglio et al., arXiv:2609.05408, Sep 2026).

The paper puts a 12-qubit parameterised circuit on frozen Chronos embeddings to classify power-grid events (PSML-5) and
beats a classical MLP of 1.2-3 M parameters with 3,240 quantum parameters; its finding is that a fixed-width register
saturates on information intake, and extra qubits help only when they carry new inputs ("wings"). Here the head sits on the
gate-A feature set of qest.actfc (the drivers of the Baltic imbalance) and predicts the activation targets the plan pays for:
m_up / m_dn (mFRR price crosses DA +- 5) and the aFRR shares. Walk-forward exactly as qest.actfc (monthly refits on days
before the month, D-2 lag). Controls on identical inputs: an MLP head (classical, same training loop) and the LightGBM
forecaster of record (qest.actfc). Scored with qest.actfc.score (AUC, log-loss, Brier, within-day Spearman).

Circuit (statevector in torch, 12 qubits = 4,096 amplitudes, batched): a linear pre-encoder maps the standardised features to
12 x R angle vectors; R = 2 data re-uploading blocks, each RY(theta_in) RZ(w) RY(w) per qubit then a CZ ring; readout Z on
the first k = 5 qubits into a linear layer -> one logit per target (the four targets share the circuit; one head each).
Post-selection and wings are left out (noiseless, the core only). Parameters: pre-encoder F x 24 + 2 x 24 + 5 x 4 + 4.

usage: CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m qest.qhead [--split val|test] [--months 3]  -> results/qhead_<split>.csv
"""
import argparse
import math
import sys
import time

import numpy as np
import pandas as pd
import torch

from qest import actfc
from qest.data import frame
from qest.paths import RESULTS

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
NQ, R, K = 12, 2, 5
TARGETS = actfc.TARGETS


class QHead(torch.nn.Module):
    def __init__(self, nf, nq=NQ, r=R, k=K, nt=len(TARGETS)):
        super().__init__(); self.nq, self.r, self.k = nq, r, k
        self.pre = torch.nn.Linear(nf, nq * r); self.w = torch.nn.Parameter(torch.randn(r, nq, 2) * .1); self.out = torch.nn.Linear(k, nt)
        idx = torch.arange(1 << nq); self.register_buffer('bits', ((idx[:, None] >> torch.arange(nq)) & 1).float())  # (N, nq)
        cz = torch.ones(1 << nq)
        for q in range(nq):
            cz = cz * torch.where((self.bits[:, q] * self.bits[:, (q + 1) % nq]).bool(), -1.0, 1.0)
        self.register_buffer('cz', cz)

    def _ry(self, psi, q, th):  # th (B,)
        B = psi.shape[0]; N = 1 << self.nq; c, s = torch.cos(th / 2)[:, None, None], torch.sin(th / 2)[:, None, None]
        v = psi.view(B, N >> (q + 1), 2, 1 << q); a0, a1 = v[:, :, 0], v[:, :, 1]
        return torch.stack([c * a0 - s * a1, s * a0 + c * a1], 2).reshape(B, N)

    def _rz(self, psi, q, th):
        ph = torch.exp(-.5j * th)[:, None]; return torch.where(self.bits[None, :, q].bool(), psi * ph.conj(), psi * ph)

    def forward(self, x):
        B = x.shape[0]; N = 1 << self.nq; ang = torch.tanh(self.pre(x)).view(B, self.r, self.nq) * math.pi
        psi = torch.zeros(B, N, dtype=torch.complex64, device=x.device); psi[:, 0] = 1
        for l in range(self.r):
            for q in range(self.nq):
                psi = self._ry(psi, q, ang[:, l, q]); psi = self._rz(psi, q, self.w[l, q, 0].expand(B)); psi = self._ry(psi, q, self.w[l, q, 1].expand(B))
            psi = psi * self.cz
        p = psi.abs() ** 2; z = 1 - 2 * (p @ self.bits[:, :self.k])  # <Z_q> on the first k qubits
        return self.out(z)


class MLPHead(torch.nn.Module):
    def __init__(self, nf, hidden=64, nt=len(TARGETS)):
        super().__init__(); self.net = torch.nn.Sequential(torch.nn.Linear(nf, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, nt))

    def forward(self, x):
        return self.net(x)


def _train(model, X, Y, W, init, epochs, lr, bs=1024, seed=0, wd=1e-2):
    """Weighted BCE against the four targets (shares are soft labels), boosted from the climatology logit `init`, as actfc.
    AdamW with weight decay: both heads over-fitted the 17k-row months without it (log-loss above the climatology's)."""
    torch.manual_seed(seed); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd); n = len(X)
    g = torch.Generator(device='cpu').manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            b = perm[i:i + bs]; xb, yb, wb, ib = X[b], Y[b], W[b], init[b]
            logit = model(xb) + ib; ok = ~torch.isnan(yb)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(logit[ok], yb[ok], reduction='none') * wb[:, None].expand_as(yb)[ok]).sum() / wb[:, None].expand_as(yb)[ok].sum()
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def fit_predict(fit_days, pred_days, kind='q', X=None, half_life=actfc.HALF_LIFE, epochs=None, lr=None, seed=0):
    """Walk-forward predictions (monthly refits before each month on days before it, D-2 lag), the qest.actfc protocol."""
    f = frame(); X = actfc.features() if X is None else X; Y = actfc.targets(f); h = f.local.dt.hour
    pred_days = sorted(pred_days); months = sorted({pd.Timestamp(d).to_period('M') for d in pred_days})
    out = pd.DataFrame(index=f.index[f.day.isin(pred_days)], columns=list(TARGETS), dtype=float)
    cols = [c for c in X.columns if X[c].notna().mean() > .5]; Xn = X[cols].astype(float)
    epochs = epochs or (8 if kind == 'q' else 15); lr = lr or (1e-3 if kind == 'q' else 5e-4)
    for mo in months:
        pdays = [d for d in pred_days if pd.Timestamp(d).to_period('M') == mo]
        cut = pd.Timestamp(pdays[0]) - pd.Timedelta(days=actfc.LAG_DAYS)
        fdays = sorted({d for d in set(fit_days) | set(pred_days) if pd.Timestamp(d) < cut})
        if len(fdays) < 10:
            continue
        fm, pm = f.day.isin(fdays), f.day.isin(pdays)
        age = (pd.Timestamp(max(fdays)) - f.day).dt.days.to_numpy(); w = .5 ** (age / half_life)
        mu, sd = Xn[fm].mean(), Xn[fm].std().replace(0, 1); Z = ((Xn - mu) / sd).fillna(0).clip(-5, 5)
        clim = {}; init = np.zeros((len(f), len(TARGETS)))
        for j, k in enumerate(TARGETS):
            y = Y[k]; m = fm & y.notna(); ww = pd.Series(w[m.to_numpy()], index=y.index[m]); clim[k] = (y[m] * ww).groupby(h[m]).sum() / ww.groupby(h[m]).sum()
            p0 = np.clip(h.map(clim[k]).fillna(.5).to_numpy(), 1e-3, 1 - 1e-3); init[:, j] = np.log(p0 / (1 - p0))
        Xt = torch.as_tensor(Z[fm].to_numpy(), dtype=torch.float32, device=DEV); Yt = torch.as_tensor(np.stack([Y[k][fm].to_numpy(float) for k in TARGETS], 1), dtype=torch.float32, device=DEV)
        Wt = torch.as_tensor(w[fm.to_numpy()], dtype=torch.float32, device=DEV); It = torch.as_tensor(init[fm.to_numpy()], dtype=torch.float32, device=DEV)
        model = (QHead(len(cols)) if kind == 'q' else MLPHead(len(cols))).to(DEV)
        model = _train(model, Xt, Yt, Wt, It, epochs, lr, seed=seed)
        with torch.no_grad():
            Xp = torch.as_tensor(Z[pm].to_numpy(), dtype=torch.float32, device=DEV); Ip = torch.as_tensor(init[pm.to_numpy()], dtype=torch.float32, device=DEV)
            P = torch.sigmoid(torch.cat([model(Xp[i:i + 2048]) for i in range(0, len(Xp), 2048)]) + Ip).cpu().numpy()
        out.loc[pm, list(TARGETS)] = P
        print(f'{kind} {mo}: fit {len(fdays)} days, {int(fm.sum())} rows, predicted {len(pdays)} days', flush=True)
    return out


def run(split='val', months=None, seeds=(0,)):
    from qest.data import days
    f = frame(); pred = days(f, split); fit = days(f, 'train') if split in ('train', 'val') else days(f, ('train', 'val'))
    if months:
        pred = [d for d in pred if pd.Timestamp(d).to_period('M') in sorted({pd.Timestamp(x).to_period('M') for x in pred})[:months]]
    X = actfc.features(); rows = []
    for kind in ('q', 'mlp'):
        for s in seeds:
            t = time.time(); P = fit_predict(fit, pred, kind, X=X, seed=s); dt = time.time() - t
            P.to_parquet(RESULTS / f'qhead_{kind}{s}_{split}.parquet'); S = actfc.score(P, split); S['method'] = f'{kind}{s}'; S['s'] = dt; rows.append(S)
    if (RESULTS / f'actfc_{split}.parquet').exists():  # the LightGBM forecaster of record, same days
        A = pd.read_parquet(RESULTS / f'actfc_{split}.parquet'); A = A.loc[A.index.isin(f.index[f.day.isin(pred)])]
        S = actfc.score(A, split); S['method'] = 'lightgbm (actfc)'; S['s'] = float('nan'); rows.append(S)
    C = actfc.climatology(fit, pred); S = actfc.score(C, split); S['method'] = 'climatology'; S['s'] = float('nan'); rows.append(S)
    T = pd.concat(rows); T.to_csv(RESULTS / f'qhead_{split}.csv'); pd.set_option('display.width', 250); print(T.round(4).to_string()); return T


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--split', default='val'); ap.add_argument('--months', type=int, default=0); ap.add_argument('--seeds', default='0')
    a = ap.parse_args(); run(a.split, a.months or None, tuple(int(s) for s in a.seeds.split(',')))

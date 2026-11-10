"""Joint activation scenarios for the day plan: a conditional Markov chain of the TSO's mFRR activations, sampled exactly.

Finding (9 Nov 2026, Q1-Q2 fit, Q3 held out): gate-A features barely move the per-MTU activation probability (log-likelihood
per bit -0.663 vs -0.672 for a constant), but the pattern within a day has strong structure: an active MTU stays active
(P(UP | UP 15 min ago) = 0.86) and UP and DOWN almost never coincide. The quantile scenarios of qest.stoch (comonotonic
q10 / q50 / q90) cannot express runs of activations, which is what drains or fills a battery. This module samples whole-day
patterns from a fitted joint model instead.

Model, per day: state z_t in {00, 10, 01, 11} = (UP active, DOWN active) in MTU t (mFRR price beyond DA +- MARGIN);
P(z_{t+1} | z_t, x_{t+1}) a multinomial logistic in the gate-A features x and the current state; the first state from the
data's start-of-day distribution. Maximum likelihood, so rates and run lengths are matched by construction (a pseudo-
likelihood Ising fit of the same ladder locked into all-UP / all-DOWN phases: tried, dropped). Sampling is exact and forward.
A window of 8 MTUs is a 4^8-state energy table E(z) = -log P(z | z_0): the ground truth the quantum-enhanced MCMC sampler
is benchmarked against (qest.qscen_q).

usage: .venv/bin/python -m qest.qscen [--check]   -> results/qscen_params/<month>.pkl
"""
import pickle
import sys

import numpy as np
import pandas as pd

from qest.paths import RESULTS

MARGIN = 5.0
FEATS = ['mfrr_prem', 'imb_prem', 'aus', 'ads', 'da', 'sinh', 'cosh', 'wknd', 'up2', 'dn2', 'pu', 'pd', 'pas', 'pds']  # pu.. : qest.actfc driver-based marginals (logit), the conditioning that makes scenarios informative
STATES = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])  # z -> (u, d)
_LAG = {}


def lag_rates():
    """Each delivery day's mFRR UP / DOWN activation rate two days earlier (balancing data is public with a D-2 lag at gate A)."""
    if not _LAG:
        from qest.data import frame
        f = frame(); da = f.nps_dayahead_eur_mwh
        up = (f.mfrr_up_eur_mwh.notna() & (f.mfrr_up_eur_mwh >= da + MARGIN)).groupby(f.day).mean()
        dn = (f.mfrr_down_eur_mwh.notna() & (f.mfrr_down_eur_mwh <= da - MARGIN)).groupby(f.day).mean()
        r = pd.DataFrame({'up2': up.shift(2), 'dn2': dn.shift(2)}).bfill().fillna(.5)
        _LAG.update({pd.Timestamp(d).date(): (float(a), float(b)) for d, a, b in zip(r.index, r.up2, r.dn2)})
    return _LAG


def features(fc, idx):
    """Gate-A features per MTU from the quantile forecast frame fc (q50 columns) and the MTU index."""
    loc = idx.tz_convert('Europe/Tallinn'); hr = loc.hour; g = lambda c: fc[c].to_numpy() if c in fc else np.zeros(len(idx)); L = lag_rates()
    lag = np.array([L.get(d, (.5, .5)) for d in loc.date])
    X = pd.DataFrame({'mfrr_prem': np.clip(g('mfrr_q50') - g('da_q50'), -200, 400) / 100, 'imb_prem': np.clip(g('imb_q50') - g('da_q50'), -300, 300) / 100,
                      'aus': np.nan_to_num(g('afrr_up_s_q50')), 'ads': np.nan_to_num(g('afrr_dn_s_q50')), 'da': np.clip(g('da_q50'), -50, 400) / 100,
                      'sinh': np.sin(2 * np.pi * hr / 24), 'cosh': np.cos(2 * np.pi * hr / 24), 'wknd': (idx.dayofweek >= 5).astype(float),
                      'up2': lag[:, 0], 'dn2': lag[:, 1]}, index=idx)
    from qest.strategies import actfc_pred
    A = actfc_pred(idx); lg = lambda p: np.log(np.clip(p, .02, .98) / (1 - np.clip(p, .02, .98)))
    if A is None:
        A = pd.DataFrame(np.nan, index=idx, columns=['m_up', 'm_dn', 'a_up_s', 'a_dn_s'])
    X['pu'], X['pd'], X['pas'], X['pds'] = lg(A.m_up.to_numpy(float)), lg(A.m_dn.to_numpy(float)), A.a_up_s.to_numpy(float), A.a_dn_s.to_numpy(float)
    return X.fillna({c: 0.0 for c in X.columns if c not in ('pu', 'pd', 'pas', 'pds')})  # the actfc columns stay NaN where no forecast exists (first weeks): dropped in fit


def _states(j):
    da = j.nps_dayahead_eur_mwh
    u = (j.mfrr_up_eur_mwh.notna() & (j.mfrr_up_eur_mwh >= da + MARGIN)).astype(int); d = (j.mfrr_down_eur_mwh.notna() & (j.mfrr_down_eur_mwh <= da - MARGIN)).astype(int)
    return (u + 2 * d).to_numpy()


def _design(X, zprev):
    return np.c_[X[FEATS].to_numpy(), np.eye(4)[zprev][:, 1:]]  # features + one-hot of the previous state (00 is the base)


HALF_LIFE = 45  # days: recent activation regimes weigh more (Q3's UP rate is 0.40 against Q1-Q2's 0.26; the market shifts)
_P = {}


def fit(before=None, splits=('train', 'val', 'test')):
    """Maximum-likelihood transition model on every day strictly before `before` minus the D-2 publication lag (all data when
    None), recency-weighted. Gate-A legal: a plan for day D only sees activations up to D-2."""
    from sklearn.linear_model import LogisticRegression
    from qest.data import frame
    from qest.evaluate import forecasts
    f = frame(); F = pd.concat([forecasts(s, 'chronos') for s in splits]); j = f.join(F, how='inner'); j = j[j.split.isin(splits)]
    if before is not None:
        j = j[j.day < pd.Timestamp(before) - pd.Timedelta(days=2)]
    z = _states(j); X = features(j, j.index); days = j.day.to_numpy(); first = np.r_[True, days[1:] != days[:-1]]
    ok = np.isfinite(X[FEATS].to_numpy()).all(1)  # days without an actfc forecast (the first weeks of the data) are left out
    age = (days.max() - days) / np.timedelta64(1, 'D'); w = 0.5 ** (age / HALF_LIFE)
    m = LogisticRegression(max_iter=2000, C=1.0).fit(_design(X, np.r_[0, z[:-1]])[~first & ok], z[~first & ok], sample_weight=w[~first & ok])
    pi0 = np.bincount(z[first & ok], weights=w[first & ok], minlength=4); pi0 = pi0 / pi0.sum()
    return dict(model=m, pi0=pi0, n_days=int(first.sum()), aus_ratio=float(j.afrr4s_up_share[z % 2 == 1].mean() / max(j.afrr4s_up_share[z % 2 == 0].mean(), 1e-3)),
                ads_ratio=float(j.afrr4s_down_share[z >= 2].mean() / max(j.afrr4s_down_share[z < 2].mean(), 1e-3)))


def params(day=None):
    """The model for delivery day `day`, refitted once a month (cache in results/qscen_params/)."""
    key = 'all' if day is None else pd.Timestamp(day).strftime('%Y-%m')
    if key not in _P:
        d = RESULTS / 'qscen_params'; d.mkdir(exist_ok=True); p = d / f'{key}.pkl'
        if p.exists():
            _P[key] = pickle.loads(p.read_bytes())
        else:
            _P[key] = fit(None if day is None else pd.Timestamp(day).replace(day=1)); p.write_bytes(pickle.dumps(_P[key]))
    return _P[key]


def transitions(P, X):
    """(T, 4, 4) matrices: row = previous state, column = next, for each MTU of the day (MTU 0's row is unused)."""
    T = len(X); M = np.zeros((T, 4, 4))
    for zp in range(4):
        pr = P['model'].predict_proba(_design(X, np.full(T, zp))); M[:, zp, P['model'].classes_] = pr
    return M


def sample(P, X, n, seed=0):
    """n exact forward samples of the day's (u, d) pattern: (n, T, 2) ints."""
    rng = np.random.default_rng(seed); M = transitions(P, X); T = len(X); z = np.zeros((n, T), int)
    z[:, 0] = rng.choice(4, n, p=P['pi0'])
    for t in range(1, T):
        c = M[t][z[:, t - 1]].cumsum(1); z[:, t] = (rng.random(n)[:, None] > c).sum(1).clip(0, 3)
    return STATES[z]


def window_energy(P, X, t0, z0, w=8):
    """E(z_{t0+1..t0+w}) = -log P(path | z_{t0} = z0) over all 4^w paths: the exact target for a sampler benchmark."""
    M = np.log(np.clip(transitions(P, X), 1e-12, 1)); idx = np.indices([4] * w).reshape(w, -1)  # (w, 4^w)
    E = -M[t0 + 1][z0, idx[0]]
    for k in range(1, w):
        E = E - M[t0 + 1 + k][idx[k - 1], idx[k]]
    return E, idx


def scenario_px(P, px50, y):
    """One scenario's px dict from the bits y (T, 2): mFRR probabilities become the bits, aFRR shares scale with them."""
    p = dict(px50); u, d = y[:, 0].astype(float), y[:, 1].astype(float)
    p['m_up_q'], p['m_dn_q'] = u, d
    p['a_up_s'] = np.clip(np.asarray(px50['a_up_s']) * np.where(u > 0, P['aus_ratio'], 1.0), 0, 1)
    p['a_dn_s'] = np.clip(np.asarray(px50['a_dn_s']) * np.where(d > 0, P['ads_ratio'], 1.0), 0, 1)
    return p


def day_scenarios(ctx, px50, n=8, seed=0):
    P = params(ctx.idx[0].tz_convert('Europe/Tallinn').date()); Y = sample(P, features(ctx.fc, ctx.idx), n, seed)
    return [scenario_px(P, px50, y) for y in Y], np.full(n, 1 / n)


def _runlen(y):
    import itertools
    return float(np.mean([len(list(g)) for k, g in itertools.groupby(y) if k == 1] or [0]))


def _check():
    """The sampler reproduces the held-out split's activation statistics (rates, run lengths, UP-DOWN exclusion); the
    held-out log-likelihood beats the independent model; the window energy table is a normalised distribution."""
    from qest.data import frame
    from qest.evaluate import forecasts
    P = fit(before='2026-07-01'); f = frame(); F = forecasts('val', 'chronos'); j = f.join(F, how='inner'); j = j[j.split == 'val']
    z = _states(j); X = features(j, j.index); days = j.day.to_numpy(); up, dn = z % 2, (z >= 2).astype(int)
    U, D = [], []
    for d in pd.unique(days):
        m = days == d; Y = sample(P, X[m], 2, seed=7); U.append(Y[:, :, 0].ravel()); D.append(Y[:, :, 1].ravel())
    U, D = np.concatenate(U), np.concatenate(D)
    first = np.r_[True, days[1:] != days[:-1]]; pr = P['model'].predict_proba(_design(X, np.r_[0, z[:-1]])[~first])
    ll = float(np.mean(np.log(np.clip(pr[np.arange(len(pr)), np.searchsorted(P['model'].classes_, z[~first])], 1e-9, 1))))
    print('with actfc conditioning' if np.isfinite(X.pu).all() else 'WITHOUT actfc for some rows')
    base = np.bincount(z, minlength=4) / len(z); ll0 = float(np.mean(np.log(base[z])))
    print(f'qscen check: val UP {up.mean():.2f} / sampled {U.mean():.2f}; DOWN {dn.mean():.2f} / {D.mean():.2f}; run UP {_runlen(up):.1f} / {_runlen(U):.1f}; '
          f'both {np.mean(up * dn):.3f} / {np.mean(U * D):.3f}; held-out log-lik per MTU {ll:.3f} vs independent {ll0:.3f}')
    assert abs(U.mean() - up.mean()) < .15 and abs(D.mean() - dn.mean()) < .15 and abs(_runlen(U) - _runlen(up)) < .35 * _runlen(up) and ll > ll0 + .3  # Q3 shifted vs Q1-Q2
    E, _ = window_energy(P, X[days == days[0]], 10, 1, w=6); assert abs(np.exp(-E).sum() - 1) < 1e-6, np.exp(-E).sum()
    print('qscen check ok')


if __name__ == '__main__':
    _check() if '--check' in sys.argv else fit()

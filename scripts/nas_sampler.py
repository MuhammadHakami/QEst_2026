"""NAS + PostNAS for the quantum-proposal MCMC sampler (qest.qscen_q), leakage-free.
Search (Optuna TPE) over the proposal circuit: mixer weight range [g_lo, g_hi], evolution time range [t_lo, t_hi], Trotter depth.
Objective on Q2 windows only (June chain, fitted on data before June): integrated autocorrelation of the UP count x a mild
depth cost, subject to TV to the exact marginals. PostNAS: shrink the Trotter depth of the winner until mixing degrades by
more than 10 %. Held-out score: Q3 windows (chain fitted before each month), default vs NAS vs PostNAS vs single-flip.
usage: CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/nas_sampler.py [--trials 40]
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.getcwd()); os.environ['QEST_WALK'] = '1'
import numpy as np, pandas as pd, torch, optuna
from qest import qscen, qscen_q as Q
from qest.data import frame
from qest.evaluate import forecasts
from qest.paths import RESULTS

DEFAULT = dict(g_lo=.25, g_hi=.60, t_lo=2.0, t_hi=20.0, trotter=8)


def windows(split, lo, hi, n, seed):
    f = frame(); F = forecasts(split, 'chronos'); j = f.join(F, how='inner'); j = j[(j.day >= lo) & (j.day <= hi)]
    X = qscen.features(j, j.index); dd = j.day.to_numpy(); rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        d = rng.choice(pd.unique(dd)); Xd = X[dd == d]; t0 = int(rng.integers(0, 96 - Q.W - 1)); z0 = int(rng.integers(0, 4))
        E = Q.energies(qscen.params(pd.Timestamp(d)), Xd, t0, z0); out.append((E, Q.exact_marginals(E)))
    return out


def score(prop, wins, steps=200, burn=50, quantum=True):
    Q.PROP.update(prop); tv, ia, acc = [], [], []
    for w, (E, ex) in enumerate(wins):
        S, a = Q.mcmc(E, steps, quantum, chains=128, seed=w); keep = S[burn:]
        tv.append(float(np.abs(Q.marginals(keep) - ex).sum(1).mean() / 2)); acc.append(a)
        ups = ((keep[:, :, None] >> (2 * torch.arange(Q.W, device=Q.DEV))) & 1).sum(2).float().cpu().numpy()
        ia.append(float(np.mean([Q.iact(ups[:, c]) for c in range(0, ups.shape[1], 16)])))
    return float(np.mean(tv)), float(np.mean(ia)), float(np.mean(acc))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--trials', type=int, default=40); a = ap.parse_args()
    tr = windows('train', pd.Timestamp('2026-06-01'), pd.Timestamp('2026-06-30'), 6, 0)
    def obj(t):
        g_lo = t.suggest_float('g_lo', .05, .5); t_lo = t.suggest_float('t_lo', .5, 8)
        prop = dict(g_lo=g_lo, g_hi=g_lo + t.suggest_float('g_w', .05, .5), t_lo=t_lo, t_hi=t_lo + t.suggest_float('t_w', 1, 24), trotter=t.suggest_int('trotter', 2, 12))
        tv, ia, acc = score(prop, tr); t.set_user_attr('tv', tv); t.set_user_attr('acc', acc); t.set_user_attr('prop', prop)
        return ia * (1 + .02 * prop['trotter']) + 200 * max(tv - .12, 0)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=0)); st.enqueue_trial(dict(g_lo=.25, g_w=.35, t_lo=2.0, t_w=18.0, trotter=8))
    st.optimize(obj, n_trials=a.trials); best = st.best_trial.user_attrs['prop']
    # PostNAS: fewest Trotter steps within 10 % of the winner's autocorrelation on the search windows
    base = score(best, tr)[1]; post = dict(best)
    for k in range(best['trotter'] - 1, 0, -1):
        c = dict(best, trotter=k)
        if score(c, tr)[1] <= 1.10 * base: post = c
        else: break
    te = windows('val', pd.Timestamp('2026-07-01'), pd.Timestamp('2026-09-30'), 16, 1); rows = []
    for name, prop, q in (('default (Layden ranges)', DEFAULT, True), ('NAS', best, True), ('PostNAS', post, True), ('single-bit flips', DEFAULT, False)):
        t0 = time.time(); tv, ia, acc = score(prop, te, quantum=q); rows.append(dict(sampler=name, tv=tv, iact=ia, acc=acc, trotter=prop['trotter'] if q else 0, s_per_window=(time.time() - t0) / len(te), **({k: round(v, 3) for k, v in prop.items() if k != 'trotter'} if q else {})))
    R = pd.DataFrame(rows); R.to_csv(RESULTS / 'nas_sampler.csv', index=False); pd.set_option('display.width', 220); print(R.round(3).to_string())
    (RESULTS / 'nas_sampler_best.json').write_text(json.dumps(dict(prop=post, nas=best, search='Q2 (June) windows; held-out Q3 in nas_sampler.csv'), indent=1))

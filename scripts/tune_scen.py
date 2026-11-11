"""Optuna (TPE) on the scenario plan's knobs, leakage-free: tuned on Q2 days only (every 2nd day of Apr-Jun, walk-forward
calibration), then applied unchanged to Q3 and Q4, which the tuner never sees. Knobs: scenario count (4 / 8), aFRR and mFRR
bid margins, kappa_up / kappa_dn (haircut on activation energy), tol (gate-C SoC drift). Official settlement, battery + hybrid.
usage: .venv/bin/python scripts/tune_scen.py [--trials 64] [--workers 8]  -> results/tune_scen.csv, results/tune_best_milp_scenf_t.json
"""
import argparse, json, os, subprocess, sys
sys.path.insert(0, os.getcwd())
import optuna
import pandas as pd
from qest import market
from qest.data import MARGINS, frame
from qest.evaluate import run
from qest.paths import RESULTS

REAL = bool(os.environ.get('QEST_REAL'))  # tune under the sim2real preset (set QEST_FAT=1 QEST_ACT_CAP=25 too)
TAG = '_real' if REAL else ''
STORE = f'sqlite:///{RESULTS}/tune_scen{TAG}.db'; STUDY = 'scenf_q2' + TAG
f = frame(); Q2 = sorted(d for d in f.day.unique() if pd.Timestamp('2026-04-01') <= pd.Timestamp(d) <= pd.Timestamp('2026-06-30'))[::2]
os.environ['QEST_WALK'] = '1'; os.environ['QEST_DAYS'] = ','.join(str(pd.Timestamp(d).date()) for d in Q2)


def objective(trial):
    s = trial.suggest_categorical('plan', ['milp_scenf4', 'milp_scenf'])
    a = trial.suggest_categorical('afrr', [m for m in MARGINS if m > 0]); m = trial.suggest_float('mfrr', 5, 300, log=True)
    params = dict(kappa_up=trial.suggest_float('kappa_up', .3, 1), kappa_dn=trial.suggest_float('kappa_dn', 0, 1), tol=trial.suggest_float('tol', .01, .3, log=True))
    cfg = market.Config(settlement='baltic' if REAL else 'official'); units = market.portfolio(['battery', 'hybrid'], official=not REAL)
    R, tar = run('train', [s], units, cfg, margins=dict(afrr_up=a, afrr_down=a, mfrr_up=m, mfrr_down=m), params=params)
    return float(R.total.sum())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--trials', type=int, default=64); ap.add_argument('--workers', type=int, default=8); ap.add_argument('--worker', action='store_true'); a = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if a.worker:
        optuna.load_study(study_name=STUDY, storage=STORE).optimize(objective, n_trials=a.trials, callbacks=[lambda s, t: print(t.number, round(t.value or 0), t.params, flush=True)]); sys.exit()
    st = optuna.create_study(study_name=STUDY, storage=STORE, direction='maximize', load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=0, multivariate=True, n_startup_trials=16))
    if not st.trials:  # the inherited Chronos-2 knobs, for both scenario counts
        B = json.loads((RESULTS / 'tune_best_milp_chronos.json').read_text())
        for s in ('milp_scenf4', 'milp_scenf'):
            st.enqueue_trial(dict(plan=s, afrr=B['margins']['afrr_up'], mfrr=B['margins']['mfrr_up'], **B['params']))
    ps = [subprocess.Popen([sys.executable, __file__, '--worker', '--trials', str(-(-a.trials // a.workers))]) for _ in range(a.workers)]
    [p.wait() for p in ps]
    st = optuna.load_study(study_name=STUDY, storage=STORE); st.trials_dataframe().sort_values('value', ascending=False).to_csv(RESULTS / f'tune_scen{TAG}.csv', index=False)
    b = st.best_trial.params; out = dict(plan=b['plan'], margins=dict(afrr_up=b['afrr'], afrr_down=b['afrr'], mfrr_up=b['mfrr'], mfrr_down=b['mfrr']), params=dict(kappa_up=b['kappa_up'], kappa_dn=b['kappa_dn'], tol=b['tol']), q2_eur=st.best_value, q2_days=len(Q2),
                                         inherited=[t.value for t in st.trials[:2]])
    (RESULTS / f'tune_best_scen_q2{TAG}.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))

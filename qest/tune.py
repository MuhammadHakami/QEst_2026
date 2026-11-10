"""Tune the MILP strategy's trading knobs with Optuna (TPE) on validation only (Q3, forecasts fitted on Q1-Q2), then use
them unchanged on test. Official rules (slides 10 + 12), battery + hybrid.

Knobs: aFRR energy-bid margin (on the PICASSO margin grid), mFRR margin (EUR/MWh beyond DA), kappa_up / kappa_dn
(haircut on the activation energy the day-ahead plan leans on), tol (SoC drift, share of energy, before gate C prices
a direction out). The old 4 x 4 margin grid is trial 0's neighbourhood.
usage: .venv/bin/python -m qest.tune [--strategy milp|milp_chronos] [--trials 60] [--workers 8]
  -> results/tune_optuna[_<strategy>].csv, results/tune_best[_<strategy>].json (milp keeps the unsuffixed names)
"""
import argparse
import json
import subprocess
import sys

import optuna

from qest import market
from qest.data import MARGINS
from qest.evaluate import run
from qest.paths import RESULTS

STORE = f'sqlite:///{RESULTS}/tune_optuna.db'
STRAT = 'milp'


def sfx(s):
    return '' if s == 'milp' else f'_{s}'


def objective(trial):
    a = trial.suggest_categorical('afrr', [m for m in MARGINS if m > 0])
    m = trial.suggest_float('mfrr', 5, 300, log=True)
    params = dict(kappa_up=trial.suggest_float('kappa_up', 0, 1), kappa_dn=trial.suggest_float('kappa_dn', 0, 1),
                  tol=trial.suggest_float('tol', .01, .3, log=True))
    cfg = market.Config(); units = market.portfolio(['battery', 'hybrid'], official=cfg.official)
    R, tar = run('val', [STRAT], units, cfg, margins=dict(afrr_up=a, afrr_down=a, mfrr_up=m, mfrr_down=m), params=params)
    trial.set_user_attr('imbalance', float(R.imbalance.sum())); trial.set_user_attr('imb_mwh', float(R.imb_mwh.sum()))
    return float(R.total.sum() + tar.sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--trials', type=int, default=60); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--worker', action='store_true'); ap.add_argument('--strategy', default='milp'); a = ap.parse_args()
    global STRAT; STRAT = a.strategy; study = f'{STRAT}_official_q3'
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if a.worker:  # one process of the pool: HiGHS holds the GIL, so parallelism is by process on a shared SQLite study
        st = optuna.load_study(study_name=study, storage=STORE)
        st.optimize(objective, n_trials=a.trials, callbacks=[lambda s, t: print(t.number, round(t.value or 0), t.params, flush=True)])
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    st = optuna.create_study(study_name=study, storage=STORE, direction='maximize', load_if_exists=True,
                             sampler=optuna.samplers.TPESampler(seed=0, multivariate=True, n_startup_trials=12))
    if not st.trials:  # seed with the previous grid's winner and with the no-haircut default
        st.enqueue_trial(dict(afrr=25, mfrr=100.0, kappa_up=1.0, kappa_dn=1.0, tol=.05))
        st.enqueue_trial(dict(afrr=25, mfrr=100.0, kappa_up=.5, kappa_dn=.5, tol=.05))
    per = -(-a.trials // a.workers)
    ps = [subprocess.Popen([sys.executable, '-m', 'qest.tune', '--worker', '--strategy', STRAT, '--trials', str(per)]) for _ in range(a.workers)]
    for p in ps:
        p.wait()
    st = optuna.load_study(study_name=study, storage=STORE); df = st.trials_dataframe().sort_values('value', ascending=False)
    df.to_csv(RESULTS / f'tune_optuna{sfx(STRAT)}.csv', index=False)
    b = st.best_trial.params; out = dict(margins=dict(afrr_up=b['afrr'], afrr_down=b['afrr'], mfrr_up=b['mfrr'], mfrr_down=b['mfrr']),
                                         params=dict(kappa_up=b['kappa_up'], kappa_dn=b['kappa_dn'], tol=b['tol']), val_eur=st.best_value)
    (RESULTS / f'tune_best{sfx(STRAT)}.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()

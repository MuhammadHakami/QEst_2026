"""Charts for the rebuilt deck: one real day bid by bid, forecast skill, the forecaster x decider euro matrix, euros per
market against hindsight. Titles and explanations live on the slide, not on the plot; legends sit outside the data.
usage: .venv/bin/python -m qest.deckfigs [--day 2026-10-24]   -> results/figs/deck_*.png
"""
import argparse
import json

import numpy as np
import pandas as pd

from qest.figures import AQUA, BLUE, FIG, GREY, INK2, ORANGE, VIOLET, YELLOW, plt, save
from qest.paths import RESULTS

MAGENTA, GREEN, RED = '#e87ba4', '#008300', '#e34948'
MODELS = [('milp@chronos', 'Chronos-2 + MILP (classical)', BLUE), ('qaoa@chronos', 'Chronos-2 + LR-QAOA (CUDA-Q)', ORANGE),
          ('milp@qrc', 'Quantum reservoir + MILP', VIOLET), ('oracle', 'Hindsight (real prices, same bid path)', AQUA)]


def day_run(d):
    """Plan + settle one delivery day for each model, keeping per-MTU traces."""
    from qest import market, qdecide, strategies
    from qest.data import afrr_margin_table, days, frame
    from qest.evaluate import forecasts, unit_gen
    from qest.power import generation
    from qest.strategies import DayContext, calibration
    f = frame(); g = generation(); d = pd.Timestamp(d)
    B = json.loads((RESULTS / 'tune_best_milp_chronos.json').read_text()); cal = calibration(f, days(f, ('train', 'val')), B['margins']); cal.update(B['params'])
    cfg = market.Config(); units = market.portfolio(['battery', 'hybrid'], official=True)
    m = (f.day == d).to_numpy(); day = f[m]; idx = day.index; atab = afrr_margin_table().reindex(f.index).fillna(0).loc[idx]
    act, fc = unit_gen(g, units); A = {k: v.reindex(idx).fillna(0).to_numpy() for k, v in act.items()}; F = {k: v.reindex(idx).fillna(0).to_numpy() for k, v in fc.items()}
    qdecide.register('chronos'); out = {}
    for name, _, _ in MODELS:
        kind = 'chronos' if name == 'oracle' else name.split('@')[1]
        ctx = DayContext(idx, forecasts('test', kind).reindex(idx), F, cal, real=day if name == 'oracle' else None,
                         gen_real=A if name == 'oracle' else None, afrr_tab=atab if name == 'oracle' else None)
        fn = strategies.STRATEGIES['milp'] if name.startswith('milp@') else strategies.STRATEGIES[name]
        plan = fn(units, ctx, cfg); res = market.settle_day(units, plan, day, A, cfg, afrr_tab=atab)
        out[name] = dict(plan=plan, res=res, fc=ctx.fc)
    return day, out


def day_figs(d):
    day, out = day_run(d)
    t = day.index.tz_convert('Europe/Tallinn'); h = t.hour + t.minute / 60
    fig, ax = plt.subplots(figsize=(16, 4.2))
    ax.plot(h, day.nps_dayahead_eur_mwh, color='#0b0b0b', lw=2.4, label='Day-ahead, real')
    ax.plot(h, out['milp@chronos']['fc']['da_q50'], color=BLUE, lw=2, label='Chronos-2 forecast (gate A)')
    ax.plot(h, out['milp@qrc']['fc']['da_q50'], color=VIOLET, lw=2, label='Quantum reservoir forecast (gate A)')
    ax.plot(h, day.mfrr_up_eur_mwh, color=GREY, lw=1.2, label='mFRR energy, real')
    ax.set_ylabel('€/MWh'); ax.set_xlabel('Hour, Tallinn'); ax.set_xticks(range(0, 25, 3)); ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
    fig.tight_layout(); save(fig, 'deck_day_prices')
    fig, ax = plt.subplots(3, len(MODELS), figsize=(16, 9), sharex=True, sharey='row')
    for j, (name, lab, col) in enumerate(MODELS):
        P = out[name]['plan']['battery']; cap = P.get('cap', {}); tr = out[name]['res']['battery']['trace']
        ax[0, j].step(h, sum(cap.get(k, 0 * h) for k in ('afrr_up', 'mfrr_up')), where='post', color=col, lw=2)
        ax[0, j].step(h, -sum(cap.get(k, 0 * h) for k in ('afrr_down', 'mfrr_down')), where='post', color=col, lw=2, alpha=.45)
        ax[1, j].step(h, P['da'], where='post', color=col, lw=2)
        ax[2, j].plot(h, tr['soc'], color=col, lw=2); ax[2, j].set_xticks(range(0, 25, 6)); ax[2, j].set_xlabel('Hour')
        ax[0, j].set_title(lab.split(' (')[0], fontsize=14, color=col)
    ax[0, 0].set_ylabel('Reserve MW, gate A\n(UP +, DOWN −)'); ax[1, 0].set_ylabel('DA position MW\ngate B (+ sell)'); ax[2, 0].set_ylabel('SoC MWh\n(settled)')
    fig.tight_layout(); save(fig, 'deck_day_bids')
    comp = [('cap', 'Capacity'), ('da', 'Day-ahead'), ('mfrr', 'mFRR energy'), ('afrr', 'aFRR energy'), ('imbalance', 'Imbalance'), ('tariff_mtu', 'Grid tariff'), ('leftover', 'Energy left')]
    fig, ax = plt.subplots(figsize=(16, 6)); w = .2; x = np.arange(len(comp))
    for i, (name, lab, col) in enumerate(MODELS):
        r = out[name]['res']; v = [sum(r[u][k] for u in r if k in r[u]) / 1e3 for k, _ in comp]
        ax.bar(x + (i - 1.5) * w, v, w * .9, color=col, label=f"{lab}: €{sum(r[u]['total'] for u in r) / 1e3:,.0f}k")
    ax.axhline(0, color=INK2, lw=1); ax.set_xticks(x, [c for _, c in comp]); ax.set_ylabel('€ thousand, battery + hybrid')
    ax.legend(loc='upper left', bbox_to_anchor=(0, -.12), ncol=2); fig.tight_layout(); save(fig, 'deck_day_eur')
    return {name: sum(o['res'][u]['total'] for u in o['res']) for name, o in out.items()}


def forecast_fig():
    S = pd.read_csv(RESULTS / 'qforecast_skill.csv'); E = pd.read_csv(RESULTS / 'qforecast_ens.csv')
    lab = dict(da='Day-ahead', cap_afrr_up='aFRR cap UP', cap_afrr_down='aFRR cap DOWN', cap_mfrr_up='mFRR cap UP', cap_mfrr_down='mFRR cap DOWN',
               mfrr='mFRR energy', imb='Imbalance', afrr_up_p='aFRR UP price', afrr_dn_p='aFRR DOWN price', afrr_up_s='aFRR UP share', afrr_dn_s='aFRR DOWN share')
    ser = [('qrc_q4', S, 'Quantum reservoir (CUDA-Q)', VIOLET, 'o'), ('rff_q4', S, 'Random Fourier features (classical twin)', GREY, 's'),
           ('ens_q4', E, 'Chronos-2 + quantum reservoir', ORANGE, 'D'), ('ens_rff_q4', E, 'Chronos-2 + classical twin', YELLOW, 'v')]
    fig, ax = plt.subplots(figsize=(16, 8)); y = np.arange(len(S))[::-1]
    base = E.chronos_q4.to_numpy()
    for c, D, l, col, mk in ser:
        ax.scatter(D[c].to_numpy() / base, y, s=110, color=col, marker=mk, label=l, zorder=3, edgecolor='#fcfcfb', linewidth=1.5)
    ax.axvline(1, color=BLUE, lw=2.5, label='Chronos-2 (classical baseline)'); ax.set_xscale('log')
    ax.set_xticks([.9, 1, 1.25, 1.5, 2, 3], ['0.9×', '1×', '1.25×', '1.5×', '2×', '3×']); ax.minorticks_off()
    ax.set_yticks(y, [lab[k] for k in S.target]); ax.set_xlabel('Q4 mean absolute error ÷ Chronos-2 error (left = better)')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1)); fig.tight_layout(); save(fig, 'deck_forecast')


def matrix_fig():
    T = pd.read_csv(RESULTS / 'qdecide_matrix_test.csv', index_col=0)
    base = T.loc['milp@chronos', 'total']
    names = dict(milp='MILP (classical)', samples='Best of 384 LP samples (classical)', exact='Exact QUBO (classical)', qaoa='LR-QAOA, 24 qubits',
                 wsqaoa='Warm-start QAOA', pce='PCE, 5 qubits', sqa='Simulated quantum annealing', sa='Simulated annealing (classical)')
    fcs = [('chronos', 'Chronos-2 forecast', BLUE, 'o'), ('ens', 'Chronos-2 + quantum reservoir', ORANGE, 'D'), ('qrc', 'Quantum reservoir forecast', VIOLET, 's')]
    R = pd.read_parquet(RESULTS / 'days_test_qdecide_chronos.parquet'); dd = R.groupby(['strategy', 'day']).total.sum().unstack(0)
    se = (dd.drop(columns='milp@chronos').sub(dd['milp@chronos'], axis=0)).sum().std() if False else (dd['qaoa@chronos'] - dd['milp@chronos']).std() * np.sqrt(len(dd))
    fig, ax = plt.subplots(figsize=(16, 7.5)); order = list(names); y = np.arange(len(order))[::-1]
    ax.axvspan(-2 * se / base * 100, 2 * se / base * 100, color='#e4e3df', zorder=0, label='±2 s.e. of a 25-day total (noise)')
    for fc, l, col, mk in fcs:
        v = [T.loc[f'{s}@{fc}', 'total'] / base * 100 - 100 for s in order]
        ax.scatter(v, y, s=120, color=col, marker=mk, label=l, zorder=3, edgecolor='#fcfcfb', linewidth=1.5)
    ax.axvline(0, color=INK2, lw=1); ax.set_yticks(y, [names[s] for s in order])
    ax.set_xlabel('Q4 € vs Chronos-2 + MILP (€1.81 M), %'); ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1)); fig.tight_layout(); save(fig, 'deck_matrix')
    return se


def markets_fig():
    T = pd.read_csv(RESULTS / 'qdecide_matrix_test.csv', index_col=0); O = pd.read_csv(RESULTS / 'summary_test_official.csv', index_col=0)
    rows = {'milp@chronos': T.loc['milp@chronos'], 'qaoa@chronos': T.loc['qaoa@chronos'], 'milp@qrc': T.loc['milp@qrc'], 'oracle': O.loc['oracle']}
    comp = [('cap', 'Capacity'), ('da', 'Day-ahead'), ('mfrr', 'mFRR energy'), ('afrr', 'aFRR energy'), ('imbalance', 'Imbalance'), ('tariff', 'Grid tariff'), ('leftover', 'Energy left')]
    fig, ax = plt.subplots(figsize=(16, 6.5)); w = .2; x = np.arange(len(comp))
    for i, (name, lab, col) in enumerate(MODELS):
        r = rows[name]; tariff = r['tariff'] + r['tariff_mtu'] if 'tariff_mtu' in r else r['tariff']
        v = [(tariff if k == 'tariff' else r[k]) / 1e6 for k, _ in comp]
        ax.bar(x + (i - 1.5) * w, v, w * .9, color=col, label=f"{lab}: €{r['total'] / 1e6:.2f} M")
    ax.axhline(0, color=INK2, lw=1); ax.set_xticks(x, [c for _, c in comp]); ax.set_ylabel('€ million, Q4 (1–25 Oct)')
    ax.legend(loc='upper left', bbox_to_anchor=(0, -.12), ncol=2); fig.tight_layout(); save(fig, 'deck_markets')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--day', default='2026-10-24'); a = ap.parse_args()
    forecast_fig(); se = matrix_fig(); markets_fig(); tot = day_figs(a.day)
    print('noise s.e. of 25-day total EUR', round(se), '| day totals', {k: round(v) for k, v in tot.items()})


if __name__ == '__main__':
    main()

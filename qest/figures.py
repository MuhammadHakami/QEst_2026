"""The deck's charts (PNG, 1600 px wide) from results/ and data/: every number drawn comes from those files.
usage: .venv/bin/python -m qest.figures   -> results/figs/*.png
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qest.paths import RAW, RESULTS

FIG = RESULTS / 'figs'
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, AQUA, YELLOW, VIOLET, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#6250d6', '#9a9893'
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'font.size': 15, 'axes.grid': True,
                     'grid.color': GRID, 'grid.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 17, 'axes.titleweight': 'bold', 'legend.frameon': False, 'font.family': 'DejaVu Sans'})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(FIG / f'{name}.png', dpi=200); plt.close(fig)


def usd():
    from qest.fetch_fx import eurusd
    return eurusd()['latest']


def assets_map():
    """Where the assets are: ERA5 cloud-pattern match (r per 0.25 deg cell), the solar park, the battery, every wind plant."""
    from qest.sites import BATTERY, SOLAR, locate_solar, wind_plants, PORTFOLIO_WIND
    r = pd.DataFrame(locate_solar(), columns=['r', 'lat', 'lon'])
    fig, ax = plt.subplots(figsize=(8, 5.2))
    sc = ax.scatter(r.lon, r.lat, c=r.r, cmap='Blues', s=150, marker='s', vmin=.4, vmax=.76, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, shrink=.8); cb.set_label('match r: park forecast vs ERA5 cloud pattern')
    W = wind_plants()
    for n, p in W.items():
        if p.get('mw'):
            ax.scatter(p['lon'], p['lat'], s=8 + 2 * p['mw'], facecolor='none', edgecolor=AQUA, lw=2)
    pw = W[PORTFOLIO_WIND]
    ax.scatter([SOLAR['lon']], [SOLAR['lat']], s=260, marker='*', color=ORANGE, edgecolor=SURF, lw=1.5, zorder=5)
    ax.scatter([BATTERY['lon']], [BATTERY['lat']], s=200, marker='D', color=VIOLET, edgecolor=SURF, lw=1.5, zorder=5)
    ax.scatter([pw['lon']], [pw['lat']], s=200, marker='^', color=AQUA, edgecolor=SURF, lw=1.5, zorder=5)
    ax.annotate(f"Pihlaka PV 42.3 MW\n(peak r = {r.r.max():.2f})", (SOLAR['lon'], SOLAR['lat']), (SOLAR['lon'] + .5, SOLAR['lat'] + .35), fontsize=13, arrowprops=dict(arrowstyle='-', color=INK2))
    ax.annotate('Kiisa battery\n100 MW / 200 MWh', (BATTERY['lon'], BATTERY['lat']), (BATTERY['lon'] - 2.6, BATTERY['lat'] - .55), fontsize=13, arrowprops=dict(arrowstyle='-', color=INK2))
    ax.annotate('Paldiski wind 45 MW', (pw['lon'], pw['lat']), (pw['lon'] - 1.6, pw['lat'] + .35), fontsize=13, arrowprops=dict(arrowstyle='-', color=INK2))
    ax.set_xlabel('longitude E'); ax.set_ylabel('latitude N'); ax.set_aspect(1 / np.cos(np.radians(58.7)))
    save(fig, 'assets_map')


def sim_week():
    """One October week of the simulated Estonia: prices, PV and wind, actual vs the D-1 forecast."""
    from qest.data import frame
    from qest.power import generation
    f = frame(); g = generation(); w = slice('2026-10-06', '2026-10-12')
    loc = f.local.loc[w]
    fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    axs[0].plot(loc, f.nps_dayahead_eur_mwh.loc[w], color=BLUE, lw=2, label='day-ahead')
    axs[0].plot(loc, f.imbalance_eur_mwh.loc[w].clip(-300, 600), color=GREY, lw=1, alpha=.8, label='imbalance (clipped)')
    axs[0].set_ylabel('EUR/MWh'); axs[0].legend(ncol=2, loc='upper left')
    axs[1].plot(loc, g.pv_actual.loc[w], color=ORANGE, lw=2, label='Pihlaka PV (metered proxy)')
    axs[1].plot(loc, g.pv_d1.loc[w], color=INK2, lw=1.2, ls='--', label='D-1 forecast')
    axs[1].set_ylabel('MW'); axs[1].legend(ncol=2, loc='upper left')
    axs[2].plot(loc, g.wind_actual.loc[w], color=AQUA, lw=2, label='Paldiski wind (ERA5)')
    axs[2].plot(loc, g.wind_d1.loc[w], color=INK2, lw=1.2, ls='--', label='D-1 forecast')
    axs[2].set_ylabel('MW'); axs[2].legend(ncol=2, loc='upper left')
    save(fig, 'sim_week')


def sim2real():
    """Wind sim2real: Estonia's whole OSM fleet through our power curves vs Elering's measured renewable output."""
    from qest.data import frame
    from qest.forecast import national_wind
    f = frame(); a = national_wind('actual').reindex(f.index); e = f.ee_real_production_renewable
    d = pd.DataFrame(dict(a=a, e=e, split=f.split)).dropna(); d = d.resample('D').mean(numeric_only=True).dropna()
    r = np.corrcoef(d.a, d.e)[0, 1]
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.scatter(d.e, d.a, s=26, color=AQUA, alpha=.8, edgecolor=SURF, lw=.5)
    m = max(d.a.max(), d.e.max()); ax.plot([0, m], [0, m], color=INK2, lw=1, ls='--')
    ax.set_xlabel('Elering measured renewables, daily mean MW'); ax.set_ylabel('our simulated wind fleet, MW')
    ax.text(.03, .93, f'r = {r:.2f} per day', transform=ax.transAxes, fontsize=16, weight='bold'); save(fig, 'sim2real')
    return r


NAMES = dict(oracle='hindsight (not tradable)', milp='MILP · LightGBM', milp_persist='MILP · persistence', milp_chronos='MILP · Chronos-2',
             milp_rl='MILP + PPO knobs', milp_da='MILP, day-ahead only', reserve_rule='reserve rule', rule='cheap/dear rule', idle='idle', naive_da='naive DA')


def strategies(split='test', settlement='official'):
    d = pd.read_parquet(RESULTS / f'days_{split}_{settlement}.parquet')
    rl = RESULTS / 'rl_ppo.csv'
    if rl.exists():
        x = pd.read_csv(rl); x = x[x.split == split]
        d = pd.concat([d, pd.DataFrame(dict(strategy='milp_rl', unit='all', day=pd.to_datetime(x.day), total=x.eur))])
    t = d.groupby('strategy').total.sum().sort_values() * usd() / 1e6
    t = t.drop([k for k in ('naive_da',) if k in t and 'idle' in t and abs(t[k] - t['idle']) < 1e-9])
    fig, ax = plt.subplots(figsize=(10, 5.6))
    col = [GREY if k == 'oracle' else BLUE if k.startswith('milp') else '#b9b7b1' for k in t.index]
    ax.barh([NAMES.get(k, k) for k in t.index], t.values, color=col, height=.62)
    for i, v in enumerate(t.values):
        ax.text(v + t.max() * .01, i, f'${v:.2f} M', va='center', fontsize=14, color=INK)
    ax.set_xlabel('USD millions, 1-25 Oct 2026 (Q4 test)'); ax.grid(axis='y', visible=False)
    ax.set_xlim(0, t.max() * 1.15); save(fig, f'strategies_{split}')
    days = d[d.strategy != 'milp_rl'].groupby(['day', 'strategy']).total.sum().unstack().cumsum() * usd() / 1e6
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, c in (('oracle', GREY), ('milp', BLUE), ('milp_chronos', VIOLET), ('milp_persist', AQUA), ('milp_da', YELLOW), ('rule', '#b9b7b1')):
        if k in days:
            ax.plot(days.index, days[k], color=c, lw=2.5 if k == 'milp' else 2, ls='--' if k == 'oracle' else '-', label=NAMES[k])
    ax.set_ylabel('cumulative USD millions'); ax.legend(ncol=2, fontsize=13); save(fig, f'cumulative_{split}')
    return t


def participation():
    out = {}
    for split in ('val', 'test'):
        d = pd.read_parquet(RESULTS / f'days_{split}_official.parquet').groupby(['day', 'strategy']).total.sum().unstack()
        out[split] = (d.milp - d.milp_da) * usd() / 1e3
    g = out['test']
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(g.index, g.values, color=[BLUE if v > 0 else ORANGE for v in g.values], width=.75)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_ylabel('USD k per day'); ax.set_title(f'Reserves on top of day-ahead: won {int((g > 0).sum())} of {len(g)} Q4 days '
                                                f'(Q3: {int((out["val"] > 0).sum())} of {len(out["val"])})')
    save(fig, 'participation'); return {k: (int((v > 0).sum()), len(v), float(v.sum())) for k, v in out.items()}


QLABEL = dict(exact='exact (classical)', sa='simulated annealing', tabu='tabu search', random_polish='random + polish', random_4096='random, 4096 shots',
              qaoa_p1='QAOA p=1', qaoa_p3='QAOA p=3', sqa='simulated QA', bfdcqo='BF-DCQO', falqon='FALQON', pce='PCE')
QUANT = ('qaoa_p1', 'qaoa_p3', 'sqa', 'bfdcqo', 'falqon', 'pce')


def quantum(mode='retro'):
    rows = []
    for lvl in ('P1', 'P2', 'P3'):
        p = RESULTS / f'qubo_{lvl}_test_{mode}.csv'
        if not p.exists():
            continue
        r = pd.read_csv(p); ex = r.exact_eur.sum()
        for k in ['exact', 'sa', 'tabu', 'random_polish', 'random_4096', *QUANT]:
            if f'{k}_eur' in r:
                rows.append(dict(level=lvl, qubits=int(r.qubits.iloc[0]), solver=k, eur=r[f'{k}_eur'].sum(), pct=r[f'{k}_eur'].sum() / ex * 100,
                                 opt=r[f'{k}_opt'].mean() if f'{k}_opt' in r else 1.0, s=r[f'{k}_s'].mean() if f'{k}_s' in r else r.exact_s.mean(),
                                 popt=r[f'{k}_popt'].mean() if f'{k}_popt' in r else np.nan, n=len(r)))
    T = pd.DataFrame(rows)
    if T.empty:
        return T
    lv = list(T.level.unique()); sol = [s for s in QLABEL if s in set(T.solver)]
    M = T.pivot(index='solver', columns='level', values='pct').reindex(index=sol, columns=lv)
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    ax.imshow(M.fillna(np.nan).to_numpy(), cmap='Blues', vmin=0, vmax=130, aspect='auto'); ax.grid(False)
    for (r, c), v in np.ndenumerate(M.to_numpy()):
        txt = 'not run: 2^30 states' if np.isnan(v) else (f'{v:.2f} %' if v < 99.995 else '100 %')
        ax.text(c, r, txt, ha='center', va='center', fontsize=14 if not np.isnan(v) else 12, color=SURF if (not np.isnan(v) and v > 75) else INK2,
                weight='bold' if (not np.isnan(v) and v >= 99.995) else 'normal')
    ax.set_xticks(range(len(lv))); ax.set_xticklabels([f'{L} · {int(T[T.level == L].qubits.iloc[0])} qubits' for L in lv]); ax.xaxis.tick_top()
    ax.set_yticks(range(len(sol))); ax.set_yticklabels([QLABEL[s] for s in sol])
    for sp in ax.spines.values(): sp.set_visible(False)
    save(fig, f'quantum_gap_{mode}')
    return T


def payback(value_per_pct_usd_yr):
    """Break-even: improvement over the classical MILP a quantum service must deliver to repay its cost in a year."""
    cost = np.array([30e3, 100e3, 300e3]) * usd()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.linspace(0, 3, 200); ax.plot(x, x * value_per_pct_usd_yr / 1e3, color=BLUE, lw=3, label='yearly value of a better plan')
    for c, ls in zip(cost, (':', '--', '-.')):
        ax.axhline(c / 1e3, color=ORANGE, lw=2, ls=ls)
        be = c / value_per_pct_usd_yr; ax.text(3.02, c / 1e3, f'€{c / usd() / 1e3:.0f}k → break-even at +{be:.2f} %', va='center', fontsize=13)
    ax.set_xlabel('gain over the classical plan, % of yearly portfolio EUR'); ax.set_ylabel('USD k per year'); ax.set_xlim(0, 3)
    ax.legend(loc='upper left'); save(fig, 'payback')


def forecast_skill(split='test'):
    """Gate-A forecast error, DA and capacity prices: Chronos-2 zero-shot vs LightGBM vs persistence."""
    from qest.data import days, frame
    from qest.evaluate import forecasts
    from qest.forecast import skill
    D = days(frame(), split); K = ['da', 'cap_afrr_up', 'cap_afrr_down', 'cap_mfrr_up', 'cap_mfrr_down']
    S = {n: skill(forecasts(split, k), D).mae.reindex(K) for n, k in (('Chronos-2 (zero-shot)', 'chronos'), ('LightGBM', 'lgbm'), ('persistence', 'persist'))}
    rel = pd.DataFrame(S).div(pd.DataFrame(S)['persistence'], axis=0) * 100
    fig, ax = plt.subplots(figsize=(10, 4.8)); w = .27
    lab = ['DA price', 'aFRR UP cap', 'aFRR DOWN cap', 'mFRR UP cap', 'mFRR DOWN cap']
    for i, (n, c) in enumerate(zip(rel, (VIOLET, BLUE, '#b9b7b1'))):
        ax.bar(np.arange(len(K)) + (i - 1) * w, rel[n], width=w * .92, color=c, label=n)
    ax.set_xticks(range(len(K))); ax.set_xticklabels(lab); ax.set_ylabel('MAE, % of persistence (lower is better)')
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(.5, 1.12)); ax.grid(axis='x', visible=False); save(fig, f'forecast_{split}')
    return pd.DataFrame(S)


def quantum_popt(T):
    """Probability mass the simulated quantum state puts on the optimum, vs uniform sampling (P1, P2)."""
    t = T[T.solver.isin(QUANT) & T.popt.notna()]
    fig, ax = plt.subplots(figsize=(10, 4.8)); lv = list(t.level.unique()); sol = [s for s in QUANT if s in set(t.solver)]; w = .8 / max(len(lv), 1)
    for i, L in enumerate(lv):
        x = t[t.level == L].set_index('solver').reindex(sol); q = int(T[T.level == L].qubits.iloc[0])
        ax.bar(np.arange(len(sol)) + (i - (len(lv) - 1) / 2) * w, x.popt * 100, width=w * .92, color=[BLUE, AQUA][i], label=f'{L} · {q} qubits (uniform {100 / 2 ** q:.3g} %)')
    ax.set_yscale('log'); ax.set_xticks(range(len(sol))); ax.set_xticklabels([QLABEL[s] for s in sol])
    ax.set_ylabel('P(optimum) in the state, %'); ax.legend(); ax.grid(axis='x', visible=False); save(fig, 'quantum_popt')


def quantum_time(T):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for s_, c in (('exact', INK), ('sa', BLUE), ('tabu', AQUA), ('qaoa_p3', ORANGE), ('bfdcqo', VIOLET), ('pce', YELLOW)):
        x = T[T.solver == s_].sort_values('qubits')
        if len(x):
            ax.plot(x.qubits, x.s * 1e3, 'o-', color=c, lw=2, ms=8, label=QLABEL[s_])
    ax.set_yscale('log'); ax.set_xlabel('qubits (P1, P2, P3)'); ax.set_ylabel('ms per instance (simulated)'); ax.legend(fontsize=12); save(fig, 'quantum_time')



QN = dict(qaoa_x_p1='QAOA p=1', qaoa_x_p3='QAOA p=3', bfdcqo='BF-DCQO', xy_ring_p1='XY mixer p=1', nas_best='NAS best',
          nas_lean='NAS lean', postnas='NAS + PostNAS', warm_only='warm start, no circuit', nas_uniform='NAS, uniform start', random_onehot='4096 random valid plans')


def qnas_pareto(levels=('P1', 'P2', 'P3')):
    """The architecture search on Q3: every trial's P(optimum) against its two-qubit gates, the Pareto front, PostNAS."""
    lv = [L for L in levels if (RESULTS / f'qnas_{L}_nas.csv').exists()]
    fig, axs = plt.subplots(1, len(lv), figsize=(5.6 * len(lv), 4.9), squeeze=False)
    for ax, L in zip(axs[0], lv):
        T = pd.read_csv(RESULTS / f'qnas_{L}_nas.csv'); P = pd.read_csv(RESULTS / f'qnas_{L}_postnas.csv')
        sp = T.arch.map(lambda a: json.loads(a)['space'])
        for k, c, lab in (('x', ORANGE, 'X mixer (plain QAOA)'), ('xy', BLUE, 'XY one-hot mixer')):
            x = T[sp == k]; ax.scatter(x.values_1, x.values_0 * 100, s=46, color=c, alpha=.75, edgecolor=SURF, lw=.8, label=lab)
        F = T[T.pareto].sort_values('values_1'); ax.step(F.values_1, F.values_0 * 100, where='post', color=INK2, lw=1.5, label='Pareto front')
        ax.scatter(P.values_1, P.values_0 * 100, s=46, marker='D', facecolor='none', edgecolor=VIOLET, lw=1.5, label='PostNAS (frozen angles)')
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('two-qubit gates per circuit'); ax.set_title(L, loc='left')
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.set_xticks([t_ for t_ in (20, 50, 100, 200, 500, 1000) if F.values_1.min() / 2 <= t_ <= T.values_1.max() * 1.5])
    axs[0][0].set_ylabel('P(optimum) in the state, %'); axs[0][0].legend(fontsize=11, loc='lower right'); save(fig, 'qnas_pareto')


def qnas_test(levels=('P1', 'P2', 'P3')):
    """Q4, scored once: P(optimum) of the picked circuits against the hand-built ones, two-qubit gates printed."""
    lv = [L for L in levels if (RESULTS / f'qnas_{L}_test.csv').exists()]
    fig, axs = plt.subplots(1, len(lv), figsize=(5.6 * len(lv), 4.9), squeeze=False, sharey=False)
    for ax, L in zip(axs[0], lv):
        D = pd.read_csv(RESULTS / f'qnas_{L}_test.csv').drop_duplicates('name'); D = D[D.name != 'nas_lean'][::-1]
        if L == 'P3': D = D[D.name != 'nas_best']  # identical to NAS + PostNAS at P3 (PostNAS kept the circuit)
        col = [VIOLET if n == 'postnas' else GREY if n in ('warm_only', 'random_onehot') else BLUE if n.startswith(('nas', 'xy')) else ORANGE for n in D.name]
        y = np.arange(len(D)); ax.barh(y, D.popt * 100, color=col, height=.7)
        for yi, (pv, g) in enumerate(zip(D.popt * 100, D.cz)):
            ax.text(pv * 1.08, yi, f'{pv:.1f} %  ·  {g:.0f} CZ', va='center', fontsize=12, color=INK2)
        ax.set_xscale('log'); ax.set_yticks(y); ax.set_yticklabels([QN[n] for n in D.name]); ax.grid(axis='y', visible=False)
        ax.set_xlim(.03, 3e4); ax.set_xticks([.1, 1, 10, 100]); ax.set_xticklabels(['0.1', '1', '10', '100'])
        ax.set_xlabel('P(optimum), %'); ax.set_title(L, loc='left')
    save(fig, 'qnas_test')



HN = dict(oracle='hindsight (not tradable)', milp_chronos='Chronos-2 + MILP (classical baseline)',
          qpolicy='+ quantum policy sets the knobs (VQC-PPO)', qday_qaoa='+ quantum day pattern (25-qubit LR-QAOA)',
          milp_chronos_qrc='+ quantum reservoir forecast fix', qday_random='control: random day pattern',
          milp_chronos_rff='control: random-Fourier forecast fix', qday_surrogate='control: best surrogate pattern (exact)',
          benders_sqa='+ annealing Benders (SQA master)', benders_sa='control: Benders, SA master', benders_exact='control: Benders, exact master')


def hybrid(split='test'):
    """Q4 money: the Chronos-2 baseline against every quantum / hybrid strategy and its classical control."""
    d = pd.read_parquet(RESULTS / f'days_{split}_hybrid.parquet'); o = pd.read_parquet(RESULTS / f'days_{split}_official.parquet')
    d = pd.concat([d, o[o.strategy == 'oracle']]); q = pd.read_csv(RESULTS / 'rl_qpolicy.csv'); q = q[q.split == split]
    d = pd.concat([d, pd.DataFrame(dict(strategy='qpolicy', unit='all', day=pd.to_datetime(q.day), total=q.eur))]
                  + [pd.read_parquet(f) for f in sorted(RESULTS.glob(f'days_{split}_benders_*.parquet'))])
    t = d.groupby('strategy').total.sum() * usd() / 1e6; base = t['milp_chronos']
    t = t.sort_values()
    fig, ax = plt.subplots(figsize=(11, 6.8))
    ctrl = ('qday_random', 'milp_chronos_rff', 'qday_surrogate', 'benders_sa', 'benders_exact')
    col = [GREY if k == 'oracle' else BLUE if k == 'milp_chronos' else '#c9c7c1' if k in ctrl else ORANGE if k == 'benders_sqa' else VIOLET for k in t.index]
    ax.barh([HN[k] for k in t.index], t.values, color=col, height=.62)
    for i, (k, v) in enumerate(t.items()):
        lab = f'${v:.2f} M' + ('' if k in ('milp_chronos', 'oracle') else f'  ({(v / base - 1) * 100:+.1f} %)')
        ax.text(v + t.max() * .01, i, lab, va='center', fontsize=13, color=INK)
    ax.set_xlabel('USD millions, Q4 test (1-25 Oct), official scoring'); ax.grid(axis='y', visible=False)
    ax.set_xlim(0, t.max() * 1.22); save(fig, f'hybrid_{split}')
    c = d.groupby(['day', 'strategy']).total.sum().unstack().cumsum() * usd() / 1e6
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, cl, ls in (('oracle', GREY, '--'), ('milp_chronos', BLUE, '-'), ('qpolicy', VIOLET, '-'), ('qday_qaoa', ORANGE, '-'), ('milp_chronos_qrc', AQUA, '-')):
        ax.plot(c.index, c[k], color=cl, lw=2.5 if k == 'milp_chronos' else 2, ls=ls, label=HN[k].lstrip('+ '))
    ax.set_ylabel('cumulative USD millions'); ax.legend(fontsize=12); save(fig, f'cumulative_hybrid_{split}')
    return t, base


if __name__ == '__main__':
    assets_map(); sim_week(); print('sim2real r', sim2real())
    print(forecast_skill().round(2).to_string()); print(strategies()); print(participation())
    T = quantum(); print(T.to_string()); quantum_popt(T); quantum_time(T)
    qnas_pareto(); qnas_test(); t, base = hybrid(); print(t.round(3).to_string())
    payback(base * 1e6 * 365 / 25 / 100)  # 1 % of a year at the baseline's Q4 pace, USD

td = lambda x, b=False, c=INK: f'<td style="padding:12px 16px;border-bottom:1px solid {LINE};font-size:24px;{"font-weight:700;" if b else ""}color:{c};text-align:{"left" if x[:1].isalpha() or x[:1] == "~" else "right"}">{x}</td>'
th = lambda x: f'<th style="padding:10px 16px;border-bottom:2px solid {INK};font-size:20px;color:{MUT};text-align:left;font-weight:600">{x}</th>'
tbl = lambda heads, rows: f"<table style=\"border-collapse:collapse;width:100%\"><tr>{''.join(th(h) for h in heads)}</tr>" + ''.join('<tr>' + ''.join(td(*c) if isinstance(c, tuple) else td(c) for c in r) + '</tr>' for r in rows) + '</table>'
M = lambda v: f'€{v/1e6:.2f} M'
stat = lambda v, t, col: f'<div style="flex:1;padding:30px;border-radius:16px;background:#F1F3F6"><p style="{H};font-size:72px;font-weight:700;color:{col}">{v}</p><p style="font-size:25px;color:{MUT};line-height:1.35;margin-top:8px">{t}</p></div>'
bar = lambda t, pct, col, note='': f'<div style="display:flex;align-items:center;gap:22px;padding:10px 0;border-bottom:1px solid {LINE}"><p style="width:560px;font-size:26px;font-weight:600">{t}</p><div style="flex:1;height:28px;background:#EDEFF2;border-radius:0 6px 6px 0"><div style="width:{max(pct,0.6)}%;height:100%;background:{col};border-radius:0 6px 6px 0"></div></div><p style="width:110px;text-align:right;font-size:30px;font-weight:700">{pct:g} %</p><p style="width:330px;font-size:21px;color:{MUT}">{note}</p></div>'
# season (291 traded days, 8 Jan - 25 Oct 2026, battery + hybrid), verified against results/*.parquet on 9 Nov
BASE, QAOA, NQS, SAMP, REACH, HIND = 37_602_146, 37_553_783, 37_409_048, 37_497_099, 46_478_701, 105_177_733
S['gap'] = sec('gap', h2(f"Where the €{(HIND-BASE)/1e6:.1f} M between our plan and hindsight sits: in the TSO's activations, not in prices",
  f'Season, both sites: Chronos-2 + MILP {M(BASE)} · hindsight {M(HIND)}. Each bar: the same MILP given one input group from the realised day (Q4; season in the note).') + f"""
{bar('aFRR + mFRR activations', 87, ORANGE, 'season 82 % · €93.1 M')}
{bar('mFRR activations only', 48.8, ORANGE)}
{bar('aFRR activations only', 34.1, ORANGE)}
{bar('Day-ahead price', 8.0, BLUE, 'season 6 % · the whole EPF literature')}
{bar('Capacity prices', 0.7, BLUE)}
{bar('Hybrid PV output', 0, AQUA, 'nothing')}
<div style="display:flex;gap:20px;margin-top:6px">
{stat(M(REACH), f'reachable optimum: perfect prices, forecast activations. The baseline already captures {100*BASE/REACH:.0f} % of it; the €{(HIND-REACH)/1e6:.1f} M above it is foresight no bidder has at D-1', BLUE)}
{rule('Activations: noise at D-1, persistent at T−25', 'P(UP now | UP 30 min ago) = 0.80. But inside this market model the money is fixed at 13:00: re-planning at T−25 is −0.5 %, bid gating −4 to −10 %, scenario / CVaR plans −17 to −25 %. Without an intraday market, D-day information cannot move it.')}
</div>""", gap=22)
CAP = [("none (organisers' rules)", 1_812_548, 2_199_338, 5_748_379, 6_335_393), ('50 MW', 1_478_017, None, 3_205_962, 4_139_778), ('25 MW', 1_328_436, 1_798_623, 2_428_645, 3_214_004)]
f0 = lambda v: f'{v:,.0f}' if v else '—'
S['real'] = sec('real', h2('Making the environment Estonian: Elering activates the MW it needs, not our whole bid',
  "The settlement activated the full won MW whenever the price crossed the bid; MARI (mFRR, since Oct 2024) and PICASSO (aFRR, since Apr 2025) activate in merit order to the TSO's need. A volume cap brackets it until the Baltic dashboard's activated volumes are in; the planner is told the cap.") + tbl(
  ['Activation volume cap (Q4)', 'Chronos-2 + MILP', 'Reachable (perfect prices)', 'Perfect activations', 'Hindsight', 'Gap that is price', 'Gap that is foresight'],
  [[(c, True), f0(b), f0(r), f0(a), f0(h), f'{100*(r-b)/(h-b):.0f} %' if r else '—', f'{100*(h-r)/(h-b):.0f} %' if r else f'~{100*(a-b)/(h-b):.0f} %'] for c, b, r, a, h in CAP]) + f"""
<div style="display:flex;gap:20px">
{rule('Q4 euros, both sites', 'At 25 MW the ceiling is 2.4× the baseline, not 3.5×; the forecastable part of the gap grows to a quarter. Ramps (mFRR 93 %, aFRR 97 % of the MW over an MTU) move every line ~2 % and change nothing structural.')}
{rule('Audited row by row', 'BBCM capacity auction (marginal pricing, €4,000/MW/h cap), SDAC 15-min, T−25 energy bids, single imbalance price, gate-A information set. research/environment_audit.md.')}
{rule('Still missing', 'Activated volumes and reserve demand per MTU (Baltic Transparency Dashboard, geo-blocked here; CSV route ready) and intraday prices (Nord Pool, paid). With them the cap becomes the real series.')}
</div>""", gap=24)
S['distill'] = sec('distill', h2('The IonQ move, applied to bidding: sample the QPU once, distil it, decide all season without it',
  'arXiv:2611.08129 trains a neural state on QPU shadows and drops the QPU from production (13,200 GPU h → 4 GPU h + 6,680 CPU core h). Ours: LR-QAOA samples of 1,517 Q3 windows train a conditional transformer that proposes every 12-qubit window of every day.') + f"""
<div style="display:flex;gap:20px">
{stat('5,878 → 1,891 → 0', 'QPU circuits a season: live partitioned QAOA every day → one Q3 training run → the season on the model', ORANGE)}
{stat('564 M → 1.9 M', 'QPU shots if the decision is re-made every 15 min (96 a day, both units): live → once. The classical side is 2 GPU h → 5 CPU core h', ORANGE)}
{stat('$811 → $0', 'per decision on IonQ Forte ($18 IQM, $7 Rigetti, $16–65 IBM live); decision quality kept: optimum hit 0.76 live vs 0.74 distilled, Q4 €1,800,367 vs €1,802,092', AQUA)}
</div>
{tbl(['Route, 2026 season (582 day-units)', 'Circuits', 'IonQ Forte', 'IQM Garnet', 'Rigetti Cepheus', 'IBM (1–4 s a job)', 'Recurs'],
 [[('Live partitioned LR-QAOA, 1,000 shots a window', True), '5,878', '$472 k', '$10.3 k', '$4.3 k', '$9.4–37.6 k', 'yearly'],
  [('Live monolithic LR-QAOA, 24 qubits', True), '582', '$93 k', '$1.9 k', '$0.7 k', '$0.9–3.7 k', 'yearly'],
  [('Distilled: Q3 training windows', True, AQUA), '1,891', '$152 k', '$3.3 k', '$1.4 k', '$3.0–12.1 k', 'once'],
  [('Distilled: the season on the model (4 vCPU)', True, AQUA), '0', '$0', '$0', '$0', '$0', '—']])}
<p style="font-size:21px;color:{MUT}">List prices Nov 2026: Braket $0.30 a task + per shot (Forte $0.08, Garnet $0.00145, Cepheus $0.000425); IBM $96 a QPU-minute. The noise calibration a, learned under a ridge (paper Eq. 15), lands at 0.90 and recovers 8 % of the noisy-to-ideal gap: our bit strings have no symmetry sector for it to grip. A classical twin trained on classical samples does as well: the value is amortising the sampler, not quantum data.</p>""", gap=22)
Y = 365 / 291
PB = [('Chronos-2 + MILP (classical)', BASE, '—', '≈ €1–3k compute', '—', 'baseline', BLUE),
      ('Chronos-2 + LR-QAOA, 24 qubits, live every day', QAOA, '−€48k ± €163k', '$0.7k–93k QPU', '−€49k to −€141k', 'never', ORANGE),
      ('Chronos-2 + distilled QPU (0 QPU calls)', NQS, '−€193k ± €159k', '$1.4k–152k once', '−€194k to −€345k', 'never', VIOLET),
      ('Best of 384 QUBO samples (classical control)', SAMP, '−€105k ± €139k', '≈ €1–3k', 'not significant', '—', MUT),
      ('Reachable optimum (perfect prices)', REACH, f'+€{(REACH-BASE)/1e6:.2f} M', '—', "the forecaster's prize", '—', AQUA)]
S['payback'] = sec('payback', h2(f'Expected 2026 revenue: €{BASE/1e6:.1f} M classical, €{NQS/1e6:.1f}–{QAOA/1e6:.1f} M hybrid; the QPU bill is paid back by nothing yet',
  f'291 traded days (8 Jan–25 Oct), battery + hybrid, official settlement; annualised ×{Y:.2f}. ± is the 95 % interval of the daily differences over 291 days. QPU cost spans Rigetti → IonQ Forte list prices; the distilled route pays once.') + tbl(
  ['Pipeline', 'Revenue, season', 'Annualised', 'vs baseline', 'Cost / yr', 'Net vs baseline / yr', 'Payback'],
  [[(a, True, c), f'€{v:,.0f}', M(v * Y), u, k, nn, (pp, True)] for a, v, u, k, nn, pp, c in PB]) + f"""
<div style="display:flex;gap:20px">
{rule('Break-even uplift', f'€30k / €{BASE/1e6:.1f} M = {100*30e3/BASE:.2f} % (€{30e3/291:,.0f} a day) · €300k = {100*300e3/BASE:.2f} % (€{300e3/291:,.0f} a day). The live QPU sits at −0.13 %, the distilled one at −0.51 %.')}
{rule('Under the Estonian cap (25 MW, Q4)', 'MILP €1,328,436 · distilled €1,315,734 (−0.96 %) · live QAOA €1,313,247 (−1.14 %): the ranking survives the faithful market.')}
{rule('What would pay', f'The price lever: €{(REACH-BASE)/1e6:.1f} M a season sits between the baseline and the reachable optimum. Cross-zonal Chronos-2 (FI / LV / LT prices) cut DA error 4.7 % and earned ±0 %: rank order, not level, moves the plan.')}
</div>""", gap=22)
S['verdict'] = sec('verdict', h2('Verdict: parity in decisions, a QPU made free at decision time, no euro advantage yet') + f'''
<div style="display:flex;gap:24px">
{stat('−0.13 %', f'LR-QAOA (24 qubits, CUDA-Q) vs the MILP over the 2026 season: {M(QAOA)} vs {M(BASE)}, inside the ±€163k noise', ORANGE)}
{stat('0 QPU calls', f'distilled sampler: the live QPU\\'s optimum-hit rate, {M(NQS)} (−0.51 %), 4× faster per decision, 300× fewer shots over a season', VIOLET)}
{stat('81 %', f'of the reachable optimum already captured ({M(BASE)} of {M(REACH)}); hindsight\\'s €{HIND/1e6:.0f} M is mostly activation foresight', BLUE)}
</div>
<div style="display:flex;gap:20px">
{rule('Why parity', "At 24–25 binaries a GPU enumerates the QUBO in a second; every decider returns the MILP's pattern on ~80 % of days. One forecast, one optimum: the sampler cannot add information.")}
{rule('Where the qubits did a measured job', 'Quantum-proposal MCMC on 16-qubit activation windows: TV to exact 0.072 vs 0.091, autocorrelation 27 vs 52 steps (Layden-style). It matters once aFRR, the neighbours and Finland couple the target beyond a chain.')}
{rule('Next', "Load Elering's activated volumes (CSV inbox ready) so the cap becomes the real series; run the distilled training set on IQM / IBM for about $3k; score forecasters by within-day rank, not MAE.")}
</div>''')

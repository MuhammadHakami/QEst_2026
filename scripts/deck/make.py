import json, os, pathlib
D = pathlib.Path(__file__).parent / 'project'; (D / 'slides').mkdir(parents=True, exist_ok=True)
BG, INK, MUT, BLUE, ORANGE, AQUA, VIOLET, LINE = '#FBFBF8', '#14213D', '#4A5568', '#2a78d6', '#eb6834', '#1baf7a', '#6250d6', '#D9DDE3'
H = "font-family:'Space Grotesk', Arial, sans-serif"
B = {k: f'/_blob/{v}' for k, v in dict(prices='e867ff42a5e7db0cbf42db716dad9a0b', bids='11513ee98556b268afe13f2a4ab82187', deur='dc8628fcf8830837ff57985b37424699',
     fc='64ab58864cf1d74f31e6b628f452cc84', matrix='5df24deb8145c5499b74f34cb34b3096', markets='c812b5124ab45a3e194e555e2c5282bd',
     solar='610327499d49458290b304dacfca0ee6', bat='69251f0cba07e2933d05623f580cd510', wind='34ee5a00f46beaa59f22edf37df9efb4').items()}
def sec(i, body, pad='96px 112px 112px', gap=40):
    return f'<section id="{i}" data-transition="fade" style="background:{BG};color:{INK};font-family:Inter, Arial, sans-serif;padding:{pad};display:flex;flex-direction:column;gap:{gap}px">\n{body}\n</section>\n'
def h2(t, sub=None):
    s = f'<h2 style="{H};font-size:64px;font-weight:700;line-height:1.08">{t}</h2>'
    return s + (f'\n<p style="font-size:30px;color:{MUT};line-height:1.35;max-width:1600px">{sub}</p>' if sub else '')
def img(k, h=None):
    return f'<img src="{B[k]}" alt="" style="width:100%;{f"max-height:{h}px;" if h else ""}object-fit:contain;display:block">'
S = {}
S['cover'] = sec('cover', f'''<p style="font-size:30px;color:{BLUE};font-weight:600;letter-spacing:.04em">QEST 2026 · TEAM KRATTRADE</p>
<h1 style="{H};font-size:104px;font-weight:700;line-height:1.02;max-width:1500px">Quantum against classical, bid for bid, on Estonia's real 2026 market</h1>
<p style="font-size:36px;color:{MUT};line-height:1.35;max-width:1450px">Chronos-2 + MILP is the classical baseline. Quantum forecasters and deciders from the 2026 literature replace each part on NVIDIA CUDA-Q, a QPU is distilled IonQ-style into a model that needs no QPU at decision time, and everything is settled on the organisers' rules over the 291 traded days of 2026.</p>''', pad='160px 128px', gap=48)
gate = lambda c, t, h, d, bg=BG: f'<div style="flex:1;display:flex;flex-direction:column;gap:10px;padding:28px;border-radius:14px;background:{bg};border-top:8px solid {c}"><p style="font-size:24px;color:{c};font-weight:700">{t}</p><h3 style="font-size:34px;font-weight:600">{h}</h3><p style="font-size:24px;color:{MUT};line-height:1.35">{d}</p></div>'
rule = lambda t, d: f'<div style="flex:1;padding:22px 26px;border:2px solid {LINE};border-radius:14px"><p style="font-size:24px;font-weight:700">{t}</p><p style="font-size:22px;color:{MUT};line-height:1.35;margin-top:8px">{d}</p></div>'
S['gates'] = sec('gates', h2('Three gates, one battery: every MW is committed before the price is known') + f'''
<div style="display:flex;gap:20px">
{gate(BLUE, 'A · D-1 09:00 → results ~09:30', 'Reserve capacity', 'aFRR / mFRR, UP and DOWN, whole MW ≥ 1. Paid €/MW/h to stand by. Won MW are held back from day-ahead.', '#EAF1FB')}
{gate(BLUE, 'B · D-1 13:00 → results ~14:00', 'Day-ahead', 'One position for each of the 96 MTUs, in the room the reserves leave free. Pay-as-cleared.', '#EAF1FB')}
{gate(BLUE, 'C · T−25 min, every MTU', 'Energy bids', 'Each won MW needs an energy bid. UP activates if price ≥ bid, DOWN if price ≤ bid. No changes after T−25.', '#EAF1FB')}
{gate(ORANGE, 'T → T+15 · settlement', 'Delivery', 'Metered − committed settles at the imbalance price. Imports pay the grid tariff (€30 day / €20 night).', '#FCEEE7')}
</div>
<div style="display:flex;gap:20px">
{rule('Score (slide 12)', 'day-ahead + capacity + reserve energy + energy left − grid tariff − imbalance')}
{rule('Information', 'Gate A sees prices to D-1, balancing to D-2, the D-1 weather run. Nothing later.')}
{rule('Hindsight (slide 9)', 'Real prices, no forecasts, no imbalance. Ours runs the same MILP → bids → settlement path, so it is a ceiling, not a different game.')}
</div>''')
site = lambda k, t, rows: f'<div style="flex:1;display:flex;flex-direction:column;gap:14px">{img(k)}<p style="font-size:28px;font-weight:700">{t}</p>' + ''.join(f'<div style="display:flex;justify-content:space-between;font-size:22px;border-bottom:1px solid {LINE};padding:6px 0"><span style="color:{MUT}">{a}</span><b>{b}</b></div>' for a, b in rows) + '</div>'
S['sites'] = sec('sites', h2('The real sites, detected in Land Board orthophotos and rebuilt in 3D') + f'''
<div style="display:flex;gap:28px">
{site('solar', 'Pihlaka solar, Kehra', [('Park detected', '46.8 ha'), ('Module area', '19.3 ha'), ('DC rating at 21 %', '41–49 MWp'), ('Dataset', '42.3 MW')])}
{site('bat', 'Hertz 1 battery, Kiisa', [('Compound detected', '1.96 ha'), ('Containers', '54 (operator)'), ('Rating', '100 MW / 200 MWh'), ('Imagery', 'before completion')])}
{site('wind', 'Paldiski wind park', [('Turbines confirmed', '14 / 14'), ('Offset to OSM', '1–35 m'), ('Rated', '45 MW'), ('Hub / rotor', '100 / 100 m')])}
</div>
<p style="font-size:22px;color:{MUT}">Interactive ground view: claude.ai/artifact/9hWyuvPBKpLbhn9oXEr7NX · Orthophoto Maa-amet (CC BY 4.0) · modules by colour + parallel-row texture; turbines by rotor blobs</p>''')
P = [('Gate variational', 'QAOA, XY-QAOA, PCE, copula-QAOA', 'C1 C4 C5 C6 C7 K6 K7 O1 O4 O5 O7 O8 O9 O12 O14 …', 18, BLUE),
     ('Quantum annealing', 'D-Wave QA, hybrid CQM, annealing Benders', 'K3 K5 O10 S4', 4, ORANGE),
     ('Gate non-variational', 'BF-DCQO, FALQON, adiabatic + SQD, QMCMC', 'O2 O13 S5 S6', 4, AQUA),
     ('Quantum ML', 'reservoir computing, qRBM, QLSTM, quantum DQN', 'O3 O6 S1 S2 S3', 5, VIOLET),
     ('Linear algebra / other', 'HHL-type power flow, challenge mixes, surveys', 'K1 K2 C2 C3 C8 K4 K8 K9 O11', 9, '#9a9893')]
row = lambda n, a, ids, c, col: f'<div style="display:flex;align-items:center;gap:28px;padding:14px 0;border-bottom:1px solid {LINE}"><div style="width:120px;height:26px;background:{col};border-radius:0 6px 6px 0;width:{c*40}px"></div><p style="width:110px;font-size:34px;font-weight:700">{c}</p><div style="flex:1"><p style="font-size:28px;font-weight:600">{n}</p><p style="font-size:22px;color:{MUT}">{a} · {ids}</p></div></div>'
S['taxonomy'] = sec('taxonomy', h2('40 methods from the sources doc, read paper by paper and grouped before use', '17 of 40 ran on a real QPU (IBM, D-Wave, IonQ, Rigetti, Quantinuum); none on IQM or Pasqal. The doc had no quantum price forecaster: QRC papers fill that lane.') + ''.join(row(*p) for p in P) + f'''
<div style="display:flex;gap:20px;margin-top:8px">
{rule('Run here as forecasters', 'S1 / S2 quantum reservoir computing (CUDA-Q, 10 qubits)')}
{rule('Run here as deciders', 'C4 / O17 LR-QAOA · O9 warm-start QAOA · O5 / O7 PCE · K3 / K5 / O10 annealing (SQA) · S4 annealing Benders')}
{rule('Notebook', 'research/methods_landscape.ipynb: taxonomy, hardware, baselines, fit ranking')}
</div>''')
box = lambda t, items, col: f'<div style="flex:1;border:2px solid {col};border-radius:16px;padding:26px;display:flex;flex-direction:column;gap:12px"><p style="font-size:26px;font-weight:700;color:{col}">{t}</p>' + ''.join(f'<p style="font-size:24px;line-height:1.3">{x}</p>' for x in items) + '</div>'
S['pipeline'] = sec('pipeline', h2('Quantum competes in both halves of the pipeline, then takes the same bid path') + f'''
<div style="display:flex;gap:24px;align-items:stretch">
{box('1 · Forecast at gate A', ['<b>Chronos-2</b> (classical baseline)', '<b>Quantum reservoir</b>: 10-qubit Ising dynamics, 1000 shots, linear readout', '<b>Blend</b>: Chronos-2 + reservoir, weights set on Q3', '<span style="color:#4A5568">control: random Fourier features</span>'], VIOLET)}
{box('2 · Decide the day', ['<b>MILP</b> (HiGHS, classical baseline)', '<b>LR-QAOA</b> 24–25 qubits · <b>warm-start QAOA</b>', '<b>PCE</b>: 24 bits on 5 qubits', '<b>SQA</b>: D-Wave emulation (OpenJij)', '<b>Distilled QPU</b>: QAOA samples of 12-qubit windows → 3,953-param transformer, 0 QPU calls after training (IonQ arXiv:2611.08129)', '<span style="color:#4A5568">controls: SA, exact QUBO, best LP sample</span>'], ORANGE)}
{box('3 · Bid and settle', ['Quantum picks each hour: charge or discharge', 'HiGHS sizes every MW: capacity, DA, energy bids', 'Gate-C bids, activations, imbalance, tariff', 'Official rules, 25 days, scored once'], BLUE)}
</div>
<div style="display:flex;gap:20px">
{rule('Simulators', 'NVIDIA CUDA-Q 0.16 on GPU (cuStateVec) · OpenJij · Qiskit / Cirq available')}
{rule('Next: hardware', 'the same CUDA-Q kernels retarget to iqm, pasqal and ibm backends')}
{rule('The QUBO', 'LP value of the day pattern, fitted from 384 LP checks (held-out R² 0.93), shared by all solvers')}
</div>''')
S['forecast'] = sec('forecast', h2('Alone, the quantum reservoir loses to Chronos-2; blended in, it trims balancing-price errors', 'Q4 error ÷ Chronos-2 error per target. Blend gains on imbalance (64.6 → 62.2 €/MWh) and mFRR (67.4 → 64.4); the classical-twin blend reaches 62.7 / 65.1.') + img('fc', 760))
S['day'] = sec('day', h2('24 Oct, one real day: forecasts against reality, and the euros they earned', 'Chronos-2 expected 100–150 €/MWh; the day cleared at 20–50. Hindsight earns €147k; the forecast-driven plans earn €50–56k.') + img('prices', 360) + img('deur', 470), gap=24)
S['bids'] = sec('bids', h2('The same day, bid by bid: what each model committed at gates A and B') + img('bids', 820), gap=24)
S['markets'] = sec('markets', h2('Q4, battery + hybrid: euros by market for each model, against hindsight', 'Forecast-driven plans buy day-ahead energy to sell it in mFRR. With perfect information, the same path takes €4.97 M: forecast skill is the gap, not optimisation.') + img('markets', 700))
S['matrix'] = sec('matrix', h2('Every quantum decider ties the classical MILP; the forecaster decides the money', 'Q4 € vs Chronos-2 + MILP. Grey band: ±2 standard errors of a 25-day total (±€18k). Quantum-reservoir forecasts cost 3.7–5.3 % whoever decides.') + img('matrix', 740))
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

AC = [('Whole year, 291 days', 37_480_487, 38_476_470, '+€996k ± €1.38 M', '146 / 291', 30_277_111, 30_333_928, '+€57k ± €266k'),
      ('Whole year, calibration gate (Brier)', 37_480_487, 38_185_900, '+€705k ± €704k', '113 / 291', 30_277_111, 30_310_007, '+€33k ± €108k'),
      ('Whole year, ranking gate (Spearman)', 37_480_487, 38_688_597, '+€1.21 M ± €0.78 M', '135 / 291', 30_277_111, 30_375_085, '+€98k ± €116k'),
      ('Q3 alone, 92 days', 8_112_684, 9_307_055, '+€1.19 M ± €0.78 M', '—', 5_964_831, 6_182_285, '+€217k ± €171k'),
      ('Q3 alone, ranking gate', 8_112_684, 8_943_649, '+€831k ± €535k', '59 / 92', 5_964_831, 6_036_478, '+€72k ± €85k'),
      ('Q4 alone, 25 days', 1_821_939, 1_869_940, '+€48k ± €191k', '13 / 25', 1_327_486, 1_324_795, '−€3k ± €46k')]
S['activations'] = sec('activations', h2("Modelling the TSO instead of waiting for it: public drivers of the Baltic imbalance forecast the activations",
  "Elering's open API serves the block's drivers at 15 min: flows on both Estlinks and the Latvian border, D-1 transfer capacities, scheduled trade. LightGBM on these plus D-1 weather, wind-fleet ramps and D-2 balancing lags, boosted from a recency-weighted hour-of-day climatology. Walk-forward: refitted before each month on outcomes public by then; every month's calibration from the months before it. Nothing after D-1 09:00 is used.") + f"""
<div style="display:flex;gap:20px">
{stat('0.57 → 0.58', 'AUC for mFRR UP activation on Q3 and Q4, against 0.56 and 0.48 for the hour-of-day climatology the plan used (Q4 climatology below chance: regime shift)', ORANGE)}
{stat('0.163 vs 0.212', 'Brier score of the aFRR UP share on Q3: driver model vs Chronos-2 zero-shot (climatology 0.172); Q4 0.153 vs 0.187', ORANGE)}
{stat('+3.2 % · +0.3 %', 'Whole-year revenue, walk-forward, with the model used only when its trailing ranking skill beats the climatology: organisers’ rules (±2.1 %, significant), and under the realistic 25 MW cap (±0.4 %, noise). Q3 alone, ungated: +14.7 % / +3.6 %', AQUA)}
</div>
{tbl(['Walk-forward cell', 'Baseline', '+ activation model', 'Gain', 'Paired 95 %', 'Days won', 'Baseline, 25 MW cap', '+ model, 25 MW cap', 'Gain', 'Paired 95 %'],
 [[(c, True), f'€{b:,.0f}', f'€{a:,.0f}', (f'{100*(a/b-1):+.1f} %', True, AQUA if a > b else ORANGE), ci, w, f'€{bc:,.0f}', f'€{ac:,.0f}', (f'{100*(ac/bc-1):+.1f} %', True, AQUA if ac > bc else ORANGE), cic] for c, b, a, ci, w, bc, ac, cic in AC])}
<div style="display:flex;gap:20px">
{rule('The honest reading', 'The skill is real and it earns in Q3 under both settlements, the only gate-A lever that has. Ungated, spring loses: the model is useless with one month of history and good with six. Under realistic activation volumes the yearly gain stays inside noise.')}
{rule('Daily participation, trading-system style', 'Per product, the model is used only when its trailing 30-day skill on already-public outcomes beats the climatology. A calibration test (Brier) gives back most of Q3; a ranking test (Spearman, what the plan profits from) keeps it: no negative month, +3.2 % a year, significant. It admits the aFRR shares most days from May and the mFRR model almost only in October.')}
{rule('Where the uncertainty work goes next', 'Pushed through the activation chain the gain dilutes (+1 %); two-stage scenario plans still lose 16–32 % because of their recourse model. Better marginals first, settlement-faithful recourse second, joint scenarios (the sampler’s job) third.')}
</div>""", gap=20)
SC_YEAR = [('4 exact chain scenarios', 41_280_072, '+€2.59 M ± €0.90 M', '193 / 291'), ('16 exact', 40_740_475, '+€2.05 M ± €0.81 M', '192 / 291'),
           ('8 from single-flip MCMC', 40_538_346, '+€1.85 M ± €0.85 M', '167 / 291'), ('8 exact', 40_273_730, '+€1.59 M ± €0.88 M', '174 / 291'),
           ('8 from quantum-enhanced MCMC', 40_236_167, '+€1.55 M ± €0.82 M', '194 / 291')]
SC_CAP = [('Whole year, 291 days, 8 exact', 30_375_085, 32_023_349, '+€1.65 M ± €0.53 M', '204 / 291'), ('Whole year, 16 exact', 30_375_085, 31_927_225, '+€1.55 M ± €0.60 M', '204 / 291'),
          ('Q3, 92 days, 8 exact', 6_036_478, 6_537_381, '+€501k ± €183k', '65 / 92'), ('Q3, 8 from quantum-enhanced MCMC', 6_036_478, 6_567_639, '+€531k ± €179k', '67 / 92'),
          ('Q4, 25 days, 8 exact', 1_327_164, 1_435_138, '+€108k ± €72k', '17 / 25'), ('Q4, 8 from quantum-enhanced MCMC', 1_327_164, 1_460_775, '+€134k ± €78k', '19 / 25')]
ACTR = 38_688_597
S['scenarios'] = sec('scenarios', h2("The uncertainty object pays: planning over joint activation scenarios beats the best deterministic plan in both settlements",
  "Two-stage MILP (qest.stoch): the day-ahead position and the capacity offers are decided once, the battery's response per scenario. Recourse mirrors settlement: an in-the-money bid delivers its whole activated MW, what the SoC cannot back is a deviation at the imbalance price, and under the cap the TSO activates at most its volume. Scenarios are exact draws of the fitted activation chain, rescaled to the plan's own activation probability at its bid margin. Walk-forward, every month from the months before it.") + f"""
<div style="display:flex;gap:20px">
{stat('+6.7 %', f'2026 season, organisers’ rules: €41.28 M against €{ACTR/1e6:.2f} M for the rank-gated deterministic plan (+€2.59 M ± €0.90 M, 193 of 291 days); +10.1 % over Chronos-2 + MILP', AQUA)}
{stat('+5.4 %', 'whole year under the realistic 25 MW activation cap: €32.02 M against €30.38 M (+€1.65 M ± €0.53 M, 204 of 291 days), positive in every month; Q3 +8.3 %, Q4 +8.1 %. The aFRR energy leg more than doubles at unchanged capacity and imbalance', AQUA)}
{stat('not fidelity', '4 exact scenarios beat 16; the single-flip MCMC sampler beats the quantum-enhanced one; all five variants sit within each other’s intervals. The plan pays for scenarios, not for their accuracy', ORANGE)}
</div>
<div style="display:flex;gap:24px;align-items:flex-start">
<div style="flex:5">{tbl(['Whole year, organisers’ rules, 291 days', 'Revenue', 'vs rank-gated plan', 'Paired 95 %', 'Days won'],
 [[(a, True), f'€{v:,.0f}', (f'{100*(v/ACTR-1):+.1f} %', True, AQUA), ci, w] for a, v, ci, w in SC_YEAR])}</div>
<div style="flex:5">{tbl(['25 MW activation cap', 'Deterministic', 'Scenario plan', 'Gain', 'Paired 95 %', 'Days won'],
 [[(a, True), f'€{b:,.0f}', f'€{v:,.0f}', (f'{100*(v/b-1):+.1f} %', True, AQUA), ci, w] for a, b, v, ci, w in SC_CAP])}</div>
</div>
<div style="display:flex;gap:20px">
{rule('Why it took four recourse models', 'Free recourse (price out at will): −31 %. Faithful delivery: −11 %. Faithful + the settlement’s SoC guard modelled: −32 %, and valuing the deterministic plan inside that model gave a third of what settlement paid, so the model, not the optimiser, was wrong. Faithful delivery + scenarios rescaled to the bid margin: +3.5 % on the probe days, then the season.')}
{rule('Where the qubits sit in this', 'The scenario source. The exact chain is free; a quantum-proposal MCMC mixes faster per step (autocorrelation 12–22 vs 15–35 steps) but a classical tempering ladder improves the marginals more (TV 0.109 → 0.037) in a quarter of a second, and in euros the samplers are indistinguishable. The qubits are for the target that is not a chain: aFRR, both borders, the Finnish price coupled in.')}
{rule('Commercial reading', 'Zero QPU calls are needed for this gain: the information is in the drivers, the value in the recourse model. A QPU sampler enters only when the joint target outgrows exact sampling, and then as a one-time, distilled source (IonQ’s route), never per decision.')}
</div>""", gap=20)
Y = 365 / 291
PB = [('Chronos-2 + MILP (classical)', BASE, '—', '≈ €1–3k compute', '—', 'baseline', BLUE),
      ('Chronos-2 + LR-QAOA, 24 qubits, live every day', QAOA, '−€48k ± €163k', '$0.7k–93k QPU', '−€49k to −€141k', 'never', ORANGE),
      ('Chronos-2 + distilled QPU (0 QPU calls)', NQS, '−€193k ± €159k', '$1.4k–152k once', '−€194k to −€345k', 'never', VIOLET),
      ('Best of 384 QUBO samples (classical control)', SAMP, '−€105k ± €139k', '≈ €1–3k', 'not significant', '—', MUT),
      ('Activation model + ranking gate (classical)', ACTR, '+€1.21 M ± €0.78 M', '≈ €1–3k compute', '+€1.2 M', 'immediate', AQUA),
      ('Scenario plan, faithful recourse, 4 exact scenarios (classical)', 41_280_072, '+€3.80 M vs baseline', '≈ €2–5k compute', '+€3.8 M', 'immediate', AQUA),
      ('Reachable optimum (perfect prices, deterministic)', REACH, f'+€{(REACH-BASE)/1e6:.2f} M', '—', "the forecaster's prize", '—', MUT)]
S['payback'] = sec('payback', h2(f'Expected 2026 revenue: €{BASE/1e6:.1f} M baseline, €41.3 M with the scenario plan, €{NQS/1e6:.1f}–{QAOA/1e6:.1f} M with a QPU deciding; the QPU bill is paid back by nothing yet',
  f'291 traded days (8 Jan–25 Oct), battery + hybrid, official settlement; annualised ×{Y:.2f}. ± is the 95 % interval of the daily differences over 291 days. Q1–Q2 calibration here is split-wise (in-sample by ~0.3 %); the activation slide uses the stricter walk-forward. QPU cost spans Rigetti → IonQ Forte list prices (USD taken as EUR); the distilled route pays once.') + tbl(
  ['Pipeline', 'Revenue, season', 'Annualised', 'vs baseline', 'Cost / yr', 'Net vs baseline / yr', 'Payback'],
  [[(a, True, c), f'€{v:,.0f}', M(v * Y), u, k, nn, (pp, True)] for a, v, u, k, nn, pp, c in PB]) + f"""
<div style="display:flex;gap:20px">
{rule('Break-even uplift', f'€30k / €{BASE/1e6:.1f} M = {100*30e3/BASE:.2f} % (€{30e3/291:,.0f} a day) · €300k = {100*300e3/BASE:.2f} % (€{300e3/291:,.0f} a day). The live QPU sits at −0.13 %, the distilled one at −0.51 %.')}
{rule('Under the Estonian cap (25 MW, Q4)', 'MILP €1,328,436 · distilled €1,315,734 (−0.96 %) · live QAOA €1,313,247 (−1.14 %): the ranking survives the faithful market.')}
{rule('What pays: information and recourse, not the solver', 'The activation model with a ranking gate: +3.2 % over the year (±2.1 %), +0.3 % under the cap. Planning over its scenarios with a settlement-faithful recourse: a further +6.7 % (±2.3 %) uncapped and +5.4 % (±1.8 %) under the cap over the year. Every quantum decider on the same inputs: ±0.7 %, inside noise.')}
</div>""", gap=22)
S['verdict'] = sec('verdict', h2('Verdict: the puzzle is solved at half the qubits; the euros came from modelling the TSO and the recourse; in euros the qubits tie') + f'''
<div style="display:flex;gap:24px">
{stat('99.9 % · 15 qubits', 'organisers’ P3 puzzle (30 qubits, two hours linked by SoC), 300 Q4 hours: partitioned alternating hour / site blocks reach 99.92 % of the exact optimum and return it on 99.3 % of hours (PCE on 6 qubits: 84.5 %, 1.7 %; tabu search 99.92 %, 90.7 %). Every gate-model solver ties: the decomposition is the gain', AQUA)}
{stat('0 QPU calls', f"distilled sampler: the live QPU's optimum-hit rate, {M(NQS)} (−0.51 %), 4× faster per decision, 300× fewer shots over a season", VIOLET)}
{stat('89 %', f"of the deterministic reachable optimum captured by the scenario plan (€41.3 M of {M(REACH)}); hindsight's €{HIND/1e6:.0f} M is activation foresight", BLUE)}
</div>
<div style="display:flex;gap:20px">
{rule('Why parity', "At 24–25 binaries a GPU enumerates the QUBO in a second; every decider returns the MILP's pattern on ~80 % of days. One forecast, one optimum: the sampler cannot add information.")}
{rule('Where the qubits were measured, honestly', 'Quantum-proposal MCMC on 16-qubit activation windows: fastest decorrelation per step (12–22 vs 15–35), but a classical tempering ladder does more for the marginals at a 200th of the time, and whole-day scenarios from either sampler earn the same in the plan. Deciders on the best inputs: ±0.7 %, the QUBO surrogate is the ceiling (QAOA hits its optimum on 94–98 % of days).')}
{rule('Next', "Elering's activated volumes (CSV inbox ready) to make the cap the real series; a joint target beyond the chain (aFRR + borders + FI price) where the quantum sampler has a job; the distilled training set on IQM / IBM for about $3k.")}
</div>''')
order = ['cover', 'gates', 'sites', 'taxonomy', 'pipeline', 'forecast', 'day', 'bids', 'markets', 'matrix', 'gap', 'real', 'activations', 'scenarios', 'distill', 'payback', 'verdict']
for k in order: (D / 'slides' / f'{k}.html').write_text(S[k])
deck = {"v": 4, "createdOnFiles": {"v": 1, "at": "2026-11-07T15:30:00Z"}, "lists": "css", "title": "KratTrade · QEst 2026", "order": order,
        "sections": {"s1": {"description": "The market and the real sites", "start": "cover"}, "s2": {"description": "Methods: taxonomy and pipeline", "start": "taxonomy"},
                     "s3": {"description": "Results on Q4, 1-25 Oct 2026", "start": "forecast"}, "s4": {"description": "The gap, the Estonian market, the activation model, the scenario plan, the distilled QPU, the bill", "start": "gap"}},
        "faces": {"space-grotesk": {"family": "Space Grotesk", "href": "https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300..700&family=Inter:wght@400;600&display=swap"},
                  "inter": {"family": "Inter", "href": "https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300..700&family=Inter:wght@400;600&display=swap"}}, "designSystems": []}
(D / 'deck.json').write_text(json.dumps(deck)); print('ok', len(order))

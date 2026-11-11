# History: QEST 2026 quantum hackathon (team KRATTRADE)

QEST Hack, Tallinn, 11–13 Nov 2026. The KratTrade challenge, in the company's application:
> Batteries, solar and wind parks and demand response are managed as one portfolio. For every hour, we need to decide
> which asset to offer to which market, taking into account weather forecast uncertainty and imbalance costs.

## 1. Before this folder (28 Oct – 5 Nov 2026): the Makkah portfolio

Our first entry, built on the Makkah water twin. The repo is `~/Documents/product/mawani_general/sage_globe/water_destilation_dt`
(github.com/MuhammadHakami/quantum_energy_life_cycle, which GitHub reported as not found on 5 Nov, so it is private or renamed).
Its brief, research notes and notebooks are copied into `docs/background/` (image links in
QUANTUM_HACKATHON.md point at that repo).
- `energy/` sizes and dispatches a PV + BESS + molten salt + ORC + Taif wind + DR portfolio for 2029. It covers a
  day-ahead stochastic MILP, a hybrid ADMM with hour QUBOs, QAOA / WS-QAOA / BF-DCQO / PCE / SQA / Rydberg MWIS, QML
  forecasts, and IBM FakeFez / IQM Apollo noise rehearsals.
- **Main finding:** the stochastic MILP gets within $0.27 M/yr of perfect foresight, so better solvers have little left
  to win. ADMM + warm-start QAOA matches ADMM + exact solves (98.6–99.5 % of the bound on dry days), and quantum alone
  reaches 31–67 %. The 2026 literature agrees: hybrid VQA does not beat random sampling on full AC-OPF-UC (arXiv 2607.15543).
- `KratTrade_Starter_Solution.ipynb` solves the organisers' starter (6-qubit QUBO, QAOA p = 1/2, noise models).
  It is in `docs/background/`.
- The 2026 state-of-the-art lists are in `docs/background/SOTA_2026.md` and `QUANTUM_METHODS_2026.md`, with 31 repos
  cloned in the old repo's `research/third_party/`. There is also a published doc, "Quantum for Energy: 2026 State of
  the Art" (claude.ai/artifact/C2R1rHLTwiqh4mu9GvwnVV). It lists energy-focused 2026 challenges, conference papers and
  repos, every link checked.

## 2. The real competition data (5 Nov 2026): this folder

At kick-off KratTrade released the dataset (`dataset/`; slides in `slides/`, screenshots of the deck "Joint
optimisation of a hybrid portfolio (v3)"). It covers Estonia (EE), delivery days 1 Jan – 25 Oct 2026, in three files:
- `prices_15min`: day-ahead, mFRR, aFRR, imbalance and capacity prices.
- `solarpark_forecast_15min`: a 42.3 MW park's latest and D-1 forecasts.
- `afrr_picasso_4s`: PICASSO aFRR prices every 4 s.

The organisers' own deck artifact (claude.ai/artifact/TExUxuHaY6DD7BaCNUdQrp) is not shared with our account. Slides
5–10 and 12 (aFRR detail, imbalance, grid tariff, scoring) were never seen, so those rules come from public sources (below).

The user asked for: the Estonian environment of the dataset's year, with real weather, the exact locations of the wind
farms, solar panels and batteries, and a scenario synced to the dataset. The time-series split trains on Q1–Q3 and tests
on Q4. Several algorithms are trained and cross-compared. The environment pays for taking part, penalises taking part
without delivering, makes earlier orders riskier but better paid, and saturates late ones.

### 2a. How the assets were localised (reproducible: `python -m qest.sites`)

**Solar park, 42.3 MW: Pihlaka, Kehra (59.322 N, 25.307 E).** The dataset gives no location.
1. Its device count ramps from 0 to 4 in March–April 2026, so the park was commissioning then. The latest forecast exists
   from 7 Apr; the D-1 forecast exists only for Jul–Aug ('v1-recon') and Oct ('v1').
2. Solar-noon timing of clear days gives about 23.5 E (±1). Tracker or tilt geometry blurs it, so this only bounds the band.
3. **Decisive step: a cloud-pattern match.** Take hourly forecast anomalies (Apr–Oct, by month × hour) and correlate them
   with ERA5 GHI anomalies on a 0.25° grid over Estonia (270 points, `data/raw/ghi_grid.json`). The peak is r = 0.752 at
   59.25–59.5 N, 25.25–25.75 E (north Estonia, east of Tallinn). Western cells drop to r ≈ 0.45.
4. OSM solar plants in that cell include two newly mapped ways (1568593429/30) at 59.32 N 25.31 E by Kehra. That is the
   Pihlaka park (KC Solar2, 45 MW, EBRD 2024, built 2025), consistent with the commissioning ramp.
5. pvlib fitted on train days: fixed south at 25° tilt, DC/AC 1.32, clipped at 42.26 MW (the dataset's maximum).
   RMSE 4.9 MW against ERA5. Our D-1 PV forecast scores MAE 3.05 MW on Q3 (the dataset's own D-1: 2.58).

**Battery, 100 MW / 200 MWh (slide 11): Hertz 1, Kiisa (59.2431 N, 24.7007 E).** This is Estonia's real park (Evecon,
Corsica Sole, Mirova), in operation 3 Feb 2026. It connects by 330 kV cable at Elering's Kiisa substation (OSM way
34935436). The battery's own outline isn't mapped. A testing fault there tripped EstLink 1 and 2 in Jan 2026.

**Hybrid (slide 11): PV 20 MW + battery 20 MW / 40 MWh on one 20 MW connection.** A toy site with no real location. It
uses the Pihlaka per-MW profile, and pvlib on ERA5 before the park existed.

**Wind: every Estonian plant from OSM** (Overpass bbox query; `data/raw/osm_wind.json`: 30 plants, 233 turbines).
- Rotor and hub come from turbine tags or from the model's datasheet (`qest/sites.py: MODELS`).
- Largest plants: Sopi-Tootsi 255 MW, Aidu 100, Aulepa 48, Paldiski 45, Narva 39, Saarde 39.
- The portfolio park is Paldiski 45 MW (`PORTFOLIO_WIND`; any plant can be chosen).
- Each plant ≥ 5 MW has its own weather file (17 sites).
- **Sim2real check:** the whole fleet through the generic curves correlates with Elering's measured renewable production
  at r = 0.88 per 15 min (0.93 daily), but runs about 1.6× high. One level factor, 0.633, is fitted on train days
  (`power.wind_scale`). Paldiski's capacity factor is then 22–29 %.

**Weather (keyless).** Open-Meteo ERA5 hourly is used as "what happened". The previous-day runs API gives the forecast
issued D-1 00 UTC. Elering's dashboard API (`/api/system/with-plan`) gives Estonia's measured and planned production,
consumption and solar.

### 2b. Market rules and their sources

- **Gates.** These come from the slides seen.
  - A, D-1 09:00: reserve capacity, whole MW, an obligation.
  - B, D-1 13:00: day-ahead, 96 MTUs, 0.1 MW.
  - C, T-25: energy bids. mFRR UP activates if price ≥ bid; aFRR follows the PICASSO 4-s merit order.
  - No intraday in the toy problems.
- **Starter notebook settlement.** Shortfall 150 €/MWh; surplus at 50 % of DA.
- **Slide 11.** Batteries start at 50 % SoC. Leftover energy is worth the day's mean DA price. Imports pay the grid
  tariff. PV can't exceed its forecast. No financial cycles.
- **Grid tariff (slide 10 unseen).** Elering's price list from 5 May 2026 (decision 7-3/2026-025): 14.50 €/MWh at 330 kV,
  23.67 at 110 kV, plus the 3.73 €/MWh balancing capacity fee. Storage pays only on net monthly consumption
  (Electricity Market Act, from 1 Jan 2026).
- **Capacity saturation.** Linear to zero at twice the Baltic need. That need is aFRR 96–120 MW (Elering); mFRR UP
  604 MW and DOWN 691 MW (AST FRR dimensioning 2026–2035). Nord Pool's aggregated bid curves (the slide's plot) need a
  login: the API at dataportal-api.nordpoolgroup.com/api/AggregatedBidCurves (marketCode=NPSDA) returns 401.

### 2c. What broke and what fixed it

1. **The oracle didn't deliver activations it knew about.** The MILP counted conversion losses per flow while the
   settlement netted them, so SoC drifted. Fix: SoC on the battery's net inverter power (`nc`/`nd`).
2. **The hybrid oracle was short on DOWN activations.** Settlement spilled planned-curtailed PV during them. Fix: the
   plan's PV use is settled, and output is never spilled against a DOWN activation.
3. **Forecast-driven plans lost €20 M on Q3 in penalties.** Static mFRR bids at DA + €5 drained the battery.
   - Fix part 1: PICASSO merit-order activation from the 4-s data (`data.afrr_margin_table`).
   - Fix part 2: a SoC-aware gate-C policy that prices a direction out on drift.
   - Fix part 3: margins tuned on Q3, aFRR ± €25 and mFRR ± €100 (`qest.tune`).
4. **LightGBM could not follow the Q2→Q3 price drop.** Fix: per-target choice of LightGBM or persistence on Q3, plus
   conformal widening of the 10–90 % band.
5. **Unsold PV and wind were curtailed under the starter rule.** Fix: surplus is delivered whenever its settlement value is positive.

### 2d. Results (Q4 test, 1–25 Oct, scored once)

| Strategy | Baltic settlement | Starter settlement |
|---|---:|---:|
| oracle (hindsight) | €5.72 M | €5.72 M |
| milp (LightGBM) | €2.15 M (37.5 %) | €1.32 M |
| milp_persist | €1.86 M | €1.52 M |
| reserve_rule | €1.00 M | €0.82 M |
| rule | €0.95 M | €0.77 M |
| idle (all real time) | €0.76 M | €0.37 M |
| naive_da | €0.72 M | €0.53 M |

**Quantum track.** The slide-11 puzzle runs on every Q4 hour, with option values from the forecasts and plans scored on
the realised hour (`results/qubo_*_test.csv`).

| Level | Qubits | Instances | Exact | Annealing | QAOA p = 1 | QAOA p = 2 | Valid rate | Hindsight |
|---|---:|---:|---|---|---|---|---|---|
| P1 | 9 | 600 | 49.0 % | 48.8 % | 48.6 % | 48.8 % | QAOA 36–38 % (random 7.8 %) | €1.57 M |
| P2 | 15 | 600 | 53.0 % | 51.2 % | 52.9 % | 53.0 % | QAOA p = 2 29 % (random 6.5 %) | €2.41 M |
| P3 | 30 | 300 pairs | 59.3 % | 36.3 % | not run | not run | (random 0.4 %) | €1.77 M |

Columns Exact to QAOA p = 2 are the share of the hindsight value each method realises.
- **P2 timing:** QAOA takes 2–8 s per instance, against under 1 ms for exact enumeration of 125 states.
- **P3 annealing** finds the forecast optimum in only 0.6 % of samples. The two-hour SoC-link penalties make the landscape
  hard, which is where a better solver could matter.
- **P3 QAOA** was not run at 30 qubits, because the statevector is too large.
- Like the Makkah study, the forecast is the limit, not the solver: exact solves reach only 49–59 % of hindsight.

## 3. The organisers' deck, official scoring, the frontier (7 Nov 2026)

The deck PDF ("Joint optimisation of a hybrid portfolio slides.pdf", v3, 14 pages) arrived, closing the old open item 1.
What it changed:
- **Slide 10:** battery 100 MW / 200 MWh (the "200" is MWh), hybrid PV 20 MW from the park forecast scaled to 20 MW,
  100 % round-trip efficiency, no cycling cost, import tariff EUR 30/MWh Mon–Fri 07–22 Tallinn and EUR 20 otherwise
  (Estonian holidays coded in `market.EE_HOLIDAYS_2026`), exports free.
- **Slide 12 (scoring):** every capacity bid accepted at the published price (no saturation, no forfeiture), in-the-money
  mFRR bids fully activated at the single mFRR price (UP and DOWN prices are identical in the data), aFRR per 4-s tick,
  imbalance = (metered - committed) x price, prospective only. Implemented as `Config(settlement='official')`, the default.
- **Slide 9:** P1–P3 are retrospective (prices known), scored as the gap to the classical optimum. `qest.qubo --mode retro`.

What was built and found:
1. **Diagnosis.** Under official rules the forecast MILP sold 700–1,370 MWh/day DA counting on forecast aFRR DOWN
   activations to recharge; Q4 delivered a third to two-thirds of them, so the battery ran empty into imbalance.
   Knob added: kappa_up / kappa_dn (haircut on planned activation energy) and the SoC tolerance.
2. **Optuna** (`qest.tune`, TPE, 64 trials, 8 processes on a SQLite study, Q3 only): EUR 6.16 M -> 6.81 M on Q3
   (mFRR margin 124.5, aFRR 25, kappa_up 0.99, kappa_dn 0.60, tol 0.092). On Q4 the tuned MILP scored 1.54 M against
   1.59 M untuned: the Q3 optimum did not transfer. Reported as is.
3. **Chronos-2** (`qest.chronos`, amazon/chronos-2 zero-shot, D-1 weather covariates, two lag groups, CUDA GPU 1):
   best on DA (Q3 MAE 32.7 vs LightGBM 40.3; Q4 36.0 vs 42.5) and all four capacity prices; level on balancing prices.
   `milp_chronos` is the best tradable strategy: Q4 EUR 1.65 M (USD 1.85 M), 33.3 % of hindsight. On Q3 it scored 5.64 M
   against milp's 6.81 M: the knobs were tuned for LightGBM forecasts.
4. **Participation** (`milp_da`, the MILP without reserves): reserves win 23/25 Q4 days and 83/92 Q3 days. Answer:
   take part every day; the MILP sizes the offer per MTU.
5. **RL** (`qest.gym_env`, Gymnasium 1.4 + SB3 2.9 PPO, 2,000 steps on Q3 days): the agent picks the MILP knobs per day
   from 16 gate-A features. It collapsed to kappa_up 1, kappa_dn 0, tol 0.077: Q3 EUR 7.07 M (in-sample), Q4 1.49 M.
6. **Quantum, retrospective** (`qest.qubo`, solvers ported from the Makkah repo into `qest/qsota.py`, exact GPU
   statevector): QAOA p=1/3, simulated QA, BF-DCQO and FALQON hit the optimum on 600/600 P1 and P2 hours; random with the
   same 4,096 shots reaches 81.9 % on P2 (P1 is too easy to separate anything). P(optimum) in the state at P2: BF-DCQO
   15 % (4,900x uniform), QAOA p=3 0.9 %. P3 (30 qubits): only PCE runs (QUBO-coefficient mode added, 30 bits on 6
   qubits) at 84.5 %; simulated annealing 46.1 %, tabu 99.9 %, random-4096 30.8 %. A bug found and fixed on the way:
   plans breaking the connection or SoC-link penalties were scored as if feasible; they now score 0.
7. **Payback:** EUR 24.1 M/yr on the two sites (best tradable, Q4 annualised), so 1 % better = EUR 241 k (USD 269 k)/yr;
   EUR 30 k–300 k breaks even at +0.12–1.24 % over the classical plan. On the puzzle the gain is 0: exact is optimal.
8. **Deck** (12 slides, charts from `qest.figures`): claude.ai/artifact/EJRo1JSfz4RVBn8Hx3EMJD (private until shared).
   USD at the ECB rate 1.1165 (6 Nov 2026, `qest.fetch_fx`).

## 4. Chronos-2 as the baseline, quantum and hybrid layers, annealing, research (7 Nov 2026, evening)

The user set Chronos-2 + MILP as the one classical baseline; everything else quantum or hybrid. NAS / PostNAS are for
the quantum circuits.
- **Baseline:** Optuna tuned `milp_chronos`'s own knobs on Q3 (`tune_best_milp_chronos.json`: kappa_dn 0.03, tight
  margins): Q3 EUR 8.09 M (5.64 M on the LightGBM knobs). Q4 once: **EUR 1,812,548 ($2.02 M), 36 % of hindsight**.
- **Quantum / hybrid layers on it, Q4 (`summary_test_hybrid.csv`, `rl_qpolicy.csv`, `days_test_benders_*.parquet`):**
  VQC policy (`qrl`, 8 qubits, PPO) -0.7 %; annealing Benders (`qbenders`, OpenJij SQA master + HiGHS LP cuts, after
  DOE winner Qtyche / HQCMCBD) -0.9 % (SA master -0.9 %, exact master +0.2 %: the 0.9 % is the annealer + 7-bit
  encoding); QRC forecast fix (`qrc`) -10.5 % (RFF control -8.8 %; DA MAE 36.05 -> 35.63, linear 35.49); 25-qubit
  LR-QAOA day pattern (`qday`) -15.9 % (exact surrogate -14.2 %, random patterns -4.2 %: the surrogate loses money).
- **Puzzle (`qnas`, `qmcmc`, `qanneal`):** NAS picked an XY one-hot mixer (P3 simulable: 15,625 states). P2: NAS uniform
  start 15.1 % on the optimum at 106 CZ vs BF-DCQO 15.0 % at 520 CZ; PostNAS 106 -> 71 CZ, 54.6 -> 56.0 %; warm start
  alone 58.7 % (the circuit adds nothing over it). Digitised QA, XY driver, 64 steps: P2 19.6 %, P3 99.81 % of exact
  (tabu 99.92 %, 4096 random valid plans 99.34 %). OpenJij SQA on the raw QUBO 85 / 80 / 54 % (SA 67 / 57 / 31 %).
  QMCMC beats a local chain only at P1 (full space).
- **Bug found and fixed:** fixed-angle (ramp) schedules started in the mixer's TOP eigenstate, so they annealed to the
  cost's maximum (qday's first LR-QAOA run, NAS ramp trials, the first qanneal run). The cost phase now flips sign for
  ramp schedules (`qnas.solve`, `qday.patterns`); OpenJij QUBOs are scaled by the spread of the better half of plans.
- **Research (`research/`):** `fetch.py` cloned 12 repos (competition entries, HQCMCBD, QHDOPT, DQAOA, ...) and 16
  papers; `README.md` maps every challenge / conference row to its code, paper and what we took; `methods.csv` +
  `methods_landscape.ipynb` group 34 entries into 7 families with charts (none of them ran on IQM or Pasqal).

**User direction (7 Nov, late):** quantum methods must not just consume Chronos-2 forecasts: they forecast and decide
alongside it, Chronos-2 being one classical competitor. Categorise the doc's methods from the papers before using them:
`research/methods.csv` (40 entries: 34 from the doc + 6 marked supplements; paradigm / hybrid pattern / encoding /
pipeline stage / real-hardware evidence / baseline / outcome / fit) and `research/methods_landscape.ipynb`. Reading the
papers corrected the briefs: REGRID-QAOA ran on ibm_marrakesh, the stochastic-UC pipeline and the hierarchical QA-MPC on
D-Wave, PCE portfolios had a trapped-ion check; 2610.10772 is a Birkhoff (matching) decomposition, not grid dispatch.
The doc has no quantum price-forecasting entry (supplements S1-S3 fill that lane). Deck still to rebuild: market
mechanics, satellite-detected assets in a three.js ground view, per-market EUR by model vs reality, no text on plots.

## 4b. Quantum methods on real simulators, competing across the pipeline (7 Nov 2026, night)

User direction: quantum runs on established simulators now (NVIDIA CUDA-Q / cuQuantum preferred; Cirq, Qiskit allowed),
real QPUs later; quantum competes in forecasting and in bidding, not on top of Chronos-2. Installed CUDA-Q 0.16
(`nvidia` target = cuStateVec; also `iqm`, `pasqal`, `ibm` targets), cuQuantum, Cirq + qsim. CUDA-Q conventions found:
`get_state` puts qubit 0 in the least significant bit, `get_unitary` in the most significant; `sample` strings are q0 first.

- **`qest.qforecast`** (S1/S2 QRC): 10-qubit Trotterised-Ising reservoir (CUDA-Q kernel; unitary from `get_unitary`,
  batched on the GPU, 1000-shot readout), ridge readout, gate-A inputs only. Q4 MAE: loses to Chronos-2 on 9 of 11
  targets (DA 47.3 vs 36.0). Blended with Chronos-2 (weight per target on Q3) it helps 7 of 11 (imbalance 64.6 -> 62.2,
  mFRR 67.4 -> 64.4); the same blend with the classical RFF twin gets 62.7 / 65.1, so most of the gain is blending.
- **`qest.qdecide`**: per forecaster/day/unit, a quadratic surrogate of the LP value of the hourly charge/discharge
  pattern (384 LP samples, held-out R2 0.93) is the shared QUBO; LR-QAOA (24-25 qubits, CUDA-Q), warm-start QAOA (cop-QAOA
  independent case), PCE (5 qubits), OpenJij SQA / SA, exact enumeration, best-sampled pattern; K = 8 candidates -> LP
  check -> MILP sizes the MW -> official settlement. Q4 EUR (results/qdecide_matrix_test.csv), vs Chronos-2 + MILP
  1,812,548: QAOA -0.03 %, WS-QAOA -0.06 %, PCE -0.36 %, exact QUBO -0.47 %, SA -0.56 %, SQA -0.79 %, best sample +0.49 %;
  daily differences sd ~EUR 1.8k vs means of +-0.4k: all within noise. QRC forecasts cost 3.8 % whatever decides; the
  blend ties Chronos-2 (-0.05 %). SQA needed the QUBO x100 (chosen on a Q3 day) for OpenJij's default schedule.
- **Real assets from imagery** (`qest.fetch_ortho`, `qest.detect`, `qest.scene`): Maa-amet orthophotos (CC BY 4.0).
  Pihlaka: colour + parallel-row coherence -> park 46.8 ha, modules 19.3 ha -> 40-49 MWp (dataset 42.3 MW). Hertz 1:
  imagery predates completion (pads, a first row of containers); compound 1.96 ha detected, 54 containers (operator)
  placed on the smoothest slab on the compound axis. Paldiski: 14/14 OSM turbines confirmed by rotor blobs (1-35 m).
  three.js ground view: results/figs/scene/scene.html.
- **Payback** (deck slide 'payback'): Q4 scaled x365/25, battery + hybrid. Baseline Chronos-2 + MILP EUR 26.46 M/yr,
  classical running cost ~EUR 1-3k/yr (estimate: open-source models, small VM). Best quantum (LR-QAOA) -EUR 9k/yr
  +-263k (95 %), WS-QAOA -16k +-296k, quantum-reservoir blend -13k +-1.16 M: at EUR 30k-300k/yr the net is -39k to -309k,
  payback never. Break-even uplift 0.11 % (EUR 82/day) to 1.13 % (EUR 822/day); with daily differences of sd EUR 1.8k,
  19 days resolve 1.13 %, ~1,900 days 0.11 %.

## 4c. Partitioned qubit reuse, after Daedalus (8 Nov 2026)

Daedalus (arXiv:2611.05735, ETH) is a classical neural-operator paper; its "92x larger" is model parameters on one
GPU. Its structure (overlapping partitions, per-partition frame normalisation, one frozen encoder pre-trained on
synthetic data, a coupling processor, chunked inference) maps onto domain decomposition for small QPUs: `qest/qpart.py`
(mapping table and lean list in research/partitioned_quantum.md). Every qubit decider runs partitioned (`p_qaoa`,
`p_qaoa_add`, `p_wsqaoa`, `p_pce`, `p_sqa` in qdecide), and the reservoir forecaster has a partitioned variant
(`qforecast --part`: 32 inputs, 4 windows through the same 10-qubit reservoir).
- Reach: 24 qubits cover 2,208 binaries (92 x 24) in 370 circuits, energy -1251.02 vs SA -1250.45 (SA 100x faster).
- Day QUBOs: hit rate 0.84 monolithic (24-25 qubits) vs 0.76 at 12-qubit windows, 0.74 for 4-qubit PCE.
- Euros, Q4: partitioned deciders -0.61 to -0.75 % vs MILP; vs monolithic LR-QAOA EUR -11.6k +/- 9.4k (1.2 s.e.).
  Partitioning fits the method to small devices; it does not add euros.
- Partitioned reservoir (Q4 MAE, `results/qforecast_skill_part.csv`): beats the plain 10-input reservoir on 3 of 11
  targets (DA 47.3 -> 44.4), its classical twin (RFF on the same 32 inputs) on 8 of 11, Chronos-2 on 1 (mFRR, 67.38 vs
  67.41). Blended with Chronos-2 it ties the plain blend (better on 3 of 11). Ridge picked alpha 100, the grid's top,
  everywhere: a wider grid may help.

## 4d. Distilled QPU, after IonQ's shadow-trained NQS (8 Nov 2026)

arXiv:2611.08129 distils a QPU's measurement data once into a small autoregressive transformer that then replaces it.
`qest/qdistill.py` does that for the partitioned solver: QAOA samples of 1,891 Q3 windows (ideal, noisy CUDA-Q
trajectories at 0.3 % per CNOT, and a classical Boltzmann twin) train one conditional 3,953-parameter transformer;
the noise-aware loss (measured = a model + (1 - a) uniform) is scanned on held-out Q3 days. Q4: hit 0.78-0.82 for the
distilled sources vs 0.76 live QAOA (uniform 0.44) with zero QPU calls; euros -0.67 to -0.88 % vs MILP, the twin equal to
the quantum-trained model. Write-up: research/distilled_quantum.md. Gotcha: import torch._dynamo before CUDA-Q.

## 4e. Estonia 2026 scenario twin (8 Nov 2026)

`run.sh twin` then `run.sh serve` gives http://localhost:18977/ (systemd user unit `qest-twin`, served from results/figs/twin).
The pieces:
- `qest.assets`: the asset register, the per-turbine wake simulation, the solar simulation, the photo checks and the
  sim-vs-Elering fit.
- `qest.twin`: the page builder.
- `qest.twin --traces`: the battery dispatch.

The wake model and simulation pattern are adapted from the sage_globe water twin (energy/wind.py) and copied in, as the
folder must stay self-contained.

**Assets.** All are filtered to Estonia's border polygon (OSM admin_level 2).
- **Wind:** 211 OSM turbines in 30 plants, corrected by public records.
  - Paldiski: 18 × GE 2.5-100. OSM filed 4 of them under Pakri.
  - Narva: Enercon E-82, 107 m hub.
  - Saarde: Vestas V150. OSM maps 4 of its 9.
  - Sopi-Tootsi: Nordex N163, 159 m hub. All 38 positions match the pads in the photo.
  - Untagged turbines: rotor sized at 350 W/m² specific power. The OSM reader's 100 m default had made 0.1 MW units run at 87 %.
- **Solar:** 925 OSM plants. Capacity is the OSM tag, else the module area measured in the plant's close-up
  (blue-grey modules with row texture, 21 % efficiency, DC/AC 1.32).
- **Battery:** Hertz 1, placed on the compound detected in the photo.
- **Grid:** 110-330 kV lines, substations and cables (EstLink 1-2 included). Each asset is linked to the nearest
  110-330 kV substation, an assumption; Hertz 1's link to Kiisa 330 kV is public.

**Simulation, hourly, 1 Jan – 25 Oct 2026.**
- **Wind:** each turbine runs inside its park's wakes (PyWake 2.6.20 TurbOPark, its own GenericWindTurbine), on ERA5
  hub wind at air density.
- **Solar:** pvlib at fixed south 25°, using each plant's 0.25° ERA5 cell (Erbs split).
- **Battery:** the 15-minute settlement traces. The re-run reproduces the qdecide euros to EUR 0.0 on all three splits.

**Sim to real (hourly).**
- **Wind:** against Elering's measured renewable (which excludes solar): r = 0.89 on Jan–Jun (fit) and 0.85 on Jul–Oct
  (unseen). Measured = 0.66 × simulated + 16 MW. The 0.66 is steady month by month (0.68–0.78), so it is a level offset
  (ERA5 100 m winds high, curtailment and availability unseen), not commissioning.
- **Solar:** r ≈ 0.95 against Elering's measured solar. OSM plants cover about half of Estonia's solar output; rooftops
  are mostly unmapped.
- **Data gap:** Elering's solar reads 0 for whole days on 3 Oct and 20–25 Oct; these are treated as missing.

**Photos.** Close-ups at 0.31 m median (z18; coarser only where a plant's outline needs it), in 12 atlases. 203 of 209
turbines and 317 of 925 solar plants show visibly. The solar fleet (447 MW counted) then gives measured = 1.71 × simulated,
r = 0.95 on Jul–Oct: OSM's plants are about 58 % of Estonia's solar.

**3D solar.** 411k tables in all. 267 plants use the module cells found in their photo; 655 fill their OSM outline
(6 m rows); 3 are points.

**Hindsight.** `qest.twin --traces oracle` adds the perfect-foresight plan's traces and daily euros. On battery +
hybrid it makes EUR 105.2 M over 8 Jan–25 Oct against the baseline's EUR 37.6 M.

**Other assets.** All four 2026 batteries: Hertz 1 (dispatch), Auvere 26.5 MW / 53 MWh, and Hertz 2 and Kirikmäe
drawn as under construction. 20 other power plants and the substation outlines, from OSM.

**Page.**
- **Map:** full-screen Land Board orthophoto (z18, 0.31 m) or Esri imagery, streamed as you zoom (display only).
- **Dock:** Makkah-twin style, with day steps, an hour track and the 2026 year with its spans and event ticks.
- **Live panels:** simulated fleets against measured, Estonia's production, consumption and net import, and Hertz 1's
  state per strategy.
- **Estonia 3D:** streamed photo ground and every turbine (maker shape, wind-facing, speed by output). Solar tables sit
  at the photo's module cells. Pylons and conductors, cables, substations and Hertz 1's 54 containers lit by dispatch.

**Pitfalls.**
- The Land Board blocks clients that open many connections: fetch with 2 threads.
- Esri tiles may be shown but not redistributed.

- **3D zoom.** three.js r160 zoomed a fixed 5 % per wheel event, so trackpads and smooth-scroll mice raced. The page now uses the water twin's vendored r182 (`qest/vendor`, MIT), which scales zoom by the wheel delta: 5.3 % per mouse notch, 0.2 % per small trackpad step (measured). The near plane follows distance/2000, and the camera stays at least 4 m above ground.
- **Clock.** The speed buttons are 1×, 10×, 60×, 360×, 1440× and 2440× real time; 1440× is one day per minute. The clock shows minutes between the hourly data. The 24 h line spans the dock, above the year track.

- **Fog off was full fog.** Fog-off set near = far = 1e9, and GLSL smoothstep is undefined when its edges are equal (many GPUs return full fog). Now near 1e9, far 2e9. The button shows the current state. 9 Nov: fog is off by default (`?fog=1` starts with it on), and the Simulation panels (live, forecasts, strategies, selection, layers) follow the viewer into Estonia 3D: one `aside#side` moves into the active view, the 3D-only blocks (`.only3d`: fog and fly-to chips at the top, the scene notes at the bottom) show only there.
- **Sharp tiles.** 3D: a zoom-7 quadtree over Estonia, culled to the view. Each tile splits until its pixel covers no more than 1.15 screen pixels at its own distance, or the source's finest zoom (Land Board z18, 0.31 m). Parents stay underneath until children arrive. Loads go nearest first, 6 in flight, and up to 1,500 tiles are cached. 2D: the zoom is round(9 + log2(scale·dpr) + 0.35); it was 0.8× the scale, up to 2.2× blur. The tile cap is 700 (200 dropped the fine layer on 2× screens).

- **Tile cache.** `qest.serve` replaces `http.server`. It serves the page and proxies Land Board tiles through `data/raw/tiles` (the cache qest.assets already filled, 11.2k tiles). Tiles the Land Board lacks (a 404, or its ~116-byte all-black JPEG where it has no photo: anything of 400 bytes or less counts as missing, so the page falls back to Esri at sea and abroad) are remembered as `.404` files, at most 6 upstream fetches run at once, and browsers keep tiles a year. `--warm` fetches Estonia z8–12 and z13–17 around every wind plant, battery and the 150 largest solar plants, one at a time (10.4k tiles, about 1 tile/s from the Land Board). The page uses the cache when `/tiles/ping` answers, so the artifact still streams direct. When idle, it prefetches the next zoom under the nearest tiles, 4 at a time.
- **Panel plots** (after the Makkah twin's web/energy.js):
  - **Next hour:** forecast→real tiles with band bullets.
  - **Per signal:** a day card with the 10–90 % band, dashed forecast, real line to the clock, now line, hits strip and an in-band ring. Signals: Chronos-2 DA (band), imbalance q50, Pihlaka PV D-1, Paldiski wind D-1.
  - **Strategy rows:** cumulative euros to date vs the baseline (dashed), % vs baseline, % of hindsight.
  - **Selected strategy:** a dispatch card with battery MW bars vs hindsight outlines, SoC vs hindsight, and the day's euros by market. The remainder (penalties, end SoC, cycling, tariff) is shown as "other" so the parts sum to the total.
  - **URL:** `?day=YYYY-MM-DD` opens a day.

## 5. Open items

1. **Wind level factor caps output.** `power.wind_scale` (0.633) multiplies power, so Paldiski peaks near 58 % of
   nameplate. Scale wind speed (or fit per plant) instead; then delete `generation.parquet`, the wind fleet caches and the
   forecast caches (fleet wind is a forecast feature). The official score (battery + hybrid) is unaffected.
2. **Hardware.** P1/P2 BF-DCQO and QAOA on IBM Heron / IQM Garnet; P3 PCE on 6 qubits. Measure P(optimum) under noise.
3. **Where quantum could pay.** A scenario-based whole-day problem (96 MTUs x weather scenarios x SoC across days) sized
   past what HiGHS solves before 09:00; judge against the 0.12 % break-even. Pasqal (Rydberg MIS) for that size.
4. **Annealer master.** Benders with an exact master ties Chronos-2; the SQA master loses 0.9 %. More slack bits and reads, or a native D-Wave run.
5. **Saturation.** Nord Pool curves need a login; the official rules accept every bid at the published price anyway.
6. **NAS / PostNAS** were not run: the only neural model is Chronos-2, pretrained and zero-shot. Optuna covers the knobs.

## 4f. Where the EUR 67.6 M to hindsight sits (9 Nov 2026)

Season (291 days, both sites): Chronos-2 + MILP EUR 37.60 M, LR-QAOA 37.55 M, best sampled 37.50 M, hindsight 105.18 M.
Partial oracles (`qest.strategies` `oracle_*`, one input group real, Chronos-2 for the rest): perfect aFRR + mFRR
activations close 87 % of the gap on Q4 and 82 % over the season, mFRR alone 49 % (Q4), aFRR alone 34 %, DA price 8 % (Q4) /
6 % (season), capacity prices 0.7 %, PV 0; the gate-C policy is worth 3 %. Activations are near-unpredictable at D-1 (imbalance within-day Spearman 0.33) but persistent close
to delivery (P(UP active | UP active 30 min ago) = 0.80, DOWN 0.84). Gating gate-C bids on the last known state (`milp_c1..4`)
loses 4-10 %. A proper T-25 re-plan (`qest.replan`, `milp_rp*`: 4-h MILP every MTU on persistence-blended activation
probabilities, with and without deliberate deviations) is -0.5 % on Q4 and -0.9 to -1.4 % even with the current MTU's own
state known: inside this market model the 82 % is decided at D-1 13:00 (DA position, capacity split); nothing that arrives
on D can move it without an intraday market, which the dataset lacks. Write-up: research/gap_to_hindsight.md.
Also this date: `qdistill` now learns the calibration scalar a under a ridge (paper's Eq. 15) with a 3-seed committee per
setting; a lands at 0.90 and recovers only a small part of the noisy-to-ideal gap (0.58 -> 0.54 held-out, ideal model 0.17),
so the earlier "92 % recovered" (a fixed a = 0.05 that sharpened) is withdrawn. Q4 distilled hit 0.74, EUR 1,802,092
(-0.58 % vs MILP). A "guided MILP" that takes the quantum pattern only when it beats the free MILP on the planned objective
collapses to the baseline by construction (tested on 50 day-units, removed). `qdistill --cost` prices a season of each route across
vendors: live partitioned 5,878 circuits = $472 k on IonQ Forte but $10 k on IQM Garnet, $4 k on Rigetti, $9-38 k on IBM
(time-billed, 1-4 s a job assumed); distilled $1.4-152 k once then $0; classical compute is cents either way. The first
version of this note quoted IonQ Forte alone, the dearest vendor by 50x; corrected. The distilled strategy `p_nqs@chronos` is replayed over 2026 for the twin (results/days_*_qdecide_chronos_nqs.parquet, traces in
twin_traces.parquet): season EUR 37,409,048, -0.51 % vs the MILP's 37,602,146 (Q4 -0.58 %, Q3 -0.70 %, Q1-Q2 -0.45 %).

## 4g. Environment fidelity and the stochastic plan (9 Nov 2026)

research/environment_audit.md lines up `qest.market` against the Estonian market row by row. The decisive gap: we activate
the whole won MW whenever the price crosses our bid; Elering activates only the MW it needs (MARI since Oct 2024, PICASSO
since Apr 2025), so both the baseline's and hindsight's activation revenue are inflated and the ceiling needs the activated
volumes to be trusted. Those live on the Baltic Transparency Dashboard, which Cloudflare closes to this network (403 to
urllib, the `baltic` client and headless Chrome). `qest.fetch_btd` ingests either the API (`baltic-py`, pinned) or Download
Center CSVs from `data/raw/btd/inbox/`; `market.activated_volumes` caps activations once the files exist (no-op until then,
twin check unchanged). Capacity market facts from Elering's BBCM evaluation report 2025: live 5 Feb 2025, marginal pricing,
cap 4,000 EUR/MW/h, EE imports most FRR up, EE lacks mFRR down suppliers, Kiisa plant 250 MW up / resistors 40 MW down as
TSO demand reduction. Intraday prices: no free source; Nord Pool API is paid. `qest.stoch`: two-stage Mean-CVaR day plan over
five gate-A quantile scenarios (one scenario = plan_storage to 1e-4, checked); strategies `milp_stoch`, `milp_cvar`,
`milp_cvar6`.
Later the same day: `qest.qscen` (conditional Markov chain of mFRR activations, exact sampler, monthly recency-weighted refits;
a pseudo-likelihood Ising fit of the same ladder locked into one phase and was dropped), `qest.qscen_q` (quantum-enhanced
MCMC on 16-qubit windows vs single flips vs exact: TV 0.072 / 0.091, autocorrelation 27 / 52 steps). Scenario plans on Q4:
every variant loses 17-25 % to the deterministic median plan, and the Markov marginals inside the deterministic plan are -0.4 % (table in research/gap_to_hindsight.md s7). Activation-cap
bracket (`QEST_ACT_CAP`): at 50 / 25 MW the Q4 gap is 3.0 / 2.7 M and activation knowledge still closes 76-82 %.
Volume-aware planner: `plan_storage(vol_up=, vol_dn=)` caps the activated MW of an offer by the TSO's volume (g_* variables);
`strategies._vol_caps` feeds it (realised caps to the oracle, their mean to the forecast plans). Cap-aware bracket on Q4: at
50 / 25 MW the baseline is 1.48 / 1.33 M, hindsight 4.14 / 3.21 M, activation knowledge 65 / 58 % of the gap and price
knowledge the rest. Twin check unchanged (uncapped path).
Under the 25 MW cap the quantum deciders keep their standing: MILP 1,328,436, distilled 1,315,734 (-0.96 %), live QAOA
1,313,247 (-1.14 %) on Q4 (`qdecide --tag _cap25`; `qdecide.main` now imports torch._dynamo first so distilled and live
solvers can share one run).
Reachable ceiling under the 25 MW cap (plans volume-aware, Q4): baseline 1,328,436, perfect DA price 1,782,522 (+34 %),
perfect DA + capacity prices 1,798,623, activations 2,428,645, hindsight 3,214,004: the forecastable third. Forecasters
tried for it: `qest.fc_tabpfn` (TabPFN v2 open weights, checkpoint in models/tabpfn, stacked on Chronos-2): -14 % / -10 %
worse than Chronos-2 on the Q3 DA price (raw / residual), negative; `qest.fetch_nps` + `chronos.predict(neighbours=True)`
(kind 'chronosx', strategy `milp_chronosx`): the FI / LV / LT day-ahead prices from Elering's open API (37.5k MTUs, EE column
equal to the dataset to the cent) as joint series of the D-1 group.
`chronosx` on Q3: DA MAE -4.7 %, rank correlation unchanged (0.653 -> 0.655); euros neutral (uncapped -0.8 / -1.2 %, capped
+0.8 / +0.05 % on Q3 / Q4). huggingface_hub and transformers pinned after tabpfn's resolver broke the Chronos-2 import.
Reachable optimum over the season (`strategies.oracle_prices`: perfect DA and capacity prices, Chronos-2 activations):
EUR 46,478,701 vs baseline 37,602,146 (81 % captured) vs hindsight 105,177,733; shown in the twin beside hindsight
(results/twin_reach_days.parquet, `R.reach`). Twin check unchanged.
Opt-in activation ramps (`market.FAT`, `QEST_FAT=1`; mFRR .933, aFRR .97 from the products' full activation times) shift
every Q4 line by ~2 % (baseline 1,777,256 uncapped, 1,320,719 with the 25 MW cap); default off to keep the organisers'
scoring; twin check unchanged.

## 4h. Modelling the TSO's activations from public drivers (9 Nov 2026)

Elering's open API turned out to carry the Baltic-block drivers (15-min cross-border flows, NTC / ATC, planned trade;
`qest.fetch_tx`, data/raw/elering_tx.parquet); its `balance` endpoint's regulation fields are empty for every date tried.
`qest.actfc`: LightGBM for P(mFRR UP / DOWN at our bid) and the aFRR active shares from gate-A-legal features (dataset lags,
D-1 weather and wind-fleet ramps, Elering NTC / schedules / flows, Nord Pool spreads, Elering's D-1 plan), boosted from the
recency-weighted hour-of-day climatology, monthly refits. Q3: mFRR UP AUC 0.574 vs 0.564 climatology, aFRR share Brier 0.163 vs
0.172 (Chronos-2 0.212). First version had day-of-year / month features and read October as the Q1-Q2 regime (+12.8 % Q3,
-8.2 % Q4); removed after that diagnosis. Corrected `milp_act` (same MILP, the model's marginals): Q3 +16.0 % uncapped
(+1.29 M +- 0.80 M paired), +3.4 % under the 25 MW cap (+205k +- 157k); Q4 +6.4 % / +0.6 % (within noise on 25 days). The
scenario chain `qscen` now takes the marginals as features; through it the gain dilutes (`milp_mk` +1.1 % Q3, -0.9 % Q4) and the two-stage
scenario plans still lose 16-25 % (recourse model, not information). Quantum-proposal MCMC on the conditioned chain keeps its 2x mixing
edge (TV 0.071 vs 0.088, autocorrelation 28 vs 52). Optuna re-tune of `milp_act` on Q3 (40 trials) stayed below the inherited knobs
(8.69 M vs 9.38 M): file kept as tune_best_milp_act.discarded.json. Deck v11 adds the activation slide. Write-up: research/activation_model.md.
Later the same day, leakage pass: `qest.actfc` refits now cut at month start minus the D-2 lag; `qest.evaluate` gained a
walk-forward mode (`QEST_WALK=1`: per-month calibration from earlier months only, January on itself); `milp_act_gated` uses the
model per product only when its trailing 30-day Brier (public outcomes, D-2 lag) beats the table. Whole-year walk-forward,
291 days: `milp_act` +2.7 % uncapped (+996k +- 1.38 M), +0.2 % at 25 MW (+57k +- 266k); gated +1.9 % / +0.1 %; Q3 alone +14.7 % /
+3.6 %. research/leakage_audit.md lists every input's timing. Deck v12 carries the walk-forward table.
Then `milp_act_rank`: the same trailing gate scored by Spearman rank correlation (what the plan profits from) instead of
Brier: no negative month, +3.2 % a year uncapped (+1.21 M +- 0.78 M, 135 / 291 days, significant), +0.3 % at 25 MW
(+98k +- 116k, noise); Q3 +10.2 % / +1.2 %. It admits the aFRR shares on nearly every day from May and the mFRR model
almost only in October. Deck v13 (artifact version 14) quotes it; the Brier gate stays in the table for comparison.


## 4i. The uncertainty line pays: faithful two-stage scenario plans (10 Nov 2026)

Autonomous day (user away). Research pass on 2026 entries (research/README.md "Added 10 Nov"): QEst 2025 winner Enefit
(EV charging, savings statement, pilot prize), Qtyche (QAOA + Benders, exact optimum at 14 buses), IonQ shadow-trained NQS
(2611.08129), Marshall-Egger QeMCMC + tempering (2602.06171), PCE at scale (2607.24722), the Heron benchmark where deep
QAOA is random-baseline-limited (2607.11637). Repo pushed to github.com/MuhammadHakami/QEst_2026 (two commits).
Built and measured (research/quantum_gain_2026.md): `qest.stoch` recourse 'faithful' (delivered = activated, shortfall at
the activation price) and 'guard' (settlement's SoC guard by fixed point; too pessimistic, dropped), pinned first stage for
in-model valuation, exact 25 MW cap (activated MW before the probability, one binary per MTU / direction / scenario);
`qest.qscen` mean-consistent scenarios at the plan's margin, chain fallback for Jan-Feb (fit to 1 Mar, printed), fill of
missing actfc features; `qest.qscen_q` per-chain targets, parallel tempering, day sampler, day benchmark; `qdecide`
forecaster 'actrank'; `QEST_DAYS` subsets. Results, walk-forward: deciders on the rank-gated marginals stay within
+-0.7 % of the MILP (QAOA hits the QUBO optimum 94-98 %: the surrogate is the ceiling). Scenario plans `milp_scenf*`
(rank-gated marginals, 4-16 exact chain scenarios or MCMC day samplers) beat `milp_act_rank` over the year by +4.0 to
+6.7 % (+1.5 to +2.6 M, all significant; +7 to +10 % over Chronos-2 + MILP) and under the 25 MW cap by +8.3 / +8.6 % on Q3
and +8.1 / +8.5 % on Q4 (significant). Sampler fidelity is not what pays: 4 exact scenarios beat 16, the single-flip
MCMC beats the quantum one; day-level sampler benchmark: quantum proposal equal to local on marginals, worse on
persistence, 80x the time; tempering helps both. Whole year under the 25 MW cap (11 Nov): `milp_scenf` +5.4 % (+1.65 M
+- 0.53 M, 204 / 291 days, every month positive), `milp_scenf16` +5.1 %; MCMC-sampled plans under the cap +8.8-10.2 % on
Q3 / Q4. Deck v14 (17 slides): new 'scenarios' slide, payback and verdict rewritten. Scripts copied into scripts/.

11 Nov: the organisers' P3 puzzle (30 qubits, two hours linked by SoC) solved on the quantum track: `qubo.partitioned`
(blocks pinned into each other's linear terms, alternating hour / site partitions, 4 restarts) takes every gate-model
solver (QAOA, BF-DCQO, FALQON, SQA) from PCE's 84.5 % / 0.017 hit to 99.92 % of the optimum / 0.993 hit on 300 Q4
instances with 15 and 10 qubits (tabu 99.92 % / 0.907). PCE one-hot decoding (`qsota.pce(groups=)`) changed nothing.
research/quantum_gain_2026.md s5, results/RESULTS.md.

## 4j. Forecasting SOTA, Lean, tuning, sim-to-real (11 Nov 2026)

Timestral-209M (`qest.fc_tsfm`, `QEST_FC`): moves the baseline +-6 %, the plan of record by less than its interval. Quantum
fine-tuning head (`qest.qhead`): beats its MLP control, not LightGBM. Conditional Born machine (`qest.qgen`): loses to the
chain (-0.80 vs -0.52 log-lik per MTU). Forecast collapse measured (medians keep 30-57 % of the spread). Lean 4 proofs
(`proofs/`, five theorems, clean build). Optuna on Q2 only (`scripts/tune_scen.py`): out of sample on Q3 + Q4 the tuned
scenario plan is +17.2 % over baseline (+13.5 % capped); quantum-MCMC scenarios with the NAS proposal
(`scripts/nas_sampler.py`, IACT 18.5 -> 11.7) +16.6 % / +12.8 %, tied with the exact chain. Sim-to-real preset
(`QEST_REAL=1`, SoC carried over midnight, research/sim2real.md): year 28.70 M -> 30.68 M (+6.9 %, every month positive).
Deck v15 (22 slides, story rebuilt, inline charts). Bug caught: `tsens` fell back to the LightGBM knobs. Plan:
research/plan_v2.md. Runs are launched detached (scripts/run_*.sh) since a session restart killed two jobs.
Later on 11 Nov: scenario reduction (`reduce`, `reduceq`) and joint mFRR + aFRR scenarios (`qscen.sample_afrr`, `joint` /
`jreduce`): best plan `milp_scenf_jr8_t`, out of sample Q3 + Q4 +19.1 % (+14.5 % capped); whole year 41.68 M (+11.2 %),
33.10 M capped (+9.3 %), 31.55 M under the sim-to-real preset (+9.9 %). Sim-to-real tune on Q2: +0.3 %, knobs kept.
Quantum samplers stay at parity in every comparison; verdict in research/quantum_gain_2026.md s10d.
Hardware, 11 Nov evening: IBM (`ibm_fez`, key IBM_API + CRN in .env) and IQM Garnet (IQM_MUHAMMAD_TOKEN, separate
.venv-iqm because the IQM client pins qiskit 2.1). `scripts/ibm_run.py` (P2 puzzle, LR-QAOA, random control; `--export` /
`--score` for other vendors; `--pack`), `scripts/iqm_run.py`. ibm_fez, 120 hours on 5 days, p = 2: optimum returned 85.0 %
vs 59.2 % random (McNemar p < 0.0001). Garnet 71 % vs 62 % on one day. Packing 10 puzzles on 150 qubits drops to 58 %.
Pasqal not run (password prompt). `proofs/Qproofs/Hybrid.lean`: 8 more theorems on the hybrid strategies.

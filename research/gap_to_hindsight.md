# The EUR 67.6 M between our plan and hindsight: where it is, what SOTA says, what the hybrid can do (9 Nov 2026)

Season = the 291 traded days of the dataset (8 Jan to 25 Oct 2026), both sites (Hertz 1 100 MW / 200 MWh, hybrid 20 MW PV +
20 MW / 40 MWh), official settlement. Code: `qest.strategies` (`oracle_*` partial oracles, `milp_c*` gate-C variants),
`qest.evaluate`; runs in `results/days_*_oracle_parts.parquet`, logs `results/run_oracle_parts_*.log`.

## 1. The gap by market (season)

| EUR M | Chronos-2 + MILP | Hindsight | Gap |
|---|---:|---:|---:|
| capacity | 18.67 | 21.54 | +2.87 |
| day-ahead energy | -45.71 | 13.13 | +58.84 |
| mFRR energy | 36.95 | 43.65 | +6.70 |
| aFRR energy | 20.21 | 29.13 | +8.92 |
| imbalance | 8.89 | 0.08 | -8.81 |
| leftover / tariff | -1.40 | -2.34 | -0.94 |
| **total** | **37.60** | **105.18** | **+67.58** |

Read with care: the day-ahead line is where the plan *pays* for the energy it later sells into activations. Hindsight
sources that energy at the right quarter-hours because it knows when the activations come. The accounting puts the
difference on the DA line; the information behind it is the activations.

## 2. Value of perfect information, one input at a time (Q4, 25 days)

Each row is the same MILP with one input group taken from the realised day and Chronos-2 for the rest; the gate-C policy
(`soc_aware`) is unchanged. Baseline EUR 1,812,548, hindsight 6,335,393, gap 4,522,845.

| Perfect knowledge of | EUR Q4 | Gap closed |
|---|---:|---:|
| aFRR + mFRR activations (shares, prices) | 5,748,379 | **87.0 %** |
| mFRR only | 4,018,224 | 48.8 % |
| aFRR only | 3,353,315 | 34.1 % |
| DA + capacity prices | 2,199,338 | 8.6 % |
| DA price only | 2,172,655 | 8.0 % |
| capacity prices only | 1,842,442 | 0.7 % |
| hybrid PV output | 1,810,235 | -0.1 % |
| everything, `soc_aware` gate C kept | 6,182,952 | 96.6 % (the gate-C policy itself is worth 3.4 %) |

The money is in knowing the TSO's activations. Price forecasting (the whole EPF literature) caps at 8 %.

Season (291 days, gap EUR 67.58 M): perfect activations close **82.1 %** (EUR 93.05 M), perfect DA price 6.0 %
(41.67 M), everything with the `soc_aware` gate C 97.1 % (103.24 M).

## 2b. How Estonia runs these activations (what is known when)

- Reserves are procured D-1 in the Baltic balancing capacity market (BBCM, live since Feb 2025; aFRR added April 2025
  when Elering joined PICASSO). The capacity auction closes 09:00, the day-ahead auction 13:00 (deck slide 6).
- Energy bids for each MTU close at **T-25** with the connecting TSO (MARI gate closure; the deck's gate C). mFRR energy
  is activated through MARI (Elering since 9 Oct 2024): scheduled activation once per 15-min MTU, direct activation at
  any time, marginal pricing; aFRR through PICASSO every 4 s (the dataset's 4-s ticks). The TSO decides, from the
  Baltic LFC block's imbalance, which direction and how much; a BSP never knows in advance.
- What is public while D runs: Elering's dashboard shows the system balance and production/consumption in near real
  time (the `ee_real_*` columns); Fingrid's practice, which the Baltic platforms share, is a live indicative mFRR price
  with the **final price published 15 min after the MTU ends**; EU 543/2013 Art. 17 obliges TSOs to publish activated
  volumes and prices per balancing time unit (the Baltic Transparency Dashboard carries them; third-party mirrors refresh
  hourly). Elering's own publication lag for activated mFRR/aFRR energy was not confirmed from a primary source
  (the dashboard and the AST rulebook blocked automated fetches); the 15-min figure is Fingrid's.
- So "by the time everything has cleared" (D-1 14:00) nothing about D's activations is known: they depend on D's
  imbalances. At T-25 for MTU t, the BSP knows activations up to roughly t-3 (45 min, with a 15-min publication lag) and
  the system balance a few minutes old. The persistence table in section 3 is exactly that information. A 2025 Estonian
  thesis finds a moderate link between system imbalance and mFRR volumes and only a weak link between volumes and
  prices ([EMU](https://dspace.emu.ee/items/0528658b-6ff7-4c7b-ad9a-bb39ffa53dc3)), which matches the persistence of
  *direction* and the noisiness of *price* we measure.

## 3. How predictable are activations?

Gate A (D-1 09:00): Chronos-2's within-day Spearman rank correlation is 0.64 for the DA price (MAE 33 on a 64 EUR/MWh sd)
and 0.33 for the imbalance price (MAE 71 on sd 108). At that horizon the activation pattern of D is mostly noise.

Close to delivery (Q3 + Q4, 11,236 MTUs; mFRR UP active in 40 % of MTUs, DOWN in 53 %):

| Horizon | P(UP active now, given UP active then) | DOWN |
|---|---:|---:|
| 15 min | 0.86 | 0.88 |
| 30 min | 0.80 | 0.84 |
| 45 min | 0.75 | 0.80 |
| 60 min | 0.72 | 0.78 |
| 120 min | 0.65 | 0.72 |

aFRR UP share autocorrelation 0.58 at 15 min, 0.32 at 30 min. System balance two MTUs back: P(UP) 0.42 when short vs
0.20 when long. The sign of (imbalance - DA) persists 74 % at 30 min. 60 % of the mFRR UP premium sits in 10 % of MTUs.
So activations are strongly persistent on the 30-60 min scale the T-25 gate works on, and nearly unpredictable at D-1.

## 4. What does NOT work: gating gate-C bids on the last known state

`milp_c<L>`: the plan is unchanged, but at each MTU the energy bid stays in the money only in the directions the TSO was
activating L MTUs earlier (others priced out). Q4: L = 1 -4.2 %, L = 2 -9.8 %, L = 3 -8.9 %, L = 4 -9.9 % vs the baseline.
Lowering the margin when active changes nothing (the tuned margins are already at the grid's bottom). Missing the 20 %
of activations that persistence gets wrong, plus the imbalance they cause, costs more than the pricing out saves. The
activation value is in **re-planning the rest of the day** (DA energy already bought, SoC path, which direction to
hold), not in bid timing.

## 4b. The T-25 re-plan, built and measured (`qest.replan`, strategies `milp_rp*`)

The gate-A plan stands; at every MTU a small MILP re-solves the next 4 h with the activation state public by T-25
(persistence tables from the train split blended with the gate-A forecast), choosing per direction whether the energy bid
stays in the money, and, in the `dev` variants, a deliberate deviation from the committed schedule settled at the expected
imbalance price. Q4, EUR, baseline 1,812,548:

| Variant | Information | EUR Q4 | vs baseline |
|---|---|---:|---:|
| `milp_rp` | activations 45 min back, imbalance 60 min back | 1,803,193 | -0.5 % |
| `milp_rpdev` | same, deviations allowed | 1,802,852 | -0.5 % |
| `milp_rp0` | the current MTU's own state (not public: an upper bound) | 1,796,245 | -0.9 % |
| `milp_rp0dev` | same, deviations allowed | 1,787,974 | -1.4 % |
| activation oracle at gate A (for scale) | the whole day's activations in the D-1 plan | 5,748,379 | +217 % |

Even knowing the current MTU's activation state and imbalance price exactly, gate-C levers add nothing: the `soc_aware`
rule already keeps every direction in the money whenever the SoC can back it, and being activated is never a loss in
itself. The 82 % is decided at D-1 13:00, in the DA position and the capacity split. Inside the competition's market model
(no intraday market), information that arrives on D cannot move the money; the only intraday proxy, deliberate imbalance
positions, is neutral at best here and is a BRP-rule problem in reality.

## 5. SOTA that applies (checked 9 Nov 2026)

- Rank correlation, not MAE, drives dispatch value: forecasts with Kendall tau above 0.85-0.95 capture 97-100 % of
  perfect-foresight intraday revenue, persistence 33 %; rolling MPC re-plan after DA clears
  ([arXiv:2604.12082](https://arxiv.org/abs/2604.12082), ETH, 2026). Our DA tau is ~0.46 (Spearman 0.64).
- Sequential market re-optimisation at every gate with rolling forecasts and opportunity-cost bids
  ([arXiv:2607.24075](https://arxiv.org/abs/2607.24075), 2026).
- Risk-aware two-stage stochastic DA + mFRR co-optimisation with KDE price scenarios and a Mean-CVaR objective
  ([arXiv:2609.18349](https://arxiv.org/abs/2609.18349), IECON 2026). We plan on q50 only; q10 / q90 are unused.
- Foundation models for prices: TabPFN beats market-specific EPF benchmarks statistically but not always economically
  ([arXiv:2609.00089](https://arxiv.org/abs/2609.00089), Lipiecki & Weron 2026); Chronos-2 with covariates (load and
  RES forecasts) on Belgian DA and imbalance ([arXiv:2605.17045](https://arxiv.org/abs/2605.17045)); gated-LoRA
  Chronos-2 for 15-min DA ([arXiv:2608.11359](https://arxiv.org/abs/2608.11359)); PriceFM, 96 quarter-hours across
  38 European regions ([arXiv:2508.04875](https://arxiv.org/abs/2508.04875), [code](https://github.com/runyao-yu/PriceFM)).
  Capture of perfect-foresight DA arbitrage with forecasts: 87-89 % in Spain
  ([MDPI Forecasting 8(4):61](https://www.mdpi.com/2571-9394/8/4/61)), 94-95 % in DK with LightGBM
  ([nordic-power-price-forecast](https://github.com/davidsanzdeandino/nordic-power-price-forecast)). For us the DA
  lever is worth 8 % of the gap at most, so this literature is second-order here.
- Activation / imbalance forecasting at short horizon: two-stage XGBoost (direction classifier, then price) for mFRR
  in SE2 ([KTH thesis](https://www.diva-portal.org/smash/get/diva2:1988435/FULLTEXT01.pdf)); market-rule-informed
  neural imbalance price model on 15-min data ([arXiv:2605.09061](https://arxiv.org/abs/2605.09061), 2026).
- Our quantum-for-energy landscape (claude.ai/artifact/C2R1rHLTwiqh4mu9GvwnVV, `research/methods.csv`): every 2026
  competition and paper is a decision-side method (unit commitment, QAOA / PCE / annealing variants). None forecasts
  activations or prices. That matches our own result: all deciders land within 0.3 % of each other on one forecast.

## 6. Plan, ranked by the ceiling it can reach (revised after 4b)

1. ~~Rolling re-plan at T-25~~: built, measured, dead inside this market model (section 4b). It would come back only with
   an intraday market in the settlement (XBID-style, T-30 per MTU), where the ETH and Hornek results (90-100 % of
   perfect-foresight intraday value with rank-sufficient forecasts) apply. The dataset has no intraday prices, so this
   needs data the organisers did not give.
2. **Stochastic, risk-aware day plan**: two-stage MILP over Chronos-2 quantile scenarios of activations and prices,
   Mean-CVaR objective; tune the risk knob on Q3 as `qest.tune` does. Uses the q10 / q90 we already produce.
3. **Gate-A activation forecasts with covariates**: Elering's D-1 plan, Fingrid's wind forecast, Estlink state; score
   by within-day rank correlation, not MAE. Expect a modest gain: at D-1 the activation pattern is mostly noise.
4. **DA price**: capped at 8 % of the gap. Keep Chronos-2.

Where the hybrid fits, honestly: (1) is a QUBO re-solved 96 times a day per unit; a live QPU cannot do that in cost or
latency (EUR 727 and minutes per decision), the distilled sampler does it in 0.03 s with zero QPU calls, which is the
IonQ amortisation argument applied where it matters; (2) needs joint scenario sampling, where our annealing / MCMC
samplers (`qanneal`, `qmcmc`) can propose scenario sets for the LP to price. Neither has been shown to add euros here
yet; the deciders so far trail the MILP by 0.1-0.9 %, and the gain, if any, will come from the information, not the
sampler.

## 7. Toward a reachable ceiling and a planner that uses the uncertainty (9 Nov 2026, later)

**Is the ceiling physical?** The settlement activates the whole won MW whenever the price crosses the bid; Elering activates
what it needs. Until the activated volumes are in (research/environment_audit.md s5), a uniform cap brackets the effect
(`QEST_ACT_CAP`, plans not cap-aware, Q4):

| Cap on every activation | Baseline | Activation oracle | Hindsight | Gap | Activations close |
|---|---:|---:|---:|---:|---:|
| none (as before) | 1,812,548 | 5,748,379 | 6,335,393 | 4,522,845 | 87 % |
| 50 MW | 1,730,692 | 4,032,917 | 4,768,630 | 3,037,938 | 76 % |
| 25 MW | 1,460,961 | 3,659,566 | 4,134,785 | 2,673,824 | 82 % |

The euros scale with the cap; the split does not. Note the undelivered activation energy lands on the imbalance line of
every plan (0.2 -> 1.2 M for the baseline at 25 MW), a windfall the plans did not earn. With the planner made volume-aware
(`plan_storage` with `vol_up` / `vol_dn`: the activated MW of an offer is a variable capped by the TSO's volume; the
baseline is told the cap, hindsight the realised one) the bracket becomes the physically consistent one:

| Cap, plans aware of it | Baseline | Activation oracle | Hindsight | Gap | Activations close |
|---|---:|---:|---:|---:|---:|
| 50 MW | 1,478,017 | 3,205,962 | 4,139,778 | 2,661,761 | 65 % |
| 25 MW | 1,328,436 | 2,428,645 | 3,214,004 | 1,885,568 | 58 % |

So under realistic activation volumes the Q4 ceiling is 2.4-2.8x the baseline instead of 3.5x, and price knowledge grows
to a third of the (smaller) gap. The **physically reachable ceiling** is the one built from information that exists before
the gates: perfect prices, no activation foresight. Under the 25 MW cap (plans volume-aware): baseline 1,328,436, perfect DA
price 1,782,522 (+34 %), perfect DA and capacity prices 1,798,623 (+35 %), perfect activations 2,428,645, hindsight
3,214,004. The reachable target for any forecaster, quantum or classical, is therefore about a third above the baseline;
the rest of hindsight is foresight nobody has.

Over the whole season (uncapped settlement, as the twin shows it): reachable optimum EUR 46,478,701 (Q1-Q2 33.86 M, Q3
10.42 M, Q4 2.20 M) against the baseline's 37,602,146 and hindsight's 105,177,733. **The baseline already captures 81 % of
what any forecaster could reach**; the remaining EUR 8.9 M a season is the price lever, and the EUR 58.7 M above the
reachable line is activation foresight. The twin now shows the reachable optimum beside hindsight (strategies panel and
the results KPIs, `results/twin_reach_days.parquet`), so the benchmark people see is the one a real bidder can approach.

Activation ramps (`QEST_FAT=1`: mFRR delivers 93.3 % of its MW over the MTU, aFRR 97 %, rule-derived) move every line by
about 2 % and nothing else: uncapped Q4 baseline 1,777,256, reachable 2,214,328, activations 5,671,729, hindsight 6,200,264;
with the 25 MW cap 1,320,719 / 1,799,284 / 2,387,163 / 3,149,159. The structure of the gap is unchanged.

Chasing that third, two forecasters were tried on Q3 against Chronos-2 (`qest.fc_tabpfn`, TabPFN v2 open weights, the
model Lipiecki & Weron 2026 found best): raw stacking -14 % worse on the DA price, residual stacking -10 % worse, about
even on the balancing prices. Negative, kept as a module with its failing check. The cross-zonal variant (`chronosx`:
FI / LV / LT day-ahead prices from Elering's open API as joint series, qest.fetch_nps) is the next, physical, lever. Q3 skill: DA price MAE 32.73 -> 31.19 EUR/MWh (-4.7 %), the balancing targets unchanged (they
sit in the D-2 group), within-day rank correlation 0.653 -> 0.655. Euros (`milp_chronosx` vs `milp_chronos`): uncapped Q4 -1.2 %, Q3 -0.8 %;
under the 25 MW cap Q4 +0.05 %, Q3 +0.8 %. A 5 % better MAE with the same rank correlation buys nothing, exactly the ETH
finding: the plan reads the shape, not the level. Neutral, kept as the forecaster of record for cross-zonal experiments. The majority of hindsight stays activation foresight, which no one has at D-1: that part
of hindsight is not physically reachable by any bidder, with or without qubits. What is reachable is the price third
(forecastable in principle) and the cap-aware plan itself, which the real volumes will size.

**Scenario planning (`qest.stoch`, `qest.qscen`).** A two-stage Mean-CVaR plan over gate-A scenarios: comonotonic
quantiles (`milp_stoch`, `milp_cvar`) or eight exact draws of a conditional Markov model of the activation pattern
(`milp_scen`, `milp_scen_cvar`; held-out log-likelihood -0.516 per MTU vs -0.889 independent, run lengths reproduced). Q4:

| Plan | EUR Q4 | vs baseline |
|---|---:|---:|
| deterministic median (baseline) | 1,812,548 | – |
| Markov scenarios, CVaR 0.3 | 1,513,357 | -17 % |
| Markov scenarios | 1,431,531 | -21 % |
| quantile scenarios | 1,376,917 | -24 % |
| quantile scenarios, CVaR 0.3 | 1,362,578 | -25 % |
| Markov scenarios, no delivery recourse (first version) | 1,356,918 | -25 % |
| deterministic plan with the Markov model's activation probabilities (`milp_mk`) | 1,804,792 | -0.4 % |

All of them lose. The deterministic plan buys heavily on DA (-2.7 M) and earns it back in activations (+4.0 M); the
scenario plans hedge the runs of activations the model draws, buy less, and the activations that then come find no energy.
Two mechanisms, both ours: eight binary draws make a coarse expectation per MTU, and the recourse (deliver what the SoC
can back) is more permissive than the settlement's guard, so the plan leans on it. The honest reading: with activations
this unpredictable at D-1, the median plan plus the gate-C guard is a strong policy, and risk aversion only costs
expected euros; the scenario machinery becomes useful when the scenarios carry information, i.e. with the activation
volumes and a market model where runs actually bind.

**The sampler benchmark (`qest.qscen_q`, 20 windows of 8 MTUs = 16 qubits, 400 steps x 256 chains).** Target: the exact
conditional path distribution. Quantum proposal (Layden-style, Trotterised X mixer) vs single-bit flips:

| Sampler | TV to exact marginals | acceptance | autocorrelation (steps) |
|---|---:|---:|---:|
| quantum proposal | 0.072 | 0.09 | 27 |
| local flips | 0.091 | 0.15 | 52 |

The quantum proposal mixes about twice as fast per step, as Layden et al. report for Ising targets, and the exact forward
sampler needs no steps at all because the model is a chain. So the qubits are doing a real, measured job, on a target
where a classical exact method exists. The target stops being a chain when aFRR, the neighbouring zones and the Finnish
price couple in: that is the data to fetch before the sampler can matter.

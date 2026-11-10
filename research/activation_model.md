# Modelling the TSO's activations from public drivers (9 Nov 2026)

Why: 82 % of the EUR 67.6 M between our plan and hindsight is knowing when Elering activates aFRR / mFRR
(research/gap_to_hindsight.md). Until today the plan used an hour-of-day climatology for mFRR and Chronos-2's zero-shot
share for aFRR, and "activations are noise at D-1" was measured on one model's features, not on the drivers of the Baltic
block's imbalance. This note builds the driver model, measures it, and puts it in the plan. Code: `qest.fetch_tx`,
`qest.actfc`, strategies `milp_act` / `milp_act_afrr` / `milp_act_mfrr`, `qest.qscen` conditioned on it.

## 1. Public data that was reachable after all

Elering's open dashboard API (dashboard.elering.ee, `/v3/api-docs`, no key, not geo-blocked) carries, at 15 min:

| Endpoint | Content | Legal at gate A (D-1 09:00) as |
|---|---|---|
| `transmission/cross-border` | realised flows: Finland, Latvia, Estlink 1, Estlink 2 (MW, + = import) | D-2 same MTU; D-2 schedule deviation flow - plan |
| `transmission/cross-border-capacity` | NTC and ATC in / out per border (FI, LV) | D's NTC (published before the auction; ATC moves intraday, not used) |
| `transmission/cross-border-planned-trade` | scheduled trade per border after the auctions | D-1's schedule (fixed at D-2 13:00) and its hourly steps |
| `balance` | imbalance, regulate_up / down fields | **empty** for every date tried (2024-2026): the activated-volume series is not here |

`qest.fetch_tx` writes data/raw/elering_tx.parquet (37.5k MTUs, 2025-10-01 .. 2026-10-27, every column complete). The
dataset's own `ee_plan_*` columns (Elering's D plan) enter lagged one day; the FI / LV / LT day-ahead prices
(qest.fetch_nps) as D-1 spreads to EE.

## 2. The forecaster (`qest.actfc`)

Targets per MTU: m_up / m_dn = the mFRR price crosses DA +- 5 EUR/MWh (the planner's activation indicator), a_up_s / a_dn_s =
PICASSO active share. LightGBM, boosted from the recency-weighted hour-of-day climatology (init_score = its logit) with
heavy regularisation (7 leaves, 250 min samples, lambda 20), recency half-life 60 days, refitted before each calendar month
on every day before it (gate-A legal). Features: the gate-A set of qest.forecast (D-2 balancing lags, D-1 prices, D-1
weather for PV and the wind fleet, D-2 consumption) plus wind-fleet ramps (4 and 8 MTUs), PV day sum, D-7..D-2 activation
rates, the Elering transmission features above, Elering's D-1 plan and D-2 plan error, the Nord Pool spreads. No calendar
features (day of year, month): the first version had them, and in October (never seen in a January-September fit) they,
with the low autumn PV level, made the model read Q4 as the Q1-Q2 regime (predicted mFRR UP 0.26 against 0.43 realised).
That version earned +12.8 % on Q3 and -8.2 % on Q4; the correction below was made after that diagnosis, so the Q4 numbers
of the corrected model are not a blind test in the strict sense. Q3 is.

Skill (fit through the previous quarter, monthly refits; climatology = the plan's reference):

| | Q3 actfc | Q3 climatology | Q3 Chronos-2 | Q4 actfc | Q4 climatology | Q4 Chronos-2 |
|---|---:|---:|---:|---:|---:|---:|
| mFRR UP: AUC | 0.574 | 0.564 | - | 0.581 | 0.483 | - |
| mFRR UP: log-loss | 0.676 | 0.665 | - | 0.673 | 0.694 | - |
| mFRR UP: within-day Spearman | 0.166 | 0.113 | - | 0.020 | -0.044 | - |
| mFRR DOWN: AUC | 0.550 | 0.557 | - | 0.577 | 0.490 | - |
| aFRR UP share: Brier | 0.163 | 0.172 | 0.212 | 0.153 | 0.163 | 0.187 |
| aFRR UP share: within-day Spearman | 0.149 | 0.106 | 0.119 | 0.075 | 0.076 | 0.106 |
| aFRR DOWN share: Brier | 0.156 | 0.162 | 0.205 | 0.147 | 0.156 | 0.181 |

Top features: PV and wind forecasts and the wind-fleet ramps, yesterday's Latvian schedule, D-2 flow on Estlink, the FI
spread, Elering's plan. The signal is real and small: AUC 0.57-0.58 for mFRR direction. The aFRR shares gain most.

## 3. Euros: the deterministic plan with these marginals (`milp_act`)

Same MILP, same gate-C policy, same tuned knobs as `milp_chronos`; only m_up_q / m_dn_q (mFRR probabilities) and the aFRR
shares change. Battery + hybrid, official settlement. The 25 MW column is the volume-capped settlement with the planner told
the cap (research/environment_audit.md): the realistic cell.

| Cell | Chronos-2 + MILP | + activation model | Gain | Paired 95 % | Days won |
|---|---:|---:|---:|---:|---:|
| Q3 (92 d), uncapped | 8,085,147 | 9,377,554 | **+16.0 %** | +1,292,407 +- 803,146 | 64 / 92 |
| Q4 (25 d), uncapped | 1,812,548 | 1,928,479 | +6.4 % | +115,932 +- 162,290 | 15 / 25 |
| Q3, 25 MW cap | 5,953,752 | 6,159,087 | **+3.4 %** | +205,335 +- 157,410 | 58 / 92 |
| Q4, 25 MW cap | 1,328,436 | 1,336,548 | +0.6 % | +8,112 +- 47,283 | 11 / 25 |

aFRR shares alone (`milp_act_afrr`): Q3 +10.0 % / +0.8 % capped, Q4 -4.0 % / -2.3 % capped. The mFRR probabilities are what
carries Q4. Where the uncapped gain comes from: the plan buys more day-ahead energy (Q3: 205k MWh vs 136k) and sells it into
the activations it now expects (mFRR +1.8 M, aFRR +2.9 M), with more favourable imbalance (+0.8 M) and no penalties. Under
the cap the activation energy is bounded by the TSO's volume, so the same information is worth 3.4 % on Q3 and is within
noise on 25 October days. These are the first gate-A levers that move euros in the right direction since the pipeline
was built; everything tried before (re-plans, bid gating, scenario / CVaR plans, TabPFN, cross-zonal prices) was neutral or
negative.

Knobs: `milp_act` runs with `milp_chronos`'s Optuna knobs; `qest.tune --strategy milp_act` re-tunes margin / haircut /
tolerance on Q3 only (results/tune_best_milp_act.json), see section 5.

## 4. Where the qubits go: the uncertainty, now that it carries information

The conditional Markov chain of the day's activation pattern (`qest.qscen`) now takes the driver marginals as features
(logit p_up, p_dn, shares; FEATS extended; days without a forecast are left out of the fit). The chain is the exact sampler
for the joint pattern; the quantum-enhanced MCMC (`qest.qscen_q`, 16-qubit windows, Layden-style proposals) is the sampler
for the general case, when the target stops being a chain (aFRR, both borders, the Finnish price coupled in). Results of
the scenario plans on the conditioned chain and the sampler benchmark: section 5.

## 5. The conditioned chain, the scenario plans, the sampler, the re-tune

Chain conditioned on the actfc marginals (held-out log-likelihood unchanged at -0.516 per MTU: the persistence terms
already carry most of what the drivers add at the transition level). Plans on it, EUR, baseline in brackets:

| Plan | Q3 uncapped (8,085,147) | Q4 uncapped (1,812,548) | Q3 25 MW (5,953,752) | Q4 25 MW (1,328,436) |
|---|---:|---:|---:|---:|
| `milp_act` (direct marginals) | 9,377,554 (+16.0 %) | 1,928,479 (+6.4 %) | 6,159,087 (+3.4 %) | 1,336,548 (+0.6 %) |
| `milp_mk` (chain marginals, 200 samples) | 8,174,893 (+1.1 %) | 1,795,731 (-0.9 %) | 5,981,908 (+0.5 %) | 1,322,485 (-0.4 %) |
| `milp_scen` (8 chain scenarios) | 6,259,656 (-23 %) | 1,368,622 (-24 %) | 4,071,200 (-32 %) | 998,439 (-25 %) |
| `milp_scen_cvar` (CVaR 0.3) | 6,333,018 (-22 %) | 1,467,699 (-19 %) | 4,302,414 (-28 %) | 1,122,587 (-16 %) |

Reading: the information reaches the plan through the direct marginals; pushed through the chain it is diluted (the
logistic transition model absorbs the drivers into persistence), and the two-stage scenario plan still loses a fifth to a
quarter, as before conditioning. The scenario machinery's problem is its recourse model (deliver what the SoC can back,
more permissive than the settlement's guard) and the coarse 8-draw expectation, not the information. The honest
placement of the uncertainty work is therefore: better marginals first (done, section 3), a recourse model that mirrors
the settlement second, and only then joint scenarios, where the sampler matters.

Sampler benchmark on the conditioned chain (`qscen_q`, 20 windows of 8 MTUs = 16 qubits, 400 steps x 256 chains):
quantum proposal TV to the exact marginals 0.071, acceptance 0.093, autocorrelation 28.5 steps; single-bit flips 0.088 /
0.148 / 52.1 (unconditioned chain: 0.072 / 0.09 / 27 vs 0.091 / 0.15 / 52, results/qscen_q_unconditioned.csv). The
quantum proposal mixes about twice as fast per step on the informative target too; the exact forward sampler needs no
steps because the target is a chain.

Re-tune of `milp_act`'s knobs on Q3 (Optuna, 40 trials, `qest.tune --strategy milp_act`): best 8,689,941 EUR (mFRR margin
37.8, kappa_dn 0.15, tol 0.19), below the inherited Chronos-2 knobs (9,377,554), so the inherited knobs stay; the tuner's
file is kept as results/tune_best_milp_act.discarded.json so it is not picked up automatically.

## 6. The whole year, walk-forward, no leakage (the number that counts)

The portfolio decides every day; the estimate must be a walk-forward backtest (research/leakage_audit.md). `QEST_WALK=1`:
each month's calibration table comes from the months before it (January on itself), the activation model is refitted
before each month on outcomes public by then (D-2 lag), the chain likewise; the Optuna knobs stay the Q3-tuned ones for
both plans (shared, so the difference is fair; the level of Q1-Q3 is optimistic by what the knobs are worth). 291 days,
battery + hybrid, official settlement; paired daily 95 % intervals:

| Plan | Uncapped | 25 MW cap (realistic) |
|---|---:|---:|
| Chronos-2 + MILP | 37,480,487 | 30,277,111 |
| `milp_act` | 38,476,470 (+2.7 %, +996k +- 1,383k, 146 / 291 days) | 30,333,928 (+0.2 %, +57k +- 266k) |
| `milp_act_gated` (model used only when its trailing 30-day Brier beats the table, per product, D-2 lag) | 38,185,900 (+1.9 %, +705k +- 704k, 113 / 291) | 30,310,007 (+0.1 %, +33k +- 108k) |
| `milp_act_rank` (same gate, trailing Spearman rank correlation instead of Brier) | **38,688,597 (+3.2 %, +1,208k +- 778k, 135 / 291)** | 30,375,085 (+0.3 %, +98k +- 116k, 132 / 291) |

By quarter, `milp_act`, uncapped / capped: Q1-Q2 (Feb-Jun, the model has 1-5 months of history) -1.1 % / -0.3 %; Q3 +14.7 %
(+1.19 M +- 0.78 M) / +3.6 % (+217k +- 171k); Q4 +2.6 % / -0.2 %. The Brier gate removes the spring losses (no month
below -1 %) and most of the Q3 gain (its calibration test is stricter than what the plan needs, which is ranking): +3.9 % /
+0.2 % on Q3. The rank gate keeps both: no negative month uncapped (Mar +1.0, Apr +3.0, May +2.0, Jun +2.0, Jul +12, Aug +10,
Sep +8, Oct +3 %), Q3 +10.2 % (+831k +- 535k, 59 / 92) / +1.2 % (+72k +- 85k), Q4 +3.1 % / -0.0 %. Which products it admits
(share of days, walk-forward calibration): aFRR shares 30 % of Q1 days, 98-100 % from Q2 on; the mFRR model 0-3 % of days
before October and 96 % in October, where the climatology collapsed (section 2). So the whole-year gain is mostly the aFRR
shares; the mFRR DOWN model (AUC below the climatology on Q3) keeps the mFRR pair out until the regime shift.

Reading, without decoration: the activation model's skill is real and earns in Q3 under both settlements, the only
gate-A lever that has done so. With a ranking gate, the whole-year walk-forward gain under the organisers' settlement is
+3.2 % and significant (interval +- 2.1 %), worth EUR 1.2 M on this portfolio; under realistic activation volumes it is
+0.3 % and inside noise (+- 0.4 %), positive in eight of ten months. Two things would move this: the real activated
volumes (which decide which bracket is true) and a longer history (the model is useless with one month of data and good
with six). Season files: results/days_{train,val,test}_actfc_wf*.parquet (`_wfg*` Brier gate, `_wfr*` rank gate); assembler
scratchpad/season_wf.py `<strategy> <tag>`.

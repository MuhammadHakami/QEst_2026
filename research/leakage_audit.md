# Leakage audit for the whole-year revenue estimate (9 Nov 2026)

The portfolio decides every day whether and how to participate; the revenue estimate must therefore be a walk-forward
backtest, as for a trading system: the plan for delivery day D may use only what was public at D-1 09:00 Tallinn (gate A),
every model is fitted on outcomes published before that moment, and no knob is set with later data. This note lists each
input, its information timing, where the pipeline enforces it, and what was corrected today. Mode: `QEST_WALK=1` in
`qest.evaluate` (per-month calibration from earlier months only) plus the monthly refits in `qest.actfc` and `qest.qscen`.

## 1. Inputs and their timing

| Input | Public from | Used as | Enforced in | Status |
|---|---|---|---|---|
| Day-ahead price of D-1 and earlier | D-2 13:00 | lags >= 1 day | qest.forecast (`LAG_KNOWN`), qest.chronos group D-1 | ok |
| Capacity prices (BBCM) of D-1 and earlier | D-2 ~09:30 | lags >= 1 day | same | ok |
| Balancing outcomes (mFRR, aFRR, imbalance) | with delay; organisers: D-2 at gate A | lags >= 2 days, day means / maxes lagged 2 | qest.forecast, qest.actfc (`_lag`, `_daymap`, rate7 over D-7..D-2), qest.qscen (`lag_rates`, fit cut D-2) | ok; actfc's monthly refit cut moved from month start to month start - 2 days today |
| Weather forecast for D | D-1 00 UTC run | PV / wind per plant and the national fleet, ramps, day sums | qest.power (`*_d1`), qest.forecast | ok |
| Elering realised production / consumption | near real time | day mean lagged 2 (`cons_lag2`), plan error lagged 2 | qest.forecast, qest.actfc | ok (conservative: D-1 morning values would also be legal) |
| Elering's production / consumption plan for a day | published for D after the DA auction (D-1 evening) | lagged 1 day (the D-1 plan) | qest.actfc | ok |
| Cross-border realised flows (qest.fetch_tx) | near real time | same MTU lagged 2 days; schedule deviation lagged 2 | qest.actfc | ok (conservative) |
| NTC for D (qest.fetch_tx) | D-1 morning, before the auction | level for D | qest.actfc | assumption: published before 09:00; ATC not used (it moves intraday) |
| Planned cross-border trade | fixed for D-1 at D-2 13:00; for D at D-1 13:00 | the D-1 schedule and its steps (lag 1) | qest.actfc | ok; D's own schedule is never used (after gate A) |
| FI / LV / LT day-ahead prices (qest.fetch_nps) | D-2 13:00 for D-1 | spreads lagged 1 day | qest.actfc, qest.chronos 'chronosx' | ok |
| Chronos-2 weights | pretrained before 2026 on public corpora | zero-shot | qest.chronos | no 2026 Estonian data in pretraining (released 2025) |
| TabPFN v2 weights | pretrained on synthetic priors | stacked forecaster (negative result) | qest.fc_tabpfn | ok |

## 2. Fitted objects and their fit windows

| Object | Fit window for a day D in month M | Before today |
|---|---|---|
| LightGBM quantile forecasts (kind 'lgbm') | train split for val, train + val for test | unchanged (not the baseline of record) |
| Chronos-2 context | the last CONTEXT_DAYS before D, D-2 for the balancing group | unchanged, legal |
| Hour-of-day calibration table (activation rates, premiums, aFRR shares at the margin) | **walk-forward**: all days before M's first day - 2 days; January on itself (printed) | train split on itself for Q1-Q2 (in-sample), train for Q3, train + val for Q4 |
| qest.actfc activation forecaster | all days with outcomes before M's first day - 2 days; the first month has none (falls back to the calibration table) | month start without the 2-day lag |
| qest.qscen chain | days before M's first day - 2 days | unchanged |
| Optuna knobs (aFRR / mFRR margins, kappa_up / kappa_dn, tol) | **tuned on Q3 (val) only**, applied everywhere | unchanged: the organisers' protocol (tune on Q3, score Q4 once). For Q1-Q2 this is information from the future, for Q3 it is in-sample. Both compared plans share the same knobs, so the *difference* is fair; the *level* of Q1-Q3 revenue is optimistic by whatever the knobs are worth. A walk-forward re-tune (tune on months < M, apply to M) is the remaining step; it costs about 1.5 h of CPU per tune point |

## 3. Settlement-side realism (not leakage, but it moves the level)

- Activation volume: the organisers' rules activate the whole won MW; the real TSO activates what it needs. The 25 MW
  cap (`QEST_ACT_CAP=25`, planner volume-aware) is the realistic bracket until the Baltic dashboard volumes are in
  (research/environment_audit.md). Both brackets are reported.
- Deliberate imbalance positions settle at the imbalance price without BRP penalties; the plans of record do not take them.
- Price impact: DA linear impact `da_slope`; capacity depth saturation `cap_depth`; no impact in balancing energy.

## 4. What the walk-forward changes in the numbers

Walk-forward season (291 days): baseline 37,480,487 uncapped / 30,277,111 at 25 MW (the earlier split-wise numbers were
37,602,146 / 30.3 M: the in-sample Q1-Q2 calibration was worth about 0.3 %). Activation plan +2.7 % / +0.2 %, Brier-gated
+1.9 % / +0.1 %, rank-gated +3.2 % (+- 2.1 %) / +0.3 % (+- 0.4 %), Q3 alone +14.7 % / +3.6 % (research/activation_model.md
s6). The gates themselves are leakage-free: they score the model's past predictions against outcomes public at D-2 over a
trailing 30-day window. Two gate metrics were tried on the same year and the better one is quoted: a mild selection
effect (one binary choice, motivated beforehand by the plan's use of the marginals as a ranking), stated here so the
+3.2 % is read with it. The remaining item is the Q3-tuned knobs, shared by both
plans; walking the tune forward changes the level, not the comparison.

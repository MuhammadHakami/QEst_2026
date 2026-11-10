# Environment audit: our settlement vs the Estonian market (9 Nov 2026)

Why: the 82 % of the gap to hindsight is activation knowledge (research/gap_to_hindsight.md). Before chasing it, the
settlement must activate, clear and pay the way Elering does, or the ceiling itself is an artefact. Each row: what the
market does, what `qest.market` does, the data that closes the gap, where it lives, and whether we have it.

## 1. Balancing energy activation (the 82 %)

| Real market | Our model | Gap and effect | Data / status |
|---|---|---|---|
| mFRR through MARI (Elering since 9 Oct 2024): bids by **T-25**, scheduled activation once per MTU plus direct activations, pay-as-cleared marginal price, the TSO activates only the MW it needs in merit order | an in-the-money bid is activated for the **whole won MW** whenever the 15-min price crosses DA +- margin | our 100 MW is activated far too often; Estonia's activated mFRR volume per MTU is often tens of MW. Inflates both the baseline's and hindsight's mFRR line (36.9 / 43.6 M a season) | `normal_activations_mfrr` (Baltic Transparency Dashboard, BTD): activated MW per MTU, direction, area. **Blocked**: Cloudflare denies our network (403 to urllib, the `baltic` client and a real headless Chrome). Hook is in `market.activated_volumes`: drop the Download-Center CSVs into `data/raw/btd/inbox/` and run `qest.fetch_btd --inbox` |
| aFRR through PICASSO (Elering since 9 Apr 2025): 4-s optimisation, Baltic aFRR demand ~90-120 MW an hour in total, EE a part of it | full activation of the offered MW on every in-the-money 4-s tick (the dataset's PICASSO prices) | a 100 MW aFRR bid would be most of the Baltic demand; shares and energy are overstated | `activations_afrr`, `satisfied_demand_afrr` (BTD). Same block, same hook |
| Full activation time: aFRR 5 min, mFRR 12.5 min; energy of a scheduled activation ramps | **done, opt-in** (`QEST_FAT=1`): mFRR delivers 93.3 % of the MW over its MTU (60 -> 100 % ramp in the first 5 min), aFRR 97 %, in settlement and planner (`market.FAT`); default off because the organisers' scoring has no ramp | a few % of activation energy, sized in gap_to_hindsight s7 | rule text (ACER mFRR IF Annex I), no data needed |
| Energy bid price is free per MTU; the won capacity must be offered | the same, with a tuned margin and the SoC guard | none | — |

## 2. Capacity market (BBCM, live 5 Feb 2025; aFRR added 15 Apr 2025)

| Real market | Our model | Gap and effect | Data / status |
|---|---|---|---|
| Four auctions per MTU block (aFRR up / down, mFRR up / down), D-1 09:00, **marginal pricing**, cap 4,000 EUR/MW/h, cross-zonal exchange with a 50 % CZC limit (Elering, BBCM evaluation report 2025) | we win our bid MW at the forecast price with a fixed depth saturation (`cap_depth`) | demand is finite: Estonia's FRR-up demand is large and mostly imported, mFRR-down suppliers in EE are scarce (good for a battery's DOWN bids); our 100 MW moves these prices | `reserves_demand`, `procured_reserves`, `price_procured_reserves` (BTD). Blocked, same route. The evaluation report gives the mechanics but not per-hour demand |
| TSO resources as demand-reduction: Kiisa plant up to 250 MW mFRR up, Kiisa resistors 40 MW mFRR down (2025) | not modelled | caps the price a BSP can get in EE | report, qualitative |

## 3. Day-ahead, intraday, imbalance

| Real market | Our model | Gap and effect | Data / status |
|---|---|---|---|
| SDAC, 15-min MTUs since 1 Oct 2025, gate 13:00, one price per MTU; 100 MW is ~10 % of EE load | DA bids clear at the realised price with a linear impact `da_slope` | adequate | dataset |
| Continuous intraday (XBID), T-30 per MTU; the only venue where delivery-day information is tradable | **absent** | the structural reason no intraday lever moved money (gap_to_hindsight s4b) | Nord Pool intraday prices are a paid API; no free 15-min EE source found (9 Nov). Needs a Nord Pool account or an organiser file |
| Imbalance: single price per 15-min ISP, (metered - committed) x price, BRP obligations | the same, no BRP penalty for deliberate positions | deliberate imbalance positions look free; they are not in practice | rule, not data: keep `allow_dev` strategies out of the default set |

## 4. Information timing

| Real | Ours | Status |
|---|---|---|
| gate A D-1 09:00: DA and capacity prices to D-1, balancing to D-2 (organisers), weather D-1 00 UTC | enforced in `qest.forecast` / `qest.chronos` | ok |
| T-25 gate sees: system balance (near real time), activation results ~15 min after each MTU (Fingrid practice; Elering's own lag unconfirmed), imbalance price later | `qest.replan` uses lags of 3 / 4 MTUs | ok, assumption documented |

## 5. What to fetch, in order (the Download Center at baltic.transparency-dashboard.eu, CSV, 2026-01-01 to 2026-10-26, UTC)

1. `normal_activations_mfrr`, `activations_afrr` — activation caps (section 1).
2. `reserves_demand`, `procured_reserves`, `price_procured_reserves` — capacity clearing (section 2).
3. `mfrr_bid_volumes`, `volumes_afrr`, `imbalance_volumes_v2`, `direction_of_balancing_v2`, `local_marginal_price_mfrr`.

Name each file `<report>*.csv`, drop it into `data/raw/btd/inbox/`, run `.venv/bin/python -m qest.fetch_btd --inbox`, then
`run.sh evaluate` and the twin. The official alternative is an ENTSO-E Transparency token (register, e-mail
transparency@entsoe.eu "Restful API access", up to 3 working days); `qest.fetch_btd` can gain that path once a token exists.

## 6. Measured effect of activation caps on the ceiling (uniform what-if caps, plans volume-aware; Q4)

| Cap | Baseline | Hindsight | Gap | Activation knowledge | Price knowledge (rest) |
|---|---:|---:|---:|---:|---:|
| none | 1,812,548 | 6,335,393 | 4,522,845 | 87 % | 13 % |
| 50 MW | 1,478,017 | 4,139,778 | 2,661,761 | 65 % | 35 % |
| 25 MW | 1,328,436 | 3,214,004 | 1,885,568 | 58 % | 42 % |

Tested and rejected as a stand-in for the volumes: the dataset's `ee_real_system_balance` is production minus consumption
(mean -347 MW, Estonia's net import, correlation 1.00 with that difference), not the TSO's balancing need; its correlation
with the activation direction is 0.01, so it carries nothing about activated volumes. The dataset holds no activated-volume
information at all; the uniform bracket above is the best the data allow, and the real series must come from outside.

The planner (`qest.optimize.plan_storage`) takes `vol_up` / `vol_dn` per MTU; `qest.market.activated_volumes` supplies them from
the Baltic dashboard files or from `QEST_ACT_CAP`. The real volumes replace the what-if the moment the CSVs are in the inbox.

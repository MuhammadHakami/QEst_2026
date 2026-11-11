# Sim-to-real audit: our environment against the Estonian market as it runs in 2026 (11 Nov 2026)

Purpose: the revenue we maximise should be the revenue a 100 MW / 200 MWh battery and a 20 MW hybrid would have earned in
Estonia in 2026, not the organisers' simplified score. Two settlements are therefore kept side by side: `official` (the
jury's rules, deck v3 slides 10 + 12) and the `QEST_REAL=1` preset below. Extends research/environment_audit.md (9 Nov).
Sources: Elering balancing-service pages, Baltic Balancing Roadmap 2024, BBCM rules proposal and evaluation report 2025,
BBCM procurement optimisation function (Jan 2026), Harmonised principles for Baltic LFC reserve prequalification
(12.06.2025), ENTSO-E balancing report 2026, PICASSO / MARI implementation frameworks. The two Elering / AST PDFs could
not be downloaded from this network (404 / 403); items marked (unverified) rest on search extracts and need the PDF.

## 1. Rule by rule

| Real rule (2026) | In the official score | In `QEST_REAL=1` | Status |
|---|---|---|---|
| **Time**: 15-min MTU everywhere (SDAC since 1 Oct 2025, imbalance settlement, BBCM, MARI); Tallinn delivery days incl. the 92 / 100-MTU DST days | yes | yes | ok |
| **Gate A**: BBCM capacity auction D-1 morning, before SDAC; aFRR / mFRR, up / down, per MTU; min bid 1 MW, whole MW (unverified granularity) | yes, 09:00, whole MW | same | ok |
| BBCM is pay-as-cleared with a Baltic demand (aFRR ~90-120 MW, mFRR up 720-860 / down 490-700 MW) and CZC limits; 100 MW moves the price | every bid accepted at the published price | price falls linearly with our volume (`cap_depth` 110 / 604 / 691 MW) | modelled, depth is an assumption |
| **Gate B**: SDAC 13:00 CET D-1 (12:00 CET gate; 13:00 EET), 0.1 MW steps, one price per MTU | yes | yes; price impact `da_slope` available, 0 by default | ok |
| **Gate C**: balancing energy bids to MARI / PICASSO by T-25; price free within the harmonised limits (+-15,000 EUR/MWh technical limit; a secondary source says +-10,000: unverified); the contracted capacity must be bid | bids at DA +- margin, SoC guard prices out at the cap | same | ok |
| **mFRR (MARI, Elering since 9 Oct 2024)**: scheduled activation once per MTU + direct activation; marginal price; the TSO activates only the volume it needs, in merit order | in-the-money bid fully activated | activated MW capped per direction (`QEST_ACT_CAP`, 25 MW what-if; the real series when data/raw/btd has it) | bracketed, real volumes missing |
| **aFRR (PICASSO, Elering since 9 Apr 2025)**: 4-s optimisation, cross-border marginal price per direction, settled per 15 min; demand-limited | per 4-s tick on the dataset's PICASSO prices, full volume | same ticks, volume capped as above | bracketed |
| **Full activation time**: aFRR 5 min, mFRR 12.5 min | none | ramps: mFRR delivers 93.3 % of the MW-quarter, aFRR 97 % (`QEST_FAT=1`) | modelled |
| **Imbalance**: single price per ISP, (metered - scheduled) x price | yes | yes, plus EUR 150 / MWh on undelivered activated reserve and forfeited capacity payment for that MTU | penalty level is an assumption (BBCM unavailability rules unverified) |
| **Limited energy reservoir**: the provider must deliver the full activation despite reservoir depletion; must file an energy-management strategy | reserve backed for one MTU at both ends | same (`reserve_s` = 900 s) | required duration (unverified): 60 min for mFRR would halve the offer of a 2-h battery |
| **Battery physics**: round-trip efficiency, SoC continuous across days | 100 % efficiency, SoC reset to 50 % every midnight | 95 % each way; **SoC carried over midnight**: the plan for D starts from end(D-2) + planned change over D-1, settlement from the truth | modelled (new) |
| **Grid tariff**: Elering price list from 5 May 2026: 14.50 (330 kV) / 23.67 (110 kV) + 3.73 EUR/MWh, storage on net monthly consumption only | EUR 30 / 20 per imported MWh (slide 10) | Elering's, monthly net | modelled |
| Intraday (XBID) continuous trading to T-30 | absent | absent | gap: no data (Nord Pool intraday is paid) |
| BRP obligations: deliberate imbalance positions are not allowed as a strategy | free | free in the score; the plans of record do not take them | policy |
| TSO-owned reserves (Kiisa 250 MW mFRR up) cap prices | in the realised prices | in the realised prices | ok |

## 2. What cannot be made real from here

1. Activated volumes per MTU (Baltic transparency dashboard, Cloudflare-blocked from this network): the 25 MW cap stays a
   bracket. `data/raw/btd/inbox/` takes the CSVs; `qest.market.activated_volumes` then uses them with no code change.
2. Our own price impact in balancing energy: a 100 MW bid shifts the MARI / PICASSO marginal price; no bid curves.
3. Prequalified duration requirement and the unavailability penalty: the two PDFs above.

## 3. Results under the preset

Section 4 is filled from results/days_*_actfc_real.parquet (scripts/run_real.sh): Chronos-2 + MILP, the activation plan
and the scenario plans, whole year, walk-forward. Knobs are the ones tuned under the official rules (not re-tuned for the
preset); SoC carry-over restarts at the split boundaries (1 Jul, 1 Oct).

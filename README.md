# QEST 2026 · KratTrade: Estonia's 2026 market as a trading environment

Everything lives in this folder: data, environment, strategies, results. The competition dataset (`dataset/`) covers
Estonian delivery days from 1 Jan to 25 Oct 2026. On it we rebuild the year: the real assets at their real
locations, Estonian weather, the market rules from the kick-off, and every trading strategy, settled MTU by MTU.

```
dataset/   competition files (read-only)        slides/    kick-off screenshots
qest/      the package                           data/raw   fetched once (OSM, weather, Elering)
results/   tables per split                      data/derived  built frames
run.sh     setup | fetch | build | evaluate | quantum | all
```

## The split (time series, no look-ahead)

| Split | Delivery days (Tallinn) | Days | Used for |
|---|---|---:|---|
| train | Q1–Q2: 1 Jan – 30 Jun 2026 | 181 | fitting forecasters, activation calibration |
| validation | Q3: 1 Jul – 30 Sep 2026 | 92 | model choice, bid margins, interval calibration |
| test | Q4: 1 – 25 Oct 2026 | 25 | scored once, forecasters refitted on Q1–Q3 |

Every forecast is made at gate A (D-1 09:00 Tallinn) from information published by then: day-ahead and capacity prices
up to D-1, balancing and Elering data up to D-2, weather forecasts issued D-1 00 UTC.

## The assets, located

| Asset | Where | How we know |
|---|---|---|
| Solar park, 42.3 MW (the dataset's) | Pihlaka, Kehra, 59.32 N 25.31 E | The dataset gives no place. Its forecast's cloud pattern was matched against ERA5 irradiance on a 0.25° grid over Estonia: peak r = 0.75 at 59.25–59.5 N, 25.25–25.5 E (`python -m qest.sites`). OSM has two newly mapped solar plants there (ways 1568593429/30). Pihlaka (45 MW, EBRD 2024) is commissioning as the dataset's devices come online (Mar–Apr 2026). |
| Battery, 100 MW / 200 MWh (slide 11) | Hertz 1, Kiisa, 59.24 N 24.70 E | Estonia's real 100 MW / 200 MWh park, in operation 3 Feb 2026, on Elering's Kiisa 330 kV substation (OSM way 34935436) |
| Hybrid: PV 20 MW + battery 20 MW / 40 MWh, 20 MW connection (slide 11) | at the solar park | the toy site, with the park's per-MW profile |
| Wind park, 45 MW | Paldiski tuulepark, 59.37 N 24.08 E | OSM plant + turbines; any Estonian park can be chosen (`qest/sites.py: PORTFOLIO_WIND`) |
| Estonia's wind fleet | 30 plants, 233 turbines (OSM) | one weather file per plant ≥ 5 MW; the fleet's D-1 forecast is a price feature |

Weather: Open-Meteo, keyless. ERA5 hourly is used as "what happened". Previous-day runs give the forecast issued the day
before. Elering's dashboard API gives Estonia's measured and planned production, consumption and solar.

Generation:
- **PV (pvlib):** fitted on train days to the park's own series. The fit is fixed south-facing at 25°, DC/AC 1.32,
  clipped at 42.26 MW. Q3 D-1 error is 3.05 MW MAE (the dataset's own D-1 forecast: 2.58 MW).
- **Wind:** a power curve for each turbine from its OSM rotor and hub. Hub-height wind is sheared from ERA5 100 m with
  an air-density correction, less 8 % for wakes and losses.

## The environment: the order ladder

Each Tallinn delivery day is one episode. A strategy acts at three gates, earliest first:

| Gate | Time | What | Paid |
|---|---|---|---|
| A | D-1 09:00 | reserve capacity (aFRR / mFRR, UP / DOWN), whole MW | capacity price, EUR/MW/h: the money for taking part |
| B | D-1 13:00 | day-ahead position for all 96 MTUs, 0.1 MW | one clearing price per MTU |
| C | T-25 | energy bids for the won capacity (price free, volume ≥ won MW) | mFRR: 15-min price crosses the bid; aFRR: every 4-s PICASSO tick that crosses it |
| R | real time | whatever was not sold ahead | imbalance: the "too late" market |

- **Penalties.** Capacity is an obligation. If an activation is not delivered (an empty battery, a PV shortfall), the
  MTU's capacity payment is forfeited. The undelivered energy pays the imbalance price plus 150 EUR/MWh (the starter's
  shortfall cost). A day-ahead sale that isn't produced is settled as imbalance.
- **Earlier is riskier.** Gate A commits for up to 39 h ahead on the D-1 forecast. Gate B fixes the whole day.
  Real time sells unplanned energy into a saturated market.
- **Saturation.** Our own MW lower the capacity clearing price: linearly, to zero at twice the Baltic need. That need
  is aFRR 110 MW, mFRR UP 604 MW and mFRR DOWN 691 MW (Elering; AST FRR dimensioning 2026). A 100 MW battery is
  about the whole aFRR market. The day-ahead slope is configurable (`Config.da_slope`).
  Nord Pool's aggregated bid curves (the slide's plot) need a Nord Pool login (HTTP 401), so they are not used.
- **Settlement modes.**
  - `official` (default): the organisers' deck v3, slides 10 + 12. Every capacity bid accepted at the published price,
    in-the-money mFRR bids fully activated, aFRR per 4-s tick, deviations at the imbalance price, no extra penalty,
    100 % round-trip efficiency, EUR 30/MWh import tariff Mon–Fri 07–22 Tallinn and EUR 20 otherwise (holidays included).
  - `baltic`: deviations at the published single imbalance price.
  - `starter`: the starter notebook's toy rule, with surplus at 50 % of DA and shortfall at 150 EUR/MWh.
- **Grid tariff** (`baltic`/`starter`; `official` uses slide 10's EUR 30/20). Elering's price list from 5 May 2026: 14.50 EUR/MWh at 330 kV (Kiisa) and 23.67 at 110 kV (hybrid),
  plus the 3.73 EUR/MWh balancing capacity fee. Storage pays them only on its net monthly consumption
  (Electricity Market Act, from 1 Jan 2026).
- **Batteries.** Each day starts at 50 % SoC, 100 % round-trip efficiency under `official` (95 % / 95 % otherwise). Leftover energy is worth
  the day's average DA price. A battery may not charge and discharge in the same MTU, so there are no financial cycles.

## Strategies compared

| Strategy | What it does |
|---|---|
| `idle` | sells nothing ahead: every MWh goes to real time |
| `naive_da` | PV/wind sell their D-1 forecast day-ahead |
| `rule` | + batteries charge in the 8 cheapest forecast MTUs and discharge in the 8 dearest |
| `reserve_rule` | batteries sell their full power in the best-priced forecast capacity product |
| `milp` | joint DA + reserve MILP (HiGHS) on LightGBM quantile forecasts, with SoC-aware gate-C bids |
| `milp_persist` | the same MILP on persistence forecasts |
| `milp_chronos` | the same MILP on Chronos-2 zero-shot forecasts (`qest/chronos.py`) |
| `milp_da` | the MILP without reserves (does taking part pay?) |
| PPO | Stable-Baselines3 agent in the Gymnasium env (`qest/gym_env.py`) choosing the MILP's risk knobs per day |
| `oracle` | the MILP with perfect foresight: an upper bound, not tradable |
| quantum track | the portfolio puzzle (slide 11), retrospective: P1 9 qubits, P2 15, P3 30. Exact, simulated annealing, tabu, random controls, QAOA p=1/3, simulated QA, BF-DCQO, FALQON, PCE (`qest/qubo.py`, `qest/qsota.py`, GPU statevector) |

## Self-contained

Nothing outside this folder is needed.
- **Interpreter:** `.python/` holds the Python itself (uv-managed CPython 3.12.12).
- **Packages:** `.uv-cache/` holds every wheel, so `./run.sh setup` rebuilds `.venv` offline.
- **uv:** `tools/uv` is the uv binary.
- **Data:** `data/raw/` holds every fetched input (OSM, ERA5 and D-1 weather, Elering), so only `./run.sh fetch` needs
  the network, and it skips files that exist.

Checked 7 Nov 2026 on a fresh copy without network, with an empty `$HOME` and no system uv:
`./run.sh setup && ./run.sh build && evaluate --split test`. It reproduced the Q4 numbers to the euro.
If the folder moves, run `./run.sh setup` again: a venv records absolute paths, so setup rebuilds it from the
folder's own Python and cache.

## Run

```bash
./run.sh setup      # .venv from the folder's own Python + wheel cache (offline)
./run.sh fetch      # ~20 min (Open-Meteo, Overpass, Elering)
./run.sh build
./run.sh tune       # Optuna on Q3 -> results/tune_best.json (~1 h)
./run.sh evaluate   # results/summary_<split>_<settlement>.csv
./run.sh rl         # PPO -> results/rl_ppo.csv (~45 min)
./run.sh quantum    # results/qubo_<level>_test_retro.csv (GPU 1)
./run.sh figures    # results/figs/*.png + results/RESULTS.md
./run.sh check      # the day-by-day environment (qest/env.py) end to end
```

## Results (Q4 test, 1–25 Oct 2026, scored once, official scoring)

Official scoring = the organisers' deck v3, slides 10 + 12, on the deck's two sites (battery + hybrid). Full tables (both
splits, all three settlement modes, the quantum track, RL, forecast skill) are in [results/RESULTS.md](results/RESULTS.md).
The deck is published at claude.ai/artifact/EJRo1JSfz4RVBn8Hx3EMJD (charts: `results/figs/`, `python -m qest.figures`).

| Strategy | EUR | USD (1.1165) | % of hindsight |
|---|---:|---:|---:|
| oracle (hindsight, not tradable) | 4.97 M | 5.55 M | 100 |
| **milp_chronos** (Chronos-2 zero-shot forecasts) | **1.65 M** | **1.85 M** | 33.3 |
| milp (LightGBM, Optuna knobs) | 1.54 M | 1.71 M | 30.9 |
| milp + PPO knobs (RL, trained on Q3) | 1.49 M | 1.66 M | 29.9 |
| milp_persist | 1.23 M | 1.37 M | 24.7 |
| milp_da (no reserves) | 0.46 M | 0.51 M | 9.2 |
| reserve_rule | 0.39 M | 0.44 M | 7.8 |
| rule (cheapest/dearest MTUs) | 0.26 M | 0.29 M | 5.2 |
| idle | 0.06 M | 0.07 M | 1.2 |

What the year says:
- **Take part every day.** Adding the D-1 09:00 reserve auction to the same MILP wins on 23 of 25 Q4 days and 83 of 92
  Q3 days; the MILP sizes the offer per MTU, so a weak day gets a small offer, not a skip.
- **Forecasts move the money; solvers don't.** Chronos-2, zero-shot, beats LightGBM on the DA price (Q4 MAE 36.0 vs
  42.5 €/MWh) and all four capacity prices; it is the best tradable strategy. HiGHS solves each day in ~1 s to 0.1 %.
- **Tuning.** Optuna (TPE, 64 trials, 8 processes) lifted Q3 from €6.16 M to €6.81 M with a 40 % haircut on forecast
  aFRR/mFRR DOWN activations (`results/tune_best.json`). On Q4 the untuned MILP scored €1.59 M, slightly more: the
  Q3 optimum did not transfer. PPO learned a near-constant knob set (kappa_dn = 0) and scored €1.49 M on Q4.
- **Quantum (retrospective puzzle, slide 9/12, every Q4 hour).** QAOA, simulated QA, BF-DCQO and FALQON reach the
  classical optimum on all 600 P1 and P2 hours; uniform random with the same 4,096 shots reaches 81.9 % on P2. At P3
  (30 qubits) only PCE runs (30 bits on 6 qubits): 84.5 %, against simulated annealing 46.1 % and tabu 99.9 %. Exact
  enumeration solves every instance in < 0.1 s, so quantum adds no euros here yet.
- **Payback.** One percent better planning on the two sites is worth €241 k (USD 269 k) a year; a €30 k–€300 k quantum
  service breaks even at a 0.12–1.24 % gain over the classical plan, on a problem the MILP cannot finish in time.

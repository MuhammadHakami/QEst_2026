# QEST 2026 · KratTrade: working notes for Claude

The user's competition folder for the QEST quantum hackathon (Tallinn, 11–13 Nov 2026, team KRATTRADE). Everything
stays inside this folder. Read [HISTORY.md](HISTORY.md) first: what was built, how every asset was located, what broke,
the results, and the open items. Runs and results are in [README.md](README.md) and [results/RESULTS.md](results/RESULTS.md). Scoring is `official` (organisers' deck v3, slides 10 + 12) by default.

## User rules (carried over)
- Work directly: no subagents, no workflows.
- Kill processes by PID only. A `pkill` or `pgrep -f` pattern matches the tool's own shell and kills it (exit 144).
- curl, wget and inline HTTP are blocked in Bash. Use Python script files (urllib) or the ctx_execute sandbox.
- No secrets in files or output (`IBM_QUANTUM_TOKEN`, `IQM_TOKEN` go in a gitignored `.env`).
- Never add Co-Authored-By or any Claude attribution. Commit or push only when asked.
- Responses under 500 words. Accuracy over looks: verify every number against the data, and report results honestly
  (including where the quantum method loses).
- Research recent (2026) work with code before building. Keep working through long tasks.
- GPU: use GPU 1 (`CUDA_VISIBLE_DEVICES=1`) if one is needed; GPU 0 hosts vLLM. (7 Nov: vLLM was actually on GPU 1,
  80 GB; our sims fit in its free 16 GB.)

## Layout
- `dataset/`: the competition files, read-only. `slides/`: kick-off screenshots. `docs/background/`: the Makkah-era
  brief, 2026 research, starter + solved starter notebooks.
- `qest/` modules:
  - `paths`, `data`: the 15-min frame, split, aFRR margin table.
  - `fetch_osm`, `fetch_weather`, `fetch_elering`: write `data/raw`.
  - `sites`: asset locations, `locate_solar`.
  - `power`: PV and wind, actual and D-1.
  - `forecast`: gate-A LightGBM quantiles.
  - `market`: rules and settlement.
  - `optimize`: day MILP.
  - `strategies`, `evaluate`, `tune` (Optuna), `qubo` (slide-11 puzzle) + `qsota` (GPU quantum simulators), `env`
    (reset/step), `gym_env` (Gymnasium + PPO), `chronos` (Chronos-2 forecasts), `figures`, `fetch_fx`, `report`.
  - Quantum / hybrid on the Chronos-2 baseline: `qnas` (circuit NAS + PostNAS), `qanneal` (SQA, digitised QA),
    `qmcmc`, `qrc` (reservoir forecast fix), `qrl` (VQC policy), `qday` (QAOA day pattern), `qbenders` (annealing Benders).
  - CUDA-Q (GPU) competitors across the pipeline: `qforecast` (QRC forecaster vs Chronos-2), `qdecide` (QAOA / WS-QAOA /
    PCE / SQA deciders x forecasters, strategies '<solver>@<forecaster>'), `qpart` (Daedalus-style partitioned windows
    for every qubit solver, `p_*` in qdecide; `qforecast --part` for the reservoir). Write-up: research/partitioned_quantum.md. Gap analysis: research/gap_to_hindsight.md (partial oracles `oracle_*`, `milp_c*`, `milp_rp*` via `replan`, `milp_stoch` / `milp_cvar*` via `stoch`, `milp_scen*` via `qscen` (exact joint activation scenarios; `qscen_q` benchmarks the quantum-enhanced MCMC sampler against it) in strategies); environment vs the real market: research/environment_audit.md, Baltic dashboard ingestion `fetch_btd` (blocked from this network by Cloudflare; CSV inbox route); `fetch_nps` (EE/FI/LV/LT day-ahead prices, Elering open API) feeding `chronos` kind 'chronosx'; `fc_tabpfn` (TabPFN v2, models/tabpfn; lost to Chronos-2 on Q3). Activation model: `fetch_tx` (Elering cross-border flows / NTC / planned trade, open API) + `actfc` (LightGBM boosted from the recency-weighted climatology, monthly refits) -> strategies `milp_act*`; `qscen` conditioned on it; `milp_act_gated` / `milp_act_rank` (trailing-skill gates, Brier / Spearman); research/activation_model.md. Whole-year numbers only from the walk-forward mode `QEST_WALK=1` (research/leakage_audit.md). `qdistill`
    (QPU samples distilled into a conditional transformer, IonQ arXiv:2611.08129; `p_nqs*` in qdecide; research/distilled_quantum.md). Real sites: `fetch_ortho`, `detect`, `scene`.
    `assets` (every turbine / solar plant / battery at its place, corrected models, PyWake wakes, close-up photo checks, sim vs Elering)
    and `twin` (the 2026 scenario page: simulation map, 3D parks, results; results/figs/twin; `run.sh twin`), served by `serve` (static + Land Board tile disk cache; `run.sh serve`, port 18977; three.js r182 vendored in qest/vendor). Wake model and
    simulation pattern adapted from ../../product/mawani_general/sage_globe/water_destilation_dt (energy/wind.py); copied in, not linked.
- `research/`: cloned competition / conference code (`third_party/`), papers, `README.md` (what we took from each),
  `methods.csv` + `methods_landscape.ipynb` (methods grouped, with charts).
- `run.sh setup|fetch|build|tune|evaluate|rl|quantum|qnas|hybrid|qsim|figures|check|all`. Self-contained:
  - `.python/` holds the interpreter, `.uv-cache/` the wheels (offline rebuild), `tools/uv` the uv binary, `data/raw/`
    every fetched input.
  - New packages: install with `UV_PYTHON_INSTALL_DIR=$PWD/.python UV_CACHE_DIR=$PWD/.uv-cache tools/uv pip install ...`
    and pin them in `requirements.txt`. Never use the system Python or `~/.local`.
  - After moving the folder, run `./run.sh setup`: venvs record absolute paths.
  - Set `OPENBLAS_NUM_THREADS` for parallel runs.

## Gotchas
- **Gate-A information only** (D-1 09:00 Tallinn): DA and capacity prices to D-1, balancing and Elering to D-2, weather
  forecast issued D-1 00 UTC. Never feed delivery-day values to a non-oracle strategy.
- **Splits are Tallinn delivery days.** Test is Q4 = 1–25 Oct, scored once. Tune only on Q3 (val), then refit on Q1–Q3.
- **Caches.** Forecasts are cached in `results/forecast_*.parquet` and generation in `data/derived/generation.parquet`.
  Delete them after changing `forecast.py` or `power.py`.
- **SoC books.** The MILP's SoC must follow the battery's net power, as settlement does, or plans drift into
  undelivered reserve.
- **Bids.** Energy bids sit at DA ± margin. Static near-DA bids drain the battery (€-20 M penalties on Q3), so keep the
  SoC-aware gate-C policy.
- **Overpass** is flaky: use a bbox query, and `overpass.private.coffee` first.
- **Nord Pool bid curves** return 401 without a login. Elering's API is open.
- **The organisers' deck** is in the folder as `Joint optimisation of a hybrid portfolio slides.pdf` (v3, all 13 slides):
  read it with `pdftotext -layout`. Its rules are `settlement='official'`.
- **Our deck** (16 slides, v13, 9 Nov: walk-forward season numbers, gap, Estonian market, activation model with ranking gate, distilled QPU, expected revenue) is claude.ai/artifact/EJRo1JSfz4RVBn8Hx3EMJD (source: scratchpad deck2/make.py); charts from `qest.deckfigs`. 3D sites: claude.ai/artifact/9hWyuvPBKpLbhn9oXEr7NX (`qest.scene`).
- **CUDA-Q + torch optimizers**: import `torch._dynamo` before CUDA-Q in one process (both bundle LLVM; the other order aborts).
- **CUDA-Q** qubit order: `get_state` qubit 0 = least significant bit, `get_unitary` = most significant (tested).
- **Land Board tiles** (tiles.maaamet.ee): more than ~2 parallel connections gets the whole maaamet.ee domain blocked for a while; `qest.assets` caches tiles in data/raw/tiles and runs 2 threads.
- **Self-contained**: anything taken from outside this folder (code, data, wheels) is copied in; Esri imagery is not redistributable, Maa-amet (CC BY 4.0) is.
- **Ports**: 8765 is taken by another service; serve local previews on a free high port.
- **Quantum scoring** is retrospective (slide 9): a plan counts only if it pays no QUBO penalty (`qubo.feasible`).

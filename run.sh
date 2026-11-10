#!/usr/bin/env bash
# QEST 2026 (KratTrade): the whole pipeline, self-contained in this folder.
#   ./run.sh setup      .venv from the folder's own uv (tools/uv), Python (.python/) and package cache (.uv-cache/): works offline
#   ./run.sh fetch      OSM assets, Estonian weather (ERA5 + D-1 forecasts), Elering system data -> data/raw
#   ./run.sh build      15-min frame, aFRR activation table, PV/wind generation -> data/derived
#   ./run.sh tune       Optuna (TPE) on Q3: MILP bid margins + activation haircut + SoC tolerance -> results/tune_best.json
#   ./run.sh evaluate   every strategy (incl. Chronos-2 forecasts) on Q3 and Q4: official (battery + hybrid), baltic, starter
#   ./run.sh rl         PPO (Stable-Baselines3) in the Gymnasium env, trained on Q3 -> results/rl_ppo.csv
#   ./run.sh quantum    the slide-11 puzzle, retrospective, every Q4 hour: exact, SA, tabu, QAOA, SQA, BF-DCQO, FALQON, PCE
#   ./run.sh qnas       quantum NAS + PostNAS, quantum-enhanced MCMC, quantum annealing on the puzzle (P1-P3)
#   ./run.sh hybrid     Chronos-2 baseline (Optuna) vs quantum / hybrid layers on Q4: QRC, VQC policy, QAOA day pattern, annealing Benders
#   ./run.sh qsim       CUDA-Q (GPU): quantum reservoir forecaster vs Chronos-2, quantum deciders (QAOA, warm-start QAOA,
#                       PCE, SQA) x forecasters on Q4; Land Board orthophotos, asset detection, three.js ground view
#   ./run.sh twin       2026 scenario: every Estonian turbine (PyWake wakes) / solar plant / Hertz 1 at its place, close-up photo checks,
#                       battery dispatch of the baseline + top-2 deciders, vs Elering -> results/figs/twin
#   ./run.sh serve      the twin page on http://localhost:18977/ (a systemd user unit, survives the session; caches Land Board tiles)
#   ./run.sh figures    the deck's charts -> results/figs/
#   ./run.sh all
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
UV=tools/uv; [ -x "$UV" ] || UV=$(command -v uv)
export UV_PYTHON_INSTALL_DIR="$PWD/.python" UV_CACHE_DIR="$PWD/.uv-cache"  # the interpreter and every wheel stay in this folder
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}  # GPU 1 (GPU 0 hosts vLLM): Chronos-2 and the quantum simulators
step() {
  case "$1" in
    setup) # a moved folder breaks the venv's absolute paths: rebuild it from the folder's own Python and cache (offline if need be)
           "$PY" -c "import qest" 2>/dev/null && [ "$(readlink -f "$PY")" = "$(readlink -f .python/cpython-3.12*/bin/python3.12)" ] && { echo "environment ready"; return 0; }
           rm -rf .venv
           $UV python install 3.12 --offline 2>/dev/null || $UV python install 3.12
           $UV venv .venv --python-preference only-managed --python 3.12
           $UV pip install --python "$PY" --offline -r requirements.txt 2>/dev/null || $UV pip install --python "$PY" -r requirements.txt
           "$PY" -c "import qest, numpy, pandas, highspy, lightgbm, pvlib, qiskit, dwave.samplers; print('environment ready:', __import__('sys').executable)" ;;
    fetch) $PY -m qest.fetch_osm; $PY -m qest.fetch_elering; $PY -m qest.fetch_weather ;;
    build) $PY -m qest.data; $PY -c "from qest.data import afrr_margin_table; afrr_margin_table()"; $PY -m qest.power; $PY -m qest.sites ;;
    tune) $PY -m qest.tune --trials 64 --workers 8 | tee results/run_tune_optuna.log ;;
    evaluate) for s in val test; do
                $PY -m qest.evaluate --split $s --settlement official --units battery,hybrid | tee results/run_${s}_official.log
                for m in baltic starter; do $PY -m qest.evaluate --split $s --settlement $m --strategies idle,naive_da,rule,reserve_rule,milp,milp_persist,oracle | tee results/run_${s}_${m}.log; done; done ;;
    rl) $PY -m qest.gym_env --steps 2000 | tee results/run_rl_ppo.log ;;
    quantum) for L in P1 P2 P3; do $PY -m qest.qubo --level $L --mode retro | tee results/run_qubo_${L}_retro.log; done ;;
    figures) $PY -m qest.figures; $PY -m qest.report > /dev/null ;;
    qnas) for L in P1 P2 P3; do $PY -m qest.qnas --level $L --trials 60 --post 40 | tee results/run_qnas_$L.log; done
          $PY -m qest.qmcmc | tee results/run_qmcmc.log; $PY -m qest.qanneal | tee results/run_qanneal.log ;;
    hybrid) $PY -m qest.tune --strategy milp_chronos --trials 64 --workers 8 | tee results/run_tune_optuna_milp_chronos.log
          $PY -m qest.qrc | tee results/run_qrc.log; $PY -m qest.qrl | tee results/run_rl_qpolicy.log
          $PY -m qest.qday --split test | tee results/run_test_hybrid.log
          for m in sqa sa exact; do $PY -m qest.qbenders --split test --master $m | tee results/run_benders_$m.log; done ;;
    qsim) CUDA_VISIBLE_DEVICES=1 $PY -m qest.qforecast | tee results/run_qforecast.log
          for fc in chronos qrc ens; do CUDA_VISIBLE_DEVICES=1 $PY -m qest.qdecide --split test --fc $fc | tee results/run_qdecide_test_$fc.log; done
          $PY -m qest.fetch_ortho; $PY -m qest.detect; $PY -m qest.scene ;;
    twin) for sp in train val; do CUDA_VISIBLE_DEVICES=1 $PY -m qest.qdecide --split $sp --fc chronos --solvers samples,qaoa | tee results/run_qdecide_${sp}_chronos.log; done
          CUDA_VISIBLE_DEVICES=1 $PY -m qest.twin --traces | tee results/run_twin_traces.log
          $PY -m qest.fetch_osm; $PY -m qest.fetch_ortho; $PY -m qest.assets | tee results/run_assets.log; $PY -m qest.twin ;;
    serve) systemctl --user stop qest-twin 2>/dev/null || true
          systemd-run --user --unit=qest-twin --working-directory="$PWD" "$PWD/$PY" -m qest.serve 18977 --warm  # static page + Land Board tile cache (data/raw/tiles)
          echo "KratTrade Twin: http://localhost:18977/  (stop: systemctl --user stop qest-twin)" ;;
    check) $PY -m qest.env ;;
    all) for s in setup fetch build tune evaluate rl quantum qnas hybrid qsim twin figures; do step $s; done ;;
    *) sed -n '2,18p' "$0"; exit 1 ;;
  esac
}
mkdir -p results
step "${1:-all}"

#!/bin/bash
cd "$(dirname "$0")/.."; export PYTHONPATH=$PWD QEST_WALK=1 OPENBLAS_NUM_THREADS=10
.venv/bin/python scripts/act_eval.py train _scenfj milp_scenf_jr8_t > results/run_scenfj_train.log 2>&1 &
QEST_ACT_CAP=25 .venv/bin/python scripts/act_eval.py train _scenfj_cap25 milp_scenf_jr8_t > results/run_scenfj_train_cap25.log 2>&1 &
QEST_REAL=1 QEST_FAT=1 QEST_ACT_CAP=25 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/act_eval.py train _real_j milp_scenf_jr8_t > results/run_real_j_train.log 2>&1 &
for sp in val test; do QEST_REAL=1 QEST_FAT=1 QEST_ACT_CAP=25 OPENBLAS_NUM_THREADS=3 .venv/bin/python scripts/act_eval.py $sp _real_j milp_scenf_jr8_t > results/run_real_j_$sp.log 2>&1 & done
wait; touch results/run_year_best.done

#!/bin/bash
# sim2real preset over the whole year, walk-forward (research/sim2real.md)
cd "$(dirname "$0")/.."; export PYTHONPATH=$PWD QEST_WALK=1 QEST_REAL=1 QEST_FAT=1 QEST_ACT_CAP=${QEST_ACT_CAP:-25} OPENBLAS_NUM_THREADS=6
for sp in train val test; do .venv/bin/python scripts/act_eval.py $sp _real milp_chronos,milp_act_rank,milp_scenf4,milp_scenf > results/run_real_$sp.log 2>&1 & done
wait; touch results/run_real.done

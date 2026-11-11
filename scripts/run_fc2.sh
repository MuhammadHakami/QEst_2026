#!/bin/bash
cd "$(dirname "$0")/.."; export PYTHONPATH=$PWD QEST_WALK=1 OPENBLAS_NUM_THREADS=4 QEST_FC=tsens
for sp in val test; do
  .venv/bin/python scripts/act_eval.py $sp _fc_tsens milp_chronos,milp_act_rank,milp_scenf > results/run_fc_tsens_$sp.log 2>&1 &
  QEST_ACT_CAP=25 .venv/bin/python scripts/act_eval.py $sp _fc_tsens_cap25 milp_chronos,milp_act_rank,milp_scenf > results/run_fc_tsens_${sp}_cap25.log 2>&1 &
done; wait; touch results/run_fc2.done

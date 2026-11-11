#!/bin/bash
cd "$(dirname "$0")/.."; export PYTHONPATH=$PWD QEST_WALK=1 OPENBLAS_NUM_THREADS=7
for sp in val test; do
  .venv/bin/python scripts/act_eval.py $sp _scenfj milp_scenf_jr8_t,milp_scenf_j8_t > results/run_scenfj_$sp.log 2>&1 &
  QEST_ACT_CAP=25 .venv/bin/python scripts/act_eval.py $sp _scenfj_cap25 milp_scenf_jr8_t,milp_scenf_j8_t > results/run_scenfj_${sp}_cap25.log 2>&1 &
done; wait; touch results/run_joint.done

#!/bin/bash
# plan family on another foundation model's forecasts (QEST_FC), walk-forward, both settlements; detached so a session restart does not kill it
cd "$(dirname "$0")/.."; export PYTHONPATH=$PWD QEST_WALK=1 OPENBLAS_NUM_THREADS=4
for fc in timestral tsens; do
  QEST_FC=$fc .venv/bin/python scripts/act_eval.py val _fc_$fc milp_chronos,milp_act_rank,milp_scenf > results/run_fc_${fc}_val.log 2>&1 &
  QEST_FC=$fc QEST_ACT_CAP=25 .venv/bin/python scripts/act_eval.py val _fc_${fc}_cap25 milp_chronos,milp_act_rank,milp_scenf > results/run_fc_${fc}_val_cap25.log 2>&1 &
  QEST_FC=$fc QEST_ACT_CAP=25 .venv/bin/python scripts/act_eval.py test _fc_${fc}_cap25 milp_chronos,milp_act_rank,milp_scenf > results/run_fc_${fc}_test_cap25.log 2>&1 &
done
for sp in val test; do CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m qest.qgen --bench --split $sp > results/run_qgen_$sp.log 2>&1; done &
wait; touch results/run_fc.done

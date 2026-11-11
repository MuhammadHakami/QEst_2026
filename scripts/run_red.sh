#!/bin/bash
cd "$(dirname "$0")/.."; export PYTHONPATH=$PWD QEST_WALK=1 OPENBLAS_NUM_THREADS=5
for sp in val test; do
  .venv/bin/python scripts/act_eval.py $sp _scenfr milp_scenf_r4_t,milp_scenf_r8_t > results/run_scenfr_$sp.log 2>&1 &
  QEST_ACT_CAP=25 .venv/bin/python scripts/act_eval.py $sp _scenfr_cap25 milp_scenf_r4_t,milp_scenf_r8_t > results/run_scenfr_${sp}_cap25.log 2>&1 &
done
for sp in test val; do CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/act_eval.py $sp _scenfrq milp_scenf_rq4_t > results/run_scenfrq_$sp.log 2>&1; done &
for sp in test val; do CUDA_VISIBLE_DEVICES=0 QEST_ACT_CAP=25 .venv/bin/python scripts/act_eval.py $sp _scenfrq_cap25 milp_scenf_rq4_t > results/run_scenfrq_${sp}_cap25.log 2>&1; done &
wait; touch results/run_red.done

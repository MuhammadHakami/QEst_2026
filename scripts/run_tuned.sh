#!/bin/bash
cd "$(dirname "$0")/.."; export PYTHONPATH=$PWD QEST_WALK=1 OPENBLAS_NUM_THREADS=4
for sp in val test; do
  .venv/bin/python scripts/act_eval.py $sp _scenft milp_scenf4_t,milp_scenf_ls4_t > results/run_scenft_$sp.log 2>&1 &
  QEST_ACT_CAP=25 .venv/bin/python scripts/act_eval.py $sp _scenft_cap25 milp_scenf4_t,milp_scenf_ls4_t,milp_scenf4 > results/run_scenft_${sp}_cap25.log 2>&1 &
done
for sp in test val; do CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/act_eval.py $sp _scenft_q milp_scenf_qs4_t > results/run_scenft_q_$sp.log 2>&1; done &
for sp in test val; do CUDA_VISIBLE_DEVICES=0 QEST_ACT_CAP=25 .venv/bin/python scripts/act_eval.py $sp _scenft_q_cap25 milp_scenf_qs4_t > results/run_scenft_q_${sp}_cap25.log 2>&1; done &
wait; touch results/run_tuned.done

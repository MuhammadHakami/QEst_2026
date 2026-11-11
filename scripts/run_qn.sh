#!/bin/bash
# the Q2-tuned plan on quantum-MCMC scenarios from the NAS proposal (results/nas_sampler_best.json, searched on June windows)
cd "$(dirname "$0")/.."; export PYTHONPATH=$PWD QEST_WALK=1 OPENBLAS_NUM_THREADS=6
for sp in test val; do CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/act_eval.py $sp _scenft_qn milp_scenf_qs4_t > results/run_scenft_qn_$sp.log 2>&1; done &
for sp in test val; do CUDA_VISIBLE_DEVICES=0 QEST_ACT_CAP=25 .venv/bin/python scripts/act_eval.py $sp _scenft_qn_cap25 milp_scenf_qs4_t > results/run_scenft_qn_${sp}_cap25.log 2>&1; done &
wait; touch results/run_qn.done

#!/bin/bash
# 2022 holdout (second addendum in experiments/PLAN_weekend_1106.md). GPU 0, 3 threads.
cd ~/Workspace_Hyundo/breaking-judge; source scripts/env.sh
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
for F in vq2_ft vq2; do for s in 0 1 2; do
  nice -n 10 $MLAVL_PY ours/wdsf_holdout.py --name hold_${F}_Technique_s$s --feat $F --seed $s > logs/wdsf/hold_${F}_Technique_s$s.log 2>&1 &
done; done
wait
nice -n 10 $MLAVL_PY ours/wdsf_holdout.py --eval vq2_ft vq2
echo "[$(date)] done"

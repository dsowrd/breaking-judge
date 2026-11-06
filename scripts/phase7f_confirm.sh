#!/bin/bash
# Confirmation addendum in experiments/PLAN_weekend_1106.md. GPU 0, 3 threads, 4 at a time.
cd ~/Workspace_Hyundo/breaking-judge; source scripts/env.sh
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
run() { [ -f logs/wdsf/$1/summary.json ] || nice -n 10 $MLAVL_PY ours/train_wdsf_pair.py --name $1 --feat $2 --target $3 --seed $4 > logs/wdsf/$1.log 2>&1; }
for F in vq2_ft vq2 vq; do
  for s in 1 2; do for t in Technique Execution; do run ${F}_${t}_s$s $F $t $s & done; done
  wait
done
E="nice -n 10 $MLAVL_PY ours/eval_pair.py"
for F in vq2_ft vq2 vq; do for t in Technique Execution; do
  $E logs/wdsf/${F}_${t}_s0 logs/wdsf/${F}_${t}_s1 logs/wdsf/${F}_${t}_s2
done; done
echo "[$(date)] done"

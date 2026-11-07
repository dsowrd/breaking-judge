#!/bin/bash
# Ninth addendum (experiments/PLAN_weekend_1106.md): remove the test fold's athletes from training (purge) vs as many random rounds (size),
# Swin + AST (pamfn) and pose codes + audio (vq2_ft), Technique, seeds 0-2. GPU 0, 3 threads per run, 6 at a time.
cd ~/Workspace_Hyundo/breaking-judge; source scripts/env.sh
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
P="nice -n 10 $MLAVL_PY"
L=logs/wdsf
n=0
gate() { n=$((n+1)); if [ $((n%6)) = 0 ]; then wait; fi; }
cv() { name=$1; shift; [ -f $L/$name/summary.json ] || { $P ours/train_wdsf_pair.py --name $name "$@" > $L/$name.log 2>&1 & gate; }; }
for m in purge size; do for f in vq2_ft pamfn; do for s in 0 1 2; do
  cv ath${m}_${f}_Technique_s$s --athletes $m --feat $f --target Technique --seed $s
done; done; done
wait
echo "[$(date)] done"

#!/bin/bash
# Fifth addendum, part owed: MLAVL with zero audio on the Chengdu 2026 youth holdout. vq2_ft, Technique and Execution, seeds 0-2. GPU 0, 3 threads.
cd ~/Workspace_Hyundo/breaking-judge; source scripts/env.sh
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
P="nice -n 10 $MLAVL_PY"
for t in Technique Execution; do
  for s in 0 1 2; do
    $P ours/wdsf_holdout.py --no_audio --holdout y26 --name hold26_na_vq2_ft_${t}_s$s --feat vq2_ft --target $t --seed $s > logs/wdsf/hold26_na_vq2_ft_${t}_s$s.log 2>&1 &
  done
  wait
  $P ours/wdsf_holdout.py --holdout y26 --prefix hold26_na --eval vq2_ft --target $t
done
echo "[$(date)] done"

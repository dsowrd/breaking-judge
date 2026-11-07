#!/bin/bash
# Fifth addendum: MLAVL with zero audio on vq2_ft. Cross-validation, 2022 holdout. Chengdu when its features exist. GPU 0, 3 threads.
cd ~/Workspace_Hyundo/breaking-judge; source scripts/env.sh
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
P="nice -n 10 $MLAVL_PY"
for s in 0 1 2; do
  $P ours/wdsf_holdout.py --no_audio --name hold_na_vq2_ft_Technique_s$s --feat vq2_ft --seed $s > logs/wdsf/hold_na_vq2_ft_Technique_s$s.log 2>&1 &
done
wait
for s in 0 1 2; do for t in Technique Execution; do
  [ -f logs/wdsf/na_vq2_ft_${t}_s$s/summary.json ] || $P ours/train_wdsf_pair.py --no_audio --name na_vq2_ft_${t}_s$s --feat vq2_ft --target $t --seed $s > logs/wdsf/na_vq2_ft_${t}_s$s.log 2>&1 &
done; done
wait
echo "[$(date)] done"

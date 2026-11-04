#!/bin/bash
# Phase 6b: RG pose detection on GPUs 1 and 3 once the Phase 6 Fis-V runs (scripts/phase6_fisv.sh) have freed them.
cd ~/Workspace_Hyundo/breaking-judge
while pgrep -f "bash scripts/phase6_fisv.sh" > /dev/null; do sleep 120; done
echo "start RG detection $(date)"
for s in 0 1 2 3 4 5 6 7; do
  g=$([ $s -lt 4 ] && echo 1 || echo 3)
  CUDA_VISIBLE_DEVICES=$g ~/anaconda3/envs/mlavl/bin/python -I tools/aqa/aqa_pose.py detect --set rg --shard $s --nshards 8 > logs/aqa_pose/rg_$s.log 2>&1 &
done
wait
echo "RG detection done $(date)"

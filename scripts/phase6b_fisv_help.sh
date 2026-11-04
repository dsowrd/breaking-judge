#!/bin/bash
# Phase 6b: after RG detection, GPUs 1 and 3 help finish the Fis-V detection from the other end of the list.
cd ~/Workspace_Hyundo/breaking-judge
while tmux has-session -t pose_rg 2>/dev/null; do sleep 60; done
echo "start Fis-V helpers $(date)"
for s in 0 1 2 3 4 5; do
  g=$([ $s -lt 3 ] && echo 1 || echo 3)
  CUDA_VISIBLE_DEVICES=$g ~/anaconda3/envs/mlavl/bin/python -I tools/aqa/aqa_pose.py detect --set fisv --reverse --shard $s --nshards 6 > logs/aqa_pose/fisv_help_$s.log 2>&1 &
done
wait
echo "Fis-V helpers done $(date)"

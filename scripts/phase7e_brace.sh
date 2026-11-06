#!/bin/bash
# Phase 7e: YOLO11x-pose and Keypoint R-CNN on the 26,676 BRACE manual frames, GPU 0, 3 shards. Frame ids are 1-based (offset -1,
# chosen on one video: strictest PCK 0.200 vs 0.126 at -2 and 0.122 at 0, and the manual joints sit on the dancer by eye).
cd ~/Workspace_Hyundo/breaking-judge; source scripts/env.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0
for k in 0 1 2; do
  nice -n 10 $MLAVL_PY -I tools/battle/brace_pose_eval.py run --offset -1 --shard $k --nshards 3 > logs/stdout/brace_pose_s$k.log 2>&1 &
done
wait
nice -n 10 $MLAVL_PY -I tools/battle/brace_pose_eval.py score > logs/stdout/brace_pose_score.log 2>&1
echo DONE >> logs/stdout/brace_pose_score.log

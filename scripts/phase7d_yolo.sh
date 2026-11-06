#!/bin/bash
# Phase 7d: YOLO11x-pose + ByteTrack on all WDSF turns, GPU 0, 3 shards (about 600k frames at 10 fps).
cd ~/Workspace_Hyundo/breaking-judge; source scripts/env.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for k in 0 1 2; do
  tmux new-window -d -t p7d -n s$k "CUDA_VISIBLE_DEVICES=0 nice -n 10 $MLAVL_PY -I tools/battle/wdsf_track_pose.py track --shard $k --nshards 3 > logs/stdout/yolo_track_s$k.log 2>&1"
done

#!/bin/bash
# Plan C: RG with YOLO11x-pose: detection (3 shards), athlete tracking, codebook, MLAVL --extra pose, 12 runs, 6 at a time. GPU 0 only.
cd ~/Workspace_Hyundo/breaking-judge; source scripts/env.sh
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 AQA_POSE_TAG=_y
P="nice -n 10 $MLAVL_PY -I tools/aqa"
for k in 0 1 2; do $P/aqa_pose.py detect --set rg --yolo --shard $k --nshards 3 > logs/stdout/rgy_det_s$k.log 2>&1 & done
wait
$P/aqa_pose.py track --set rg > logs/stdout/rgy_track.log 2>&1
$P/aqa_choreo.py train --set rg > logs/stdout/rgy_vq.log 2>&1
$P/aqa_choreo.py encode --set rg >> logs/stdout/rgy_vq.log 2>&1
for s in 0 1 2; do for a in Ball Clubs Hoop Ribbon; do echo "$s $a"; done; done | \
  xargs -P 6 -L 1 bash -c 'SEED=$0 TAG=yolo_pose bash scripts/run_rg.sh $1 0 --extra pose'
echo "[$(date)] done"

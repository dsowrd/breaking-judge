#!/bin/bash
# Plan B after B1 passed: re-track WDSF with the BRACE fine-tuned YOLO, same selection rule, then the same measures,
# dancer signature (main setting L8 K64) and pairwise MLAVL (A1 / A2). GPU 0. Old pose files stay as they are.
cd ~/Workspace_Hyundo/breaking-judge; source scripts/env.sh
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
export WDSF_YOLO_WEIGHTS=/data/mlavl/ckpt_live/yolo_brace/ft1/weights/best.pt WDSF_TRACK_TAG=_ft
P="nice -n 10 $MLAVL_PY -I tools/battle"
for k in 0 1 2; do $P/wdsf_track_pose.py track --shard $k --nshards 3 > logs/stdout/ft_track_s$k.log 2>&1 & done
wait
$P/wdsf_track_pose.py select > logs/stdout/ft_select.log 2>&1
export WDSF_POSE_TAG=2_ft
$P/choreo_memory.py train > logs/stdout/ft_choreo.log 2>&1
$P/choreo_memory.py encode >> logs/stdout/ft_choreo.log 2>&1
$P/criteria_validity.py measures > logs/stdout/ft_validity.log 2>&1
$P/criteria_validity.py validity >> logs/stdout/ft_validity.log 2>&1
$P/move_units.py 8 64 > logs/stdout/ft_move.log 2>&1
bash scripts/phase7f_pair.sh vq2_ft > logs/stdout/ft_pair.log 2>&1
echo "[$(date)] done" >> logs/stdout/ft_pair.log

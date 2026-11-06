#!/bin/bash
# Plan B: frames + labels, fine-tune, then the test year 2020 with the fine-tuned model. GPU 0.
cd ~/Workspace_Hyundo/breaking-judge; source scripts/env.sh
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
P="nice -n 10 $MLAVL_PY -I tools/battle"
for k in 0 1 2; do $P/brace_pose_train.py prepare --shard $k --nshards 3 > logs/stdout/brace_prep_s$k.log 2>&1 & done
wait
$P/brace_pose_train.py train > logs/stdout/brace_train.log 2>&1
W=/data/mlavl/ckpt_live/yolo_brace/ft1/weights/best.pt
V=$(awk -F, '$3==2020{print $1}' /data/mlavl/datasets/brace/repo/videos_info.csv | paste -sd,)
$P/brace_pose_eval.py run --offset -1 --weights $W --videos $V --tag _ft > logs/stdout/brace_ft_eval.log 2>&1
$P/brace_pose_eval.py score --tag _ft > logs/stdout/brace_ft_score.log 2>&1
echo "[$(date)] done" >> logs/stdout/brace_ft_score.log

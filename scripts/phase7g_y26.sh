#!/bin/bash
# Chengdu 2026 youth holdout (third addendum in experiments/PLAN_weekend_1106.md): download, features, turns, pose on both models,
# codes from the saved codebooks, measures, MLAVL training. GPU 0. Nothing in the main WDSF folders is written.
cd ~/Workspace_Hyundo/breaking-judge; source scripts/env.sh
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
Y=/data/mlavl/datasets/wdsf_y26 V=/data/mlavl/videos/wdsf_y26 Z=/data/mlavl/features/WDSF_Y26 W=/data/mlavl/features/WDSF
mkdir -p $V $Z
HEIGHT=720 OUT=$V bash tools/battle/download_brace_videos.sh $(cat $Y/video_ids.txt)
P="nice -n 10 $MLAVL_PY -I tools/battle"
$P/extract_features.py --videos $V --out $Z
$P/extract_features_pamfn.py extract --set wdsf_y26 --tm_dim 64 --audio_sr 32000 --fbank_sr 32000
export WDSF_F=$Z WDSF_D=$Y WDSF_V=$V
$P/wdsf_turn_detector.py segment
for f in choreo2_upper choreo2_lower choreo2_ft_upper choreo2_ft_lower; do ln -sf $W/$f.pt $Z/$f.pt; done
$P/criteria_validity.py beats
# fine-tuned pose
WDSF_YOLO_WEIGHTS=/data/mlavl/ckpt_live/yolo_brace/ft1/weights/best.pt WDSF_TRACK_TAG=_ft $P/wdsf_track_pose.py track
WDSF_TRACK_TAG=_ft $P/wdsf_track_pose.py select
WDSF_POSE_TAG=2_ft $P/choreo_memory.py encode
WDSF_POSE_TAG=2_ft $P/criteria_validity.py measures
# COCO pose
$P/wdsf_track_pose.py track
$P/wdsf_track_pose.py select
WDSF_POSE_TAG=2 $P/choreo_memory.py encode
WDSF_POSE_TAG=2 $P/criteria_validity.py measures
unset WDSF_F WDSF_D WDSF_V
run() { nice -n 10 $MLAVL_PY ours/wdsf_holdout.py --holdout y26 --name hold26_$1_$2_s$3 --feat $1 --target $2 --seed $3 > logs/wdsf/hold26_$1_$2_s$3.log 2>&1; }
for F in vq2_ft vq2; do for s in 0 1 2; do for t in Technique Execution; do run $F $t $s & done; done; wait; done
nice -n 10 $MLAVL_PY -I tools/battle/holdout_measures.py y26
for t in Technique Execution; do nice -n 10 $MLAVL_PY ours/wdsf_holdout.py --holdout y26 --target $t --eval vq2_ft vq2; done
echo "[$(date)] done"

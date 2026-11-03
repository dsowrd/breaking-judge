#!/usr/bin/env bash
# Diagnostic: hand-labelled battles with held-out detector turns (tools/battle/wdsf_turns_cv.py), same pose MLAVL.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
$MLAVL_PY -I tools/battle/wdsf_turns_cv.py
export WDSF_TURNS=$MLAVL_DATA/datasets/wdsf/wdsf_turns_handcv.csv
run() { $MLAVL_PY ours/train_wdsf_pair.py --name $1 --feat pose "${@:2}" > logs/wdsf/$1.log 2>&1; }
run handcv_Execution_s0 --target Execution --seed 0 &
run handcv_total_s0 --target total --seed 0 &
run handcv_Technique_s0 --target Technique --seed 0 &
wait
echo "[$(date)] done"

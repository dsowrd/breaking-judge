#!/bin/bash
# Phase 7d after track selection: codebook on the new pose, then the Phase 7 / 7b measures with WDSF_POSE_TAG=2. GPU 0.
# The old results keep their files; first the old signature AUC is recomputed with the same code for comparison.
cd ~/Workspace_Hyundo/breaking-judge; source scripts/env.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0
set -x
nice -n 10 $MLAVL_PY -I tools/battle/move_units.py 8 64 > logs/stdout/p7d_old_L8K64.log 2>&1
export WDSF_POSE_TAG=2
nice -n 10 $MLAVL_PY -I tools/battle/choreo_memory.py train
nice -n 10 $MLAVL_PY -I tools/battle/choreo_memory.py encode
nice -n 10 $MLAVL_PY -I tools/battle/criteria_validity.py measures
nice -n 10 $MLAVL_PY -I tools/battle/criteria_validity.py validity
nice -n 10 $MLAVL_PY -I tools/battle/move_units.py
echo PHASE7D_DONE

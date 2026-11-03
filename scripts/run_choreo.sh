#!/bin/bash
# Phase 6: Bailando-style choreographic memory (tools/battle/choreo_memory.py), GPU 0 only
set -e
cd ~/Workspace_Hyundo/breaking-judge
PY=~/anaconda3/envs/mlavl/bin/python
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -I tools/battle/choreo_memory.py extract --workers 4
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 $PY -I tools/battle/choreo_memory.py train
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 $PY -I tools/battle/choreo_memory.py encode

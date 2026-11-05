#!/bin/bash
# Phase 6c (2026-11-05): best RG result with Bailando-derived inputs (--extra, ours/extra_inputs.py), 3 seeds each.
# Starts when the Phase 6b RG ablations (tmux p6c_rgA / p6c_rgB) and the beat features (tmux beat) are done.
#   A: pose+beat (main MLAVL + Bailando), pose+posecont, flow (reference)
#   B: beat only, pose+beat+flow (best effort)
cd ~/Workspace_Hyundo/breaking-judge
while tmux has-session -t p6c_rgA 2>/dev/null || tmux has-session -t p6c_rgB 2>/dev/null || tmux has-session -t beat 2>/dev/null; do sleep 300; done
n=$(ls /data/mlavl/features/RG/beat | wc -l); echo "[$(date)] start; RG beat files $n"
bash scripts/rg_sweep.sh p6d_rgA x_pose_beat "--extra pose,beat" x_pose_posecont "--extra pose,posecont" x_flow "--extra flow"
bash scripts/rg_sweep.sh p6d_rgB x_beat "--extra beat" x_pose_beat_flow "--extra pose,beat,flow"

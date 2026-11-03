#!/usr/bin/env bash
# Phase 6 on Fis-V: MLAVL + choreographic memory (graph / add), paper settings, seeds 0-2.
# Seed 0 (4 runs) starts at once on GPU 3; seeds 1-2 start when the RG sweep (tmux session p6_rg) has ended.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
CK="--choreo_ckpt ./ckpt/choreo_fisv.pt"
go() { SEED=$1 TAG=choreo_$2 bash scripts/run_fisv.sh $3 $4 --choreo $2 $CK > /dev/null 2>&1; }
for m in graph add; do for sc in TES PCS; do go 0 $m $sc 3 & done; done
wait
while tmux has-session -t p6_rg 2>/dev/null; do sleep 300; done
for m in graph add; do for sc in TES PCS; do go 1 $m $sc 1 & go 2 $m $sc 3 & done; done
wait
echo "[$(date)] done"

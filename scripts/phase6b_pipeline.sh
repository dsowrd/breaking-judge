#!/bin/bash
# Phase 6b pipeline (MLAVL + pose-based choreographic memory on the paper datasets), runs unattended:
#   wait for pose detection (tmux pose_fisv / pose_rg / pose_fisv_help) -> track -> VQ-VAE train + encode (per dataset)
#   -> MLAVL --choreo {graph,add} --choreo_src pose, paper settings, seeds 0-2:
#      RG via rg_sweep.sh (session p6b_rg, GPUs 0/1/3 = seeds 0/1/2), Fis-V 12 runs spread over GPUs 0/1/3.
# Log: logs/aqa_pose/pipeline.log
cd ~/Workspace_Hyundo/breaking-judge
source scripts/env.sh
PY=$MLAVL_PY
for s in pose_fisv pose_rg pose_fisv_help; do while tmux has-session -t $s 2>/dev/null; do sleep 120; done; done
nf=$(ls /data/mlavl/features/FISV/pose_det/*.npz | wc -l); nr=$(ls /data/mlavl/features/RG/pose_det/*.npz | wc -l)
echo "[$(date)] detection done: fisv $nf/500 rg $nr/1000"
if [ "$nf" -lt 500 ] || [ "$nr" -lt 1000 ]; then          # stragglers (e.g. a killed shard): finish them on GPU 0
  CUDA_VISIBLE_DEVICES=0 $PY -I tools/aqa/aqa_pose.py detect --set fisv
  CUDA_VISIBLE_DEVICES=0 $PY -I tools/aqa/aqa_pose.py detect --set rg
fi
rm -f /data/mlavl/features/{FISV,RG}/pose_det/*.busy
for set in fisv rg; do $PY -I tools/aqa/aqa_pose.py track --set $set; done
CUDA_VISIBLE_DEVICES=0 $PY -I tools/aqa/aqa_choreo.py train --set fisv > logs/aqa_pose/vq_fisv.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY -I tools/aqa/aqa_choreo.py train --set rg > logs/aqa_pose/vq_rg.log 2>&1 &
wait
tail -n 3 logs/aqa_pose/vq_fisv.log logs/aqa_pose/vq_rg.log
for set in fisv rg; do CUDA_VISIBLE_DEVICES=0 $PY -I tools/aqa/aqa_choreo.py encode --set $set || exit 1; done
echo "[$(date)] memory ready; launching MLAVL runs"
bash scripts/rg_sweep.sh p6b_rg choreo_pose_graph "--choreo graph --choreo_src pose" choreo_pose_add "--choreo add --choreo_src pose"
declare -A G=([0]=0 [1]=1 [2]=3)
for seed in 0 1 2; do for m in graph add; do for sc in TES PCS; do
  SEED=$seed TAG=choreo_pose_$m bash scripts/run_fisv.sh $sc ${G[$seed]} --choreo $m --choreo_src pose > /dev/null 2>&1 &
done; done; done
wait
while tmux has-session -t p6b_rg 2>/dev/null; do sleep 300; done
echo "[$(date)] all Phase 6b runs done"

#!/bin/bash
# Plan A (experiments/PLAN_weekend_1106.md): pairwise MLAVL on the Bailando pose codes of the new pose. GPU 0, 3 threads, 4 at a time.
cd ~/Workspace_Hyundo/breaking-judge; source scripts/env.sh
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
F=${1:-vq2}
for t in total Technique Vocabulary Originality Execution Musicality; do nice -n 10 $MLAVL_PY ours/probe_diff.py --feat $F --target $t --k 8 --video_only > /dev/null; done
run() { [ -f logs/wdsf/$1/summary.json ] || nice -n 10 $MLAVL_PY ours/train_wdsf_pair.py --name $1 --feat $F "${@:2}" > logs/wdsf/$1.log 2>&1; }
for s in 0 1 2; do run ${F}_total_s$s --target total --seed $s & done
run ${F}_Technique_s0 --target Technique --seed 0 &
wait
for t in Execution Vocabulary Originality Musicality; do run ${F}_${t}_s0 --target $t --seed 0 & done
wait
E="nice -n 10 $MLAVL_PY ours/eval_pair.py"
for t in total Technique Execution Vocabulary Originality Musicality; do $E logs/wdsf/probe_${F}_vonly_k8_$t; done
$E logs/wdsf/${F}_total_s0 logs/wdsf/${F}_total_s1 logs/wdsf/${F}_total_s2
for t in Technique Execution Vocabulary Originality Musicality; do $E logs/wdsf/${F}_${t}_s0; done
echo "[$(date)] done"

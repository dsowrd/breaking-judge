#!/usr/bin/env bash
# Phase 6 (MLAVL + Bailando): after scripts/run_choreo.sh (choreographic memory + vq/ features), compare on the same
# folds and strict evaluation: linear probe on vq, vocabulary statistics, token sequence judge, pairwise MLAVL on the
# code vectors (vq) and on code vectors + hand pose statistics (vqpose). GPU 0 only, 4 threads per run.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
while [ "$(ls $MLAVL_DATA/features/WDSF/vq 2>/dev/null | grep -c 'npz$')" -lt 313 ] || pgrep -f "choreo_memory.py" >/dev/null; do sleep 60; done
T="total Technique Vocabulary Originality Execution Musicality"
echo "[$(date)] probes + vocabulary stats"
for t in $T; do
  for feat in vq vqpose; do $MLAVL_PY ours/probe_diff.py --feat $feat --target $t --k 8 --video_only > /dev/null; done
  $MLAVL_PY ours/train_vocab_pair.py --model stats --target $t --name vstats_$t > logs/wdsf/vstats_$t.log 2>&1
done
echo "[$(date)] token sequence judge"
seq_run() { [ -f logs/wdsf/$1/predictions.csv ] || $MLAVL_PY ours/train_vocab_pair.py --model seq --name $1 "${@:2}" > logs/wdsf/$1.log 2>&1; }
for t in $T; do
  for s in 0 1 2; do seq_run vseq_${t}_s$s --target $t --seed $s & done
  wait
done
echo "[$(date)] pairwise MLAVL on vq / vqpose"
run() { [ -f logs/wdsf/$1/summary.json ] || $MLAVL_PY ours/train_wdsf_pair.py --name $1 "${@:2}" > logs/wdsf/$1.log 2>&1; }
for s in 0 1 2; do run vq_total_s$s --feat vq --target total --seed $s & done
run vq_Technique_s0 --feat vq --target Technique --seed 0 &
wait
run vq_Execution_s0 --feat vq --target Execution --seed 0 &
run vq_Vocabulary_s0 --feat vq --target Vocabulary --seed 0 &
run vq_Originality_s0 --feat vq --target Originality --seed 0 &
run vq_Musicality_s0 --feat vq --target Musicality --seed 0 &
wait
for s in 0 1 2; do run vqpose_total_s$s --feat vqpose --target total --seed $s & done
run vqpose_Execution_s0 --feat vqpose --target Execution --seed 0 &
wait
echo "[$(date)] strict evaluation"
E="$MLAVL_PY ours/eval_pair.py"
for t in $T; do
  echo "=== $t"
  $E logs/wdsf/probe_vq_vonly_k8_$t; $E logs/wdsf/probe_vqpose_vonly_k8_$t; $E logs/wdsf/vstats_$t
  $E logs/wdsf/vseq_${t}_s0 logs/wdsf/vseq_${t}_s1 logs/wdsf/vseq_${t}_s2
done
$E logs/wdsf/vq_total_s0 logs/wdsf/vq_total_s1 logs/wdsf/vq_total_s2
for t in Technique Execution Vocabulary Originality Musicality; do $E logs/wdsf/vq_${t}_s0; done
$E logs/wdsf/vqpose_total_s0 logs/wdsf/vqpose_total_s1 logs/wdsf/vqpose_total_s2
$E logs/wdsf/vqpose_Execution_s0
echo "[$(date)] done"

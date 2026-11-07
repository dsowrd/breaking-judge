#!/bin/bash
# Eighth addendum: Video Swin + pose codes (the RG combination) on breaking, Swin-only reference seeds, and the benchmark's missing cells. GPU 0, 3 threads per run, 6 at a time.
cd ~/Workspace_Hyundo/breaking-judge; source scripts/env.sh
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
P="nice -n 10 $MLAVL_PY"
L=logs/wdsf
n=0
gate() { n=$((n+1)); if [ $((n%6)) = 0 ]; then wait; fi; }
cv() { name=$1; shift; [ -f $L/$name/summary.json ] || { $P ours/train_wdsf_pair.py --name $name "$@" > $L/$name.log 2>&1 & gate; }; }
# the question
for t in Technique Execution total; do for s in 0 1 2; do cv swin_vq2_ft_${t}_s$s --feat swin_vq2_ft --target $t --seed $s; done; done
for t in Technique Execution; do for s in 1 2; do cv pairv_${t}_s$s --target $t --seed $s; done; done
wait
for h in 2022 y26; do
  x=$([ $h = 2022 ] && echo hold || echo hold26)
  for s in 0 1 2; do
    $P ours/wdsf_holdout.py --holdout $h --name ${x}_swin_vq2_ft_Technique_s$s --feat swin_vq2_ft --target Technique --seed $s > $L/${x}_swin_vq2_ft_Technique_s$s.log 2>&1 &
  done
  wait
  $P ours/wdsf_holdout.py --holdout $h --eval swin_vq2_ft vq2_ft --target Technique
done
# benchmark cells
for t in Vocabulary Originality Musicality; do for s in 1 2; do cv vq2_ft_${t}_s$s --feat vq2_ft --target $t --seed $s; done; done
for t in total Vocabulary Originality Musicality; do for s in 0 1 2; do cv na_vq2_ft_${t}_s$s --no_audio --feat vq2_ft --target $t --seed $s; done; done
wait
echo "[$(date)] done"

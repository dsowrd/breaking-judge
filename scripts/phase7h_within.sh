#!/bin/bash
# Seventh addendum: MLAVL trained on differences between rounds of one battle. vq2_ft, seeds 0-2. GPU 0, 3 threads per run, 6 runs at a time.
cd ~/Workspace_Hyundo/breaking-judge; source scripts/env.sh
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
P="nice -n 10 $MLAVL_PY"
L=logs/wdsf
for f in pamfn vq2_ft; do for t in Technique Execution; do
  $P ours/probe_diff.py --feat $f --target $t --k 8 --video_only --within
done; done
for t in Technique Execution; do $P ours/probe_diff.py --feat pamfn --target $t --k 8 --audio_only --within; done
# cross-validation
for s in 0 1 2; do
  $P ours/train_wdsf_pair.py --within only --no_audio --name wt_na_vq2_ft_Technique_s$s --feat vq2_ft --target Technique --seed $s > $L/wt_na_vq2_ft_Technique_s$s.log 2>&1 &
  $P ours/train_wdsf_pair.py --within only --name wt_vq2_ft_Technique_s$s --feat vq2_ft --target Technique --seed $s > $L/wt_vq2_ft_Technique_s$s.log 2>&1 &
done
wait
for s in 0 1 2; do
  $P ours/train_wdsf_pair.py --within only --no_audio --name wt_na_vq2_ft_Execution_s$s --feat vq2_ft --target Execution --seed $s > $L/wt_na_vq2_ft_Execution_s$s.log 2>&1 &
  $P ours/train_wdsf_pair.py --within only --name wt_vq2_ft_Execution_s$s --feat vq2_ft --target Execution --seed $s > $L/wt_vq2_ft_Execution_s$s.log 2>&1 &
done
wait
for s in 0 1 2; do
  $P ours/train_wdsf_pair.py --within mix --no_audio --name wm_na_vq2_ft_Technique_s$s --feat vq2_ft --target Technique --seed $s > $L/wm_na_vq2_ft_Technique_s$s.log 2>&1 &
done
wait
# holdouts, Technique
for h in 2022 y26; do
  x=$([ $h = 2022 ] && echo hold || echo hold26)
  for s in 0 1 2; do
    $P ours/wdsf_holdout.py --within only --no_audio --holdout $h --name ${x}_wtna_vq2_ft_Technique_s$s --feat vq2_ft --target Technique --seed $s > $L/${x}_wtna_vq2_ft_Technique_s$s.log 2>&1 &
    $P ours/wdsf_holdout.py --within only --holdout $h --name ${x}_wt_vq2_ft_Technique_s$s --feat vq2_ft --target Technique --seed $s > $L/${x}_wt_vq2_ft_Technique_s$s.log 2>&1 &
  done
  wait
done
echo "[$(date)] done"

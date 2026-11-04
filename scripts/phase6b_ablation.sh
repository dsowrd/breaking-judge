#!/bin/bash
# Phase 6b ablations on Fis-V (paper settings, seeds 0-2, TES + PCS), started when scripts/phase6b_pipeline.sh ends.
#  A. our module, on the better of --choreo graph / add (by Fis-V avg rho of the pose runs):
#     upper / lower (one body half), shuffle (clip order within the video), swap (codes of another video),
#     posecont (continuous pose statistics, no VQ; add)
#  B. MLAVL components, no choreo: no_audio, no_text_graph
# 42 runs, round-robin over GPUs 0/1/3, at most 7 at a time per GPU. Log: logs/aqa_pose/ablation.log
cd ~/Workspace_Hyundo/breaking-judge
source scripts/env.sh
while tmux has-session -t p6b_pipe 2>/dev/null; do sleep 300; done
avg() { $MLAVL_PY scripts/summarize_runs.py --seeds logs/stdout/fisv_*_choreo_pose_$1_*.log 2>/dev/null \
        | grep "Fis-V avg" | sed -E 's/[^*]*\*\*[^*]*\*\*[^*]*\*\*([0-9.]+)\*\*.*/\1/'; }
G=$(avg graph); A=$(avg add)
MODE=$($MLAVL_PY -c "import sys; g,a=(float(x or 0) for x in sys.argv[1:]); print('graph' if g>=a else 'add')" "$G" "$A")
echo "[$(date)] pose runs: graph $G add $A -> ablations on $MODE"
JOBS=()
for seed in 0 1 2; do for sc in TES PCS; do
  for ab in upper lower shuffle swap; do JOBS+=("$seed $sc abl_${ab} --choreo $MODE --choreo_src pose --choreo_ablate $ab"); done
  JOBS+=("$seed $sc abl_posecont --choreo add --choreo_src posecont")
  for ab in no_audio no_text_graph; do JOBS+=("$seed $sc abl_${ab} --ablate $ab"); done
done; done
GPUS=(0 1 3)
for k in 0 1 2; do
  g=${GPUS[$k]}
  for ((i = k; i < ${#JOBS[@]}; i += 3)); do echo "$g ${JOBS[$i]}"; done \
    | xargs -P 7 -L 1 bash -c 'g=$0 seed=$1 sc=$2 tag=$3; shift 3; SEED=$seed TAG=$tag bash scripts/run_fisv.sh $sc $g "$@" > /dev/null 2>&1' &
done
wait
echo "[$(date)] ablations done"

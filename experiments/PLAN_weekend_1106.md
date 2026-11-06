# Weekend plan, fixed before the runs (2026-11-06 evening)

Goal (정현도): research results and, for the course, the MLAVL + Bailando combination; both if possible.
Rule: hypotheses, main settings and stop rules below are fixed now. Every run listed is reported, also when it fails.
Results on seeds, folds and settings not listed here are exploratory and are labelled so. GPU 0 only.

## A. Does cleaner pose help the combined model on breaking? (course + research)
Pairwise MLAVL with the Bailando pose codes from the new pose (`--feat vq2`), inner-validation epochs, same folds and strict
evaluation as the old pose (`--feat vq`, Phase 6): total s0-2, each criterion s0, plus the linear probe on the codes.
- A1: round accuracy above the second-dancer rule (one-sided p < 0.05, battle bootstrap) on at least one of total / Technique / Execution.
- A2: holds on rounds with unseen athletes (Phase 5c: appearance inputs dropped to 37-41% there).

## B. Does a pose model fine-tuned on breaking fix power moves? (research)
One configuration: YOLO11x-pose on BRACE manual keypoints; test year 2020, validation 2018, train 2011-2017 (no year overlap).
Other people in the frame get pseudo labels from the pretrained model (score >= 0.5). Backbone frozen, 40 epochs at most, 8 h limit.
- B1: on 2020, PCK@0.1 for power moves (no detection = 0) improves by at least 0.05 over the pretrained model, video bootstrap CI above 0.
- If B1 holds: re-track WDSF (3.5 h), same selection rule, and repeat: dancer signature AUC (goal >= 0.7), the same pre-specified
  measures, main setting L8 K64, A1 / A2. New frames for human grading (정현도 decides when).
- If B1 fails: no re-tracking; pose stays as it is.

## C. Does a better pose model raise the RG result? (course)
RG with YOLO11x-pose (`AQA_POSE_TAG=_y`), same athlete tracking rule, codebook and `--extra pose`, 4 apparatus x 3 seeds.
- C1: vs reproduction 0.775 and vs Keypoint R-CNN pose add 0.798, paired t over 12 runs.
- Fis-V only if C1 beats 0.798 with p < 0.05. The midterm deck changes only then, with an independent fact check, by Monday night.

## Stop rules
No hyperparameter sweep on any of these. No new measures chosen by looking at judge scores. The submitted midterm numbers stay unless
C1 passes. Nothing is deleted. /data free space stays above 200 GB.

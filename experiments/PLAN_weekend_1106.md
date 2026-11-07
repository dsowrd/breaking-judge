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

## Addendum, fixed before these runs (2026-11-07 07:45, after seeing the single-seed `vq2_ft` results)
`vq2_ft` Technique s0 passed A1 on one seed (0.584, rule 0.523, p = 0.027), one of three targets, no correction. To confirm:
Technique and Execution, seeds 1 and 2, on `vq`, `vq2`, `vq2_ft`, same settings (`scripts/phase7f_confirm.sh`).
- Primary: `vq2_ft` Technique 3-seed ensemble above the second-dancer rule, one-sided p < 0.05, all rounds. Fails if not.
- Secondary: `vq2_ft` minus `vq2` on the same rounds (paired battle bootstrap); Execution; unseen athletes. Reported either way.
No other target, seed or setting is added on this question.

## Second addendum, fixed before any prediction on these rounds (2026-11-07): 2022 holdout
Plan A/B and the confirmation used only 5-criteria rounds for Technique. Three 2022 events (Kitakyushu, Manchester, Seoul, B-Boys and
B-Girls) use six sliders that include Technique, and their 47 matched battles (about 113 rounds with turns) were never used to train or
evaluate a Technique model or a criterion measure. Only the label distribution has been looked at. The older `total` runs saw these
rounds with their total score, a different target.

Train on all 5-criteria Technique rounds, epoch chosen by the same inner validation event, seeds 0, 1, 2, on `vq2_ft` and `vq2`; predict
the holdout (`scripts/phase7g_holdout.sh`, `ours/wdsf_holdout.py`). z-scoring from training rounds only.
- Primary: `vq2_ft` 3-seed ensemble above the second-dancer rule on the holdout, one-sided p < .05, battle bootstrap. Fails if not.
- Secondary, reported either way: above 50%; `vq2_ft` − `vq2` (paired); athletes absent from training.
- Pose-error test (designed now, on rounds that did not suggest it). Blue-minus-red measures vs Technique-specific, where specific =
  Technique minus the mean of the six sliders, Spearman, battle bootstrap: inverted (`_ft`) expected positive. Joint rank regression
  old + new: energy and spin coefficient expected larger on the old pose, inverted larger on the new one.
- Power: at about 113 rounds the accuracy CI is about +/- 9 points and a rho's about +/- 0.18, so a failure does not refute the 5-criteria
  result; a pass is strong because these rounds played no part in choosing anything. Six-slider Technique may not mean exactly the same
  as five-slider Technique (other sliders absorb some of it).
No other target, seed, feature or setting is added on this question.

Added before reading any holdout prediction (2026-11-07, runs still going), after an exploratory check on the 5-criteria rounds:
- Beyond dancer strength: strength = blue minus red mean Technique margin in 5-criteria rounds (all training rounds, so nothing from
  2022). Joint rank regression of the holdout margin on the `vq2_ft` ensemble prediction and strength, battle bootstrap. The prediction's
  coefficient is expected positive. Secondary, reported either way. Strength's own accuracy is reported as the identity reference.

## Third addendum, fixed before these videos were downloaded (2026-11-07): Chengdu 2026 youth holdout
WDSF World Championships Chengdu 2026, Youth, B-Boys and B-Girls (September 2026): 32 battles, 78 rounds, five sliders, matched 32 / 32
by exact names (`tools/battle/wdsf_y26_prep.py`). Newest event in the data, in no training set, and mostly dancers the model has not
seen. Own folders (`videos/wdsf_y26`, `features/WDSF_Y26`, `datasets/wdsf_y26`); nothing refit: saved turn detector (no hand labels),
saved YOLO, saved codebooks (`vq2_ft`, `vq2`), z-score from training rounds. Same training as the 2022 holdout, seeds 0, 1, 2.
- Primary: `vq2_ft` Technique 3-seed ensemble above the second-dancer rule, one-sided p < .05, battle bootstrap. Fails if not.
- Secondary, reported either way: above 50%; `vq2_ft` Execution above the rule; `vq2_ft` − `vq2` (paired); athletes absent from training;
  inverted (`_ft`) vs Technique-specific expected positive; joint old + new pose as in the second addendum.
- Power: at about 75 rounds the accuracy CI is about +/- 11 points, so a failure does not refute the earlier result. Youth may differ from
  adults (the model trained on adults only) and the segmentation here has no hand check.
No other target, seed, feature or setting is added on this question.

## Fourth addendum, fixed before computing it (2026-11-07): within-battle test, no identity possible
Both holdouts and cross-validation cannot fully exclude a model that recognises dancers (2022: 109 of 111 rounds have an athlete in
training). Within one battle the two dancers are the same and the blue rule is constant, so only the performance can change between
rounds. For every pair of rounds of one battle: does the model order them as the judges' margins do (blue minus red, target's mean)?
Ties dropped, battle bootstrap, chance 50%.
- Main: `vq2_ft` Technique, out-of-fold predictions on the 637 five-slider rounds, one-sided p < .05 above 50%.
- Also reported either way: the same on the 2022 holdout and Chengdu 2026, Execution, `vq2` and `vq`, the five simple measures.
No training, no new setting. A fail would mean this model gives no evidence of reading the performance apart from who dances.

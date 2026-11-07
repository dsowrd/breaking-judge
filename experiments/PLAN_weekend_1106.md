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

## Fifth addendum, fixed before these runs (2026-11-07): what carries the within-battle signal?
Fourth-addendum main test passed (`vq2_ft` Technique 0.562, p = .006; 2022 holdout 0.609, p = .022). But a linear probe on AST audio alone
reaches 0.583 (Technique) and 0.590 (Execution) on the same test, above video alone (0.549, 0.515). The audio of a live battle holds music,
crowd, MC and possibly the reaction when votes are shown, and the MLAVL input includes that audio. So the pass may not be about movement.
1. **MLAVL without audio** (zero audio input, `--no_audio`), `vq2_ft`, same settings, seeds 0, 1, 2, Technique and Execution, plus the 2022
   holdout and Chengdu. Main: Technique within-battle above 50%, one-sided p < .05, cross-validation. This decides whether pose alone
   passes. Secondary: overall vs rule; the holdouts; difference from the run with audio (paired).
2. **Where in time is the audio signal?** Linear probe (same k = 8, same folds) on AST statistics in windows relative to each turn: 15 s before
   the start, first half, second half, 15 s after the end. Same rounds. Expectation if votes leak: after-the-end dominates. Expectation if the
   crowd reacts to the dancing: first and second half carry it too. Before-the-start is a negative control for the same round's dancing.
3. **Loudness.** Blue minus red RMS (dB) in the same windows vs margin, no fitting.
Nothing here changes a model already reported. No hyperparameter is added.

## Sixth addendum, fixed before computing it (2026-11-07): do judges weigh the end of a turn?
Audio carries the within-battle signal in the second half of a turn and not in the first. Is that only the crowd building up, or does the
movement in the second half also matter more? Linear probe (k = 8, same folds, turn pooled mean and std), video features only, in the first
half, second half and last quarter of each turn: Video Swin and `vq2_ft`. Technique and Execution, overall and within-battle.
Hypothesis: second half above first half, paired battle bootstrap on within-battle accuracy, one-sided p < .05 on Technique, Video Swin.
Reported either way. No model is chosen from this.

## Seventh addendum, fixed before these runs (2026-11-07): train on the within-battle signal itself
Between-battle differences hold 90% of the margin variance, and dancer strength alone reaches 0.70-0.80, so the usual loss mostly rewards
knowing who is the stronger dancer. A model that only recognises dancers predicts the same value for every round of a battle, so a loss on
the difference between two rounds of one battle gives it nothing to gain. New training mode `--within only` in `ours/train_wdsf_pair.py`
and `ours/wdsf_holdout.py`: each sample is two rounds of one battle, loss 10 * MSE((d_a - d_b), clip((m_a - m_b) / S_w, -1, 1)), S_w = 3 std
of the within-battle margin differences of all rounds, MLAVL auxiliary terms unchanged, same number of turns per update and updates per
epoch, epoch chosen on within-battle accuracy of the same inner validation event. `--within mix` adds the usual per-round MSE. Everything
else as before: `vq2_ft`, same folds, seeds 0, 1, 2 (`scripts/phase7h_within.sh`).
- Main: `--within only --no_audio`, Technique, cross-validation, 3-seed ensemble, within-battle accuracy above 50%, one-sided p < .05,
  battle bootstrap. Usual training scored 0.543 (p = .055) on this test.
- Secondary, reported either way: paired difference from usual training (same pairs, with and without audio); with audio; Execution; `mix`
  on the main setting; round accuracy against the rule (expected lower, because identity is removed); correlation of the prediction with
  dancer strength (expected below the usual training's +.175); Technique on the 2022 and Chengdu holdouts, with and without audio.
- Linear counterpart, reported either way: `ours/probe_diff.py --within` (features and margins centred by battle before the fit) on Swin
  only, AST only, `vq2_ft` codes only, k = 8, same folds, Technique and Execution.
No other target, seed, feature, weight or setting is added on this question.

## Eighth addendum, fixed before these runs (2026-11-07): the RG combination on breaking, and a complete benchmark
Course goal: the model that improved RG (Video Swin + Bailando pose codes, `--extra pose`, codes added to the visual tokens) has not been run on breaking with the
fine-tuned pose. Breaking runs so far use codes alone (`vq2_ft`) or Swin alone (`pairv`). New `--feat swin_vq2_ft`: Swin as in the paper, z-scored codes through the same `--extra pose` projection,
everything else as the usual pairwise training (`scripts/phase7i_combo.sh`): seeds 0, 1, 2, inner-validation epoch, same folds.
- Main: Technique, cross-validation, 3-seed ensemble, round accuracy above the blue rule, one-sided p < .05, battle bootstrap.
- Secondary, reported either way: paired vs `vq2_ft` and vs Swin-only MLAVL (three new seeds if not already present at the same settings); within-battle; rounds with an unseen athlete; Execution and total; Technique on the 2022 and
  Chengdu holdouts; correlation with dancer strength. Expected risk: Swin brings identity back (Phase 5c: unseen athletes 37-41%).
Benchmark completion, no question attached: `vq2_ft` seeds 1, 2 on Vocabulary, Originality, Musicality; zero-audio `vq2_ft` seeds 0, 1, 2 on total and those three. Filled into `experiments/benchmark_wdsf.csv` as they are, no selection.
No other target, seed or setting is added on this question.

## Ninth addendum, fixed before these runs (2026-11-07 evening): does the video model lean on who the dancer is?
The eighth addendum (and Phase 5c) suggest Swin input learns athlete identity while Bailando pose codes do not: on rounds whose athletes are absent from training,
Technique Swin .45, Swin + codes .48, codes .60 (50 rounds, exploratory). Test this directly with the training set instead of a small subset.
`ours/train_wdsf_pair.py --athletes purge`: per fold, drop every training round in which either athlete also appears in the test fold. Training shrinks to 127-290 rounds
(from 477-566), so the control `--athletes size` drops the same number of training rounds at random (seeded by fold). Test rounds are unchanged (all 637 Technique rounds).
Runs (`scripts/phase7j_purge.sh`): Technique, `--feat vq2_ft` (codes + audio) and `--feat pamfn` (Swin + AST, = `pairv`), purge and size, seeds 0, 1, 2; 12 runs, usual settings and folds, GPU 0.
- Main: identity reliance = round accuracy(size) - accuracy(purge), 3-seed ensembles on the same rounds. Prediction: reliance(Swin) - reliance(codes) > 0, one-sided p < .05, battle bootstrap.
- Secondary, reported either way: each of the four ensembles vs the blue rule; within-battle accuracy of each; reliance of each model on its own; Swin purge vs codes purge paired.
- No other target, seed or setting is added on this question. If the main test fails, the identity reading of the eighth addendum stays exploratory.

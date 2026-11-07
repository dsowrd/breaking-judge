# Experiment log

One line per run. Fill in every column; if a value is unknown write `?` and fix it later.
Numbers that differ from the paper are recorded as they are, with a guess at the cause in *Notes*.

Columns: Date · Commit · Dataset/score · Key args · GPU · Wall time · Max VRAM · best ρ (epoch) · best MSE (epoch) · Notes

## Paper targets (MLAVL, CVPR 2025)

| Data | ρ | MSE |
| --- | --- | --- |
| RG | Ball 0.826 / Clubs 0.829 / Hoop 0.871 / Ribbon 0.866 / Avg 0.849 | 5.57 / 4.20 / 4.11 / 3.99 / Avg 4.47 |
| FS1000 | TES 0.92 / PCS 0.89 / SS 0.90 / TR 0.90 / PE 0.88 / CO 0.89 / IN 0.88 / Avg 0.90 | 64.89 / 6.39 / 0.23 / 0.24 / 0.50 / 0.25 / 0.26 / Avg 10.39 |
| Fis-V | TES 0.766 / PCS 0.863 / Avg 0.823 | 19.44 / 7.17 / 13.31 |

Average ρ: report both arithmetic mean and Fisher-z mean.

## Phase 0 · setup

| Date | Commit | Dataset/score | Key args | GPU | Time | Max VRAM | best ρ (ep) | best MSE (ep) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Phase 1 · RG reproduction

| Date | Commit | Dataset/score | Key args | GPU | Time | Max VRAM | best ρ (ep) | best MSE (ep) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-11 | d092717 | RG / Ball | `scripts/run_rg.sh Ball 0` (paper: lr 1e-4, dropout 0.3, α 1.0, α_ce 0.1, clip 68, in_dim 1024, 620 ep) | 0 (4 runs concurrently) | 1:32:54 | ~1.8 GB (RSS 1.6 GB) | 0.754 (129), MSE at that ckpt 7.01 | 5.42 (145), ρ at that ckpt 0.753 | paper 0.826 / 5.57. Train ρ 0.99 from ep ~300: overfits, test ρ oscillates 0.66–0.72 after ep 200 |
| 2026-10-11 | d092717 | RG / Clubs | `scripts/run_rg.sh Clubs 0` (340 ep) | 0 (4 concurrent) | 0:52:58 | ~1.8 GB (RSS 1.6 GB) | 0.785 (226), MSE 5.29 | 4.47 (225), ρ 0.779 | paper 0.829 / 4.20 |
| 2026-10-11 | d092717 | RG / Hoop | `scripts/run_rg.sh Hoop 0` (450 ep) | 0 (4 concurrent) | 1:07:27 | ~1.8 GB (RSS 1.7 GB) | 0.780 (163), MSE 6.39 | 4.68 (288), ρ 0.766 | paper 0.871 / 4.11. Largest gap (−0.09) |
| 2026-10-11 | d092717 | RG / Ribbon | `scripts/run_rg.sh Ribbon 0` (710 ep) | 0 (4 concurrent) | 1:41:04 | ~1.8 GB (RSS 1.6 GB) | 0.792 (355), MSE 5.39 | 5.36 (354), ρ 0.789 | paper 0.866 / 3.99 |
| 2026-10-11 | d092717 | **RG / Avg** | paper settings, seed 0, 1 run each | 0 | 4 runs in 1:41 wall | | **0.778** (arithmetic) / **0.778** (Fisher z) | **4.98** (mean of best-MSE ckpts); 6.02 at the best-ρ ckpts | paper 0.849 / 4.47 → ρ −0.071, MSE +0.51. Logs `logs/stdout/rg_*_1011_1548.log`, ckpts backed up to `/data/mlavl/ckpt_backup/`. Suspected causes below |
| 2026-10-11 | 394eed5 | RG / Ball,Clubs,Hoop,Ribbon | as above + `--lr 1e-2` (README lr) | 0 (4 concurrent) | 17:31–17:50, **stopped at 72–80 of 340–710 epochs** | ~1.8 GB | partial: Ball 0.255 / Clubs 0.385 / Hoop 0.412 / Ribbon 0.401 (test ρ oscillates around 0, often negative) | – | README lr 1e-2 does not train with the paper's other settings: at the same epoch lr 1e-4 reaches 0.69–0.78. Stopped early by 정현도 to free GPU 0. Conclusion: use the paper lr; README value is wrong or assumed a different setup. Logs `logs/stdout/rg_*_1011_1731.log` |
| 2026-10-11 | 1c0f97c | RG / Ball s1 | paper settings, `SEED=1` | 3 (4 concurrent) | 1:51:08 | ~1.8 GB | 0.768 (37), MSE 8.39 | 5.43 (87), ρ 0.763 | |
| 2026-10-11 | 2a2a346 | RG / Ball s2 | `SEED=2` | 3 | 1:28:42 | ~1.8 GB | 0.751 (42), MSE 12.02 | 5.72 (185), ρ 0.746 | |
| 2026-10-11 | 1c0f97c | RG / Clubs s1 | `SEED=1` | 3 | 1:00:17 | ~1.8 GB | 0.758 (128), MSE 5.42 | 5.09 (156), ρ 0.735 | |
| 2026-10-11 | 2a2a346 | RG / Clubs s2 | `SEED=2` | 3 | 0:59:28 | ~1.8 GB | 0.704 (229), MSE 6.51 | 5.69 (179), ρ 0.698 | worst RG run |
| 2026-10-11 | 1c0f97c | RG / Hoop s1 | `SEED=1` | 3 | 1:18:03 | ~1.8 GB | 0.824 (45), MSE 9.44 | 4.29 (151), ρ 0.803 | |
| 2026-10-11 | 2a2a346 | RG / Hoop s2 | `SEED=2` | 3 | 1:11:19 | ~1.8 GB | 0.755 (305), MSE 5.30 | 5.22 (307), ρ 0.747 | |
| 2026-10-11 | 1c0f97c | RG / Ribbon s1 | `SEED=1` | 3 | 1:59:03 | ~1.8 GB | 0.789 (255), MSE 5.99 | 5.44 (222), ρ 0.779 | |
| 2026-10-11 | 2a2a346 | RG / Ribbon s2 | `SEED=2` | 3 | 1:34:40 | ~1.8 GB | 0.842 (561), MSE 4.66 | 4.36 (380), ρ 0.839 | best RG run |

**RG, 3 seeds (0, 1, 2), paper settings.** `scripts/summarize_runs.py --seeds logs/stdout/rg_*_1011_1548.log logs/stdout/rg_*_s[12]_*.log`

| Apparatus | paper ρ | ours ρ, mean ± std | paper MSE | ours best MSE, mean ± std | ours MSE at best-ρ ckpt |
| --- | --- | --- | --- | --- | --- |
| Ball | 0.826 | 0.758 ± 0.009 | 5.57 | 5.52 ± 0.17 | 9.14 ± 2.59 |
| Clubs | 0.829 | 0.749 ± 0.041 | 4.20 | 5.08 ± 0.61 | 5.74 ± 0.67 |
| Hoop | 0.871 | 0.786 ± 0.035 | 4.11 | 4.73 ± 0.47 | 7.04 ± 2.14 |
| Ribbon | 0.866 | 0.808 ± 0.030 | 3.99 | 5.05 ± 0.60 | 5.35 ± 0.67 |
| **Avg** | **0.849** | **0.775** (Fisher z 0.776) | **4.47** | **5.10** | 6.82 |

- Seed spread is large: ±0.03–0.04 ρ for Clubs/Hoop/Ribbon, single runs range 0.704 (Clubs s2) to 0.842 (Ribbon s2). The paper's per-apparatus numbers are within about 1–2 std of our best single runs for Hoop/Ribbon but not for Ball/Clubs.
- Seed 0 alone gave 0.778; the 3-seed mean 0.775 is essentially the same, so the −0.07 gap to the paper is not a seed artefact.
- Best-ρ epochs are often very early (37–45 for Ball s1/s2, Hoop s1) with a high MSE at that checkpoint (8–12): ρ and MSE optima do not coincide; the paper probably reports ρ and MSE from separate checkpoints too (the code tracks both), but note the early peaks are noisy.
- Per-run ckpts (`_best`, `_best_mse`) backed up to `/data/mlavl/ckpt_backup/`.

Suspected causes of the RG gap (to check in order):

1. ~~**Features differ from the authors'.**~~ **Ruled out (2026-11-05):** the upstream README says the RG / Fis-V features are the PAMFN release, the upstream `RGDataset` hard-codes `{App}_rgb_VST.npy` (1024-d), and the PAMFN release has no 768-d visual feature (VST 1024, I3D flow 1024, UMT (T, 2, 1024); 768-d are only the AST / MAST audio). The README's `--in_dim 768` is the FS1000 example copied into an RG command. The authors used the same features as we do. Original note: MLAVL's README uses `--in_dim 768` for RG, but the only public RG features (PAMFN) are 1024-d Video Swin. The authors probably extracted their own 768-d RGB features (and possibly their own audio features); we cannot match them. This alone could explain most of the gap.
2. **Over-fitting on 200 training videos.** Train ρ reaches 0.99 while test ρ plateaus at ep 150–350; the best epochs (129–355) are well before the paper's epoch counts (340–710). Paper numbers are probably best-epoch numbers as well (the code saves `_best.pkl`), so this is not a reporting difference, but a stronger regulariser (dropout 0.5, weight decay) or fewer epochs may help.
3. **Single seed (0).** Test set is 50 videos per apparatus; ρ varies by ±0.03 between neighbouring epochs. The paper may report the best of several seeds. Repeat with seeds 1, 2 during the 10/26–10/31 trip.
4. **Trainable params 6.72M vs paper 3.82M**: the released code is not exactly the paper's configuration (to be itemised per module).
5. **README vs paper hyper-parameters**: README says lr 1e-2 and epochs 250/400/500/150 (Ball/Clubs/Hoop/Ribbon), the paper says lr 1e-4 and 620/340/450/710. The lr 1e-2 run above tests the first; if it is better, the README is what the authors actually ran.

## Phase 2 · Fis-V reproduction (FS1000 if the features are obtained)

| Date | Commit | Dataset/score | Key args | GPU | Time | Max VRAM | best ρ (ep) | best MSE (ep) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-11 | 93a141b | Fis-V / TES | `scripts/run_fisv.sh TES 1` (paper: lr 3e-4, dropout 0.7, α 0.5, α_ce 0.1, clip 124, in_dim 1024, 420 ep, seed 0) | 1 (2 concurrent) | 3:36:15 | ~2.7 GB (RSS 2.0 GB) | 0.723 (56), MSE at that ckpt 25.73 | 20.65 (165), ρ at that ckpt 0.667 | paper 0.766 / 19.44. Best ρ at ep 56 of 420, afterwards 0.63–0.67 (train ρ 0.99): over-fits early |
| 2026-10-11 | 93a141b | Fis-V / PCS | `scripts/run_fisv.sh PCS 1` (440 ep, seed 0) | 1 (2 concurrent) | 3:45:42 | ~2.7 GB (RSS 1.9 GB) | 0.852 (48), MSE 8.00 | 8.00 (48), ρ 0.852 | paper 0.863 / 7.17. Close to the paper; best at ep 48, then 0.78–0.83 |
| 2026-10-11 | 93a141b | **Fis-V / Avg** | seed 0 | 1 | | | **0.787** (arithmetic) / **0.796** (Fisher z) | **14.33** (mean of best-MSE ckpts); 16.86 at the best-ρ ckpts | paper 0.823 / 13.31 → ρ −0.036, MSE +1.0. Logs `logs/stdout/fisv_*_1011_1554.log`, ckpts in `/data/mlavl/ckpt_backup/`. Fis-V gap is smaller than RG's (−0.071): Fis-V has 400 training videos vs 200, and the feature question (in_dim 768 vs 1024) applies here too since the README also says 768 (2026-11-05: ruled out, the authors used the same PAMFN features; see Phase 1 cause 1) |
| 2026-10-11 | f6fc37b | Fis-V / TES s1 | paper settings, `SEED=1` | 0 (2 concurrent) | 3:28:04 | ~2.7 GB (RSS 2.1 GB) | 0.736 (16), MSE 36.59 | 21.34 (63), ρ 0.684 | best ρ at epoch 16 (!), very noisy early peak; logs `logs/stdout/fisv_*_s1_1011_1747.log` |
| 2026-10-11 | f6fc37b | Fis-V / PCS s1 | `SEED=1` | 0 (2 concurrent) | 3:36:22 | ~2.7 GB (RSS 2.0 GB) | 0.847 (65), MSE 10.59 | 8.29 (64), ρ 0.826 | |
| 2026-10-11 | f9d91f2 | Fis-V / TES s2 | `SEED=2` | 3 (2 concurrent) | 2:57:23 | ~2.7 GB (RSS 1.8 GB) | 0.711 (45), MSE 27.34 | 20.30 (123), ρ 0.683 | started 21:10 after the RG seed runs freed GPU 3; logs `logs/stdout/fisv_*_s2_1011_2110.log` |
| 2026-10-11 | f9d91f2 | Fis-V / PCS s2 | `SEED=2` | 3 (2 concurrent) | 3:04:31 | ~2.7 GB (RSS 1.9 GB) | 0.858 (50), MSE 9.01 | 7.52 (248), ρ 0.834 | best Fis-V PCS run |

**Fis-V, 3 seeds (0, 1, 2), paper settings.** `scripts/summarize_runs.py --seeds logs/stdout/fisv_*_1011_1554.log logs/stdout/fisv_*_s[12]_*.log`

| Score | paper ρ | ours ρ, mean ± std | paper MSE | ours best MSE, mean ± std | ours MSE at best-ρ ckpt |
| --- | --- | --- | --- | --- | --- |
| TES | 0.766 | 0.723 ± 0.013 | 19.44 | 20.77 ± 0.53 | 29.89 ± 5.86 |
| PCS | 0.863 | 0.852 ± 0.006 | 7.17 | 7.93 ± 0.39 | 9.20 ± 1.30 |
| **Avg** | **0.823** | **0.788** (Fisher z 0.797) | **13.31** | **14.35** | 19.54 |

- Seed spread is much smaller than on RG (±0.006–0.013 ρ): 400 training / 100 test videos vs 200 / 50.
- PCS reproduces within 0.011 ρ; TES is 0.043 below the paper in every seed. TES (technical element score) depends on recognising specific jumps/spins, which is where the authors' text-guided features would matter most and where a feature mismatch (README `--in_dim 768` vs public 1024-d VST) would hurt most. (2026-11-05: feature mismatch ruled out, same PAMFN features as the authors; the TES gap is unexplained.)
- Best-ρ epochs are very early (16–65 of 420/440) and the MSE at those checkpoints is poor (TES 26–37); after epoch ~100 the model has train ρ 0.99 and test ρ settles 0.03–0.08 below the early peak. Fewer epochs or stronger regularisation is the first thing to try in Phase 3; the early peaks also mean the "best ρ" protocol is optimistic and noisy.
- All `_best` / `_best_mse` ckpts backed up to `/data/mlavl/ckpt_backup/` (975 MB total so far).

## Phase 3 · improvements (public features only)

### 3-1 · dropout (regularisation), RG

Motivation: every reproduction run over-fits early (train ρ 0.99; best test ρ at epoch 16–355 of 340–710). `--dropout` is the dropout inside the Transformer layers (S²CE encoder, AVCF cross-temporal decoder, DPG grading decoder; attention + FFN). Paper value 0.3 (RG). Baseline (3 seeds): ρ 0.775 / MSE 5.10.

| Date | Commit | Dataset/score | Key args | GPU | Time | Max VRAM | best ρ (ep) | best MSE (ep) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-12 | 585983c | RG / 4 apparatus × seeds 0,1,2 | paper settings + `--dropout 0.5` (`scripts/phase3_dropout.sh`) | 0/1/3 = seed 0/1/2; 4 concurrent per GPU | 12:04–14:21, 1:07–2:17 per run | ~1.8 GB | avg 0.776 (see table) | avg 5.33 | 12 runs, logs `logs/stdout/rg_*_s?_dp0.5_1012_1204.log`. Final train ρ still 0.98–0.99 |
| 2026-10-12 | 9b9d21c | RG / 4 apparatus × seeds 0,1,2 | paper settings + `--dropout 0.7` | 0/1/3 = seed 0/1/2 | 13:11–15:59, 1:07–1:41 per run | ~1.8 GB | avg 0.751 | avg 6.40 | 12 runs, logs `logs/stdout/rg_*_s?_dp0.7_1012_*.log`. Final train ρ 0.93–0.96. Requested by 정현도 ("dropout 올려서 Phase 3 첫 실험 돌려") |

**Result, 3 seeds each (mean ± std of per-apparatus best test ρ / best test MSE):**

| Apparatus | dropout 0.3 (paper, baseline) | dropout 0.5 | dropout 0.7 |
| --- | --- | --- | --- |
| Ball | 0.758 ± 0.009 / 5.52 ± 0.17 | 0.771 ± 0.016 / 5.93 ± 0.73 | 0.735 ± 0.021 / 7.39 ± 0.36 |
| Clubs | 0.749 ± 0.041 / 5.08 ± 0.61 | 0.745 ± 0.024 / 5.24 ± 0.43 | 0.722 ± 0.037 / 6.83 ± 0.85 |
| Hoop | 0.786 ± 0.035 / 4.73 ± 0.47 | 0.792 ± 0.006 / 4.79 ± 0.28 | 0.785 ± 0.008 / 5.18 ± 0.10 |
| Ribbon | 0.808 ± 0.030 / 5.05 ± 0.60 | 0.798 ± 0.030 / 5.37 ± 0.63 | 0.763 ± 0.031 / 6.20 ± 0.80 |
| **Avg** | **0.775 / 5.10** | **0.776 / 5.33** | **0.751 / 6.40** |
| final train ρ | 0.99 | 0.98–0.99 | 0.93–0.96 |

Conclusion: **Transformer dropout is not the lever.** 0.5 is indistinguishable from the baseline (ρ +0.001, MSE +0.23, all within seed std); 0.7 hurts (ρ −0.024, MSE +1.3) and only then does the train ρ drop below 0.99. The model memorises the 200 training videos through something dropout inside the Transformer layers does not touch (frozen 1024-d clip features → 1×1 Conv projections, the graph module, the grading queries). Next regularisation candidates, in order: (a) fewer epochs / early-stopping-style evaluation protocol, (b) weight decay 1e-4 → 1e-3/1e-2, (c) dropout on the input features / after the projection (`PositionalEncoding` has a dropout=0 slot), (d) temporal augmentation (random crop range is only T−68 ≤ 9 clips on RG; try random clip masking or mixup across videos). Keep 0.3 for all later Phase 3 experiments.

### 3-2 · weight decay, RG

Motivation: dropout did nothing (3-1). The optimiser is plain Adam with coupled L2 `weight_decay` (main.py `get_optim`), paper value 1e-4. Try 1e-3 and 1e-2. Baseline (3 seeds): ρ 0.775 / MSE 5.10. Launcher: `scripts/rg_sweep.sh mlavl_p3_wd wd1e-3 "--weight-decay 1e-3" wd1e-2 "--weight-decay 1e-2"`.

| Date | Commit | Dataset/score | Key args | GPU | Time | Max VRAM | best ρ (ep) | best MSE (ep) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-12 | 5f35949 | RG / 4 apparatus × seeds 0,1,2 | paper settings + `--weight-decay 1e-3` | 0/1/3 = seed 0/1/2; 4 concurrent per GPU | 18:04–20:22 | ~1.8 GB | avg 0.772 | avg 5.21 | 12 runs, logs `logs/stdout/rg_*_s?_wd1e-3_1012_1804.log`. Final train ρ 0.99 |
| 2026-10-12 | 3e573ea | RG / 4 apparatus × seeds 0,1,2 | paper settings + `--weight-decay 1e-2` | 0/1/3 = seed 0/1/2 | 19:10–22:05 | ~1.8 GB | avg 0.773 | avg 5.30 | 12 runs, logs `logs/stdout/rg_*_s?_wd1e-2_1012_*.log`. Final train ρ 0.98–0.99. Requested by 정현도 ("weight decay 올려서 다음 실험 돌려") |

**Result, 3 seeds each (mean ± std of per-apparatus best test ρ / best test MSE):**

| Apparatus | wd 1e-4 (paper, baseline) | wd 1e-3 | wd 1e-2 |
| --- | --- | --- | --- |
| Ball | 0.758 ± 0.009 / 5.52 ± 0.17 | 0.767 ± 0.010 / 5.55 ± 0.27 | 0.780 ± 0.015 / 5.21 ± 0.25 |
| Clubs | 0.749 ± 0.041 / 5.08 ± 0.61 | 0.748 ± 0.034 / 4.96 ± 0.45 | 0.749 ± 0.028 / 5.03 ± 0.35 |
| Hoop | 0.786 ± 0.035 / 4.73 ± 0.47 | 0.787 ± 0.036 / 4.81 ± 0.45 | 0.795 ± 0.021 / 4.61 ± 0.31 |
| Ribbon | 0.808 ± 0.030 / 5.05 ± 0.60 | 0.784 ± 0.019 / 5.52 ± 0.41 | 0.769 ± 0.024 / 6.34 ± 0.28 |
| **Avg** | **0.775 / 5.10** | **0.772 / 5.21** | **0.773 / 5.30** |
| final train ρ | 0.99 | 0.99 | 0.98–0.99 |

Conclusion: **no effect.** Averages move by ≤0.003 ρ, within seed std, and train ρ stays at 0.99 even at 1e-2 (100× the paper value). Ball/Hoop tick up and Ribbon ticks down with larger decay; these are opposite-signed, apparatus-level noise. Note that this is *coupled* L2 inside Adam, which Adam's per-parameter scaling largely neutralises (the AdamW argument); the code already has `--optim adamw`, so a decoupled-decay test (AdamW, wd 0.05–0.1) is the proper follow-up if weight decay is pursued further. Together with 3-1 the picture is that generic parameter regularisation does not address this over-fitting; the next candidates act on the data/input side (fewer epochs, input-feature dropout, temporal augmentation) or on the evaluation protocol.

### 3-3 · input-side regularisation (ours), RG

Motivation: parameter regularisation (3-1 dropout, 3-2 L2) did nothing; the model memorises the fixed clip features. Two input-side knobs (`feat:` commit, `models/model.py` forward, both off by default):
- `--clip_mask p`: during training each clip is zeroed with probability p, same mask for video and audio (keeps them aligned). RG random-crop gives ≤9 clips of augmentation, so this is the first real temporal augmentation.
- `--input_dropout p`: `nn.Dropout` on the projected 256-d V/A tokens right before the shared S²CE encoder.

Settings: clip_mask 0.1, clip_mask 0.25, input_dropout 0.3, input_dropout 0.5; 3 seeds each (48 runs). Launcher:
`scripts/rg_sweep.sh mlavl_p3_input cm0.1 "--clip_mask 0.1" cm0.25 "--clip_mask 0.25" id0.3 "--input_dropout 0.3" id0.5 "--input_dropout 0.5"`. Baseline (3 seeds): ρ 0.775 / MSE 5.10.

| Date | Commit | Dataset/score | Key args | GPU | Time | Max VRAM | best ρ (ep) | best MSE (ep) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-13 | bf5dec5 | RG / 4 apparatus × seeds 0,1,2 | paper settings + `--clip_mask 0.1` | 0/1/3 = seed 0/1/2; 4 concurrent per GPU | 10:29–11:54 | ~1.8 GB | avg 0.765 | avg 5.33 | 12 runs, logs `logs/stdout/rg_*_s?_cm0.1_1013_1029.log` |
| 2026-10-13 | bf5dec5 | RG / 4 apparatus × seeds 0,1,2 | paper settings + `--clip_mask 0.25` | 0/1/3 | ~11:55–13:45 | ~1.8 GB | avg 0.753 | avg 5.60 | 12 runs, logs `logs/stdout/rg_*_s?_cm0.25_1013_*.log` |
| 2026-10-13 | bf5dec5 | RG / 4 apparatus × seeds 0,1,2 | paper settings + `--input_dropout 0.3` | 0/1/3 | ~13:45–16:05 | ~1.8 GB | avg 0.774 | avg 5.11 | 12 runs, logs `logs/stdout/rg_*_s?_id0.3_1013_*.log` |
| 2026-10-13 | bf5dec5 | RG / 4 apparatus × seeds 0,1,2 | paper settings + `--input_dropout 0.5` | 0/1/3 | ~16:05–18:20 | ~1.8 GB | avg 0.771 | avg 5.17 | 12 runs, logs `logs/stdout/rg_*_s?_id0.5_1013_*.log`. Requested by 정현도 ("1번 가자") |

**Result, 3 seeds each (mean ± std of per-apparatus best test ρ / best test MSE):**

| Apparatus | baseline (paper) | clip_mask 0.1 | clip_mask 0.25 | input_dropout 0.3 | input_dropout 0.5 |
| --- | --- | --- | --- | --- | --- |
| Ball | 0.758 ± 0.009 / 5.52 | 0.758 ± 0.008 / 5.70 | 0.737 ± 0.016 / 6.16 | 0.750 ± 0.006 / 5.74 | 0.757 ± 0.011 / 5.72 |
| Clubs | 0.749 ± 0.041 / 5.08 | 0.739 ± 0.024 / 5.14 | 0.702 ± 0.030 / 5.89 | 0.745 ± 0.030 / 5.07 | 0.738 ± 0.016 / 5.11 |
| Hoop | 0.786 ± 0.035 / 4.73 | 0.789 ± 0.024 / 4.71 | 0.791 ± 0.017 / 4.72 | 0.796 ± 0.025 / 4.69 | 0.794 ± 0.005 / 4.70 |
| Ribbon | 0.808 ± 0.030 / 5.05 | 0.774 ± 0.027 / 5.78 | 0.782 ± 0.035 / 5.62 | 0.804 ± 0.033 / 4.93 | 0.795 ± 0.033 / 5.17 |
| **Avg** | **0.775 / 5.10** | **0.765 / 5.33** | **0.753 / 5.60** | **0.774 / 5.11** | **0.771 / 5.17** |
| final train ρ | 0.99 | 0.99 | 0.99 | 0.99 | 0.98–0.99 |

Conclusion: **no gain from input-side regularisation either.** Input dropout 0.3/0.5 is a wash (−0.001 / −0.004 ρ, within seed std); clip masking *hurts* monotonically (−0.010 at 0.1, −0.022 at 0.25, driven by Ball/Clubs/Ribbon) and train ρ stays 0.99 regardless, so masking 25 % of clips still does not stop memorisation — the model fits through whichever clips remain. Across 3-1/3-2/3-3 (dropout, L2, input dropout, clip masking; 96 runs) nothing moves the RG average by more than seed noise, which strongly suggests the −0.07 gap to the paper is not a regularisation problem but an input problem (the public 1024-d PAMFN features vs whatever 768-d features the authors used). (2026-11-05: wrong — the authors used the same PAMFN features; the gap's cause is unknown. Known differences: the released code has 6.99M trainable parameters at `--in_dim 1024` vs 3.82M in the paper, and the paper does not state how many runs / which run it reports.) **Phase 3 hyper-parameter sweeps stop here** (정현도, 2026-10-13); the improvement story for the mid-term presentation moves to the task-level changes (multi-task head, high-score weighting) and to the battle domain (Phase 4/5), where the baseline is ours anyway. Keep `--clip_mask 0 --input_dropout 0` (defaults).

### 3-4 · multi-task TES+PCS head and high-score weighting, Fis-V (2026-10-17/18)

Diagnosis first (`tools/score_bias.py` on the reproduction checkpoints): test predictions are compressed toward the mean (slope prediction ~ label 0.51–0.54 for TES, 0.61–0.81 for PCS); the bottom fifth of TES is over-predicted by +8 to +10 points, which dominates the MSE. Trainer `ours/train_fisv.py` (upstream protocol: paper settings, test every epoch, best-ρ / best-MSE epoch per score; + linear calibration fitted on the training-set predictions of the best-ρ checkpoint → "calibrated MSE"). Queue `scripts/phase3_fisv.sh`: 6 settings × seeds 0–2 = 18 runs, all on GPU 0, 6 at a time, 2026-10-17 19:37 → 2026-10-18 ~21:00 (GPU 0 shared with the audio re-extraction and the battle runs), ~1.5 GB each.

| setting | score | best ρ (mean ± std) | MSE at best ρ | calibrated MSE | best MSE | slope | last-epoch ρ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| single-task (control, `st`) | TES | 0.716 ± 0.008 | 31.95 | 20.48 | 20.57 | 0.45 | 0.617 |
| multi-task (`mt`) | TES | **0.752 ± 0.007** | 40.34 | 22.17 | 19.01 | 0.45 | 0.629 |
| LDS (`lds`) | TES | 0.701 ± 0.011 | 59.77 | 23.96 | 20.55 | 0.85 | 0.631 |
| multi-task + LDS (`mtlds`) | TES | 0.739 ± 0.017 | 32.95 | **19.13** | **18.54** | 0.81 | 0.631 |
| single-task (control, `st`) | PCS | 0.846 ± 0.003 | 9.10 | 7.92 | 7.96 | 0.63 | 0.802 |
| multi-task (`mt`) | PCS | 0.832 ± 0.006 | 8.92 | 7.53 | 7.59 | 0.64 | 0.797 |
| LDS (`lds`) | PCS | **0.849 ± 0.005** | 8.56 | **7.01** | **6.73** | 0.72 | 0.809 |
| multi-task + LDS (`mtlds`) | PCS | 0.844 ± 0.006 | 8.04 | 7.51 | 7.01 | 0.72 | 0.796 |
| paper (Table 3) | TES / PCS | 0.766 / 0.863 | | | 19.44 / 7.17 | | |
| our reproduction (upstream `main.py`, 3 seeds) | TES / PCS | 0.723 / 0.852 | | | 20.77 / 7.93 | | |

Reading:
- **Multi-task helps TES ranking: +0.036 ρ over the control (≈ 5 seed std), 0.752 vs paper 0.766**, the closest we have come. Predicting PCS from the same grade embeddings regularises the TES ranking. It costs PCS −0.014 (the PCS head is a second regressor on embeddings shaped by the TES loss).
- **LDS fixes the compression** (slope 0.45 → 0.81–0.85 on TES, 0.63 → 0.72 on PCS) and lowers the MSE: PCS best MSE 6.73 and TES 18.54 (mtlds) are below the paper's 7.17 / 19.44. It does not improve ranking on its own (TES −0.015).
- Post-hoc calibration alone gives most of the MSE gain for the unweighted models (TES 31.95 → 20.48), so part of the MSE gap is a scale problem, not a ranking problem.
- Best per score: TES ρ from `mt`, PCS ρ and MSE from `lds`; `mtlds` is the best single compromise (ρ 0.739 / 0.844, avg 0.792 vs reproduction 0.788, best MSE 18.54 / 7.01).
- Caveat (as for the reproduction): best-ρ / best-MSE epochs are chosen on the test set, as in the upstream protocol; last-epoch numbers (0.63 / 0.80) are much lower and show the same ordering is not guaranteed without early stopping.


## Phase 4/6 · battle winner prediction (BRACE, 81 battles)

Labels: `ours/battle/winners.csv` (Wikipedia brackets, 31/81 checked by 정현도 against the videos, all correct).
Metric: pairwise accuracy over the 81 battles, 95% bootstrap CI (10,000 resamples of battles). A = dancer who danced first.

### Training-free baselines (2026-10-13, `tools/battle/baselines.py`)

Pose statistics from the BRACE keypoints, aggregated over all of a dancer's segments in the battle; prediction = the dancer with the larger value wins.

| baseline | accuracy | 95% CI | 2011 | 2013 | 2014 | 2017 | 2018 | 2020 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random (expected) | 50.0% | | | | | | | |
| first dancer wins | 59.3% | 48.1–70.4% | 38% | 60% | 50% | 60% | 60% | 79% |
| more dancing time | 50.6% | 39.5–61.7% | 50% | 53% | 50% | 47% | 47% | 57% |
| higher mean joint speed (energy) | 58.0% | 46.9–69.1% | 75% | 67% | 29% | 73% | 53% | 57% |
| higher 90th-pct joint speed | 56.8% | 45.7–67.9% | 75% | 67% | 36% | 80% | 40% | 50% |
| larger powermove share | 59.3% | 48.1–70.4% | 88% | 47% | 50% | 67% | 47% | 71% |

- Every rule's CI contains 50%: with 81 battles, anything below ~61% is not distinguishable from chance. A learned model needs ≥ ~65% to be clearly better than the first-dancer rule.
- Winners and losers are almost identical on these statistics (mean energy 0.0476 vs 0.0464, powermove share 0.39 vs 0.37, dancing time 78.6 s vs 78.9 s): judges are not rewarding "more" or "faster", which is the motivation for MLAVL-style assessment (musicality, vocabulary) rather than motion energy.
- Year-to-year swings (energy rule 29% in 2014, 75% in 2011) are large because a year has only 8–15 battles; leave-one-year-out results must be pooled over all 81 battles, never reported per fold alone.

### Siamese MLAVL, first run (2026-10-13, `scripts/run_battle.sh`, commit 43f98ef)

P(A wins) = σ(k·(s_A − s_B)), BCE + 0.1 × MLAVL aux losses, clip_num 64, 40 epochs, lr 1e-4, batch 16, random A/B swap, leave-one-year-out (6 folds), model after the last epoch (no selection on the test year). Breaking vocabulary = unreviewed draft, vanilla-CLIP text features. GPU 0/1/3, ~35 min per seed (6 folds), ~2 GB VRAM.

| init | acc (mean ± std over 3 seeds) | AUC | log loss | seed-ensemble acc (95% CI) | seed agreement |
| --- | --- | --- | --- | --- | --- |
| scratch | 55.6% ± 2.0 | 0.544 | 1.058 | 58.0% (46.9–69.1) | 0.71 |
| Fis-V PCS ckpt | 54.7% ± 1.5 | 0.569 | 0.924 | 55.6% (44.4–65.4) | 0.91 |
| RG Ribbon ckpt | **60.9% ± 1.2** | **0.605** | 0.987 | **61.7% (50.6–72.8)** | 0.84 |
| *first dancer wins (rule)* | *59.3%* | | | *(48.1–70.4)* | |
| *higher joint-speed energy (rule)* | *58.0%* | | | *(46.9–69.1)* | |

- Best setting (RG init) is the first one whose CI lower bound clears 50%, but it is not distinguishable from the first-dancer rule (59.3%); none of the three is a demonstrated improvement over the training-free baselines.
- Initialising from an AQA model helps only from RG (rhythmic gymnastics: whole-body motion to music), not from Fis-V PCS; scratch is noisiest (seed agreement 0.71).
- Over-confidence / over-fitting: final train loss 0.14–0.42 while test log loss is 0.92–1.06 (> 0.693 of a coin flip). The model memorises ~70 training battles in 40 epochs.
- 2018 is the hardest year in every setting (33–60%).
- Next: fewer epochs / smaller lr for the init runs, freeze the backbone part (S²CE + MAG²) and train only the grading head, add the first-dancer prior as a learned bias (the model is order-invariant by design), and the reviewed vocabulary.

### Siamese MLAVL, round 2: less fitting (2026-10-13/14, RG Ribbon init, 3 seeds each)

| setting | acc (mean ± std) | AUC | log loss | seed-ensemble acc (95% CI) | final train loss |
| --- | --- | --- | --- | --- | --- |
| RG init, 40 ep, lr 1e-4 (round 1) | **60.9% ± 1.2** | **0.605** | 0.987 | **61.7% (50.6–72.8)** | 0.14–0.42 |
| 10 epochs | 56.0% ± 2.3 | 0.580 | 0.845 | 58.0% (46.9–69.1) | 0.53–0.76 |
| 20 epochs, lr 5e-5 | 55.6% ± 2.7 | 0.580 | 0.881 | 56.8% (45.7–67.9) | 0.44–0.66 |
| freeze proj/S²CE/MAG²/AVCF, train text proj + grading decoder (2.97 M params) | 48.6% ± 1.2 | 0.511 | 0.696 | 50.6% (39.5–61.7) | 0.83–0.93 |
| 10 epochs + first-dancer bias (rerun after fix) | 56.4% ± 2.1 | 0.579 | 0.871 | 56.8% (45.7–67.9) | 0.54–0.76 |
| 40 epochs + first-dancer bias (rerun after fix) | 60.9% ± 0.6 | 0.610 | 1.169 | 61.7% (50.6–72.8) | |

- Training less makes the predictions less over-confident (log loss 0.99 → 0.85) but does not raise accuracy; freezing the RG-trained front end leaves the model at chance (log loss 0.696 = it outputs ~0.5), so the RG features are not directly usable for battles and the front end has to adapt.
- All settings so far lie in 55–61%, inside one another's CIs: with 81 battles, settings cannot be ranked yet. The round-1 RG-init number is the best point estimate, not a proven winner.
- **Bug found (fixed in 5a825e8):** the logit scale k and the order bias were trained at the model lr (1e-4); with Adam a scalar moves ~lr per step, so in ≤ 200 steps they stayed at their initial values (bias printed 0.00 in every fold, results identical to "10 epochs"). They now have their own lr (`--scalar_lr 1e-2`). The invalid bias run was discarded (logs moved to `logs/stdout/old/`). Earlier accuracies are unaffected (k does not change the sign of s_A − s_B); their log losses were computed with k ≈ 10 throughout. Rerunning: `rg_e10bias` and `initRG_bias` (40 ep).
- **Order-bias rerun (2026-10-14 00:40):** the learned first-dancer offset settles at b = 0.03–0.29 (≈ +1–7 points of probability for the first dancer), i.e. the model treats the first-dancer effect as small once it sees the performances; accuracy is unchanged (60.9% with 40 epochs, 56.4% with 10). With the scale now learnable, k grows to 16–18 in the 40-epoch runs and the log loss gets worse (1.17): the 40-epoch model is over-confident, which the fixed k = 10 had been hiding.
- **Status after rounds 1–2:** best = RG-initialised siamese MLAVL, 60.9% ± 0.6–1.2 (seed-ensemble 61.7%, CI 50.6–72.8%), AUC 0.61. That is above chance but not distinguishable from the first-dancer rule (59.3%). Hyper-parameter changes move results only within 55–61%. The remaining levers are information the model does not have yet (pose / kick-beat, the reviewed breaking vocabulary) and more training pairs (round-level pairs: every battle has 2–4 rounds per dancer).


### Round 3: vocabulary v1 and round-level pairs (2026-10-14, RG Ribbon init, 40 epochs, 3 seeds)

Reference changed (정현도, 2026-10-14): the task is judging who danced better, so results are compared with chance (50%) and content-based rules, not with the dancing order (first-dancer win rate), which the order-invariant model cannot use anyway. `tools/battle/compare_runs.py` now reports p(acc ≤ 50%) and the paired difference to the energy rule.

| run | vocab | unit | acc mean ± std | AUC | log loss | ensemble acc (95% CI) | p(acc≤50%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| initRG (round 1) | draft0 | battle | 60.9% ± 1.2 | 0.605 | 0.987 | 61.7% (50.6–72.8) | 0.015 |
| v1_battle | v1 | battle | 57.2% ± 1.5 | 0.620 | 1.156 | 58.0% (46.9–69.1) | 0.071 |
| d0_round | draft0 | round pairs | 46.5% ± 0.6 | 0.500 | 1.483 | 46.9% (35.8–58.0) | 0.711 |
| v1_round | v1 | round pairs | 48.1% ± 1.7 | 0.492 | 1.500 | 43.2% (32.1–54.3) | 0.891 |

- Round-level pairs (each round of A vs the same-index round of B, labelled with the battle winner) remove the signal: judges vote on the whole battle, single rounds do not carry it, and the extra (noisy) pairs make the model fit noise (train loss 0.08, test log loss 1.5). Dropped.
- Vocabulary v1 vs draft0: −3.7 points accuracy, +0.015 AUC; indistinguishable with 81 battles. Both are kept as options; the decision needs more battles.
- With 81 battles every comparison between settings is inside the noise. Next: more battles (203 extra World Final battles being downloaded and segmented automatically), and the pose / kick-beat branch on BRACE (running).

### Round 4: pose / kick-beat branch (2026-10-14, BRACE, RG Ribbon init, 40 epochs, 3 seeds)

`--pose`: 115-d clip features from BRACE keypoints + kick-beat alignment, added to the video tokens through a gate that starts at 0 (`ours/pose_branch.py`).

| run | vocab | acc mean ± std | AUC | log loss | ensemble acc (95% CI) | p(acc≤50%) |
| --- | --- | --- | --- | --- | --- | --- |
| initRG | draft0 | 60.9% ± 1.2 | 0.605 | 0.987 | 61.7% (50.6–72.8) | 0.015 |
| d0_pose | draft0 | 58.0% ± 1.7 | 0.581 | 1.297 | 61.7% (50.6–71.6) | 0.016 |
| v1_battle | v1 | 57.2% ± 1.5 | 0.620 | 1.156 | 58.0% (46.9–69.1) | 0.071 |
| v1_pose | v1 | 58.0% ± 1.0 | 0.596 | 1.328 | 60.5% (49.4–70.4) | 0.029 |

- No gain. The learned pose gate ends between −0.21 and +0.19 with a sign that changes across folds and seeds: the model does not find a consistent use for pose with ~70 training battles. Kick-beat alignment alone does not separate winners either (winner 0.468 vs loser 0.472; "higher BAS wins" 44%).
- Pose stays BRACE-only (the extra battles have no keypoints); it is revisited only if the larger dataset shows headroom.

### Data extension: 203 more World Final battles (2026-10-14)

정현도: "합당한 결과가 나올때까지 계속 진행해. 데이터를 더 모으고, 라벨링은 알아서 해줘".

- **Videos and labels** (`tools/battle/collect_extra_battles.py`, `ours/battle/extra_battles.csv`): official Red Bull BC One channel (2,107 videos) matched to the Wikipedia brackets 2004–2025 → 284 battles, 81 = BRACE, **203 new** (201 downloaded at 720p, 13 GB; 2 permanently 403). Winner from the bracket (agrees with BRACE labels on 81/81). Dancing order: first name in the YouTube title (= BRACE first dancer on 81/81); the Wikipedia `<sup>1</sup>` order marks agree on only 41/81 and are not used. First dancer won 61.6% of the new battles.
- **Features**: Video Swin + AST for all 201 (`/data/mlavl/features/BCONE_EXTRA`).
- **Who dances when** (no annotation outside BRACE), `tools/battle/segment_turns.py`, leave-one-year-out on BRACE:
  - BiGRU dancing detector: 92.0% clip accuracy.
  - 2-means on whole-frame Swin features to split the two dancers: 50% (chance) – frames are mostly crowd/stage.
  - Turn-start head + joint decoding of 4/6/8 alternating turns (DP, BRACE turn-count prior): dancer assignment 83.9% mean, 92.5% median; but some videos come out phase-flipped (whole battle A↔B). Confidence (weakest chosen turn start ≥ 0.4) keeps 70% of BRACE at 89%, and 66 of the 201 new battles.
- **Cost of segmentation errors** (`--brace_auto_turns`, BRACE trained/tested with held-out-year automatic turns, RG init, draft0, 3 seeds): **47.3% ± 2.3** (ensemble 44.4%, CI 34.5–55.6) vs 60.9% with the human round annotation. Turn errors, especially phase flips that invert the winner label for a battle, wipe out the signal; adding the 203 battles with these turns would add more noise than signal.
- **Appearance identity** (`tools/battle/appearance.py`): CLIP ViT-B/16 embedding of the dancer crop at each clip centre. With the BRACE annotated dancer boxes (oracle), 2-means per clip separates the dancers only 64.8% (median 60%); breaking poses (inverted, spinning, blurred) dominate the embedding. Single clips are not enough; next: use it at the turn level (mean of ~20 clips per turn) to verify/repair the alternating labelling and filter flipped videos.

### Round 5: BRACE + 89/128 extra battles, and linear probes (2026-10-14)

Turn-level appearance consistency (`segment_turns.py turn_consistency`): mean crop embedding of each turn vs the other A / B turns. On BRACE, consistency ≥ 0.8 keeps 49/81 videos, dancer assignment 91.7%, **no phase-flipped video** (12% without the filter). Applied to the extra battles: 89 kept (≥ 0.8), 128 (≥ 0.6).

Siamese MLAVL, RG init, draft0, 40 epochs, 3 seeds:

| train | test | acc mean ± std | ensemble acc (95% CI) | p(acc≤50%) |
| --- | --- | --- | --- | --- |
| BRACE 81 (round 1) | BRACE 81 | 60.9% ± 1.2 | 61.7% (50.6–72.8) | 0.015 |
| BRACE + 89 extra | BRACE 81 | 56.4% ± 2.3 | 54.3% (43.2–65.4) | 0.217 |
| BRACE + 128 extra | BRACE 81 | 54.7% ± 4.5 | 51.9% (40.7–63.0) | 0.370 |
| BRACE + 89 extra | all 170 (seed 0) | 53.5% | (45.9–60.6) | |

Linear probes (`tools/battle/linear_probe.py`, `tools/battle/probe_sources.py`): per-dancer mean/std of the features, antisymmetric logistic regression on f(A) − f(B), PCA in-fold, leave-one-year-out, 170 battles (81 BRACE + 89 extra):

| source | all 170 | unanimous 5–0 (n=61) | split 3–2 (n=44) |
| --- | --- | --- | --- |
| Video Swin, k=32 | 55.3% (47.6–62.9) | 63.9% (50.8–75.4) | 59.1% |
| AST audio, k=32 | 53.5% (45.9–61.2) | 44.3% | 54.5% |
| Swin + AST, k=32 | 48.8–49.4% | | |
| dancer-crop CLIP, k=8 | 55.3% (47.6–62.9) | 65.6% (52.5–77.0) | 38.6% |
| BRACE pose (81 only), k=16 | 58.0% (46.9–69.1) | 52.2% (n=23) | 61.9% (n=21) |
| crop + video, trained on 4–1/5–0 only, k=16 | 57.1% (50.0–64.7) | 60.7% | 54.5% |

**Conclusion of rounds 1–5.** No model or probe gets a convincing winner signal from the available representations: on 170 battles everything lies in 53–58% with CIs from ~47% to ~64%. The best single BRACE number (60.9%) did not survive more data and is best read as a selection effect over ~15 tried settings. There is a weak, consistent hint that clear (unanimous) battles are predictable above chance (60–66% across sources, CI lower bounds ≈ 47–53%) while split decisions are not. The bottleneck is the input representation (frozen Kinetics/AudioSet/CLIP features pooled over clips carry little "quality" information), not the head, the regulariser or the amount of BRACE-style data.

All-battle siamese run (`d0_extra08_all`, 3 seeds): 53.5% for every seed on 170 battles (BRACE 53.1–58.0%, extra 49.4–53.9%).

### Round 6: estimated pose for all battles + judging-criteria features (2026-10-15)

- **Pose** (`tools/battle/estimate_pose.py`): Keypoint R-CNN (COCO-17) every 3rd frame of the dancing clips of all 282 battle videos (BRACE + extra), up to 6 persons per frame. The dancer is among the detections in 93% of BRACE frames. Dancer selection: hand-made rule (tracking + size + centre + appearance box) picks the wrong person in 20.6% of frames; a learned selector (gradient boosting on candidate features + Viterbi tracking, trained on BRACE keypoints, never on winners) 12.0% (held-out years).
- **Judging-criteria features** (`tools/battle/judge_features.py`), per dancer over their own turns, normalised by bbox height and per second: dancing time, joint speed (mean, p90), inverted share, floor share, spin speed (mean, p90), freezes (rate, time share, inverted share), jerk, kick-beat BAS, freeze-on-beat, pose-cluster entropy / clusters used, travel, extension.

Winner vs loser, 170 battles (81 BRACE + 89 extra with consistent turns), paired sign test: no feature differs (all p ≥ 0.077; closest: freeze time, winner higher in 57.1% of battles, p = 0.077). Means are almost identical (inverted 0.166 vs 0.155, BAS 0.478 vs 0.479, pose entropy 2.771 vs 2.752, speed 4.13 vs 4.27).

Leave-one-year-out antisymmetric logistic regression on the 17 differences: 46.5–50.6% on 170 battles; 47.9–53.6% on all 280 (no turn filter).

**Human reference.** From the bracket vote counts of 259 battles: 5–0 113, 4–1 81, 3–2 64 (24.8% split decisions); a single judge agrees with the panel majority 83.8% of the time. That is the ceiling a model of "the judges' decision" could approach, and it shows the label itself is partly subjective.

**Conclusion after rounds 1–6.** Across the siamese MLAVL (frozen Swin/AST/CLIP features, ~15 settings), linear probes of every feature source, and interpretable judging-criteria statistics from estimated pose, battle outcomes are not predictable above chance on 170–280 battles; winners and losers are statistically indistinguishable on motion, difficulty, musicality-alignment and variety measures. What decides breaking battles (originality, character, responding to the opponent, round-by-round context) is not captured by these signals.

### Round 7: diagnosis and dancer-centred features (2026-10-15)

**Is a battle outcome predictable at all? Dancer-history Elo** (`tools/battle/elo_baseline.py`; only who the dancers are and their earlier World Final results, never the video): 60.4% (55.3–65.6) on 331 bracket matches 2004–2025; "more previous World Final wins wins" 66.5% (n = 173); on our 170 video battles with both dancers seen before 57.1% (49.4–64.7). Outcomes carry a dancer-strength signal of roughly 60–66%; the video models (53–57%) do not reach it, so the gap is in the representation.

**Dancer-centred Video Swin** (`tools/battle/extract_crop_features.py`): each 32-frame clip cropped to a square of 1.4 × the clip's median dancer box (learned selector on estimated pose), 224 px, same Swin-B. Cosine to the whole-frame feature 0.875.
- Linear probe on 170 battles: 46.5–52.4% (whole frame 50.6–55.3%); unanimous 50.8–59.0%.
- Siamese MLAVL (RG init, draft0, BRACE + 89 extra, test on all 170, 3 seeds in parallel): 52.9 / 54.1 / 55.3%, ensemble 52.4% (44.7–60.0); whole frame 53.5 ×3, ensemble 54.7% (47.1–61.8).
- No gain: giving Kinetics-pretrained Video Swin a close-up of the dancer does not make it encode performance quality.

### Round 8: label noise (2026-10-15/16; whole-frame features, RG init, draft0, BRACE + 89 extra, test on all 170, 3 seeds)

| run | seeds | AUC | ensemble acc (95% CI) | unanimous 5–0 (n=61) | 4–1 (n=50) | split 3–2 (n=44) |
| --- | --- | --- | --- | --- | --- | --- |
| hard labels (d0_extra08_all) | 53.5 / 53.5 / 53.5% | 0.563 | 54.7% (47.1–61.8) | | | |
| `--soft_labels` (target = A's vote share) | 54.1 / 53.5 / 55.3% | 0.573 | 54.7% (47.1–61.8) | 57.4% | 62.0% | 43.2% |
| `--train_min_votes 4` (train on 5–0 / 4–1 only) | 54.1 / 56.5 / 53.5% | 0.549 | 54.7% (47.1–62.4) | 55.7% | 54.0% | 50.0% |

No gain overall. The pattern seen in the probes repeats: clear decisions 55–62%, split decisions at or below chance.

**Summary of the improvement attempts (rounds 7–8).** (1) Outcomes are predictable from dancer strength (Elo 60%, past wins 66%), (2) a dancer close-up does not make Kinetics features encode it, (3) cleaner / softer labels do not help. Remaining lever: expert round-level labels (정현도) and/or a representation trained for dance quality.

### Round 9: expert evidence from real judges + fixed expert rubric (2026-10-16)

**What real judges decide on** (`tools/battle/olympics_judging.py`, Paris 2024 results book, 144 rounds × 9 judges = 1,296 votes, per-criterion signed percentages; `ours/battle/olympics_criteria.csv`):

| criterion | agrees with the judge's overall pick | share of the overall margin | summed over judges, predicts the round winner |
| --- | --- | --- | --- |
| Execution | 87.8% | 25.2% | 88.2% |
| Vocabulary | 82.3% | 22.5% | 83.3% |
| Technique | 77.0% | 19.4% | 72.9% |
| Musicality | 75.7% | 17.9% | 70.8% |
| Originality | 71.3% | 15.0% | 68.1% |

Judges on the majority side of a round: 80.9%; 5–4 / 4–5 rounds 13.9%.

**Fixed rubric** (`tools/battle/rubric_scores.py`): each criterion measured by observable cues on the dancer's own turns (CLIP zero-shot prompts on the dancer crops + estimated-pose statistics + beat alignment), standardised, weighted by the judges' margin shares above. Nothing fitted on our winners.

| score | 170 (BRACE + consistent extra) | BRACE 81 | all 280 | unanimous (n=61) |
| --- | --- | --- | --- | --- |
| Execution proxy | 48.8% | 45.7% | 51.1% | 49.2% |
| Vocabulary proxy | 54.1% | 56.8% | 52.1% | 60.7% |
| Technique proxy | 55.3% | 59.3% | 55.7% | 63.9% |
| Musicality proxy | 51.8% | 55.6% | 51.1% | 41.0% |
| Originality proxy | 60.0% (52.4–67.1) | 58.0% | 50.7% | 54.1% |
| **Weighted rubric** | 55.9% (48.2–62.9) | **63.0% (51.9–72.8)** | 54.6% (48.9–60.4) | 50.8% |
| 5 criteria, logistic LOYO | 55.9% | 54.3% | 52.5% | 63.9% |

- The rubric reaches 63.0% on BRACE (human turns), but not on the extra battles (automatic turns) or the full 280; the originality signal on the 170 set (60%) does not hold on 280 (50.7%). No result is robust across subsets.
- The most decisive criterion for real judges, **Execution (88%)**, is exactly the one our proxies cannot measure (48.8%, chance): crashes, slips and control show up as brief events that whole-clip CLIP similarity and pose statistics average away. This is the clearest statement of the measurement gap.

### Round 10: back to the paper's formulation – score regression (2026-10-16)

정현도: "원래 논문의 점수 방식으로 하는걸로 돌아가볼까". Breaking battles have no performance score, so the score is the dancer's **tournament placement** in that year (5 champion, 4 runner-up, 3 semifinal, 2 quarterfinal, 1 top-16 exit, from the Wikipedia brackets); one sample = one dancer's turns in one battle (340 performances from the 170 battles). Confound: the battle's round alone gives Spearman ρ 0.60 (a finalist is ≥ 4), so the metrics that count hold the round fixed: ρ within (year, round) groups, ρ on Top-16 battles (every dancer once; null 95% interval −0.15..0.15), and within-battle pair accuracy.

| model (leave-one-year-out) | ρ within round | ρ Top-16 | pair accuracy |
| --- | --- | --- | --- |
| linear probe, Swin / AST / both, PCA 16–64 (`tools/battle/placement_probe.py`) | −0.00 … 0.10 | −0.07 … 0.02 | 44.7–55.3% |
| MLAVL regression, RG init, 40 ep (`ours/train_placement.py`), seeds 0/1/2 | −0.02 / −0.02 / 0.03 | −0.07 / −0.03 / −0.06 | 51.2 / 53.5 / 55.3% |
| same, seed ensemble | 0.02 | −0.06 | 55.9% |

No signal: the score formulation of the paper does not change the outcome; predicting how far a dancer goes from one performance is as hard as predicting the battle. (Dancer-grouped CV gives ρ −0.2…−0.3 in the probe, the known regression-to-the-training-mean artefact of grouped CV; not evidence of an inverse signal.) Runs on GPU 0 only (policy since 2026-10-16).

### Round 11: zero-shot scoring with the reproduced models (2026-10-17)

정현도: "점수만 뽑고 점수 비교로 그냥 승패를 검증하는건 어때?" – the original model's input/output, no training on battles (`tools/battle/zero_shot_scores.py`). Each of the 18 reproduction checkpoints (RG Ball/Clubs/Hoop/Ribbon, Fis-V TES/PCS × seeds 0–2, `_best.pkl`) with its own vocabulary and settings scores each dancer's turns (same 1024-d Swin + 768-d AST clip features; ≤ 136 clips in one pass at real length, longer sequences on windows of the training clip number, averaged); higher score wins. GPU 0, 9 min for 280 battles × 18 checkpoints.

| scorer | all 280 | 170 (BRACE + consistent extra) | BRACE 81 | unanimous (n=61) |
| --- | --- | --- | --- | --- |
| single checkpoints | 44.3–56.8% | 41.8–56.5% | 39.5–63.0% | 41.0–62.3% |
| RG ensemble (12) | 51.8% (45.7–57.5) | 49.4% | 53.1% | 57.4% |
| Fis-V TES ensemble (3) | 50.7% | 51.8% | 49.4% | 59.0% |
| Fis-V PCS ensemble (3) | 50.4% | 50.0% | 51.9% | 54.1% |
| all 18 ensemble | 51.1% (45.0–56.8) | 50.0% | 54.3% | 57.4% |

Agreement between scorers (Spearman of the A − B score difference over the 280 battles): seeds of the same model 0.39–0.74 (Ball 0.69, Ribbon 0.69, TES 0.69, PCS 0.74), RG vs Fis-V models −0.12, all pairs 0.19.

Reading: the AQA models produce consistent, non-random judgements of breaking (seeds agree), but what a rhythmic-gymnastics model and a figure-skating model call "better" disagree with each other and neither agrees with breaking judges (≈ 50%). Quality as learned from gymnastics / skating scores does not transfer to battles; this also separates the earlier battle-training results from over-fitting: without any training on battles the result is the same chance level.

### Round 12: zero-shot again with PAMFN-matched audio (2026-10-18)

Phase 5b step 1. Audio for all 282 battles re-extracted the PAMFN way (`tools/battle/extract_features_pamfn.py extract --audio_only --tm_dim 64 --audio_sr 32000 --fbank_sr 32000` → `clips_pamfn/`; per-dimension mean-profile correlation with PAMFN 0.89 vs 0.45 before). Video features kept (K400 and K600 Swin-B share a feature space, see commit 3ac7ed7). GPU 0 shared with the Phase 3 Fis-V queue: ~2–3 min per video, finished overnight. Then round 11 repeated unchanged (`zero_shot_scores.py clips_pamfn`).

| scorer | all 280 | 170 | BRACE 81 | unanimous (n=61) |
| --- | --- | --- | --- | --- |
| single checkpoints | 43.9–58.2% | 40.6–58.2% | 38.3–61.7% | 44.3–63.9% |
| RG ensemble (12) | 51.4% (45.7–57.1) | 48.8% | 53.1% | 59.0% |
| Fis-V TES ensemble (3) | 50.7% | 51.8% | 46.9% | 59.0% |
| Fis-V PCS ensemble (3) | 51.1% | 51.2% | 50.6% | 49.2% |
| all 18 ensemble | 51.1% (45.4–57.1) | 50.6% | 54.3% | 57.4% |

Reading: matching the audio input to the training distribution changes nothing (all within ±1 point of round 11). The input mismatch was not why the reproduced models fail on battles; what they learned as "quality" does not transfer. Next: fine-tune with the matched features (`pamfn_all`, same setting as `d0_extra08_all`), then step 2 (backbones and prompts adapted to breaking).

Fine-tuning with the matched features (`pamfn_all`: RG Ribbon s2 init, draft0 vocabulary, BRACE + 89 consistent extra battles, test on all 170, leave-one-year-out, 40 epochs, seeds 0–2 on GPU 0, 2026-10-18 13:39 → ~21:00, ~7 h per run because GPU 0 was shared):

| features | seeds | seed ensemble (170) | p(acc ≤ 50%) | BRACE 81 | extra 89 |
| --- | --- | --- | --- | --- | --- |
| old `clips` (`d0_extra08_all`, round 5) | 53.5 / 53.5 / 53.5% | 54.7% (47.1–61.8) | 0.13 | 53.1% | 56.2% |
| PAMFN-matched audio `clips_pamfn` | 54.1 / 55.3 / 54.1% | 52.9% (45.3–60.0) | 0.24 | 50.6% | 55.1% |

Same chance level. Step 1 of Phase 5b (input matching) is closed: neither zero-shot nor fine-tuning moves. Moving on to step 2 (adapting the backbones and the prompts to breaking).

### Round 13: adapting the backbones and the prompts to breaking (Phase 5b step 2, 2026-10-18/19)

**Video: Video Swin-B K600 fine-tuned on BRACE move types** (`tools/battle/finetune_vst.py`, ckpt `ckpt/vst_move.pt`). Dancer-centred crops of the 32-frame grid clips inside the 1,352 annotated segments (cache: 256 px JPEG at box scale 1.6, 4.6 GB on /data; train: random 75–100 % crop + flip, eval: centre = box scale 1.4 = the extraction crop). BRACE's own sequence-level split: 6,363 train clips / 905 segments, 2,733 test clips / 413 segments. Trained: last stage (features.6) + norm, 25.4M params, AdamW lr 5e-5 (head 5e-4), one-cycle, 15 epochs, batch 16, fp16, GPU 0, ~2 h.

| model | test clip acc | test segment acc |
| --- | --- | --- |
| frozen K400 whole frame, linear probe (leave-one-year-out) | 0.643 | 0.736 |
| frozen K400 dancer crop, linear probe (leave-one-year-out) | 0.699 | 0.819 |
| fine-tuned K600 dancer crop, epoch 1 (kept) | 0.815 | **0.903** |
| same, epoch 14 | 0.783 | 0.869 |

Best after 1–2 epochs, then over-fits (train loss keeps falling). Move labels only; winner labels are never used and the extra battles have none.

**Audio: AST fine-tuned toward beat position and tempo** (`tools/battle/finetune_ast.py`). PAMFN preprocessing (each clip sees 0.64 s = 64 fbank frames; frozen output = `clips_pamfn` audio, r = 0.99999998). Targets from librosa beats on all 282 videos: beat in each of 8 bins of 80 ms (BCE), log2 tempo from the median inter-beat interval within ±5 s (MSE). Held out: BRACE 2020 + every 10th extra video (66,766 / 9,083 clips). AdamW lr 5e-5 (heads ×20), batch 64, ~10–20 min per run.

| trained AST layers | params | best held-out beat AUC (epoch) | tempo within 4 % |
| --- | --- | --- | --- |
| heads only (frozen = linear probe) | 0 | 0.619 (7) | 0.18 |
| last 2 | 14.2M | 0.664 (7) | 0.16 |
| last 4 | 28.4M | 0.698 (6) | 0.13 |
| last 6 (kept, `ckpt/ast_beat_b6.pt`) | 42.5M | **0.735 (5)** | 0.14 |

Frozen AudioSet AST barely encodes where the beat falls inside a clip; fine-tuning half the network brings it to 0.735. Tempo does not improve (librosa's tempo is ambiguous between half and double time, a label problem). AST's time resolution (16-frame patches, stride 10 → 5 time positions per 0.64 s) bounds the 80 ms task.

Battle features from these backbones (`scripts/extract_ft.sh`, 4 shards on GPU 0): `clips_ft` (fine-tuned dancer-crop video + PAMFN audio), `clips_ftva` (+ fine-tuned AST audio).

**Prompts: learnable breaking prompts** (`--learn_text`: the 90 draft0 CLIP text vectors become parameters, lr 1e-3, loss + 1.0 × relative squared distance to the CLIP init). Run `pamfn_lt_all` (= `pamfn_all` + `--learn_text`, 3 seeds, GPU 0, 2026-10-18 22:10 → 10-19 01:05):

| run | seeds (170) | seed ensemble | p(acc ≤ 50%) | BRACE 81 | extra 89 |
| --- | --- | --- | --- | --- | --- |
| `pamfn_all` (fixed prompts) | 54.1 / 55.3 / 54.1% | 52.9% (45.3–60.0) | 0.24 | 50.6% | 55.1% |
| `pamfn_lt_all` (learnable prompts) | 56.5 / 54.7 / 53.5% | 55.3% (47.6–62.4) | 0.10 | 55.6% | 55.1% |

The prompts move little (relative drift 3.5 % on average, max 7 %, cosine to init ≥ 0.997) and the result stays within chance (+2.4 points, inside the seed spread). Nothing in the battle labels pulls the text anchors toward a different meaning.

**Battle runs on the adapted features** (`scripts/phase5b_battle.sh`, same setting as `pamfn_all`, 3 seeds each, 9 runs on GPU 0, 2026-10-19 ~03:45 → 10:25; extraction `scripts/extract_ft.sh` ~1.5 h after the torch.stack fix):

| run | features | seeds (170) | seed ensemble | p(acc ≤ 50%) | AUC | BRACE 81 | extra 89 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `d0_extra08_all` | whole frame K400 + HF AST | 53.5 / 53.5 / 53.5% | 54.7% (47.1–61.8) | 0.13 | 0.566 | 53.1% | 56.2% |
| `pamfn_all` | + PAMFN audio | 54.1 / 55.3 / 54.1% | 52.9% (45.3–60.0) | 0.24 | 0.574 | 50.6% | 55.1% |
| `pamfn_lt_all` | + learnable prompts | 56.5 / 54.7 / 53.5% | 55.3% (47.6–62.4) | 0.10 | 0.579 | 55.6% | 55.1% |
| `ft_all` | move-fine-tuned dancer-crop Swin + PAMFN audio | 47.6 / 47.6 / 50.0% | 48.2% (40.6–55.9) | 0.71 | 0.491 | 42.0% | 53.9% |
| `ftva_all` | + beat-fine-tuned AST | 44.7 / 48.2 / 44.7% | 45.9% (38.2–53.5) | 0.87 | 0.480 | 45.7% | 46.1% |
| `ftva_lt_all` | + learnable prompts | 44.1 / 51.2 / 46.5% | 46.5% (38.8–53.5) | 0.84 | 0.478 | 46.9% | 46.1% |

Reading: adapting the backbones to breaking makes them better at breaking *recognition* (move type 0.82 → 0.90, beat position 0.62 → 0.735) but not at judging: winner prediction drops to 46–48% (AUC < 0.5, all CIs contain 50%). Move type and beat position are the same kind of information for both dancers in a battle (both do toprock, footwork and power moves to the same music), so features specialised on them carry even less of what separates the two. Step 2 of Phase 5b is closed without a gain; 13 rounds and ~40 settings are all at chance on 170–280 battles.

## Phase 5c · real judge scores (WDSF), score regression as in the paper (2026-10-19/20)

Decision 2026-10-19 (정현도): "우연 수준이면 안돼 … 논문 그대로 점수 방식으로". Labels: WDSF per-round, per-judge comparative
scores (overall % and 5 criteria, positive towards blue), `tools/battle/wdsf_scrape.py` / `wdsf_parse.py`; videos: official
@breakingforgold battle uploads matched by name + event + year (`wdsf_match_videos.py`), 313 videos downloaded. Panel
reliability (split-half, Spearman-Brown): total 0.87, criteria 0.76–0.82.

### Turn segmentation (2026-10-20)

Hand labels (정현도, review page https://claude.ai/artifact/Gh1wPRbMT7mmESY54etB3C): 22 battles from all 14 events, 106 turns
(`ours/wdsf/wdsf_turn_labels.json`). **Red dances first in every round (48 / 48)** → athletes follow from the turn order.
Turn length median 41 s (19–62 s). One video lacks round 3 (cut on YouTube).

| detector (BiGRU of segment_turns.py, K400 + AST clips) | mean IoU | IoU ≥ 0.7 | IoU ≥ 0.5 | start err median | videos all IoU ≥ 0.5 |
| --- | --- | --- | --- | --- | --- |
| BRACE-trained, unchanged (4 videos by eye) | – | – | – | – | fails (8 s turns, no decode) |
| WDSF labels only | 0.673 | 0.66 | 0.74 | 2.8 s | 0.59 |
| BRACE detector fine-tuned on WDSF | 0.708 | 0.73 | 0.80 | 2.4 s | 0.73 |
| joint BRACE + WDSF (3x) | 0.785 | 0.81 | 0.88 | 2.2 s | 0.77 |
| joint + min turn 15 s | 0.812 | 0.83 | 0.91 | 1.9 s | 0.86 |
| **joint + min turn 15 s + 3-seed ensemble** | **0.861** | **0.89** | **0.98** | **1.8 s** | **0.91** |

Leave-one-event-out over the 22 labelled battles; the number of turns is fixed from the scores (2 × rounds). Applied to all
313 videos (`wdsf_turn_detector.py apply --variant joint --n_seeds 3 --min_turn_s 15`): 1,520 turns, median 39 s
(`ours/wdsf/wdsf_turns.csv`, hand labels where available).

### MLAVL as in the paper: one turn → the round's score from its athlete's side (2026-10-20, `ours/train_wdsf.py`)

RG settings (lr 1e-4, dropout 0.3, CE 0.1, N = 6), PAMFN-style features (clips_pamfn), breaking vocabulary draft0, clip_num 32,
60 epochs, batch 32, 5 event-grouped folds (Adult + Youth of an event together), model after the last epoch, GPU 0
(7 runs in parallel, 4 threads each, ~2.5 h). Reference: the second dancer of a round (blue) wins **54.4%** of rounds.

| model | target | turns / rounds | rho (turn score) | MSE (zero-predictor) | round acc | hand-labelled rounds |
| --- | --- | --- | --- | --- | --- | --- |
| linear probe, one turn (PCA 64, ridge) | total | 1512 / 751 | 0.078 | 37.36 (37.08) | 0.562 | |
| linear probe, difference of the two turns (PCA 8) | total | – / 751 | margin 0.154 | | 0.570 | |
| same | Execution | – / 640 | margin 0.225 | | 0.583 | |
| same | **Technique** | – / 640 | margin **0.298** | | **0.595** | |
| MLAVL s0 / s1 / s2 | total | 1512 / 751 | 0.054 / 0.033 / 0.048 | 49.3 / 48.3 / 47.9 (37.08) | 0.535 / 0.510 / 0.523 | 0.50 / 0.64 / 0.56 (n=52) |
| MLAVL s0 | Technique | 1288 / 640 | 0.045 | 5.03 (3.35) | 0.516 | |
| MLAVL s0 | Vocabulary | 1288 / 641 | 0.041 | 2.81 (1.96) | 0.524 | |
| MLAVL s0 | Originality | 1288 / 639 | 0.013 | 2.60 (1.67) | 0.507 | |
| MLAVL s0 | Execution | 1288 / 640 | 0.023 | 3.91 (2.71) | 0.525 | |
| MLAVL s0 | Musicality | 1288 / 641 | 0.042 | 2.90 (2.03) | 0.516 | |

Reading: the paper's per-performance formulation does not work on comparative scores: round accuracy 51–53.5% is below the
second-dancer rule (54.4%), MSE above the zero predictor, and folds swing from 38% (Hong Kong + Wuxi) to 64% (Brisbane +
Shanghai Asian). The information is there, though: the antisymmetric probe on the difference of the two turns of a round
reaches 59.5% (Technique, margin rho 0.30). Next: pairwise MLAVL (shared weights, loss on the round margin), `ours/train_wdsf_pair.py`.

### Pairwise (comparative) formulation and strict evaluation (2026-10-20)

Decision (정현도): "비교형으로 바꿔보자, 단 결과가 확실해야 해". Strict evaluation `ours/eval_pair.py`: round accuracy with a 95%
bootstrap CI over battles, one-sided p against 50% and against the second-dancer rule for that score (share of rounds the
blue = second dancer wins; it differs per criterion), subsets: hand-labelled turns, both athletes unseen in the training
folds (no identity shortcut). Reference outside the video: athlete strength = mean margin at OTHER events → **63.6%**
(769 video rounds; both athletes known in 725).

| model | target | round acc [95% CI] | p vs 50% | rule | p vs rule | hand-labelled | athletes unseen (n=51) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| linear probe, difference of the two turns (PCA 8, ridge, antisymmetric) | total | 0.570 [0.532, 0.607] | <0.0001 | 0.543 | 0.19 | 0.654 (n=52) | 0.647 |
| same | **Technique** | **0.595 [0.554, 0.637]** | <0.0001 | 0.523 | **0.016** | 0.689 (n=45) | 0.608 |
| same | **Execution** | **0.583 [0.542, 0.621]** | <0.0001 | 0.494 | **0.004** | 0.667 (n=45) | 0.647 (p50 0.010) |
| same | Vocabulary / Originality / Musicality | 0.552 / 0.516 / 0.510 | | | | | |
| pairwise MLAVL (`ours/train_wdsf_pair.py`, shared weights, 10·MSE on the margin + aux), last epoch, 3-seed ensemble | total | 0.515 [0.477, 0.553] | 0.22 | 0.543 | 0.85 | 0.462 | 0.549 |
| same, s0 | Technique | 0.527 [0.485, 0.568] | 0.11 | 0.523 | 0.46 | 0.578 | 0.373 |
| same, s0 | Execution | 0.523 [0.481, 0.566] | 0.14 | 0.494 | 0.19 | 0.467 | 0.451 |

Reading: the video carries a reliable signal for the Technique and Execution margins (significant against chance and the
position rule, holding on unseen athletes and stronger on clean hand-labelled turns), but MLAVL, in either formulation,
does not extract it: its predictions spread 2.5x more than the probe's at lower correlation (pred sd 2.6 vs 1.0, r 0.08 vs
0.15), i.e. over-fitting ~600 training rounds with ~4M parameters, evaluated after the last of 60 epochs.
Running: pairwise MLAVL with the epoch chosen on a held-out training event (inner validation, no test-fold selection),
from scratch (3 seeds total, Technique, Execution) and from the reproduced RG Ribbon checkpoint (total, Technique),
`scripts/phase5c_pairv.sh`. Server note: `/` hit 100% (another project's HF cache); ckpt/ moved to /data/mlavl/ckpt_live
(symlink).

Pairwise MLAVL with the epoch chosen on a held-out training event (inner validation over epochs 5/10/20/30/40/60; the
test fold never used for selection), `scripts/phase5c_pairv.sh`, strict evaluation:

| run | target | round acc [95% CI] | p vs 50% | rule | p vs rule | hand-labelled | athletes unseen (n=51) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scratch, 3-seed ensemble | total | 0.510 [0.471, 0.550] | 0.32 | 0.543 | 0.88 | 0.558 | 0.510 |
| scratch s0 | Technique | 0.544 [0.502, 0.586] | 0.023 | 0.523 | 0.27 | 0.644 | **0.412** |
| scratch s0 | Execution | 0.536 [0.495, 0.577] | 0.044 | 0.494 | 0.11 | 0.622 | 0.529 |
| RG Ribbon init s0 | total | 0.521 [0.480, 0.560] | 0.16 | 0.543 | 0.81 | 0.577 | 0.529 |
| RG Ribbon init s0 | Technique | 0.556 [0.513, 0.598] | 0.006 | 0.523 | 0.15 | 0.578 | **0.373** |

Reading: early stopping and the RG initialisation move MLAVL to 54–56% on Technique (above 50%, not above the position
rule), but on rounds whose athletes never appear in training it falls to 37–41%: what MLAVL learns beyond the probe is
largely who the dancer is (clothing / body in the Swin features), which does not transfer. The linear difference probe
(PCA 8 of pooled features) has too little capacity for that shortcut and holds on unseen athletes (Technique 0.608,
Execution 0.647). Validation choice is noisy (the held-out event is Hong Kong or Brisbane; Hong Kong is near or below 50%
for every model). Next: take appearance out of MLAVL's input — pose-based tokens (the Bailando codebook plan) instead of
whole-frame Swin, or appearance-invariant / low-dimensional inputs; more hand labels for a clean test set.

### Pose (appearance-free) input (2026-10-20/21)

`scripts/wdsf_pose.sh`: Keypoint R-CNN every 3rd frame of the turns (GPU 0, 3 shards, ~6.5 h), BRACE-trained dancer
selector, per-clip pose + kick-beat features (114-d, joints relative to the box / box height, speeds, beat alignment),
z-scored; `scripts/phase5c_pose_runs.sh`: references and pairwise MLAVL with `--feat pose` (+ AST audio), inner-validation
epochs; folds frozen (`ours/wdsf/folds.json`). Strict evaluation (95% CI over battles, p vs 50% and vs the per-score
second-dancer rule):

| model | target | all rounds [95% CI] | p vs rule | hand-labelled turns | athletes unseen (n=51) |
| --- | --- | --- | --- | --- | --- |
| linear diff, Swin + AST (reference) | Technique | 0.595 [0.555, 0.634] | 0.014 | 0.667 (n=45) | 0.647 |
| linear diff, Swin + AST (reference) | Execution | 0.584 [0.544, 0.623] | 0.002 | 0.667 | 0.686 |
| linear diff, pose + AST | total | 0.537 [0.499, 0.573] | 0.60 | 0.596 (n=52) | 0.588 |
| linear diff, pose + AST | Technique | 0.578 [0.536, 0.619] | 0.047 | 0.689 | 0.627 |
| linear diff, pose + AST | Execution | 0.562 [0.521, 0.602] | 0.013 | 0.622 | 0.588 |
| linear diff, pose only | Technique / Execution | 0.553 / 0.534 | 0.20 / 0.11 | 0.600 / 0.644 | 0.549 / 0.667 |
| **pairwise MLAVL, pose + AST**, 3-seed ensemble | total | 0.525 [0.487, 0.562] | 0.77 | **0.769 [0.653, 0.880]** (n=52, p50 < 0.0001) | 0.510 |
| **pairwise MLAVL, pose + AST** s0 | Technique | 0.539 [0.498, 0.581] | 0.32 | 0.556 | 0.529 |
| **pairwise MLAVL, pose + AST** s0 | **Execution** | **0.564 [0.527, 0.602]** | **0.013** | **0.756 [0.609, 0.884]** (n=45) | **0.588** |

Reading: (1) with pose input MLAVL no longer collapses on unseen athletes (Execution 0.588 vs 0.37–0.41 with Swin), and
pose-MLAVL Execution is the first MLAVL result above the position rule (p = 0.013). (2) The gap between hand-labelled
and automatically segmented turns is now large: total 0.77 vs 0.51, Execution 0.76 vs 0.55. Pose features are computed
on the dancer inside the turn window, so a segmentation error (window shifted onto the other athlete or onto the
walk-in) corrupts the input far more than for whole-frame features. Hand-labelled rounds are few (45–52, 19–22 battles),
so the 0.77 needs a larger clean set before it is a claim → label the remaining 38 battles (review page), retrain the
turn detector, re-run pose features and the same evaluation.

### Diagnostic: exact turns or easier battles? (2026-11-03, `tools/battle/wdsf_turns_cv.py`, `scripts/diag_handcv.sh`)

The 22 hand-labelled battles were re-segmented by the turn detector trained with their own event held out (as in
`wdsf_turn_detector.py eval`, joint, 3 seeds, min 15 s), and the same pose pairwise MLAVL was retrained on that
segmentation (`WDSF_TURNS=.../wdsf_turns_handcv.csv`, seed 0). Same rounds scored, exact vs detected turns:

| target | hand rounds | exact turns (pose_*_s0) | detected turns (handcv_*_s0) | all other rounds (detected) |
|---|---|---|---|---|
| total | 51–52 | 73.1% | 78.4% | 50–51% |
| Execution | 44–45 | 75.6% | 70.5% | 52–55% |
| Technique | 44–45 | 55.6% | 47.7% | 54–55% |

- With detected turns the hand-labelled battles stay at 70–78% (total, Execution): **exact turns are not what lifts
  them.** Their margins are not larger either (median |margin| total 4.8 vs 4.2; auto rounds with |margin| above the
  hand median: 55%).
- So the earlier reading "segmentation is the bottleneck" is not supported as stated. Two explanations remain: (a)
  these 22 battles are easier / different for the model, or (b) the detector segments these battles well (IoU 0.86
  was measured on them) but the other 291 much worse. Only labelling battles that the detector never saw separates
  (a) from (b): the 38 battles on the labelling page give both the detector IoU on unseen battles and a larger clean
  test set. n = 44–52 rounds per cell, so every number above has a 95% CI of about ±13 points.

### Decisive test: 19 newly labelled (unseen) battles (2026-11-03)

정현도 labelled 19 more battles (ranks 23-41 of the labelling page; the first 22 unchanged). Merged file:
`/data/mlavl/datasets/wdsf/wdsf_turn_labels_v41.json` (copy in `ours/wdsf/`). Script: `tools/battle/eval_unseen_labels.py`.
The current turn detector never saw these 19 battles, so their `wdsf_turns.csv` segmentation can be scored directly.

| | turns | start err median | IoU mean | IoU >= 0.5 | videos all turns >= 0.5 |
|---|---|---|---|---|---|
| old 22, leave-one-event-out | 104 | 1.8 s | 0.861 | 0.98 | 0.91 |
| **new 19, unseen** | 102 | 2.5 s | **0.827** | 0.92 | 0.84 |

Segmentation generalises: explanation (b) "the detector fails on the other battles" is ruled out.

Round accuracy of the existing pose pairwise predictions (`pose_*_s*`, seed ensemble for total):

| target | old 22 (n=52/45) | **new 19 (n=51/39)** | rest (n=648/556) |
|---|---|---|---|
| total | 0.769 | **0.471** | 0.509 |
| Execution | 0.756 | 0.513 | 0.552 |
| Technique | 0.556 | 0.436 | 0.545 |

old 22 vs new 19 (total): Fisher p = 0.0023. The old 22 lead in 14 of 16 events, so it is not one event.
Blue-win rate: old 22 0.615, new 19 0.451, rest 0.545; mean |margin| 5.07 / 3.85 / 4.99.
Conclusion: the 73-78% on the hand battles is a property of those 22 battles, not of exact turns and not of the
detector. With good turns on unseen battles the model is at chance. The "clean subset" claim is withdrawn;
all 41 labelled battles together give 0.621 (n=103), with no CI lower bound above 0.5 worth claiming.

## Phase 6: MLAVL + Bailando (choreographic memory) (2026-11-03/04)

Code: `tools/battle/choreo_memory.py` (breaking: 2D pose VQ-VAE, upper / lower body, 512 codes each, one code = 0.4 s,
lower half with root velocity as in Bailando), `ours/train_vocab_pair.py` (vocabulary statistics judge, token
Transformer judge with code embeddings initialised from the memory), `ours/choreo_feat.py` + `models/model.py --choreo`
(RG / Fis-V: feature VQ-VAE on the training split, 128 codes, codes as MAG^2 nodes = `graph` or added to the visual
tokens = `add`). Runs: `scripts/run_choreo.sh`, `scripts/phase6_choreo.sh`, `scripts/rg_sweep.sh p6_rg ...`,
`scripts/phase6_fisv.sh`. Commit 5c3beca.

Pose memory (WDSF, 313 videos, danced turns only, label-free): val L1 0.100 upper / 0.149 lower vs mean pose 0.212 /
0.266; perplexity 416 / 408 of 512 (all codes used).

### Breaking, strict evaluation (all rounds, 95% battle-bootstrap CI; rule = second-dancer / blue wins)

| target | rule | probe vq | probe vq+pose | vocab stats | token Transformer (3 seeds) | MLAVL vq | MLAVL vq+pose |
|---|---|---|---|---|---|---|---|
| total | 0.543 | 0.543 | 0.562 | 0.548 | 0.516 | 0.555 [0.521, 0.591] (3 seeds) | 0.533 (3 seeds) |
| Technique | 0.523 | 0.547 | 0.562 | 0.489 | 0.503 | 0.533 | |
| Vocabulary | 0.549 | 0.555 | 0.543 | 0.535 | 0.535 | 0.537 | |
| Originality | 0.551 | 0.540 | 0.534 | 0.527 | 0.535 | 0.534 | |
| Execution | 0.494 | 0.541 | 0.536 | 0.553 (p_rule 0.033) | 0.503 | 0.505 | 0.562 (p_rule 0.009) |
| Musicality | 0.571 | 0.535 | 0.509 | 0.543 | 0.538 | 0.554 | |

Nothing beats the second-dancer rule except Execution, where the earlier hand-pose MLAVL already gave 0.564
(p_rule 0.013); with ~30 comparisons in this table one p < 0.01 is expected by chance. The vocabulary criteria
(Vocabulary, Originality) read literally from the tokens are not predicted either. Conclusion: on WDSF battles the
choreographic memory does not add a measurable signal; the problem is not the representation of single moves.

### Paper datasets first: RG (feature-level memory, 3 seeds, 2026-11-04)

Question (정현도): does the new module help where MLAVL is known to work, before judging it on breaking? RG and Fis-V
ship only Video Swin clip features, so the memory is `ours/choreo_feat.py` (VQ-VAE on the training-split clip
features, K = 128, one code per clip; `ckpt/choreo_rg.pt` val rec 0.535, perplexity 114/128), frozen in stage 2.
Paper settings otherwise; `scripts/rg_sweep.sh p6_rg ...`, GPUs 0/1/3 = seeds 0/1/2, 2026-11-04 00:58 → 08:40.
Same protocol as the reproduction (best test ρ over epochs).

| apparatus | reproduction (3 seeds) | `--choreo graph` | `--choreo add` |
|---|---|---|---|
| Ball | 0.758 ± 0.009 | 0.753 ± 0.019 | 0.736 ± 0.015 |
| Clubs | 0.749 ± 0.041 | 0.743 ± 0.040 | 0.719 ± 0.032 |
| Hoop | 0.786 ± 0.035 | 0.788 ± 0.031 | 0.778 ± 0.039 (best ρ at ep 24–84, MSE there 8.6) |
| Ribbon | 0.808 ± 0.030 | 0.807 ± 0.026 | 0.828 ± 0.027 (s2 best at ep 11, MSE 12.6) |
| **avg ρ / best MSE** | **0.775 / 5.10** | **0.773 / 5.16** | **0.765 / 5.27** (MSE @ best ρ 7.19) |

- `graph` (code nodes in MAG²) is indistinguishable from the reproduction on every apparatus (|Δ| ≤ 0.006, well
  inside one seed std).
- `add` is worse on Ball/Clubs and its Hoop/Ribbon "gains" are early-epoch peaks with poor MSE at that checkpoint
  (the best-over-epochs protocol rewards noise); not a real improvement.
- Fis-V seed 0 (2026-11-04 08:40): graph TES 0.692 / PCS 0.844, add TES 0.698 / PCS 0.843 vs reproduction seed 0
  0.723 / 0.852. Seeds 1–2 running (`logs/stdout/fisv_*_s{1,2}_choreo_*_1104_0840.log`).

Reading: a codebook learned on the same Video Swin features re-expresses information the visual tokens already
carry, so it cannot add signal. A Bailando-style memory would need a different input (pose / motion, as in
Bailando) — which RG/Fis-V do not ship (raw videos would have to be downloaded and re-posed).

### Paper datasets first: Fis-V (feature-level memory, 3 seeds, 2026-11-04)

`ckpt/choreo_fisv.pt` (val rec 0.615, perplexity 110/128); `scripts/phase6_fisv.sh`, seeds 1–2 finished 2026-11-04 ~15:00.

| score | reproduction (3 seeds) | `--choreo graph` | `--choreo add` |
|---|---|---|---|
| TES | 0.723 ± 0.013 | 0.700 ± 0.008 | 0.690 ± 0.014 |
| PCS | 0.852 | 0.848 ± 0.005 | 0.844 ± 0.001 |
| **avg ρ / best MSE** | **0.788** / 20.77, 7.93 | **0.774** / 21.15, 7.78 | **0.767** / 22.52, 7.98 |

Both variants lose TES ranking (−0.02 to −0.03, about two seed stds) and leave PCS unchanged. Together with RG:
a memory learned on the same Video Swin features adds no information and slightly hurts. Next (Phase 6b): the memory
on estimated poses, as in Bailando — raw RG / Fis-V videos downloaded 2026-11-04 (research use only),
`tools/aqa/aqa_pose.py` → `tools/aqa/aqa_choreo.py` → `--choreo_src pose`; unattended pipeline
`scripts/phase6b_pipeline.sh` (log `logs/aqa_pose/pipeline.log`).

### Paper datasets: pose-based memory (Phase 6b, 3 seeds, 2026-11-04 21:50 → 11-05 09:04)

Pose from the raw videos (`tools/aqa/aqa_pose.py`: Keypoint R-CNN every 3rd frame, single-athlete tracking; athlete
found in 95.1 % (Fis-V, min 78 %) / 97.5 % (RG, min 86 %) of sampled frames), VQ-VAE as for breaking
(`tools/aqa/aqa_choreo.py`, upper / lower K = 512, training split only; val rec Fis-V 0.062 / 0.089, RG 0.060 / 0.090
vs mean-pose 0.12–0.17, perplexity 418–463 / 512), per-clip code input `ours/choreo_pose.py` (clips with pose 96.7 % /
97.1 %). Pipeline `scripts/phase6b_pipeline.sh`; GPUs 0/1/3 = seeds 0/1/2. Best test ρ over epochs, as for the reproduction.

| RG | reproduction | feature memory graph / add | **pose graph** | **pose add** |
|---|---|---|---|---|
| Ball | 0.758 ± 0.009 | 0.753 / 0.736 | 0.765 ± 0.016 | **0.794 ± 0.017** |
| Clubs | 0.749 ± 0.041 | 0.743 / 0.719 | 0.749 ± 0.025 | **0.766 ± 0.014** |
| Hoop | 0.786 ± 0.035 | 0.788 / 0.778 | 0.789 ± 0.022 | **0.802 ± 0.019** |
| Ribbon | 0.808 ± 0.030 | 0.807 / 0.828 | 0.811 ± 0.029 | **0.832 ± 0.030** |
| **avg ρ / best MSE** | **0.775 / 5.10** | 0.773 / 5.16, 0.765 / 5.27 | 0.778 / 5.15 | **0.798 / 4.84** |

| Fis-V | reproduction | feature memory graph / add | pose graph | pose add |
|---|---|---|---|---|
| TES ρ (best MSE) | 0.723 ± 0.013 (20.77) | 0.700 / 0.690 | 0.712 ± 0.016 (20.66) | 0.719 ± 0.007 (20.24) |
| PCS ρ (best MSE) | 0.852 ± 0.006 (7.93) | 0.848 / 0.844 | 0.849 ± 0.009 (7.65) | 0.856 ± 0.008 (7.37) |
| **avg ρ** | **0.788** | 0.774 / 0.767 | 0.780 | 0.787 |

- RG pose `add`: +0.023 avg ρ and lower best MSE (4.84 vs 5.10); every apparatus improves (+0.016 to +0.036,
  about one seed std each, all four in the same direction). First gain of the new module on any dataset.
- Fis-V pose `add`: tie in ρ (PCS +0.004, TES −0.004), best MSE lower on both (20.24 / 7.37 vs 20.77 / 7.93).
- `graph` (codes as MAG² nodes) ≈ reproduction on both datasets; `add` (code vector added to the visual tokens) is
  the useful form. Pose (Bailando) instead of re-encoded Video Swin features is what changes the result.
- Ablations moved to RG (where the gain is), `--choreo add`, 3 seeds, tmux `p6c_rgA` (swap, shuffle, posecont,
  no_audio) and `p6c_rgB` (upper, lower, no_text_graph), started 2026-11-05 09:05; the Fis-V queue
  (`scripts/phase6b_ablation.sh`) was stopped before it launched anything.

### RG ablations of the pose memory (3 seeds, 2026-11-05 09:05 → 23:31)

`--choreo add` unless noted; GPU 0/1/3; best test ρ over epochs (same protocol). Δ and paired t-test over the 12
runs (4 apparatus × 3 seeds) against the bold baseline. No multiple-comparison correction (7 tests; Bonferroni 0.007
keeps only swap).

| question | setting | RG avg ρ (Fisher z) | best MSE | Δ | pos / 12 | p |
|---|---|---|---|---|---|---|
| **baseline: pose add** | | **0.798** | 4.84 | | | |
| own video's motion? | `swap` codes of another video (fixed permutation) | 0.772 (0.775) | 5.37 | −0.026 | 10 | 0.005 |
| timing? | `shuffle` clips within the video | 0.787 (0.789) | 4.99 | −0.012 | 9 | 0.026 |
| codebook needed? | `posecont` continuous pose statistics, no VQ | 0.801 (0.804) | 4.88 | +0.003 | 7 | 0.71 |
| upper or lower? | `upper` codes only | 0.788 (0.789) | 4.92 | −0.010 | 10 | 0.026 |
| | `lower` codes only | 0.796 (0.796) | 5.02 | −0.003 | 7 | 0.52 |
| **baseline: reproduction** (no pose) | | **0.775** | 5.10 | | | |
| MLAVL music? | `no_audio` | 0.777 (0.778) | 5.26 | +0.002 | 6 | 0.65 |
| MLAVL text graph? | `no_text_graph` | 0.779 (0.780) | 5.22 | +0.004 | 6 | 0.48 |

(pos / 12 = runs where the bold baseline is higher than the ablation.)
Other contrasts vs reproduction: pose add +0.023 (p = 0.003), posecont +0.026 (p = 0.018), lower +0.020 (p = 0.002),
shuffle +0.011 (p = 0.15).

Reading:
- The gain needs the athlete's own motion at the right time: another video's codes erase it (0.772 ≈ 0.775), shuffling
  the clip order keeps only half. Hoop is the exception (swap 0.811 ≥ pose 0.802).
- Lower body (hips, legs, root velocity) carries almost all of it; upper-only keeps less than half.
- The VQ step itself is not what helps: continuous pose statistics match the codebook (0.801 vs 0.798). What carries
  over from Bailando is the pose input split into upper / lower body; the codebook gives the same accuracy as
  discrete, readable 0.48 s motion units.
- In our reproduction, removing MLAVL's audio input or its action-text graph changes nothing on RG (p > 0.4). Reported
  as observed; our reproduction is below the paper (0.775 vs 0.849), so this is not a claim about the paper's setting.
- Next: Phase 6c (Bailando Beat Align per clip, `--extra`), queue `scripts/phase6c_queue.sh` (tmux `p6d_q`).

### Phase 6c: extra clip inputs on RG (Bailando Beat Align, optical flow; 3 seeds, 2026-11-05 23:34 → 11-06 10:12)

`--extra` (`ours/extra_inputs.py`): each input projected and added to the visual tokens. Beat = `tools/aqa/aqa_beat.py`
(9-d per clip: motion energy, kinematic / music beats per s, Beat Align Score σ 0.1 s, music beats followed by a
kinematic beat within 0.2 s, onset strength, speed–onset correlation, tempo, has_pose). Flow = PAMFN I3D flow (1024-d).
Queue `scripts/phase6c_queue.sh`, GPU 0/1/3. Paired t over the 12 runs.

| setting | RG avg ρ (Fisher z) | best MSE | vs pose add 0.798 | vs reproduction 0.775 |
|---|---|---|---|---|
| `--extra pose,beat` | 0.795 (0.796) | 4.96 | −0.003, p = 0.60 | +0.020, 11/12, p = 0.008 |
| `--extra pose,posecont` | 0.794 (0.796) | 5.05 | −0.004, p = 0.57 | |
| `--extra pose,beat,flow` | 0.786 (0.790) | 5.15 | −0.012, p = 0.21 | |
| `--extra beat` | 0.773 (0.776) | 5.33 | | −0.002, p = 0.78 |
| `--extra flow` | 0.771 (0.777) | 5.17 | | −0.004, p = 0.70 |

Reading: Beat Align adds nothing on top of pose and nothing alone; optical flow (another motion input) adds nothing
either, alone or with pose. The RG gain is specific to the athlete's pose; pose add (0.798) stays the best setting, so
no new Fis-V run is needed (pose add was already run there: 0.787 vs 0.788).

## Phase 7: judging criteria from motion (WDSF, 2026-11-06)

Question (정현도 approved 11/6): can interpretable motion measures track the judges' *criterion-specific* scores?
`tools/battle/criteria_validity.py` (beats → measures → validity; CPU only, ~6 min). Rounds = WDSF 5-criteria rounds
with both dancers' turns measured (automatic turn segmentation, `wdsf_turns.csv`): 641 rounds, 255 battles, 25 events
(45 rounds hand-segmented). Measure = blue − red (mean over the dancer's turns in the round); target = judge mean per
round, raw and criterion-specific (criterion − mean of the five). Spearman, battle-bootstrap 95% CI.

**Labels (2,699 rounds × 9 judges).** Criteria are not one halo: Technique–Originality 0.31, Technique–Execution 0.83.
PCA: PC1 73% (overall), PC2 14% = Technique −0.63 / Execution −0.36 vs Originality +0.59 / Musicality +0.35
(physical vs artistic). Split-half reliability (Spearman–Brown, 9 judges) raw 0.76–0.82, **criterion-specific part
0.44 (Vocabulary) – 0.73 (Technique)**: judges agree on how criteria differ, so criterion-specific validation is possible.

**Pre-specified hypotheses (measure → its own criterion-specific part):**

| measure | criterion | ρ [95% CI] | p | highest of the 5 specific parts |
|---|---|---|---|---|
| distinct upper / lower codes per 16 s | Vocabulary | +0.039 / +0.037 | 0.32 / 0.35 | no |
| (upper, lower) code-pair entropy | Vocabulary | +0.072 [−0.001, +0.141] | 0.055 | yes |
| code rarity upper / lower / transitions (leave-athlete-out) | Originality | +0.025 / −0.059 / −0.028 | ≥ 0.16 | no |
| Beat Align Score / beat hit / speed–onset corr | Musicality | −0.060 / +0.046 / +0.007 | ≥ 0.15 | no |
| inverted share | Technique | +0.016 | 0.73 | no |
| shoulder spin p90 | Technique | +0.078 | 0.070 | yes |
| **joint-speed p90 (energy)** | **Technique** | **+0.208 [+0.131, +0.287]** | **< 0.001** | **yes** |
| jerk (expected −) | Execution | +0.136 [+0.055, +0.218] | < 0.001 | no (sign reversed; tracks energy) |

- **Energy → Technique-specific** holds in all 5 event folds (0.13–0.38), after controlling duration + jerk (partial
  +0.146) or overall preference (+0.185), and is larger on the 45 rounds of the first 22 hand-labelled battles (0.49; see the correction at the end of this section: not evidence about segmentation). Energy also tracks the
  physical-vs-artistic axis (PC2, +0.204 [+0.125, +0.287]) and is negative for Originality / Musicality specific parts
  (−0.15 / −0.13): a discriminant pattern.
- Not pre-specified: **turn duration → Vocabulary-specific +0.168 [+0.092, +0.246]**, all 5 folds positive (0.05–0.28),
  unchanged after controlling overall (+0.169). Longer turns show more moves.
- **Codebook measures fail (no-go for the pre-registered codebook hypothesis):** the codes saturate (median 33 of 40
  upper codes distinct within 16 s; pair entropy 3.68 of max 3.69). 0.4 s codes are pose fragments, not moves, so
  repetition of moves is invisible to them. Beat Align does not track Musicality either.
- Reading: the physical side of judging (Technique) is measurable from 2D pose; the artistic side (Originality,
  Musicality) and Vocabulary-beyond-duration are not, with these measures. Next candidates: move-level units (code
  n-grams / longer windows, or coarser codebooks), and turn segmentation quality (hand subset is stronger).

### Phase 7b: move-level units from the codebook (2026-11-06)

`tools/battle/move_units.py`: windows of L consecutive 0.4 s codes (hop L/2) → mean upper / lower code vectors of each
half → PCA 32 → k-means K "moves" (no judge labels). Pre-specified main setting L = 8 (3.2 s), K = 64; grid L ∈ {5, 8,
12} × K ∈ {32, 64, 128} reported as sensitivity (`{D}/move_validity_grid.csv`). Same 641 rounds and battle bootstrap.

| measure → criterion-specific | main L8 K64: ρ [95% CI], p | grid (9 settings): ρ range | own criterion highest of 5 |
|---|---|---|---|
| distinct moves per 12 windows (rarefied) → Vocabulary | +0.071 [−0.006, +0.157], 0.070 | +0.044 … +0.113 | 9 / 9 |
| move entropy → Vocabulary | +0.058, 0.14 | — | yes (main) |
| novelty vs own earlier rounds → Vocabulary | −0.097 [−0.192, −0.007], 0.038 (sign reversed) | −0.067 … −0.127 | 0 / 9 |
| move rarity (leave-athlete-out) → Originality | +0.003, 0.91 | −0.093 … +0.015 | 0 / 9 |
| move-transition rarity → Originality | −0.036, 0.33 | — | no |

- Move diversity points the right way in all 9 settings, is unrelated to turn length (ρ with duration −0.04…+0.04) and
  is always most related to Vocabulary among the five specific parts, but it is small (ρ ≈ 0.07) and not significant
  in the pre-specified setting. Weak, consistent evidence only.
- Originality: no signal from any move-rarity measure. Novelty across rounds has the wrong sign.
- Units are still noisy: median 10.1 of 12 windows are different moves (K = 64), and the same dancer's move histograms in
  different battles are only slightly more similar than different dancers' (AUC 0.565, 2,181 same-dancer pairs).
  A move vocabulary needs better motion units than 2D pose on fast, inverted breaking moves gives here.

### Phase 7c: what limits the measures — pose quality, segmentation, or dancer selection? (2026-11-06)

`tools/battle/pose_diagnostics.py` (quality → strata → sheet). Per turn: mean keypoint score of the selected dancer,
median box height, implausible-jump rate (mean joint displacement > 0.5 box heights per 0.1 s: 16% of consecutive
samples on average). Rounds split into tertiles by the worse dancer.

- Pose confidence tertiles (low / mid / high): energy → Technique-specific +0.13 / +0.22 / +0.25 (similar for the jump
  rate); move diversity, Beat Align and move rarity show no trend with pose quality. Segmentation confidence
  (min_cut_p) tertiles show no trend either (energy +0.25 / +0.13 / +0.24).
- **Visual check (3 random contact sheets, 72 frames inside automatic turns; coordinates are 1920×1080-equivalent in
  `pose_est`, consistent with `appearance`, so the pipeline is fine — the first sheet was drawn without rescaling):**
  correct dancer 44, **wrong person 20 (28%, ~18–39%)**, no pose 2, unclear 6. The errors are almost all frames where
  the dancer is on the floor or inverted (power moves, freezes, floor work) and the selector takes a standing person
  (opponent, judge, audience); standing toprock is nearly always right. In many of those frames the dancer has no
  person detection at all (Keypoint R-CNN, score ≥ 0.5), so the selector cannot recover.
- Consequence: every WDSF pose-based result (Phase 5c pose inputs, Phase 6 breaking codebook, Phase 7 / 7b measures)
  saw the wrong person in roughly a quarter of the dancing frames, selectively in the technically hardest moves.
  The bottleneck is person detection / selection on inverted bodies, not keypoint precision or turn segmentation.

**Detector probe (`tools/battle/detector_probe.py`, the 22 failure frames, graded by eye):** the dancer *is* found by
Keypoint R-CNN with score ≥ 0.2 in ~21 / 22 frames (we stored only ≥ 0.5), by the same model on the 180°-rotated frame
in ~18 / 22, and by YOLO11x-pose (conf ≥ 0.25) in ~19–20 / 22 with far fewer spurious skeletons than Keypoint R-CNN at
0.2. So the failures come from the 0.5 storage threshold plus per-frame selection, not from bodies being undetectable.
Fix: keep lower-score detections and select the dancer per *track* over the whole turn (the most active track on the
stage floor) instead of per frame.

**Same visual check on the paper datasets (`tools/aqa/aqa_pose_check.py`, 2 sheets each, random frames, 2026-11-06):**
RG 48 frames: athlete 47, no pose 1 (lying on the mat), wrong person 0. Fis-V 48 frames: athlete 44, wrong person 1
(crowd, skater in a spiral), no pose 3 (close-ups). One performer and a clean background make the size-based rule work,
so the RG / Fis-V pose results (Phase 6b, 6c) are not affected by the WDSF selection problem. Keypoints look rough
during blurred skating spins (one observation that fits the TES hypothesis; not measured).

### Correction: the 0.49 on hand-labelled rounds is not evidence about segmentation (2026-11-06)

energy -> Technique_spec by group, old pose, current `wdsf_turns.csv` (battle bootstrap): first 22 labelled battles +0.491
[+0.189, +0.689] (n=45, exact turns); next 19 labelled battles +0.105 [-0.292, +0.453] (n=39, detected turns); all other
rounds +0.190 [+0.101, +0.278] (n=557, detected turns). The first 22 battles stood out in the same way for pairwise MLAVL
(see "Decisive test" above), and detected turns did not lower them there. So the inference "segmentation error halves the
effect" is withdrawn. The groups differ in both battles and turn source, so this does not show segmentation is harmless;
the 19 labels are not yet in `wdsf_turns.csv`, and a same-battles exact vs detected comparison on them is still to do.

## Phase 7d: YOLO11x-pose + track-level dancer selection (2026-11-06)

Code: `tools/battle/wdsf_track_pose.py` (`scripts/phase7d_yolo.sh`, GPU 0, 3 shards, 3.5 h, 600k frames, 1.2 GB),
`scripts/phase7d_after.sh` (`WDSF_POSE_TAG=2`; old files kept). Eligibility thresholds were set on the 22 old failure
frames and 24 development frames (seed 99). Evaluation: 72 fresh frames (seeds 10-12, random turn, then random 10 Hz
point, so frames without a pose count), `tools/battle/pose_grade_sample.py`, `pose_grade_page.py`. Two independent graders:
Claude (overlay only, committed first, 877a647; frame 28 regraded after a failed frame grab was fixed) and 정현도 (expert,
with raw frame and +-1 s context, also the move type). Grades: `ours/wdsf/pose_grades_1106_{claude,hyundo}.json`.

| of 72 frames | 정현도 | Claude |
| --- | --- | --- |
| right person, skeleton good | 35 | 55 |
| right person, skeleton off | 28 | 4 |
| wrong person | **7 (9.7%, Wilson 95% 4.8-18.7)** | 6 (8.3%) |
| unclear | 0 | 5 |
| no pose, dancer visible | 2 | 2 |

- Wrong person: 28% (20 / 72, Phase 7c, one grader) -> 9.7% (Fisher p = 0.0095). Target < 10% is met on the point estimate,
  not on the upper bound. Agreement on right / wrong / unclear / no pose 93%, kappa 0.75; all 6 of Claude's wrong frames are
  in 정현도's 7, the 7th was unclear for Claude. Pose exists on 94% of turn time (median per turn 98%).
- **Skeleton quality is the new finding.** On 28 frames the person is right and the skeleton is off (expert); Claude saw
  only 4 of them, so the first grader is not valid for joint quality. By move, among right-person frames: power 17 / 20 off,
  freeze 2 / 2, footwork 7 / 15, toprock 1 / 11 (power vs toprock Fisher p < 0.0001). "Off" had no written threshold.
- 10 / 72 frames are not dancing (walking, waiting, bowing), 9 of them within 6 s of a turn boundary. 5 of the 7 wrong-person
  frames lie in the first 7 s of a turn (both dancers on the floor, walking); seen after grading, so any trimming rule
  needs new frames to be judged.

Validity, new vs old pose, Spearman with the criterion-specific part, battle bootstrap, 636 / 641 rounds. (L8 K64 = main)

| measure -> criterion | old | new | same-round difference |
| --- | --- | --- | --- |
| inverted -> Technique | +0.016 [-0.075, +0.107] | **+0.138 [+0.050, +0.224]** | +0.123 [+0.039, +0.209] |
| spin -> Technique | +0.078 [-0.004, +0.161] | **+0.214 [+0.134, +0.288]** | +0.130 [+0.031, +0.227] |
| energy -> Technique | +0.208 [+0.131, +0.287] | +0.181 [+0.090, +0.265] | -0.024 [-0.123, +0.074] |
| jerk -> Execution | +0.136 | +0.155 [+0.074, +0.236] | |
| duration -> Vocabulary (not pre-specified) | +0.168 | +0.168 [+0.093, +0.248] | |
| move diversity -> Vocabulary, L8 K64 | +0.064 (this run) | +0.051 [-0.027, +0.141] | |
| move rarity -> Originality, L8 K64 | +0.004 | +0.001 [-0.085, +0.080] | |
| code rarity upper -> Originality | +0.025 | -0.117 [-0.205, -0.032] (wrong sign) | |
| Beat Align -> Musicality | -0.060 | +0.049 [-0.033, +0.134] | |

Per event fold (5 folds, new pose): spin +0.18..+0.30, energy +0.11..+0.24, inverted +0.04..+0.21, all positive; jerk -> Execution
-0.06..+0.32, mixed. Partial for the new pose: spin +0.111, energy +0.070 given duration + jerk; +0.191, +0.179 given overall. Coverage difference
vs Technique / Vocabulary -0.02 / +0.03: missing pose does not drive the measures. Codebook: lower val rec 0.086 vs mean
pose 0.155, perplexity 422 / 512. Dancer signature AUC 0.566 (old, same code) -> **0.596, below the 0.7 goal**; median distinct
moves 10.3 of 12. Move diversity -> Vocabulary over the 9 settings +0.031..+0.106 (L5 K64 and L8 K128 have CIs above 0; main does
not). Move-transition rarity -> Originality main +0.053, p = 0.18, L12 settings +0.076..+0.087, p = 0.03-0.04: 45 tests in the
grid, not a finding.

Reading: fixing the person made the inverted and spinning measures valid for Technique, as a wrong-person error on floor moves
would predict. It did not create move units or artistic measures. Joint errors in power moves are the next suspect for the unit
problem, not shown yet. BRACE manual keypoints (26.7k frames) can measure and fine-tune this.

### Skeleton severity of the 28 "off" frames (정현도, 2026-11-06)

Grades: `ours/wdsf/pose_grades_1106_hyundo_severity.json`. Minor = end joints only; limb = one arm or leg wrong, or left / right
swapped; structure = torso direction wrong or upside down. The levels were defined after the first grading.

| move (right-person frames) | n | good | minor | limb | structure | serious (limb + structure) |
| --- | --- | --- | --- | --- | --- | --- |
| power | 20 | 3 | 6 | 7 | 4 | 11 (55%) |
| footwork | 15 | 8 | 2 | 4 | 1 | 5 (33%) |
| freeze | 2 | 0 | 0 | 1 | 1 | 2 |
| toprock | 11 | 10 | 1 | 0 | 0 | 0 |
| transition | 7 | 7 | 0 | 0 | 0 | 0 |
| not dancing | 8 | 7 | 1 | 0 | 0 | 0 |
| all | 63 | 35 | 10 | 12 | 6 | 18 (29%) |

Serious, power vs toprock: 11 / 20 vs 0 / 11, Fisher p about 0.002.

## Phase 7e: pose accuracy on BRACE manual keypoints (2026-11-06)

`tools/battle/brace_pose_eval.py`, `scripts/phase7e_brace.sh`: 26,676 manual frames, 81 videos, GPU 0, 3 shards, 44 min. Input as in the
WDSF pipeline. The detection closest to the truth (highest OKS) is scored, so this is joint accuracy, not person choice. Frame ids are
1-based; alignment confirmed on 12 videos (PCK@0.05 at offset 0 / +1 / -1 from the corrected index: 0.184 / 0.118 / 0.096). The third
column of the manual files is not meaningful (BRACE README); an early scoring run used it as visibility and was fixed.

| YOLO11x-pose, conf >= 0.1 | frames | PCK@0.1, no detection = 0 | serious (PCK@0.2 < 0.5, incl. no detection) |
| --- | --- | --- | --- |
| toprock | 2,000 | 0.703 [0.647, 0.747] | 0.160 [0.113, 0.219] |
| footwork | 8,397 | 0.315 [0.283, 0.346] | 0.507 [0.461, 0.556] |
| powermove | 16,207 | 0.203 [0.188, 0.220] | 0.669 [0.641, 0.698] |

CIs: video bootstrap. The manual frames look selected towards hard moments (61% power moves, much occlusion; the README does not state
the selection), so read differences between moves and models, not an average error of BRACE or WDSF. Keypoint R-CNN with scores down
to 0.1 scored similarly here, but best-of-many matching favours a detector with many junk boxes, so the comparison below uses the old
storage threshold (0.5; `score --min_krcnn 0.5 --out _k50`).

| power moves, 16,207 frames | YOLO11x-pose | Keypoint R-CNN >= 0.5 |
| --- | --- | --- |
| PCK@0.05 | 0.110 | 0.131 |
| PCK@0.1 | 0.203 | 0.201 |
| PCK@0.2 | 0.329 | 0.297 |
| serious | 0.669 | 0.787 |
| torso direction off by > 60 deg, or no detection | 0.560 | 0.653 |
| nothing near the truth (OKS < 0.1) | 0.412 | 0.281 |

PCK@0.1 difference: power +0.002 [-0.003, +0.008], footwork +0.016 [+0.009, +0.022], toprock +0.041 [+0.023, +0.056]. Median joint error
on power moves (YOLO, detected frames, body scales): ankle 0.35, knee 0.26, wrist 0.20, shoulder 0.15; toprock 0.03-0.05 everywhere.

So YOLO does not localise power-move joints better than the old model: only the coarse structure is better, and it misses more frames.
The Technique gain of Phase 7d therefore probably comes from choosing the right person and from coarse structure, not from joint
accuracy; this was not separated directly. Fine-tuning is plan B in `experiments/PLAN_weekend_1106.md`.

## Phase 7f: weekend plan results (`experiments/PLAN_weekend_1106.md`, fixed before the runs)

### Plan A: pairwise MLAVL on Bailando codes from the new pose (`--feat vq2`, 2026-11-06 18:20 → 20:37, GPU 0, shared)

`scripts/phase7f_pair.sh`. Same folds, inner-validation epochs and strict evaluation as `--feat vq` (old pose). All rounds, battle bootstrap:

| target | rounds | acc | 95% CI | second-dancer rule | p vs rule | unseen athletes acc (rule, p vs rule) |
| --- | --- | --- | --- | --- | --- | --- |
| total (3 seeds, ensemble) | 748 | 0.544 | 0.507–0.580 | 0.544 | 0.507 | 0.740 (0.560, 0.059), n=50 |
| Technique (s0) | 637 | 0.549 | 0.508–0.591 | 0.523 | 0.179 | 0.600 (0.540, 0.325) |
| Execution (s0) | 637 | 0.516 | 0.478–0.553 | 0.493 | 0.237 | 0.640 (0.520, 0.113) |
| Vocabulary (s0) | 638 | 0.487 | 0.448–0.526 | 0.552 | 0.988 | 0.520 (0.520, 0.542) |
| Originality (s0) | 636 | 0.536 | 0.495–0.578 | 0.553 | 0.713 | 0.510 (0.490, 0.475) |
| Musicality (s0) | 638 | 0.553 | 0.511–0.597 | 0.574 | 0.796 | 0.520 (0.540, 0.607) |

**A1 fails** on all three pre-registered targets. **A2 is not met**: no target beats the rule at p < 0.05 on unseen athletes. Linear probe on the
codes, for reference: total 0.537, Technique 0.559 (rule 0.523, p = 0.097; unseen 0.720, rule 0.540, p = 0.039), Execution 0.507, Vocabulary 0.541,
Originality 0.509, Musicality 0.555. The probe's unseen Technique result is one of 6 targets on 50 rounds, not a plan criterion.

New vs old pose on the same 748 rounds, total ensemble (exploratory): all 0.544 vs 0.553, difference -0.009 [-0.048, +0.030]. So cleaner person
selection, which raised the Technique measures in Phase 7d, did not carry over to the trained model. Technique s0 0.549 vs 0.533 and Execution s0
0.516 vs 0.505 are single seeds.

Unseen athletes (50 rounds, 23 battles, exploratory): ensemble 0.740 vs old 0.640, difference +0.100 [-0.021, +0.220]. Single seeds are 0.700, 0.640,
0.560 (old 0.620, 0.600, 0.620), so the ensemble number should not be quoted alone. Hand-labelled 52 rounds: 0.769 vs old 0.788, rule 0.615. Only 2 rounds
are in both subsets. Rounds in neither subset (648): 0.512 vs old 0.529, rule 0.537. The first 22 labelled battles again stand out with either pose.
Simple rule on the same rounds, for scale: higher energy wins 0.550 (635 rounds), 0.587 unseen (46), 0.600 hand (45); longer turn wins 0.466.

### Same-battle hand vs detected turns (exploratory, closes the open item of the 2026-11-06 correction)

19 labelled battles, 102 turns, new pose (`pose_seq2`), same measure code, only the boundaries differ; 39 rounds (14 battles) have criterion scores.
Boundary IoU mean 0.83, median 0.89, 7.8% below 0.5. Check: the detected-boundary numbers reproduce `criteria_measures2.csv` exactly on these rounds.

| measure | turn-level rho, hand vs detected | round difference rho | -> criterion-specific | hand | detected | hand - detected [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| energy | 0.970 | 0.853 | Technique | -0.138 | -0.246 | +0.108 [-0.010, +0.269] |
| spin | 0.943 | 0.867 | Technique | +0.126 | +0.014 | +0.112 [-0.010, +0.294] |
| inverted | 0.971 | 0.961 | Technique | -0.246 | -0.278 | +0.032 [-0.076, +0.159] |
| jerk | 0.924 | 0.756 | Execution | +0.146 | +0.027 | +0.119 [-0.123, +0.324] |
| dur | 0.639 | 0.630 | Vocabulary | -0.018 | +0.261 | -0.280 [-0.484, -0.088] |

The motion measures hardly depend on the boundaries, so segmentation is not what limits them. More hand labels are not expected to raise them much; this
agrees with the earlier low priority of the labelling. Duration is the exception because the boundaries define it: on these 39 rounds the
dur -> Vocabulary link disappears with exact boundaries, which raises the question whether part of that link comes from detection. dur -> Vocabulary-specific by
group: first 22 battles, exact turns +0.023 (n=45); next 19, detected +0.261 (n=39), exact -0.018; other rounds, detected +0.150 (n=563). The first 22 alone are
not a clear contrast with the rest (about 1.4 standard errors), but the same-battles difference is. So dur -> Vocabulary (+0.17 in Phase 7d) is not established as
real and should not be reported as a judged-criteria result until it holds on exact turns.
These 39 rounds are small and unusual: energy, inverted are negative here, positive on all rounds, so only the hand minus detected column is used.

### Plan B: YOLO11x-pose fine-tuned on BRACE manual keypoints (2026-11-06 18:20 → 22:41, GPU 0, shared)

`scripts/phase7f_brace.sh`, one configuration, no sweep. Train 2011-2017, validation 2018, test 2020 (14 videos, 5,413 annotated frames, not used in
training or epoch choice). Training stopped at epoch 10 by patience 8, best epoch 2 on the 2018 validation fitness, about 3.6 h, well under the 8 h limit.
Pretrained YOLO on the same frames, no detection = 0, paired video bootstrap (1,000):

| 2020 test frames | n | PCK@0.1 pre → fine-tuned | difference [95% CI] | serious (PCK@0.2 < 0.5) pre → fine-tuned | torso > 60 deg or no detection | no detection near the truth (OKS < 0.1) |
| --- | --- | --- | --- | --- | --- | --- |
| power | 2,559 | 0.233 → 0.464 | **+0.231 [+0.214, +0.241]** | 0.606 → 0.190 | 0.494 → 0.056 | 0.359 → 0.020 |
| footwork | 2,338 | 0.344 → 0.469 | +0.124 [+0.094, +0.162] | 0.459 → 0.211 | 0.355 → 0.117 | 0.280 → 0.029 |
| toprock | 503 | 0.769 → 0.783 | +0.014 [-0.001, +0.040] | 0.082 → 0.056 | 0.058 → 0.022 | 0.054 → 0.004 |

**B1 passes** (at least +0.05, CI above 0), all 14 videos improved on power moves. Checks, exploratory:

- More found dancers explain part of the gain, not all: on frames both models find (power, 1,619 frames) PCK@0.1 0.364 → 0.499, PCK@0.2 0.580 → 0.739. Toprock unchanged, 0.812 → 0.808.
- Oracle matching: these scores use the box closest to the truth, which favours a model with more boxes. Boxes per frame with score >= 0.1 are 2.20 vs 1.79 on power. With the top-confidence box
  instead of the closest one: power 0.123 → 0.211, +0.088 [+0.056, +0.111]; footwork 0.131 → 0.142, +0.011 [-0.001, +0.032]; toprock 0.410 → 0.372, -0.038 [-0.142, +0.049]. The top box
  often is not the dancer in these frames (the same box as the closest in 43-55% on power), so the oracle numbers show what is available, not what a person-selection rule will deliver.
- Left-right swaps rise on power moves, on frames both find: 0.132 → 0.196. Our measures do not depend on left and right except the codes.
- Limits: 2020 dancers can appear in earlier years, so this is new event and video, not always new dancer; BRACE annotated frames appear selected towards hard moments; WDSF is a different camera setup,
  so the gain there is untested. By the plan, WDSF is re-tracked with the same selection rule (`scripts/phase7f_wdsf_ft.sh`), and human grading on new frames is 정현도's decision.

### Plan B follow-up: WDSF re-tracked with the fine-tuned model (2026-11-06 22:44 → 11-07 07:36, GPU 0, shared)

`scripts/phase7f_wdsf_ft.sh`, same tracker settings and selection rule. 201k + 197k + 202k frames at 8.2 fps per shard, 313 videos. Codebook lower val rec 0.076 (mean-pose 0.142), previous 0.086 (0.155), perplexity 438/512.
Tags: `yolo_tracks_ft`, `pose_seq2_ft`, `vq2_ft`, `criteria_measures2_ft.csv`.

Coverage mean 0.965 (was 0.945). Same selected person in 91.8% of the 560,181 samples both have (box IoU >= 0.3). Samples with pose in only one of them: 7,154 old, 19,904 new.

**Dancer signature, main setting L8 K64: AUC 0.613** (was 0.596, goal 0.7). **Not met.** So better power-move joints did not make the move units identify the dancer.

Pre-specified measures, same rounds, 647 rounds, battle bootstrap, criterion-specific part:

| measure -> criterion | pose2 | fine-tuned | difference [95% CI] |
| --- | --- | --- | --- |
| spin -> Technique | +0.215 [+0.139, +0.282] | +0.136 [+0.055, +0.207] | -0.079 [-0.154, -0.007] |
| energy -> Technique | +0.176 [+0.089, +0.256] | +0.102 [+0.014, +0.187] | -0.074 [-0.151, +0.006] |
| inverted -> Technique | +0.142 [+0.053, +0.227] | +0.217 [+0.135, +0.292] | +0.075 [+0.020, +0.133] |
| jerk -> Execution | +0.152 [+0.074, +0.231] | +0.117 [+0.037, +0.195] | -0.035 [-0.091, +0.018] |
| voc_ent -> Vocabulary | +0.047 [-0.031, +0.126] | +0.012 [-0.070, +0.090] | -0.035 [-0.125, +0.056] |
| rare_bi -> Originality | -0.028 [-0.113, +0.058] | -0.043 [-0.124, +0.044] | -0.015 [-0.075, +0.053] |
| bas -> Musicality | +0.052 [-0.025, +0.133] | +0.040 [-0.035, +0.122] | -0.012 [-0.055, +0.035] |
| sync -> Musicality | -0.056 [-0.132, +0.019] | -0.072 [-0.155, +0.003] | -0.017 [-0.090, +0.058] |

L8 K64 move units on the new pose: mv_distinct -> Vocabulary +0.077 [-0.015, +0.152], mv_ent +0.078 [-0.007, +0.160], mv_novel -0.060, mv_rare -> Originality +0.027, mv_rare_bi +0.048. Vocabulary, Originality and Musicality remain unmeasured.

**The surprise: cleaner pose lowers spin and energy, raises inverted.** Mean energy 2.92 → 2.13, spin 18.5 → 17.6, jerk 267 → 239, inverted 0.227 → 0.304; turn-level rho between the two poses is only energy 0.57, spin 0.59. Exploratory checks:

- In a joint regression on ranks (646 rounds), old pose holds the energy and spin signal and the new one adds nothing: energy old +0.165 [+0.073, +0.254], new +0.023 [-0.062, +0.114]; spin old +0.202 [+0.112, +0.293], new +0.025 [-0.078, +0.115]. Inverted the other way: old -0.080 [-0.237, +0.068], new +0.281 [+0.149, +0.431]. With only new-pose measures: inverted +0.191 [+0.103, +0.285], energy +0.058, spin +0.041, intervals include 0.
- Does a pose give the true speed? BRACE 2020 test, annotated frame pairs at most 2 frames apart, closest box, both models found (no judge scores). Power moves, 854 pairs: median speed truth 0.123, old 0.189, fine-tuned 0.115; median |error| 0.089 → 0.041; rho with truth 0.351 → 0.552. Footwork, 870: truth 0.121, 0.154, 0.132, rho 0.449 → 0.463. Toprock, 313: truth 0.081, 0.064, 0.065, rho 0.720 → 0.724.
- At turn level energy(old) minus energy(new) goes with inverted(new), rho +0.26, and with Technique-specific, rho +0.15.

Reading: on power moves the old pose overestimates speed, so the extra "energy" appeared where the pose fails. **The Phase 7d energy and spin links to Technique may therefore partly be pose error on hard moves, not speed.** Inverted, which only needs coarse body orientation, is the better supported measure. Not established: fine-tuned speed could also be too smooth at 10 Hz on WDSF, BRACE truth is 2D on selected frames, and this explanation came after seeing the result. It needs a test designed in advance.

Person selection on 72 new frames (seeds 20-22, same rule as the 2026-11-06 sample), overlay-only first grade by Claude, committed before any human grade: wrong person 4, unclear 1, no pose with the dancer visible 3 (`ours/wdsf/pose_grades_1107_claude.json`); earlier sample by the same grader: wrong 6, unclear 5, no pose 2. No real difference on this size; 정현도's grade is still needed.

### Plan B follow-up: MLAVL on fine-tuned pose, first pass (`--feat vq2_ft`, 2026-11-07, single seed except total)

Same settings and folds as plan A. Rule = "blue (second dancer) wins" on the same rounds.

| target | acc [95% CI] | rule | p(acc <= rule) | unseen athletes |
| --- | --- | --- | --- | --- |
| total (3-seed ens) | 0.549 | 0.544 | .425 | 0.660, p = .221 |
| Technique s0 | 0.584 [0.543, 0.625] | 0.523 | .027 | 0.600, p = .336 |
| Execution s0 | 0.537 | 0.493 | .077 | 0.500 |
| Vocabulary s0 | 0.525 | | | |
| Originality s0 | 0.505 | | | |
| Musicality s0 | 0.531 | | | |

Linear probes on the same input: Technique .573 (p = .066), Execution .549 (p = .038), total .528, Musicality .567.

### Plan B confirmation (addendum in `PLAN_weekend_1106.md`, fixed before the runs; `scripts/phase7f_confirm.sh`, 2026-11-07 07:45–09:25)

**Primary test passes.** `vq2_ft` Technique, 3-seed ensemble, all rounds: **0.586 [0.543, 0.627]**, rule 0.523, one-sided p = .025, n = 637, 268 battles.

| feature | Technique seeds 0, 1, 2 | Technique ens (rule .523) | Execution seeds 0, 1, 2 | Execution ens (rule .493) |
| --- | --- | --- | --- | --- |
| `vq2_ft` | .584 .554 .567 | **.586**, p = .025 | .537 .546 .564 | **.565**, p = .007 |
| `vq2` | .549 .564 .535 | .560, p = .110 | .516 .545 .515 | .543, p = .058 |
| `vq` | .533 .537 .530 | .548, p = .213 | .505 .516 .522 | .522, p = .180 |

Secondary tests, same rounds, paired battle bootstrap:

| contrast | Technique | Execution |
| --- | --- | --- |
| `vq2_ft` − `vq2`, all | +.025 [−.012, +.062], p = .095 | +.022 [−.016, +.060], p = .135 |
| `vq2_ft` − `vq`, all (not in the addendum) | +.039 [−.005, +.083], p = .046 | +.046 [+.002, +.088], p = .023 |
| `vq2_ft`, unseen athletes (n = 50) vs rule | .600, rule .540, p = .339 | .600, rule .520, p = .274 |
| `vq2_ft` − `vq2`, unseen | .000 [−.180, +.170] | −.060 [−.188, +.061] |

Reading:
- The rule passed as fixed, and the ordering `vq` < `vq2` < `vq2_ft` holds on both targets, which fits "better pose, better prediction". The step from `vq2` to `vq2_ft` alone is not significant.
- Limits: seed 0 triggered the test and is part of the ensemble; same rounds as the first pass; no correction for other targets; unseen athletes (n = 50) do not separate from the rule, so an identity shortcut is not excluded.

### Plan C: RG with YOLO11x-pose (`scripts/phase7f_rg.sh`, 12 runs, GPU 0, commit d166992, 2026-11-07 02:53–09:26)

COCO YOLO11x-pose (not the BRACE fine-tune), athlete found in 97.0 % of sampled frames (min 83.1 %), clips with pose 96.8 %. Wall time 1:12–3:04 per run at 6 parallel, RSS 1.7–1.8 GB.

| RG | reproduction | Keypoint R-CNN pose add | YOLO pose add |
| --- | --- | --- | --- |
| Ball | 0.758 / 5.52 | 0.794 / 5.19 | 0.797 ± 0.007 / 5.01 |
| Clubs | 0.749 / 5.08 | 0.766 / 4.77 | 0.757 ± 0.003 / 4.78 |
| Hoop | 0.786 / 4.73 | 0.802 / 4.54 | 0.811 ± 0.014 / 4.19 |
| Ribbon | 0.808 / 5.05 | 0.832 / 4.85 | 0.832 ± 0.033 / 4.79 |
| **avg ρ / best MSE** | **0.775 / 5.10** | **0.798 / 4.84** | **0.799 / 4.69** |

Paired t over 12 runs:
- YOLO − reproduction: ρ +0.024 [+0.007, +0.042], p = .012; best MSE −0.41 [−0.65, −0.16], p = .004.
- YOLO − Keypoint R-CNN: ρ +0.001 [−0.008, +0.010], p = .79; best MSE −0.15 [−0.30, +0.00], p = .051.

**C1 fails** (needed above 0.798 with p < .05), so by the plan no Fis-V run and no deck change. The useful part: a second, independent pose model gives the same gain over the reproduction, so the midterm's pose-add gain is not specific to Keypoint R-CNN.

### Does the `vq2_ft` Technique signal go beyond seed 0, dancer strength, simple measures? (exploratory, 2026-11-07)

- Seeds 1, 2 only, leaving out the seed that triggered the test: Technique 0.571 [0.528, 0.613], rule 0.523, p = .069. Execution 0.568 [0.531, 0.606], rule 0.493, p = .007.
- Dancer strength = blue minus red mean Technique margin at other events (outside the video). On the 584 rounds where both athletes have other events: **strength alone 0.700, model 0.572, rule 0.527.** Rank correlation model vs strength only +0.175.
- Joint rank regression on the margin, battle bootstrap: prediction alone +0.210 [+0.123, +0.296]; strength alone +0.541 [+0.451, +0.620]; **together prediction +0.119 [+0.049, +0.193], strength +0.520 [+0.429, +0.602].** So the model holds something strength does not, and much of what judges give in a round is who the dancer is.
- Simple measures on 582 rounds, fine-tuned pose: sign accuracy inverted 0.553, energy 0.572, spin 0.527, jerk 0.557, dur 0.485. Five measures in ridge on the same folds 0.550; model minus ridge +0.022 [−0.032, +0.076], p = .235. Joint: prediction +0.182 [+0.098, +0.274], ridge +0.093 [+0.003, +0.178].

### Phase 7g: 2022 holdout (second addendum, `scripts/phase7g_holdout.sh`, `ours/wdsf_holdout.py`, 2026-11-07, 6 runs in about 45 min, GPU 0)

Trained on all 637 five-slider Technique rounds, epoch chosen on the Brisbane event (epoch 5 in all six runs), predicted 111 rounds of Kitakyushu, Manchester and Seoul.

**Primary test passes.** `vq2_ft` 3-seed ensemble **0.667 [0.579, 0.750]**, rule 0.523, p(<=rule) = .029, p(<=50%) < .001, seeds .694 .613 .658.

| secondary | result |
| --- | --- |
| `vq2` 3-seed ensemble | 0.640 [0.556, 0.718], p(<=rule) = .080 |
| `vq2_ft` − `vq2` | +0.027 [−0.054, +0.108], p = .281 |
| athletes absent from training | only 2 rounds, no test possible |
| per event `vq2_ft` | Kitakyushu .675 (rule .525), Manchester .714 (rule .686), Seoul .611 (rule .361) |
| dancer strength on 81 rounds with a strength estimate | strength 0.802, model 0.679, rule 0.568; model vs strength rank corr +0.512 |
| joint rank regression, 81 rounds | prediction +0.220 [+0.056, +0.411], p = .006; strength +0.583 [+0.359, +0.739] |

**Limit:** 109 of 111 rounds have an athlete in training, so these results do not exclude a model that recognises dancers. Correlation with strength is much higher here (0.512) than in cross-validation (0.175). Strength is also a noisy estimate, so a positive joint coefficient could partly be a better estimate of the same thing.

Pose-error test on the same rounds (110 rounds, Technique minus the mean of six sliders):

| measure | old pose rho | fine-tuned rho | joint old | joint new | old − new |
| --- | --- | --- | --- | --- | --- |
| inverted | +0.209 [−0.002, +0.407] | +0.157 [−0.094, +0.375] | +0.335 [−0.029, +0.715] | −0.141 [−0.578, +0.288] | +0.477 [−0.283, +1.254] |
| energy | +0.114 [−0.117, +0.336] | −0.053 [−0.251, +0.155] | +0.222 [−0.060, +0.458] | −0.183 [−0.417, +0.079] | +0.405 [−0.081, +0.800] |
| spin | +0.118 [−0.102, +0.329] | +0.059 [−0.167, +0.279] | +0.141 [−0.135, +0.416] | −0.034 [−0.320, +0.245] | +0.176 [−0.326, +0.688] |

Expected: inverted positive on the fine-tuned pose (positive, interval includes 0), energy and spin larger on the old pose (yes in direction, intervals include 0), inverted larger on the new pose (**no, the opposite direction**, wide interval). So the holdout does not confirm "inverted is the better measure on the new pose"; energy and spin weakening on the new pose repeats, but is not established at this size.

### Within-battle test (fourth addendum, `tools/battle/within_battle.py`, 2026-11-07)

Pairs of rounds in one battle: same two dancers, constant blue rule, so identity cannot help. Does the prediction order the two rounds as the judges' margins do? Chance 50%, battle bootstrap.

**Main test passes: `vq2_ft` Technique 0.562 [0.515, 0.607], p = .006** (473 pairs, 262 battles; seeds .581, .543, .548). 2022 holdout 0.609 [0.505, 0.713], p = .022 (87 pairs).

| input, Technique | overall acc | within-battle | p |
| --- | --- | --- | --- |
| Linear, Video Swin + AST | .595 | **.595** [.546, .641] | < .001 |
| Linear, AST only | .569 | **.583** [.534, .631] | < .001 |
| Linear, `vq2_ft` codes only | .573 | .558 [.508, .607] | .012 |
| Linear, Video Swin only | .591 | .549 [.499, .600] | .030 |
| MLAVL, `vq2_ft` + AST, 3 seeds | .586 | .562 [.515, .607] | .006 |
| MLAVL, `vq` + AST, 3 seeds | .548 | .560 [.511, .608] | .007 |
| MLAVL, `vq2` + AST, 3 seeds | .560 | .532 [.482, .579] | .109 |
| MLAVL, Video Swin + AST, inner val, 1 seed | .544 | .512 [.465, .558] | .323 |
| MLAVL, Video Swin + AST, RG init, 1 seed | .556 | .539 [.488, .589] | .066 |

Other targets: `vq2_ft` MLAVL Execution 3 seeds .557 [.512, .600], p = .008; Vocabulary .521, Originality .514, Musicality .492 (seed 0), total .518 (3 seeds). Linear with Video Swin + AST: Execution .557, Vocabulary .562, Originality .469, Musicality .502. Linear AST only: Execution **.590**, Vocabulary .558, Originality .484, Musicality .548. Linear Video Swin only: Execution .515, Vocabulary .500, Originality .482, Musicality .475. Simple fine-tuned measures, Technique: inverted .555 (p = .013), energy .545 (p = .045), spin .502, jerk .509, dur .480.

Ceiling, split the judges of a round in two and compare the orderings (40 random splits): Technique half vs half **0.556**, Execution 0.634. Each half is noisier than the full panel, so a perfect predictor of the true round quality would score above this: with a Gaussian model and Spearman-Brown, about 0.68 for Technique and 0.78 for Execution. Battle means account for most of the margin variance (ratio of battle-mean variance to total 0.90), the within-battle part is small and noisy.

Reading: pose models and simple measures do hold within-battle information, so they are not only recognising dancers. Two checks stop this from being a pose result yet: MLAVL's input includes audio, and audio alone is as strong as anything. Fine-tuned pose is not clearly better than older pose here (`vq` .560, `vq2` .532).

### Where in time is the audio signal? (fifth addendum, `tools/battle/audio_windows.py`, 2026-11-07)

Blue minus red, same folds, within-battle accuracy. AST = linear on AST statistics in the window. dB = RMS loudness, louder turn higher, no fit.

| window | Technique AST | Technique dB | Execution AST | Execution dB | Musicality AST | Musicality dB |
| --- | --- | --- | --- | --- | --- | --- |
| 15 s before start | .564 (p = .004) | .471 | .548 (p = .029) | .487 | .503 | .488 |
| first half of turn | .507 | .503 | .505 | .470 | .477 | .483 |
| second half of turn | **.569** (p = .002) | **.559** (p = .015) | **.571** (p = .002) | **.561** (p = .007) | .524 | .541 (p = .061) |
| 15 s after end | .484 | .549 (p = .017) | .498 | .562 (p = .008) | **.583** (p < .001) | .509 |

- **Not a vote reveal:** AST after the end carries nothing for Technique and Execution, and the signal is already in the second half of the turn, before any vote appears. Musicality after the end (.583) is unexplained.
- **The first half has no signal, the second half does.** A turn that gets louder late goes with higher Technique and Execution, in the same pair of dancers. The likely source is the crowd reacting, then still loud after the end, but crowd and music are not separated here, so this is not shown. It may also be that judges respond to the crowd.
- **The "before" window is not a valid negative control, my design error:** blue's before-window lies in red's turn (red dances first), so it holds red's crowd. Its loudness sign fits that (rho −0.110, louder before blue goes with red winning).
- Consequence for any audio-visual model on live battles: audio is not only music fit, and an audio-only baseline must be reported.

### MLAVL without audio (fifth addendum, `scripts/phase7g_noaudio.sh`, zero audio input, `vq2_ft`, 3 seeds, 2026-11-07, about 45 min)

**Main test fails, narrowly.** Technique within-battle 0.543 [0.491, 0.593], p = .055 (with audio 0.562; difference −0.019 [−0.066, +0.027]).

| | Technique, no audio | Technique, audio | Execution, no audio | Execution, audio |
| --- | --- | --- | --- | --- |
| overall, cross-validation | .575 (rule .523, p = .041) | .586 (p = .025) | .549 (rule .493, p = .037) | .565 (p = .007) |
| within-battle | .543, p = .055 | .562, p = .006 | **.492**, p = .640 | .557, p = .008 |
| unseen athletes, n = 50 | .540 (rule .540) | .600 (rule .540) | .540 (rule .520) | .600 (rule .520) |
| 2022 holdout, overall | .640 (rule .523, p = .098) | .667 (p = .029) | | |
| 2022 holdout, within-battle | .575, p = .071 | .609, p = .022 | | |

- Execution's within-battle signal comes from audio: no audio minus audio −0.066 [−0.115, −0.017].
- Technique without audio stays above the rule overall and is positive but not significant within battle. So pose alone in MLAVL is not established as reading the performance apart from who dances. Linear pose codes (.558, p = .012) and inverted alone (.555, p = .013) reach it on the same test, without correction.

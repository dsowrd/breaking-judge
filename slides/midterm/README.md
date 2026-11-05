# Midterm deck (2026-11-10)

`presentations/2025198001_정현도_중간발표.pptx` is generated here; the template is the 2nd paper-review deck
(`presentations/2025198001_정현도_논문발표2.pptx`). Both decks live only locally and on the course Drive folder
(`presentations/` is git-ignored because the decks embed dataset clips).

```bash
python slides/midterm/render_clips.py      # anaconda base python (numpy, Pillow) + ffmpeg -> assets/clips/*.mp4
python slides/midterm/build2.py [out.pptx] [slides/midterm/ablation.json]   # needs python-pptx
```

- `lib.py`: loads the template and copies its styled shapes (panels, bars, bullets, tables).
- `build2.py`: the 7 slides. Every slide has a message line under the title; videos autoplay and loop on slide entry.
- `render_clips.py`: RG / Fis-V clips with the tracked pose (blue = upper codebook, red = lower, grey = torso link)
  and the current code ids. Inputs in `data/` (git-ignored) are copied from the server:
  `/data/mlavl/videos/{rg_raw,fisv_raw}/videos/*.mp4`, `/data/mlavl/features/{RG,FISV}/pose_det/*.npz`,
  `~/Workspace_Hyundo/render/codes_{rg,fisv}.npy` (per-group VQ code ids).
- `ablation.json`: RG ablation means (best test ρ, 3 seeds) filled into slide 7 as they finish.
- Fis-V #154 = 2014 Sochi Olympics ladies' short program (labels TES 39.03 / PCS 35.89 = Kim Yuna's official
  score; SOCHI 2014 boards and the CCTV caption in the frames). No Korean gymnast was found in RG (Son Yeon-jae's
  2015 Worlds scores do not match any label).

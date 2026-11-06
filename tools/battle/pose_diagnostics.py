"""Is 2D pose quality or turn segmentation what limits the criterion measures? (ours, Phase 7c diagnostics).

Per turn, from the raw detections (pose_est) and the BRACE-trained dancer selector (estimate_pose.learned_select):
  conf     mean keypoint score of the selected dancer
  box_h    median box height of the selected dancer (pixels; small dancer = harder)
  jump     share of consecutive selected samples where the mean joint displacement exceeds 0.5 box heights per 0.1 s
           (physically implausible: wrong person, left/right swap, lost limbs)
Rounds are split into tertiles of the worse of the two dancers (pose quality) and of the turn detector's cut
confidence (min_cut_p, segmentation quality); the Phase 7 / 7b correlations are recomputed per tertile with battle
bootstrap CIs. If better pose would help, the correlations should grow with pose quality.

    $MLAVL_PY -I tools/battle/pose_diagnostics.py quality     # -> {D}/turn_pose_quality.csv
    $MLAVL_PY -I tools/battle/pose_diagnostics.py strata
    $MLAVL_PY -I tools/battle/pose_diagnostics.py sheet OUT.jpg   # contact sheet of selected poses (visual check)
"""
import os
import pickle
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import criteria_validity as cv  # noqa: E402

F, D, KEY, C = cv.F, cv.D, cv.KEY, cv.C
V = '/data/mlavl/videos/wdsf'
FFMPEG = cv.FFMPEG
PAIRS = [('energy', 'Technique_spec'), ('jerk', 'Technique_spec'), ('dur', 'Vocabulary_spec'),
         ('mv_distinct', 'Vocabulary_spec'), ('bas', 'Musicality_spec'), ('mv_rare', 'Originality_spec')]


def _video(vid):
    from estimate_pose import learned_select
    clf = pickle.load(open('/data/mlavl/features/BRACE/dancer_selector.pkl', 'rb'))
    d = dict(np.load(f'{F}/pose_est/{vid}.npz'))
    ap = np.load(f'{F}/appearance/{vid}.npz')
    fps = float(np.load(f'{F}/clips_pamfn/{vid}.npz')['fps'])
    sel = learned_select(d, ap['box'], ap['has'], clf) if len(d['frame']) else np.zeros(0, int)
    i = np.where(sel >= 0)[0]
    t = d['frame'][i] / fps
    kp = d['kp'][i, sel[i]]
    box = d['box'][i, sel[i]]
    return vid, dict(t=t, conf=kp[:, :, 2].mean(1), box_h=box[:, 3], kp=kp[:, :, :2], frame=d['frame'][i],
                     fps=fps, sel=sel, all_box=d['box'], all_det=d['det_score'], all_frame=d['frame'])


def per_video():
    turns = pd.read_csv(f'{D}/wdsf_turns.csv')
    with Pool(6) as p:
        return dict(p.imap_unordered(_video, sorted(turns.video_id.unique())))


def quality():
    pv = per_video()
    rows = []
    for t in pd.read_csv(f'{D}/wdsf_turns.csv').itertuples():
        q = pv[t.video_id]
        m = (q['t'] >= t.start_s) & (q['t'] < t.end_s)
        row = dict(video_id=t.video_id, competition=t.competition, battle_id=t.battle_id, round=t.round,
                   order=t.order, athlete=t.athlete, min_cut_p=t.min_cut_p, n_pose=int(m.sum()))
        if m.sum() >= 10:
            kp, h, tt = q['kp'][m], np.maximum(q['box_h'][m], 1), q['t'][m]
            dt = np.diff(tt)
            disp = np.linalg.norm(np.diff(kp, axis=0), axis=-1).mean(1) / h[1:]
            ok = dt < 0.5
            row.update(conf=float(q['conf'][m].mean()), box_h=float(np.median(h)),
                       jump=float(np.mean(disp[ok] / dt[ok] * 0.1 > 0.5)) if ok.any() else np.nan,
                       pose_hz=float(m.sum() / max(t.end_s - t.start_s, 1e-6)))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f'{D}/turn_pose_quality.csv', index=False)
    print(df[['conf', 'box_h', 'jump', 'pose_hz', 'min_cut_p']].describe().T.round(3).to_string())
    print(df[['conf', 'box_h', 'jump']].corr('spearman').round(2).to_string())


def round_table():
    d = cv.rounds()
    mv = pd.read_csv(f'{D}/move_measures_L8_K64.csv')
    per = mv.groupby(KEY + ['athlete'])[['mv_distinct', 'mv_rare']].mean().reset_index()
    b, r = per[per.athlete == 'blue'].set_index(KEY), per[per.athlete == 'red'].set_index(KEY)
    both = b.index.intersection(r.index)
    d = d.merge((b.loc[both, ['mv_distinct', 'mv_rare']] - r.loc[both, ['mv_distinct', 'mv_rare']]).reset_index(),
                on=KEY, how='left')
    q = pd.read_csv(f'{D}/turn_pose_quality.csv')
    g = q.groupby(KEY).agg(conf=('conf', 'min'), box_h=('box_h', 'min'), jump=('jump', 'max'),
                           seg=('min_cut_p', 'min')).reset_index()
    return d.merge(g, on=KEY, how='left')


def strata():
    d = round_table()
    print(f'rounds {len(d)}; quality proxies (worse dancer of the round) Spearman:')
    print(d[['conf', 'box_h', 'jump', 'seg']].corr('spearman').round(2).to_string())
    for col, good_high in [('conf', True), ('jump', False), ('box_h', True), ('seg', True)]:
        x = d[col] if good_high else -d[col]
        d['tier'] = pd.qcut(x.rank(method='first'), 3, labels=['low', 'mid', 'high'])
        print(f'\n== split by {col} ({"higher" if good_high else "lower"} = better), tertiles low / mid / high quality')
        for m, t in PAIRS:
            cells = []
            for tier in ['low', 'mid', 'high']:
                s = d[d.tier == tier].dropna(subset=[m, t]).reset_index(drop=True)
                e, lo, hi, _ = cv.boot(s, m, t, n=500)
                cells.append(f'{e:+.3f} [{lo:+.2f},{hi:+.2f}] n={len(s)}')
            print(f'  {m:11s} -> {t:16s} ' + ' | '.join(cells))


def sheet(out, n=24, seed=0):
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(seed)
    turns = pd.read_csv(f'{D}/wdsf_turns.csv')
    pick = turns.sample(n, random_state=seed)
    tiles = []
    E = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12)]
    for t in pick.itertuples():
        _, q = _video(t.video_id)
        m = np.where((q['t'] >= t.start_s) & (q['t'] < t.end_s))[0]
        if not len(m):
            continue
        j = m[rng.integers(len(m))]
        ts = q['frame'][j] / q['fps']
        raw = subprocess.run([FFMPEG, '-v', 'error', '-ss', f'{ts:.3f}', '-i', f'{V}/{t.video_id}.mp4', '-frames:v', '1',
                              '-f', 'image2pipe', '-vcodec', 'png', '-'], capture_output=True).stdout
        import io
        im = Image.open(io.BytesIO(raw)).convert('RGB')
        sc = im.width / 1920.0                     # stored coordinates are 1920x1080-equivalent (estimate_pose.py)
        dr = ImageDraw.Draw(im)
        fi = np.where(q['all_frame'] == q['frame'][j])[0][0]
        for b, s in zip(q['all_box'][fi], q['all_det'][fi]):
            if s > 0.5:
                dr.rectangle([b[0] * sc, b[1] * sc, (b[0] + b[2]) * sc, (b[1] + b[3]) * sc], outline=(160, 160, 160), width=2)
        kp = q['kp'][j] * sc
        for a, b in E:
            col = (60, 120, 255) if a < 11 and b < 11 else (255, 70, 70) if a >= 11 and b >= 11 else (200, 200, 200)
            if a in (5, 6, 7, 9) or b in (5, 7, 9):
                col = (60, 200, 255) if a in (5, 7, 9) else col
            dr.line([tuple(kp[a]), tuple(kp[b])], fill=col, width=4)
        for k, (x, y) in enumerate(kp):
            c = (0, 255, 0) if k in (5, 7, 9, 11, 13, 15) else (255, 200, 0)   # green = left joints, yellow = right
            dr.ellipse([x - 5, y - 5, x + 5, y + 5], fill=c)
        dr.text((10, 10), f'{t.video_id} {ts:.1f}s conf {q["conf"][j]:.2f}', fill=(255, 255, 0))
        tiles.append(im.resize((480, 270)))
    W = 4
    H = (len(tiles) + W - 1) // W
    canvas = Image.new('RGB', (480 * W, 270 * H))
    for k, im in enumerate(tiles):
        canvas.paste(im, ((k % W) * 480, (k // W) * 270))
    canvas.save(out, quality=85)
    print('wrote', out, len(tiles))


if __name__ == '__main__':
    {'quality': quality, 'strata': strata, 'sheet': lambda: sheet(sys.argv[2])}[sys.argv[1]]()

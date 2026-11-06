"""Frames for grading the Phase 7d dancer selection on fresh random samples (ours, server side; no labels are read).

A random turn, then a random 10 Hz grid point of that turn. Unlike the earlier sheets the sample is NOT restricted to
points with a detection, so "no pose" counts in the denominator. For every frame: raw image, the same image with the
stored pose (green box and skeleton = selected dancer, white boxes = every tracked person), and 4 context images at
-1, -0.5, +0.5, +1 s. The selection is read back from pose_seq2 + yolo_tracks, i.e. exactly what the pipeline stored.

    $MLAVL_PY -I tools/battle/pose_grade_sample.py OUTDIR [seed ...]      # default seeds 10 11 12, 24 frames each
"""
import io
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wdsf_track_pose as tp  # noqa: E402

F, V, FFMPEG, W, H, HZ = tp.F, tp.V, tp.FFMPEG, tp.W, tp.H, tp.HZ
E = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12)]
CONTEXT = [-1.0, -0.5, 0.5, 1.0]
N = 24


def grab(vid, ts):
    raw = subprocess.run([FFMPEG, '-v', 'error', '-ss', f'{max(ts, 0):.3f}', '-i', f'{V}/{vid}.mp4', '-frames:v', '1', '-vf',
                          f'scale={W}:{H}', '-f', 'image2pipe', '-vcodec', 'png', '-'], capture_output=True).stdout
    return Image.open(io.BytesIO(raw)).convert('RGB') if raw else Image.new('RGB', (W, H))


def main(out, seeds):
    os.makedirs(out, exist_ok=True)
    tab = tp.turns_table()
    tab = tab[[os.path.exists(f'{F}/pose_seq2/{v}.npz') for v in tab.video_id]]
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for r in tab.sample(N, random_state=seed).itertuples():
            g = int(rng.integers(int(r.start_s * HZ), int(r.end_s * HZ)))
            q = np.load(f'{F}/pose_seq2/{r.video_id}.npz')
            has = bool(q['ok'][g]) if g < len(q['ok']) else False
            ts = (g + 0.5) / HZ
            z = np.load(f'{F}/yolo_tracks/{r.video_id}.npz')
            m = np.where(z['turn'] == r.turn)[0]
            near = m[np.abs(z['t'][m] - ts) <= 0.051]
            if has:                                    # draw at the detection's own time, so the skeleton fits the image
                c = near[z['tid'][near] == q['tid'][g]]
                has = bool(len(c))
            if has:
                c = c[np.argmin(np.abs(z['t'][c] - ts))]
                ts = float(z['t'][c])
                near = near[z['t'][near] == z['t'][c]]
            im = grab(r.video_id, ts)
            ov = im.copy()
            dr = ImageDraw.Draw(ov)
            for b in z['box'][near]:
                dr.rectangle(list(b), outline=(255, 255, 255), width=2)
            if has:
                kp, b = z['kp'][c], z['box'][c]
                dr.rectangle(list(b), outline=(0, 255, 0), width=5)
                for x, y in E:
                    dr.line([tuple(kp[x, :2]), tuple(kp[y, :2])], fill=(60, 170, 255) if y <= 10 else (255, 60, 60), width=5)
            k = len(rows)
            im.resize((800, 450)).save(f'{out}/{k:02d}_raw.jpg', quality=80)
            ov.resize((800, 450)).save(f'{out}/{k:02d}_pose.jpg', quality=80)
            for i, d in enumerate(CONTEXT):
                grab(r.video_id, ts + d).resize((320, 180)).save(f'{out}/{k:02d}_c{i}.jpg', quality=70)
            rows.append(dict(k=k, seed=seed, turn=int(r.turn), video_id=r.video_id, ts=round(ts, 2), has_pose=has,
                             start_s=float(r.start_s), end_s=float(r.end_s), n_people=int(len(near)),
                             conf=float(q['conf'][g]) if has else None))
            print(k, r.video_id, f'{ts:.1f}', has, flush=True)
    # shuffled so that frames of one seed do not sit together; the same order for every grader
    order = np.random.default_rng(0).permutation(len(rows)).tolist()
    json.dump([rows[i] for i in order], open(f'{out}/frames.json', 'w'), indent=0)
    print('wrote', len(rows), 'no pose', sum(not r['has_pose'] for r in rows))


if __name__ == '__main__':
    main(sys.argv[1], [int(s) for s in sys.argv[2:]] or [10, 11, 12])

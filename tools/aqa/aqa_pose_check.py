"""Visual check of the athlete chosen by aqa_pose.py track on RG / Fis-V (ours, Phase 7c): random frames with the chosen
skeleton (blue = upper, red = lower, grey = torso) and every stored person box (white), graded by eye.

    $MLAVL_PY -I tools/aqa/aqa_pose_check.py rg 0 OUT.jpg     # set, seed, output; 24 random frames
"""
import glob
import io
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

FFMPEG = os.environ.get('FFMPEG', '/home/irl/anaconda3/bin/ffmpeg')
SETS = {'fisv': ('/data/mlavl/videos/fisv_raw/videos', '/data/mlavl/features/FISV'),
        'rg': ('/data/mlavl/videos/rg_raw/videos', '/data/mlavl/features/RG')}
E = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12)]


def main(set_, seed, out, n=24):
    vids, feat = SETS[set_]
    rng = np.random.default_rng(seed)
    fs = sorted(glob.glob(f'{feat}/pose_seq/*.npz'))
    tiles = []
    for f in rng.choice(fs, n, replace=False):
        vid = os.path.basename(f)[:-4]
        z = np.load(f)
        d = np.load(f'{feat}/pose_det/{vid}.npz')
        i = int(rng.integers(len(z['ok'])))
        fr = int(z['frame'][i])
        path = next(f'{vids}/{vid}.{e}' for e in ('mp4', 'mov', 'avi', 'mkv') if os.path.exists(f'{vids}/{vid}.{e}'))
        raw = subprocess.run([FFMPEG, '-v', 'error', '-ss', f'{fr / float(z["fps"]):.3f}', '-i', path, '-frames:v', '1',
                              '-f', 'image2pipe', '-vcodec', 'png', '-'], capture_output=True).stdout
        im = Image.open(io.BytesIO(raw)).convert('RGB')
        dr = ImageDraw.Draw(im)
        j = np.where(d['frame'] == fr)[0]
        kp_sel = None
        if len(j):
            j = j[0]
            valid = np.where(d['det_score'][j] > 0)[0]
            for b in d['box'][j][valid]:
                dr.rectangle([b[0], b[1], b[0] + b[2], b[1] + b[3]], outline=(255, 255, 255), width=2)
            if z['ok'][i] and len(valid):
                # recover the chosen person: the candidate whose hip-relative, box-normalised joints match rel
                kp = d['kp'][j][valid][:, :, :2]
                hip = kp[:, 11:13].mean(1)
                r = (kp - hip[:, None]) / np.maximum(d['box'][j][valid][:, 3], 1)[:, None, None]
                c = np.abs(r - z['rel'][i]).mean((1, 2)).argmin()
                kp_sel = kp[c]
        if kp_sel is not None:
            for a, b in E:
                col = (60, 170, 255) if a <= 10 and b <= 10 else (255, 60, 60) if a >= 11 and b >= 11 else (200, 200, 200)
                dr.line([tuple(kp_sel[a]), tuple(kp_sel[b])], fill=col, width=max(3, im.width // 300))
            for x, y in kp_sel:
                dr.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 220, 0))
        im = im.resize((480, 270))
        ImageDraw.Draw(im).text((5, 5), f'{vid} f{fr}' + ('' if kp_sel is not None else ' NO POSE'), fill=(255, 255, 0))
        tiles.append(im)
    W = 4
    canvas = Image.new('RGB', (480 * W, 270 * ((n + W - 1) // W)))
    for k, im in enumerate(tiles):
        canvas.paste(im, ((k % W) * 480, (k // W) * 270))
    canvas.save(out, quality=85)
    print('wrote', out)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])

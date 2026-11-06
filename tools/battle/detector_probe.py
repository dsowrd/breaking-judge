"""Can a different person detector find the dancer where Keypoint R-CNN failed? (ours, Phase 7c follow-up).

The frames are the 72 sampled by pose_diagnostics.sheet (seeds 0, 1, 2; graded by eye on 2026-11-06). FAIL lists the
tiles where the selected person was not the dancer or no pose existed. For those frames we draw every person found by
  krcnn_lo   torchvision Keypoint R-CNN (the current detector) with score >= 0.2 instead of 0.5
  krcnn_rot  the same model on the frame rotated by 180 degrees (inverted dancers become upright), score >= 0.5
  yolo       Ultralytics YOLO11x-pose, conf >= 0.25
and grade by eye whether the dancer has a person detection with a usable skeleton.

    CUDA_VISIBLE_DEVICES=0 $MLAVL_PY -I tools/battle/detector_probe.py OUTDIR
"""
import io
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_diagnostics as pdg  # noqa: E402

D, V, FFMPEG = pdg.D, pdg.V, pdg.FFMPEG
FAIL = {0: [1, 4, 5, 10, 11, 14, 16, 17, 22, 24], 1: [2, 6, 12, 15, 16, 18, 23], 2: [1, 2, 4, 10, 18]}
E = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12)]
WEIGHTS = '/data/mlavl/cache/ultralytics'


def frames(seed, n=24):
    """the same frames pose_diagnostics.sheet drew for this seed (tile k = list index k-1)"""
    rng = np.random.default_rng(seed)
    turns = pd.read_csv(f'{D}/wdsf_turns.csv')
    out = []
    for t in turns.sample(n, random_state=seed).itertuples():
        _, q = pdg._video(t.video_id)
        m = np.where((q['t'] >= t.start_s) & (q['t'] < t.end_s))[0]
        if not len(m):
            continue
        j = m[rng.integers(len(m))]
        out.append((t.video_id, float(q['frame'][j] / q['fps'])))
    return out


def grab(vid, ts):
    raw = subprocess.run([FFMPEG, '-v', 'error', '-ss', f'{ts:.3f}', '-i', f'{V}/{vid}.mp4', '-frames:v', '1',
                          '-f', 'image2pipe', '-vcodec', 'png', '-'], capture_output=True).stdout
    return Image.open(io.BytesIO(raw)).convert('RGB')


def draw(im, people, title):
    im = im.copy()
    dr = ImageDraw.Draw(im)
    for box, kp, s in people:
        dr.rectangle(list(box), outline=(255, 255, 255), width=2)
        for a, b in E:
            dr.line([tuple(kp[a]), tuple(kp[b])], fill=(60, 220, 255) if a < 11 else (255, 70, 70), width=4)
        dr.text((box[0], box[1] - 12), f'{s:.2f}', fill=(255, 255, 0))
    dr.rectangle([0, 0, 420, 22], fill=(0, 0, 0))
    dr.text((6, 5), title, fill=(255, 255, 0))
    return im.resize((640, 360))


def main(outdir):
    import torch
    import torchvision
    from torchvision.transforms.functional import to_tensor
    from ultralytics import YOLO
    os.makedirs(outdir, exist_ok=True)
    dev = 'cuda'
    kr = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights='DEFAULT').eval().to(dev)
    os.makedirs(WEIGHTS, exist_ok=True)
    yolo = YOLO(os.path.join(WEIGHTS, 'yolo11x-pose.pt'))

    def krcnn(im, thr, rot=False):
        x = to_tensor(im.rotate(180) if rot else im).to(dev)
        with torch.no_grad():
            r = kr([x])[0]
        out = []
        for b, k, s in zip(r['boxes'].cpu().numpy(), r['keypoints'].cpu().numpy(), r['scores'].cpu().numpy()):
            if s < thr:
                continue
            k = k[:, :2]
            if rot:
                Wd, Ht = im.size
                b = np.array([Wd - b[2], Ht - b[3], Wd - b[0], Ht - b[1]])
                k = np.stack([Wd - k[:, 0], Ht - k[:, 1]], 1)
            out.append((b, k, float(s)))
        return out

    def yolo_people(im):
        r = yolo.predict(np.array(im)[:, :, ::-1], conf=0.25, verbose=False, device=0)[0]
        return [(b, k, float(s)) for b, k, s in zip(r.boxes.xyxy.cpu().numpy(), r.keypoints.xy.cpu().numpy(),
                                                    r.boxes.conf.cpu().numpy())]

    rows = []
    for seed, tiles in FAIL.items():
        fr = frames(seed)
        for k in tiles:
            vid, ts = fr[k - 1]
            im = grab(vid, ts)
            panels = [draw(im, krcnn(im, 0.2), f's{seed}#{k} krcnn>=0.2'),
                      draw(im, krcnn(im, 0.5, rot=True), f's{seed}#{k} krcnn rot180'),
                      draw(im, yolo_people(im), f's{seed}#{k} yolo11x-pose')]
            row = Image.new('RGB', (640 * 3, 360))
            for i, p in enumerate(panels):
                row.paste(p, (640 * i, 0))
            rows.append(row)
            print(seed, k, vid, f'{ts:.1f}', flush=True)
    for s in range(0, len(rows), 6):
        sheet = Image.new('RGB', (640 * 3, 360 * len(rows[s:s + 6])))
        for i, r in enumerate(rows[s:s + 6]):
            sheet.paste(r, (0, 360 * i))
        sheet.save(os.path.join(outdir, f'probe_{s // 6}.jpg'), quality=85)
    print('wrote', len(rows), 'rows')


if __name__ == '__main__':
    main(sys.argv[1])

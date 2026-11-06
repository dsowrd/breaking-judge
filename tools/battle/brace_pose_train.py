"""Fine-tune YOLO11x-pose on BRACE's manual keypoints (ours, Phase 7f, plan B in experiments/PLAN_weekend_1106.md).

Split by year, so no video or event overlaps: test 2020, validation 2018, train 2011-2017. The dancer's box comes from the manual
joints + 10%. Other people in the frame are not annotated, so pretrained detections with score >= 0.5 and IoU < 0.3 with the dancer
become pseudo labels (joints with score >= 0.5 visible, others absent); otherwise training would teach the detector to ignore people.

    $MLAVL_PY -I tools/battle/brace_pose_train.py prepare [--shard k --nshards n]    # frames + labels, needs pose_eval/{vid}.npz
    CUDA_VISIBLE_DEVICES=0 $MLAVL_PY -I tools/battle/brace_pose_train.py train
"""
import argparse
import io
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brace_pose_eval as bp  # noqa: E402

ROOT = '/data/mlavl/frames/brace_manual'
RUNS = '/data/mlavl/ckpt_live/yolo_brace'
FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
SPLIT = {2020: 'test', 2018: 'val'}


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(x2 - x1, 0) * max(y2 - y1, 0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-6)


def line(box, kp):
    W, H = bp.W, bp.H
    x1, y1, x2, y2 = np.clip(box[0], 0, W), np.clip(box[1], 0, H), np.clip(box[2], 0, W), np.clip(box[3], 0, H)
    v = [0, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H]
    for x, y, s in kp:
        v += [x / W, y / H, s] if (s > 0 and 0 <= x < W and 0 <= y < H) else [0, 0, 0]
    return '0 ' + ' '.join(f'{u:.5f}' for u in v[1:])


def prepare(a):
    from PIL import Image
    info = pd.read_csv(f'{bp.B}/repo/videos_info.csv').set_index('video_id')
    z, idx = bp.gt_index()
    vids = sorted(idx)[a.shard::a.nshards]
    n = 0
    for vid in vids:
        split = SPLIT.get(int(info.loc[vid, 'year']), 'train')
        if split == 'test':
            continue
        os.makedirs(f'{ROOT}/images/{split}', exist_ok=True)
        os.makedirs(f'{ROOT}/labels/{split}', exist_ok=True)
        d = np.load(f'{bp.OUT}/{vid}.npz')
        y = np.where(d['model'] == 0)[0]
        fr_y = d['frame'][y]
        names = dict(idx[vid])
        fps = float(info.loc[vid, 'fps'])
        for r in bp.runs(sorted(names)):
            keep = set(r)
            for fr, im in bp.read_run(vid, fps, r[0] - 1, r[-1] - 1):
                f = fr + 1
                if f not in keep:
                    continue
                gt = np.load(io.BytesIO(z.read(names[f])))['coco_joints2d'][:, :2].astype(np.float32) * bp.SC
                m = 0.1 * max(np.ptp(gt[:, 0]), np.ptp(gt[:, 1]))
                box = np.array([gt[:, 0].min() - m, gt[:, 1].min() - m, gt[:, 0].max() + m, gt[:, 1].max() + m])
                rows = [line(box, np.c_[gt, np.full(17, 2.0)])]
                for j in y[fr_y == f]:
                    if d['score'][j] >= 0.5 and iou(box, d['box'][j]) < 0.3:
                        k = d['kp'][j].copy()
                        k[:, 2] = np.where(k[:, 2] >= 0.5, 2.0, 0.0)
                        rows.append(line(d['box'][j], k))
                base = f'{vid}_{f:06d}'
                Image.fromarray(im[:, :, ::-1]).save(f'{ROOT}/images/{split}/{base}.jpg', quality=90)
                open(f'{ROOT}/labels/{split}/{base}.txt', 'w').write('\n'.join(rows) + '\n')
                n += 1
        print(vid, split, n, flush=True)


def train(a):
    from ultralytics import YOLO
    y = f'{ROOT}/brace.yaml'
    open(y, 'w').write(f'path: {ROOT}\ntrain: images/train\nval: images/val\nkpt_shape: [17, 3]\nflip_idx: {FLIP}\nnames:\n  0: person\n')
    YOLO(bp.WEIGHTS).train(data=y, imgsz=960, epochs=40, time=8, patience=8, batch=12, device=0, workers=4, freeze=10, optimizer='SGD',
                           lr0=0.002, cos_lr=True, degrees=45, fliplr=0.5, mosaic=0.5, close_mosaic=5, project=RUNS, name='ft1',
                           exist_ok=True, seed=0, plots=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['prepare', 'train'])
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1)
    a = p.parse_args()
    {'prepare': prepare, 'train': train}[a.cmd](a)

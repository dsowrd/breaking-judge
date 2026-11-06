"""How accurate is the pose on breaking? YOLO11x-pose and Keypoint R-CNN against BRACE's manual keypoints (ours, Phase 7e).

BRACE (Moltisanti et al., ECCV 2022) gives 26,676 manually annotated frames (one dancer, COCO-17, 1920x1080 video). Frames are
fed as in the WDSF pipeline: 1280x720, YOLO imgsz 960 conf 0.1; Keypoint R-CNN at 960x540. All detections are stored, and the
detection with the highest OKS is scored: this measures the joints of the right person, not the choice of the person.

    CUDA_VISIBLE_DEVICES=0 $MLAVL_PY -I tools/battle/brace_pose_eval.py run [--shard k --nshards n] [--videos a,b] [--offset 0]
    $MLAVL_PY -I tools/battle/brace_pose_eval.py score [--min_krcnn 0.5 --out _k50]
"""
import argparse
import glob
import io
import json
import os
import re
import subprocess
import time
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd

B = '/data/mlavl/datasets/brace'
V = '/data/mlavl/videos/brace'
OUT = '/data/mlavl/features/BRACE/pose_eval'
WEIGHTS = '/data/mlavl/cache/ultralytics/yolo11x-pose.pt'
FFMPEG = os.environ.get('FFMPEG', '/home/irl/anaconda3/bin/ffmpeg')
W, H = 1280, 720
SC = W / 1920.0
SIG = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10


def gt_index():
    z = zipfile.ZipFile(f'{B}/manual_keypoints.zip')
    g = defaultdict(list)
    for x in z.namelist():
        if x.endswith('.npz'):
            p = x.split('/')
            g[p[2]].append((int(re.search(r'(\d+)\.npz', p[3]).group(1)), x))
    return z, {k: sorted(v) for k, v in g.items()}


def runs(frames, gap=50):
    out, cur = [], [frames[0]]
    for f in frames[1:]:
        if f - cur[-1] > gap:
            out.append(cur)
            cur = []
        cur.append(f)
    return out + [cur]


def read_run(vid, fps, first, last):
    """frames first..last (0-based decoding index) at W x H"""
    n = last - first + 1
    p = subprocess.Popen([FFMPEG, '-v', 'error', '-ss', f'{first / fps:.5f}', '-i', f'{V}/{vid}.mp4', '-frames:v', str(n), '-vf',
                          f'scale={W}:{H}', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], stdout=subprocess.PIPE,
                         stdin=subprocess.DEVNULL)
    sz = W * H * 3
    for k in range(n):
        b = p.stdout.read(sz)
        if len(b) < sz:
            break
        yield first + k, np.frombuffer(b, np.uint8).reshape(H, W, 3)
    p.stdout.close()
    p.wait()


def run(a):
    import torch
    import torchvision
    from ultralytics import YOLO
    os.makedirs(OUT, exist_ok=True)
    info = pd.read_csv(f'{B}/repo/videos_info.csv').set_index('video_id')
    _, idx = gt_index()
    vids = sorted(idx)
    if a.videos:
        vids = a.videos.split(',')
    vids = vids[a.shard::a.nshards]
    yolo = YOLO(a.weights or WEIGHTS)
    kr = None if a.weights else torchvision.models.detection.keypointrcnn_resnet50_fpn(weights='DEFAULT').eval().to('cuda')
    t0, done = time.time(), 0
    for iv, vid in enumerate(vids):
        dst = f'{OUT}/{vid}{a.tag}.npz'
        if os.path.exists(dst):
            continue
        fps = float(info.loc[vid, 'fps'])
        want = [f for f, _ in idx[vid]]
        rec = {k: [] for k in ('frame', 'model', 'box', 'kp', 'score')}
        for r in runs(want):
            keep = set(r)
            for fr, im in read_run(vid, fps, r[0] + a.offset, r[-1] + a.offset):
                f = fr - a.offset
                if f not in keep:
                    continue
                y = yolo.predict(im, conf=0.1, iou=0.7, imgsz=960, verbose=False, device=0)[0]
                if len(y.boxes):
                    kp = np.concatenate([y.keypoints.xy.cpu().numpy(), y.keypoints.conf.cpu().numpy()[..., None]], -1)
                    n = len(kp)
                    rec['frame'].append(np.full(n, f))
                    rec['model'].append(np.zeros(n, np.int8))
                    rec['box'].append(y.boxes.xyxy.cpu().numpy())
                    rec['kp'].append(kp)
                    rec['score'].append(y.boxes.conf.cpu().numpy())
                if kr is None:
                    done += 1
                    continue
                x = torch.from_numpy(np.ascontiguousarray(im[::2, ::2, ::-1])).permute(2, 0, 1).float().div(255).to('cuda')
                with torch.no_grad():
                    q = kr([x])[0]
                s = q['scores'].cpu().numpy()
                m = s >= 0.1
                if m.any():
                    n = int(m.sum())
                    kk = q['keypoints'][m].cpu().numpy().astype(np.float32)
                    kk[..., :2] *= 2                 # back to 1280 x 720
                    kk[..., 2] = q['keypoints_scores'][m].cpu().numpy()
                    rec['frame'].append(np.full(n, f))
                    rec['model'].append(np.ones(n, np.int8))
                    rec['box'].append(q['boxes'][m].cpu().numpy() * 2)
                    rec['kp'].append(kk)
                    rec['score'].append(s[m])
                done += 1
        np.savez_compressed(dst, **{k: np.concatenate(v) if v else np.zeros((0,)) for k, v in rec.items()})
        print(f'[{iv + 1}/{len(vids)}] {vid}: {done} frames, {done / (time.time() - t0):.1f} fps', flush=True)


def metrics(gt, kp):
    """gt (17,3) and kp (n,17,3), 1280x720 px. Returns per detection OKS, and per joint normalised distance"""
    vis = gt[:, :2].sum(1) > 0     # BRACE README: the third column is not meaningful
    xy = gt[vis, :2]
    w, h = np.ptp(xy[:, 0]), np.ptp(xy[:, 1])
    scale = max(w, h, 1.0)
    area = max(w * h, 1.0)
    d2 = ((kp[:, :, :2] - gt[None, :, :2]) ** 2).sum(-1)
    e = np.exp(-d2 / (2 * area * (2 * SIG) ** 2 + 1e-9))
    return e[:, vis].mean(1), np.sqrt(d2) / scale, vis


def torso(k):
    return k[5:7, :2].mean(0) - k[11:13, :2].mean(0)


def score(a):
    z, idx = gt_index()
    seg = pd.read_csv(f'{B}/repo/annotations/segments.csv')
    rows = []
    for vid, items in idx.items():
        f = f'{OUT}/{vid}{a.tag}.npz'
        if not os.path.exists(f):
            continue
        d = np.load(f)
        s = seg[seg.video_id == vid]
        for fr, name in items:
            gt = np.load(io.BytesIO(z.read(name)))['coco_joints2d'].astype(np.float32)
            gt = np.c_[gt[:, :2] * SC, gt[:, 2]]
            t = s[(s.start_frame <= fr) & (s.end_frame >= fr)]
            typ = t.dance_type.iloc[0] if len(t) else 'none'
            m = np.where(d['frame'] == fr)[0] if len(d['frame']) else np.zeros(0, int)
            for mi, mname in enumerate(['yolo', 'krcnn']):
                c = m[(d['model'][m] == mi) & (d['score'][m] >= (a.min_yolo, a.min_krcnn)[mi])] if len(m) else m
                row = dict(video=vid, frame=fr, type=typ, model=mname, found=False)
                if len(c):
                    oks, dist, vis = metrics(gt, d['kp'][c])
                    b = int(np.argmax(oks))
                    row.update(found=bool(oks[b] >= 0.1), oks=float(oks[b]), score=float(d['score'][c][b]))
                    dv = dist[b][vis]
                    for th in (0.05, 0.1, 0.2):
                        row[f'pck{int(th * 100):02d}'] = float((dv < th).mean())
                    k = d['kp'][c][b]
                    tg, tp = torso(gt), torso(k)
                    cos = (tg @ tp) / (np.linalg.norm(tg) * np.linalg.norm(tp) + 1e-9)
                    row['torso_deg'] = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
                    sw = list(range(17))
                    for x, y in [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]:
                        sw[x], sw[y] = y, x
                    sd = np.linalg.norm(k[sw, :2] - gt[:, :2], axis=-1)[vis]
                    nd = np.linalg.norm(k[:, :2] - gt[:, :2], axis=-1)[vis]
                    row['lr_swapped'] = bool(sd.mean() < 0.8 * nd.mean())
                    row['joint_d'] = json.dumps(np.round(np.where(vis, dist[b], np.nan), 3).tolist())
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/../pose_eval{a.tag}{a.out}.csv', index=False)
    print(len(df) // 2, 'frames', df[df.model == 'yolo'].video.nunique(), 'videos')
    for m, g in df.groupby('model'):
        print('==', m)
        print(table(g).to_string())


def table(g):
    def one(s):
        ok = s[s.found]
        return pd.Series(dict(frames=len(s), found=s.found.mean(), oks=ok.oks.mean(), pck05=ok.pck05.mean(), pck10=ok.pck10.mean(),
                              pck20=ok.pck20.mean(), broken=(ok.pck20 < 0.5).mean(), torso60=(ok.torso_deg > 60).mean(),
                              lr=ok.lr_swapped.mean()))
    t = g.groupby('type').apply(one, include_groups=False)
    t.loc['all'] = one(g)
    return t.round(3)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['run', 'score'])
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1)
    p.add_argument('--videos', default=None)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--tag', default='')
    p.add_argument('--out', default='')
    p.add_argument('--weights', default=None)                 # fine-tuned YOLO; skips Keypoint R-CNN
    p.add_argument('--min_yolo', type=float, default=0.1)     # detections below this score do not exist for score
    p.add_argument('--min_krcnn', type=float, default=0.1)    # 0.5 = what the old pipeline stored
    a = p.parse_args()
    {'run': run, 'score': score}[a.cmd](a)

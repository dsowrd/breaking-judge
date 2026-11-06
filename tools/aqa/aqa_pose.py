"""2D pose of the athlete in the RG / Fis-V raw videos (ours, Phase 6b: pose-based choreographic memory on the paper
datasets).

Why: the feature-level memory (ours/choreo_feat.py) re-encodes the Video Swin features MLAVL already sees and adds
nothing (RG 0.773 vs 0.775). Bailando's memory is a vocabulary of *poses*; RG / Fis-V ship no poses, so they are
estimated here from the raw videos (Fis-V: MS_LSTM Google Drive; RG: ACTION-NET OneDrive; research use only, never
redistributed).

    CUDA_VISIBLE_DEVICES=0 $MLAVL_PY -I tools/aqa/aqa_pose.py detect --set fisv [--shard k --nshards n]
    $MLAVL_PY -I tools/aqa/aqa_pose.py track --set fisv     # one athlete per frame -> pose_seq/{id}.npz

detect: torchvision Keypoint R-CNN (COCO-17, as for the battles) on every `stride`-th frame at 960x540; keeps up to
MAXP persons per frame -> {out}/pose_det/{id}.npz (frame, kp (F,MAXP,17,3) px of the original frame, box (F,MAXP,4)
x,y,w,h, det_score (F,MAXP), fps, nframes, size).
track: a single performer, so the athlete is the large, confident, temporally consistent person: greedy tracking
with score = 1.5 area / max area + det score + keypoint score + box bottom + IoU(previous athlete box) (spectators and judges are small and static).
Output matches tools/battle/choreo_memory.py extract: rel (F,17,2) joints relative to the hip centre / box height,
root (F,2) hip centre / box height, ok (F,), hz, plus frame (F,), nframes, clip grid alignment in the trainer.
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import torch

FFMPEG = os.environ.get('FFMPEG', '/home/irl/anaconda3/bin/ffmpeg')
FFPROBE = os.environ.get('FFPROBE', '/home/irl/anaconda3/bin/ffprobe')
W, H = 960, 540
MAXP = 10    # crowds: with 6 the athlete in a wide shot can be dropped behind spectators
TAG = os.environ.get('AQA_POSE_TAG', '')     # _y: YOLO11x-pose -> pose_det_y, pose_seq_y
SETS = {'fisv': ('/data/mlavl/videos/fisv_raw/videos', '/data/mlavl/features/FISV'),
        'rg': ('/data/mlavl/videos/rg_raw/videos', '/data/mlavl/features/RG')}


def probe(path):
    out = subprocess.run([FFPROBE, '-v', 'error', '-select_streams', 'v:0', '-count_packets', '-show_entries',
                          'stream=r_frame_rate,nb_read_packets,width,height', '-of', 'csv=p=0', path],
                         capture_output=True, text=True).stdout.strip().split(',')
    w, h, r, n = out[:4]
    a, b = r.split('/')
    return int(w), int(h), float(a) / float(b), int(n)


def frames_iter(path, stride):
    p = subprocess.Popen([FFMPEG, '-v', 'error', '-threads', '2', '-i', path, '-vf',
                          f"select='not(mod(n\\,{stride}))',scale={W}:{H}", '-vsync', '0', '-f', 'rawvideo',
                          '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
    n, k = W * H * 3, 0
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield k * stride, np.frombuffer(b, np.uint8).reshape(H, W, 3)
        k += 1
    p.stdout.close()
    p.wait()


def videos(set_):
    root = SETS[set_][0]
    return sorted(f for e in ('mp4', 'mov', 'avi', 'mkv') for f in glob.glob(f'{root}/**/*.{e}', recursive=True))


def vid_id(path):
    return os.path.splitext(os.path.basename(path))[0]


def detect_yolo(a):
    """same file format as detect, YOLO11x-pose (conf 0.25), keypoint scores x 10 to match the R-CNN score range of track"""
    from ultralytics import YOLO
    model = YOLO(a.weights)
    out_dir = os.path.join(SETS[a.set][1], f'pose_det{TAG}')
    os.makedirs(out_dir, exist_ok=True)
    paths = videos(a.set)[a.shard::a.nshards]
    for k, path in enumerate(paths):
        dst = os.path.join(out_dir, vid_id(path) + '.npz')
        if os.path.exists(dst):
            continue
        w0, h0, fps, nf = probe(path)
        sx, sy = w0 / W, h0 / H
        F_, KP, BX, SC, batch = [], [], [], [], []

        def flush():
            if not batch:
                return
            res = model.predict([fr[:, :, ::-1] for _, fr in batch], conf=0.25, iou=0.7, imgsz=960, verbose=False, device=0)
            for (f, _), r in zip(batch, res):
                kp = np.zeros((MAXP, 17, 3), np.float32)
                bx = np.zeros((MAXP, 4), np.float32)
                sc = np.zeros(MAXP, np.float32)
                n = min(len(r.boxes), MAXP)
                if n:
                    o = r.boxes.conf.cpu().numpy().argsort()[::-1][:n].copy()
                    b = r.boxes.xywh.cpu().numpy()[o]
                    b[:, 0] -= b[:, 2] / 2
                    b[:, 1] -= b[:, 3] / 2
                    b[:, [0, 2]] *= sx
                    b[:, [1, 3]] *= sy
                    kk = r.keypoints.xy.cpu().numpy()[o]
                    kp[:n, :, 0], kp[:n, :, 1] = kk[..., 0] * sx, kk[..., 1] * sy
                    kp[:n, :, 2] = r.keypoints.conf.cpu().numpy()[o] * 10
                    bx[:n], sc[:n] = b, r.boxes.conf.cpu().numpy()[o]
                F_.append(f); KP.append(kp); BX.append(bx); SC.append(sc)
            batch.clear()

        for f, fr in frames_iter(path, a.stride):
            batch.append((f, fr))
            if len(batch) == 16:
                flush()
        flush()
        tmp = dst[:-4] + f'.tmp{a.shard}.npz'
        np.savez(tmp, frame=np.array(F_, np.int32), kp=np.array(KP, np.float32).reshape(-1, MAXP, 17, 3),
                 box=np.array(BX, np.float32).reshape(-1, MAXP, 4), det_score=np.array(SC, np.float32).reshape(-1, MAXP),
                 fps=fps, nframes=nf, size=np.array([w0, h0]), stride=a.stride)
        os.replace(tmp, dst)
        print(f'[{k + 1}/{len(paths)}] {vid_id(path)}: {len(F_)} frames of {nf}', flush=True)


def detect(a):
    if a.yolo:
        return detect_yolo(a)
    from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
    torch.set_num_threads(4)
    dev = torch.device('cuda')
    model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1).eval().to(dev)
    out_dir = os.path.join(SETS[a.set][1], 'pose_det')
    os.makedirs(out_dir, exist_ok=True)
    paths = videos(a.set)[::-1] if a.reverse else videos(a.set)   # --reverse: helpers start from the other end
    paths = paths[a.shard::a.nshards]
    if a.only:
        paths = [p for p in paths if vid_id(p) in a.only.split(',')]
    for k, path in enumerate(paths):
        dst = os.path.join(out_dir, vid_id(path) + '.npz')
        busy = dst + '.busy'
        if os.path.exists(dst) or os.path.exists(busy):
            continue
        open(busy, 'w').close()
        w0, h0, fps, nf = probe(path)
        sx, sy = w0 / W, h0 / H
        F_, KP, BX, SC, batch = [], [], [], [], []

        def flush():
            if not batch:
                return
            x = [torch.from_numpy(fr).to(dev).permute(2, 0, 1).float() / 255 for _, fr in batch]
            with torch.no_grad():
                res = model(x)
            for (f, _), r in zip(batch, res):
                keep = (r['labels'] == 1) & (r['scores'] >= 0.5)
                n = int(min(keep.sum().item(), MAXP))
                kp = np.zeros((MAXP, 17, 3), np.float32)
                bx = np.zeros((MAXP, 4), np.float32)
                sc = np.zeros(MAXP, np.float32)
                if n:
                    b = r['boxes'][keep][:n].cpu().numpy()
                    b[:, 2:] -= b[:, :2]
                    b[:, [0, 2]] *= sx
                    b[:, [1, 3]] *= sy
                    kk = r['keypoints'][keep][:n].cpu().numpy()
                    kp[:n, :, 0], kp[:n, :, 1] = kk[..., 0] * sx, kk[..., 1] * sy
                    kp[:n, :, 2] = r['keypoints_scores'][keep][:n].cpu().numpy()
                    bx[:n], sc[:n] = b, r['scores'][keep][:n].cpu().numpy()
                F_.append(f); KP.append(kp); BX.append(bx); SC.append(sc)
            batch.clear()

        for f, fr in frames_iter(path, a.stride):
            batch.append((f, fr))
            if len(batch) == 16:
                flush()
        flush()
        np.savez(dst, frame=np.array(F_, np.int32), kp=np.array(KP, np.float32).reshape(-1, MAXP, 17, 3),
                 box=np.array(BX, np.float32).reshape(-1, MAXP, 4), det_score=np.array(SC, np.float32).reshape(-1, MAXP),
                 fps=fps, nframes=nf, size=np.array([w0, h0]), stride=a.stride)
        os.remove(busy)
        print(f'[{k + 1}/{len(paths)}] {vid_id(path)}: {len(F_)} frames of {nf}', flush=True)


def iou(b, c):
    x1, y1 = np.maximum(b[:, 0], c[0]), np.maximum(b[:, 1], c[1])
    x2 = np.minimum(b[:, 0] + b[:, 2], c[0] + c[2])
    y2 = np.minimum(b[:, 1] + b[:, 3], c[1] + c[3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / (b[:, 2] * b[:, 3] + c[2] * c[3] - inter + 1e-6)


def track(a):
    src = os.path.join(SETS[a.set][1], f'pose_det{TAG}')
    dst = os.path.join(SETS[a.set][1], f'pose_seq{TAG}')
    os.makedirs(dst, exist_ok=True)
    stats = []
    for f in sorted(glob.glob(f'{src}/*.npz')):
        d = np.load(f)
        F = len(d['frame'])
        h0 = float(d['size'][1])
        rel = np.zeros((F, 17, 2), np.float32)
        root = np.zeros((F, 2), np.float32)
        ok = np.zeros(F, bool)
        prev, miss = None, 0
        for i in range(F):
            v = np.where(d['det_score'][i] > 0)[0]
            if not len(v):
                miss += 1
                continue
            v = v[(d['box'][i][v][:, 2] < 2.0 * d['box'][i][v][:, 3])          # not a wide strip of spectators
                  & (d['box'][i][v][:, 3] > 0.04 * h0)]
            if not len(v):
                miss += 1
                continue
            b = d['box'][i][v]
            area = b[:, 2] * b[:, 3]
            s = (1.5 * area / area.max() + d['det_score'][i][v] + d['kp'][i][v][:, :, 2].clip(0, 10).mean(1) / 10
                 + (b[:, 1] + b[:, 3]) / h0)                             # athletes stand on the floor / ice, spectators above
            if prev is not None and miss < 10:
                s = s + iou(b, prev) * (area > 0.3 * prev[2] * prev[3])  # tracking only to a similar-sized box
            j = v[int(np.argmax(s))]
            bx, kp = d['box'][i][j], d['kp'][i][j]
            hip = (kp[11, :2] + kp[12, :2]) / 2
            rel[i] = (kp[:, :2] - hip) / max(bx[3], 1)
            root[i] = hip / max(bx[3], 1)       # in box heights, as for breaking (only deltas are used)
            ok[i] = (bx[3] > 0.08 * h0) & (bx[1] + bx[3] < 0.97 * h0)   # too small, or cut off at the bottom (close-up)
            prev, miss = bx, 0
        hz = float(d['fps']) / int(d['stride'])
        np.savez(os.path.join(dst, os.path.basename(f)), rel=rel, root=root, ok=ok, hz=hz, frame=d['frame'],
                 nframes=int(d['nframes']), fps=float(d['fps']))
        stats.append(ok.mean())
    print(f'{len(stats)} videos, athlete found in {np.mean(stats):.3f} of sampled frames '
          f'(min {np.min(stats):.3f})')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['detect', 'track'])
    p.add_argument('--set', choices=list(SETS), required=True)
    p.add_argument('--stride', type=int, default=3)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1)
    p.add_argument('--only', default='')
    p.add_argument('--reverse', action='store_true')
    p.add_argument('--yolo', action='store_true')
    p.add_argument('--weights', default='/data/mlavl/cache/ultralytics/yolo11x-pose.pt')
    a = p.parse_args()
    detect(a) if a.cmd == 'detect' else track(a)


if __name__ == '__main__':
    main()

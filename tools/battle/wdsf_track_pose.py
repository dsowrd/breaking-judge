"""Person tracks and pose inside the WDSF turns with YOLO11x-pose + ByteTrack, and track-level dancer selection (ours, Phase 7d).

Phase 7c showed that about 28% of the dancing frames saw the wrong person: Keypoint R-CNN detections below score 0.5
were never stored (floor and inverted dancers often score 0.2-0.5) and the dancer was chosen frame by frame. Here every
person is tracked through each turn at 10 fps and the dancer is chosen per track (see select_turn).

    CUDA_VISIBLE_DEVICES=0 $MLAVL_PY -I tools/battle/wdsf_track_pose.py track [--shard k --nshards n] [--turns list.csv]
        -> {F}/yolo_tracks/{video_id}.npz  turn (N,) index into wdsf_turns.csv, t (N,) s, tid (N,), box (N,4) xyxy px,
           kp (N,17,3) x, y px + score, score (N,)   (one row per detection, 1280x720 pixels)
    $MLAVL_PY -I tools/battle/wdsf_track_pose.py select  [--turns list.csv]
        -> {F}/pose_seq2/{video_id}.npz  rel (n,17,2), root (n,2), ok (n,), hz, box (n,4), tid (n,), in the format of
           choreo_memory.py pose_seq on the 10 Hz grid (i + 0.5) / hz. Joints are relative to the hip centre and divided
           by a smoothed body scale (median of max(box w, box h) over +-1 s), not by the frame's box height: a lying
           dancer's box height collapses and would inflate every speed.
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

F = os.environ.get('WDSF_F', '/data/mlavl/features/WDSF')     # WDSF_F, WDSF_D, WDSF_V: another set, e.g. the 2026 holdout
D = os.environ.get('WDSF_D', '/data/mlavl/datasets/wdsf')
V = os.environ.get('WDSF_V', '/data/mlavl/videos/wdsf')
FFMPEG = os.environ.get('FFMPEG', '/home/irl/anaconda3/bin/ffmpeg')
WEIGHTS = os.environ.get('WDSF_YOLO_WEIGHTS', '/data/mlavl/cache/ultralytics/yolo11x-pose.pt')
T = os.environ.get('WDSF_TRACK_TAG', '')     # _ft: fine-tuned model -> yolo_tracks_ft, pose_seq2_ft
W, H, FPS, IMGSZ = 1280, 720, 10, 960
HZ = 10.0


def turns_table(path=None):
    t = pd.read_csv(f'{D}/wdsf_turns.csv').reset_index().rename(columns={'index': 'turn'})
    if path:
        keep = pd.read_csv(path)
        t = t[t.turn.isin(keep.turn)]
    return t


def turn_frames(vid, start, end):
    p = subprocess.Popen([FFMPEG, '-v', 'error', '-ss', f'{start:.3f}', '-i', f'{V}/{vid}.mp4', '-t', f'{end - start:.3f}',
                          '-vf', f'fps={FPS},scale={W}:{H}', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'],
                         stdout=subprocess.PIPE)
    n, k = W * H * 3, 0
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield start + k / FPS, np.frombuffer(b, np.uint8).reshape(H, W, 3)
        k += 1
    p.stdout.close()
    p.wait()


def track(a):
    from ultralytics import YOLO
    os.makedirs(f'{F}/yolo_tracks{T}', exist_ok=True)
    tab = turns_table(a.turns)
    vids = sorted(tab.video_id.unique())[a.shard::a.nshards]
    model = YOLO(WEIGHTS)
    t0, frames = time.time(), 0
    for iv, vid in enumerate(vids):
        dst = f'{F}/yolo_tracks{T}/{vid}.npz'
        if os.path.exists(dst):
            continue
        out = {k: [] for k in ('turn', 't', 'tid', 'box', 'kp', 'score')}
        for row in tab[tab.video_id == vid].itertuples():
            first = True
            for ts, fr in turn_frames(vid, row.start_s, row.end_s):
                r = model.track(fr, persist=not first, conf=0.1, iou=0.7, imgsz=IMGSZ, tracker='bytetrack.yaml',
                                verbose=False, device=0)[0]
                first = False
                frames += 1
                if r.boxes is None or r.boxes.id is None or not len(r.boxes):
                    continue
                n = len(r.boxes)
                kp = np.concatenate([r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy()[..., None]], -1)
                out['turn'].append(np.full(n, row.turn, np.int32))
                out['t'].append(np.full(n, ts, np.float32))
                out['tid'].append(r.boxes.id.cpu().numpy().astype(np.int32))
                out['box'].append(r.boxes.xyxy.cpu().numpy().astype(np.float32))
                out['kp'].append(kp.astype(np.float32))
                out['score'].append(r.boxes.conf.cpu().numpy().astype(np.float32))
        arr = {k: (np.concatenate(v) if v else np.zeros((0,))) for k, v in out.items()}
        tmp = dst[:-4] + f'.tmp{a.shard}.npz'
        np.savez_compressed(tmp, **arr)
        os.replace(tmp, dst)
        print(f'[{iv + 1}/{len(vids)}] {vid}: {frames} frames, {frames / (time.time() - t0):.1f} fps', flush=True)


def track_features(g):
    """one row per track of a turn"""
    rows = []
    for tid, d in g.groupby('tid'):
        idx = d.index.values
        rows.append(dict(tid=tid, idx=idx))
    return rows


def select_turn(t, tid, box, kp, score, dur):
    """Choose the dancer's track at every sampled time of one turn. Returns (times, row index of the chosen detection).

    Track features (constant over a track; no judge label is used). A track must be eligible, then the highest score wins:
      activity  median joint speed in body scales / s (the dancer moves, the opponent and judges mostly do not)
      size      median body scale / frame height (the broadcast camera frames the dancer)
      central   mean closeness of the box centre to the frame centre (the operator follows the dancer)
      floor     share of samples whose box bottom is in the lower 45% of the frame (kept for inspection, not used)
      cut       share of samples with the box touching the frame's bottom edge (foreground heads)
      kpconf    mean keypoint score
    A track shorter than 1 s is down-weighted. Frames with no eligible track have no pose (ok = False).
    """
    scale = np.maximum(box[:, 2] - box[:, 0], box[:, 3] - box[:, 1])
    tracks = {}
    for k in np.unique(tid):
        m = np.where(tid == k)[0]
        m = m[np.argsort(t[m])]
        s = np.maximum(scale[m], 1.0)
        mid = np.median(s)
        cx = (box[m, 0] + box[m, 2]) / 2 / W
        bottom = box[m, 3] / H
        c = kp[m, :, 2]
        hip = kp[m][:, 11:13, :2].mean(1)
        rel = (kp[m, :, :2] - hip[:, None]) / mid
        if len(m) > 2:
            dt = np.maximum(np.diff(t[m]), 0.05)
            sp = np.linalg.norm(np.diff(rel, axis=0), axis=-1).mean(1) / dt
            hipv = np.linalg.norm(np.diff(hip, axis=0), axis=-1) / mid / dt
            activity = float(np.median(np.clip(sp + 0.5 * hipv, 0, 10)))
        else:
            activity = 0.0
        tracks[k] = dict(m=m, n=len(m), activity=activity, size=float(mid / H),
                         central=float(np.exp(-((cx - 0.5) / 0.3) ** 2).mean()), floor=float((bottom > 0.55).mean()),
                         cut=float((bottom > 0.985).mean()), kpconf=float(c.mean()))
    # Eligibility first: a wrong person is worse than no pose (Phase 7c). Thresholds set on the 22 frames where the old
    # selection failed and on 24 development frames (seed 99, where activity >= 0.8 / cut <= 0.3 / central >= 0.5 dropped
    # the dancer's own track and left 7 of 24 frames without pose); the selection is judged on fresh random frames.
    for v in tracks.values():
        v['eligible'] = (v['activity'] >= 0.5 and v['size'] >= 0.18 and v['central'] >= 0.4 and v['cut'] <= 0.6
                         and v['kpconf'] >= 0.5)
        v['score'] = (min(v['activity'], 3.0) + 1.0 * min(v['size'] / 0.5, 1.5) + 0.5 * v['central']
                      + 0.5 * v['kpconf']) * min(v['n'] / 10.0, 1.0)
    times = np.unique(t)
    best = np.full(len(times), -1, np.int64)
    bs = np.full(len(times), -np.inf)
    for v in tracks.values():
        if not v['eligible']:
            continue
        pos = np.searchsorted(times, t[v['m']])
        up = v['score'] > bs[pos]
        best[pos[up]] = v['m'][up]
        bs[pos[up]] = v['score']
    return times, best, tracks


def select(a):
    os.makedirs(f'{F}/pose_seq2{T}', exist_ok=True)
    tab = turns_table(a.turns)
    for vid in sorted(tab.video_id.unique()):
        src = f'{F}/yolo_tracks{T}/{vid}.npz'
        if not os.path.exists(src):
            continue
        z = np.load(src)
        if not len(z['t']):
            continue
        c = np.load(f'{F}/clips_pamfn/{vid}.npz')
        fps = float(c['fps'])
        dur = (float(c['clip_start_frame'][-1]) + float(c['clip_len'])) / fps
        n = int(dur * HZ)
        rel = np.zeros((n, 17, 2), np.float32)
        root = np.zeros((n, 2), np.float32)
        ok = np.zeros(n, bool)
        boxes = np.zeros((n, 4), np.float32)
        tids = np.full(n, -1, np.int32)
        conf = np.zeros(n, np.float32)
        for row in tab[tab.video_id == vid].itertuples():
            m = np.where(z['turn'] == row.turn)[0]
            if not len(m):
                continue
            times, best, _ = select_turn(z['t'][m], z['tid'][m], z['box'][m], z['kp'][m], z['score'][m], row.end_s - row.start_s)
            keep = best >= 0
            times, chosen = times[keep], m[best[keep]]
            if not len(chosen):
                continue
            b, kp = z['box'][chosen], z['kp'][chosen]
            sc = np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])
            # smoothed body scale, +-1 s
            sm = np.array([np.median(sc[np.abs(times - x) <= 1.0]) for x in times])
            hip = kp[:, 11:13, :2].mean(1)
            bad = kp[:, 11:13, 2].min(1) < 0.2
            hip[bad] = np.stack([(b[bad, 0] + b[bad, 2]) / 2, (b[bad, 1] + b[bad, 3]) / 2], 1)
            r = (kp[:, :, :2] - hip[:, None]) / np.maximum(sm, 1.0)[:, None, None]
            g = np.arange(int(row.start_s * HZ), min(int(row.end_s * HZ) + 1, n))
            grid = (g + 0.5) / HZ
            j = np.clip(np.searchsorted(times, grid), 1, max(len(times) - 1, 1)) if len(times) > 1 else np.zeros(len(g), int)
            if len(times) > 1:
                j = np.where(np.abs(times[j - 1] - grid) < np.abs(times[j] - grid), j - 1, j)
            near = np.abs(times[j] - grid) <= 0.15
            rel[g[near]] = r[j[near]]
            root[g[near]] = hip[j[near]] / np.maximum(sm[j[near]], 1.0)[:, None]
            ok[g[near]] = True
            boxes[g[near]] = b[j[near]]
            tids[g[near]] = z['tid'][chosen][j[near]]
            conf[g[near]] = kp[j[near], :, 2].mean(1)
        np.savez(f'{F}/pose_seq2{T}/{vid}.npz', rel=rel, root=root, ok=ok, hz=HZ, box=boxes, tid=tids, conf=conf)
    print('done')


def sheet(a):
    """Contact sheet of chosen dancers, graded by eye. --turns csv with turn, video_id, ts draws those frames; without
    it, --n random samples from turns that have tracks (--seed)."""
    import glob
    import io
    from PIL import Image, ImageDraw
    E = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12)]
    tab = turns_table()
    rng = np.random.default_rng(a.seed)
    if a.turns:
        want = pd.read_csv(a.turns)[['turn', 'video_id', 'ts']].values.tolist()
    else:
        vids = sorted(os.path.basename(f)[:-4] for f in glob.glob(f'{F}/yolo_tracks/*.npz'))
        want = []
        while len(want) < a.n:
            v = vids[rng.integers(len(vids))]
            r = tab[tab.video_id == v].sample(1, random_state=int(rng.integers(1 << 30))).iloc[0]
            want.append([int(r.turn), v, float(rng.uniform(r.start_s, r.end_s))])
    tiles = []
    for turn, vid, ts in want:
        z = np.load(f'{F}/yolo_tracks{T}/{vid}.npz')
        m = np.where(z['turn'] == turn)[0]
        if not len(m):
            continue
        times, best, tracks = select_turn(z['t'][m], z['tid'][m], z['box'][m], z['kp'][m], z['score'][m], 0)
        i = int(np.argmin(np.abs(times - ts)))
        raw = subprocess.run([FFMPEG, '-v', 'error', '-ss', f'{times[i]:.3f}', '-i', f'{V}/{vid}.mp4', '-frames:v', '1', '-vf',
                              f'scale={W}:{H}', '-f', 'image2pipe', '-vcodec', 'png', '-'], capture_output=True).stdout
        im = Image.open(io.BytesIO(raw)).convert('RGB')
        dr = ImageDraw.Draw(im)
        here = m[z['t'][m] == times[i]]
        for b in z['box'][here]:
            dr.rectangle(list(b), outline=(255, 255, 255), width=2)
        if best[i] >= 0:
            kp = z['kp'][m][best[i]]
            b = z['box'][m][best[i]]
            dr.rectangle(list(b), outline=(0, 255, 0), width=5)
            for x, y in E:
                dr.line([tuple(kp[x, :2]), tuple(kp[y, :2])], fill=(60, 170, 255) if y <= 10 else (255, 60, 60), width=4)
        im = im.resize((480, 270))
        ImageDraw.Draw(im).text((5, 5), f'{vid} {times[i]:.1f}s', fill=(255, 255, 0))
        tiles.append(im)
    cols = 4
    canvas = Image.new('RGB', (480 * cols, 270 * ((len(tiles) + cols - 1) // cols)))
    for k, im in enumerate(tiles):
        canvas.paste(im, ((k % cols) * 480, (k // cols) * 270))
    canvas.save(a.out, quality=85)
    print('wrote', a.out, len(tiles))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['track', 'select', 'sheet'])
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1)
    p.add_argument('--turns', default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n', type=int, default=24)
    p.add_argument('--out', default='sheet.jpg')
    a = p.parse_args()
    {'track': track, 'select': select, 'sheet': sheet}[a.cmd](a)

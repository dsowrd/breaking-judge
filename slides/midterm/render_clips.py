"""Short video clips for the midterm deck (RG / Fis-V raw footage, research datasets; cited on the slides).

Inputs (copied from the server into data/, which git ignores):
  data/videos/{id}.mp4         raw dataset videos  (/data/mlavl/videos/{rg_raw,fisv_raw}/videos)
  data/pose_det/{id}.npz       Keypoint R-CNN detections (tools/aqa/aqa_pose.py detect)
  data/codes_{rg,fisv}.npy     {id: (groups, 3) = upper id, lower id, ok}, one group = 4 pose samples
                               (~/Workspace_Hyundo/render/codes_dump.py on the server)
Outputs: assets/clips/*.mp4 (H.264, no audio, ignored by git)

    python render_clips.py            # all clips
"""
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.chdir(os.path.dirname(os.path.abspath(__file__)))
OUT = 'assets/clips'
os.makedirs(OUT, exist_ok=True)
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
UPC, LOWC, TORC = (47, 128, 237), (235, 87, 87), (214, 219, 225)
UP = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6)]          # upper-codebook joints only
LOW = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12)]  # lower-codebook joints only
TOR = [(5, 11), (6, 12)]                                  # torso links join the two halves: neither codebook


def probe(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                          'stream=width,height,r_frame_rate', '-of', 'csv=p=0', path],
                         capture_output=True, text=True).stdout.strip().split(',')
    a, b = out[2].split('/')
    return int(out[0]), int(out[1]), float(a) / float(b)


def iou(b, c):
    x1, y1 = np.maximum(b[:, 0], c[0]), np.maximum(b[:, 1], c[1])
    x2, y2 = np.minimum(b[:, 0] + b[:, 2], c[0] + c[2]), np.minimum(b[:, 1] + b[:, 3], c[1] + c[3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / (b[:, 2] * b[:, 3] + c[2] * c[3] - inter + 1e-6)


def track(d):
    """Same athlete choice as tools/aqa/aqa_pose.py track; returns per sample (kp (17,3) px, box, ok)."""
    F, h0 = len(d['frame']), float(d['size'][1])
    kps, boxes, oks = np.zeros((F, 17, 3)), np.zeros((F, 4)), np.zeros(F, bool)
    prev, miss = None, 0
    for i in range(F):
        v = np.where(d['det_score'][i] > 0)[0]
        if len(v):
            v = v[(d['box'][i][v][:, 2] < 2.0 * d['box'][i][v][:, 3]) & (d['box'][i][v][:, 3] > 0.04 * h0)]
        if not len(v):
            miss += 1
            continue
        b = d['box'][i][v]
        area = b[:, 2] * b[:, 3]
        s = (1.5 * area / area.max() + d['det_score'][i][v] + d['kp'][i][v][:, :, 2].clip(0, 10).mean(1) / 10
             + (b[:, 1] + b[:, 3]) / h0)
        if prev is not None and miss < 10:
            s = s + iou(b, prev) * (area > 0.3 * prev[2] * prev[3])
        j = v[int(np.argmax(s))]
        kps[i], boxes[i] = d['kp'][i][j], d['box'][i][j]
        oks[i] = (boxes[i][3] > 0.08 * h0) & (boxes[i][1] + boxes[i][3] < 0.97 * h0)
        prev, miss = boxes[i], 0
    return kps, boxes, oks


class Pose:
    def __init__(self, vid, codes=None):
        self.d = np.load(f'data/pose_det/{vid}.npz')
        self.kp, self.box, self.ok = track(self.d)
        self.frame = self.d['frame'].astype(int)
        self.codes = codes.get(vid) if codes is not None else None

    def at(self, f):
        """keypoints at video frame f (linear interpolation between sampled frames), or None"""
        i = np.searchsorted(self.frame, f, side='right') - 1
        if i < 0 or i >= len(self.frame) - 1 or not (self.ok[i] and self.ok[i + 1]):
            return None
        w = (f - self.frame[i]) / max(self.frame[i + 1] - self.frame[i], 1)
        return (1 - w) * self.kp[i] + w * self.kp[i + 1], (1 - w) * self.box[i] + w * self.box[i + 1]

    def code(self, f):
        i = np.searchsorted(self.frame, f, side='right') - 1
        if self.codes is None or i < 0 or i // 4 >= len(self.codes):
            return None
        return self.codes[i // 4]


def frames(path, t0, dur, W, H, crop=None):
    vf = (f'crop={crop},' if crop else '') + f'scale={W}:{H}'
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-ss', f'{t0:.3f}', '-i', path, '-t', f'{dur:.3f}', '-vf', vf,
                          '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
    n = W * H * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield Image.frombytes('RGB', (W, H), b)
    p.wait()


def writer(path, W, H, fps):
    return subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                             '-r', f'{fps}', '-i', '-', '-c:v', 'libx264', '-crf', '23', '-preset', 'slow',
                             '-pix_fmt', 'yuv420p', '-movflags', '+faststart', path], stdin=subprocess.PIPE)


def draw_pose(dr, kp, sx, sy, ox=0, oy=0, lw=5):
    P = [(x * sx - ox, y * sy - oy) for x, y in kp[:, :2]]
    for a, b in TOR:
        dr.line([P[a], P[b]], fill=TORC, width=max(lw - 1, 2))
    for edges, col in ((LOW, LOWC), (UP, UPC)):
        for a, b in edges:
            dr.line([P[a], P[b]], fill=col, width=lw)
    nk = ((P[5][0] + P[6][0]) / 2, (P[5][1] + P[6][1]) / 2)
    dr.line([nk, P[0]], fill=UPC, width=lw)
    r = lw * 0.8
    for j in range(17):
        col = UPC if j < 11 else LOWC
        if j in (1, 2, 3, 4):
            continue
        dr.ellipse([P[j][0] - r, P[j][1] - r, P[j][0] + r, P[j][1] + r], fill=col, outline=(255, 255, 255))


def label(dr, xy, text, size, fg=(255, 255, 255), bg=(11, 44, 74, 215), pad=8):
    f = ImageFont.truetype(FONT, size)
    x0, y0, x1, y1 = dr.textbbox(xy, text, font=f)
    dr.rounded_rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], radius=6, fill=bg)
    dr.text(xy, text, font=f, fill=fg)


def best_window(pose, dur_f, fps, start_after=0):
    """window of dur_f frames with the athlete tracked throughout and largest on screen"""
    best, bt = -1, 0
    step = int(fps)
    for f0 in range(int(start_after * fps), int(pose.frame[-1]) - dur_f, step):
        m = (pose.frame >= f0) & (pose.frame < f0 + dur_f)
        if m.sum() < 3 or pose.ok[m].mean() < 0.97:
            continue
        s = pose.box[m][:, 3].mean()
        if s > best:
            best, bt = s, f0
    return bt


def clip_overlay(vid, codes, name, dur=8.0, W=960, H=540, start=None, ticker=True, raw=False, start_after=0):
    path = f'data/videos/{vid}.mp4'
    w0, h0, fps = probe(path)
    pose = Pose(vid, codes)
    f0 = start if start is not None else best_window(pose, int(dur * fps), fps, start_after)
    sx, sy = W / w0, H / h0
    wr = writer(f'{OUT}/{name}.mp4', W, H, fps)
    for k, im in enumerate(frames(path, f0 / fps, dur, W, H)):
        f = f0 + k
        if not raw:
            dr = ImageDraw.Draw(im, 'RGBA')
            pk = pose.at(f)
            if pk is not None:
                draw_pose(dr, pk[0], sx, sy, lw=max(int(H / 100), 4))
            c = pose.code(f)
            if ticker and c is not None and c[2] >= 0.5:
                label(dr, (24, H - 62), f'상체 #{int(c[0]):03d}   하체 #{int(c[1]):03d}', int(H / 20))
        wr.stdin.write(im.tobytes())
    wr.stdin.close(); wr.wait()
    print(name, vid, f'{f0 / fps:.1f}s +{dur}s')
    return f0


def clip_same_code(items, code, name, half=1.0, T=480, H=360):
    """2x2 grid: the same upper code in four routines; tile border turns green while that code is active"""
    tiles = []
    for vid, g in items:
        path = f'data/videos/{vid}.mp4'
        w0, h0, fps = probe(path)
        pose = Pose(vid, CODES['rg'])
        fc = int(pose.frame[min(g * 4 + 4, len(pose.frame) - 1)])          # centre of the two-group run
        bx = pose.box[g * 4 + 2]
        side = min(max(bx[3] * 1.35, 300), h0)
        cw, ch = side * T / H, side
        cx = np.clip(bx[0] + bx[2] / 2 - cw / 2, 0, w0 - cw); cy = np.clip(bx[1] + bx[3] / 2 - ch / 2, 0, h0 - ch)
        crop = f'{int(cw)}:{int(ch)}:{int(cx)}:{int(cy)}'
        f0 = fc - int(half * fps)
        ims = []
        for k, im in enumerate(frames(path, f0 / fps, 2 * half, T, H, crop=crop)):
            f = f0 + k
            dr = ImageDraw.Draw(im, 'RGBA')
            pk = pose.at(f)
            if pk is not None:
                draw_pose(dr, pk[0], T / cw, H / ch, cx * T / cw, cy * H / ch, lw=4)
            c = pose.code(f)
            on = c is not None and int(c[0]) == code
            if on:
                dr.rectangle([2, 2, T - 3, H - 3], outline=(46, 204, 113), width=6)
            label(dr, (14, 12), {'Ball': '공', 'Clubs': '곤봉', 'Hoop': '후프', 'Ribbon': '리본'}[vid.split('_')[0]], 20)
            label(dr, (T - 112, H - 44), f'상체 #{int(c[0]):03d}' if c is not None else '', 18,
                  bg=(46, 160, 90, 230) if on else (11, 44, 74, 200))
            ims.append(im)
        tiles.append(ims)
    n = min(len(t) for t in tiles)
    fps = probe(f'data/videos/{items[0][0]}.mp4')[2]
    wr = writer(f'{OUT}/{name}.mp4', 2 * T, 2 * H, fps)
    for k in range(n):
        g = Image.new('RGB', (2 * T, 2 * H))
        for i, t in enumerate(tiles):
            g.paste(t[k], ((i % 2) * T, (i // 2) * H))
        wr.stdin.write(g.tobytes())
    wr.stdin.close(); wr.wait()
    print(name, n, 'frames')


def clip_grid_raw(vids, name, dur=6.0, T=480, H=360, start_after=15):
    """2x2 grid of raw footage (one routine per apparatus), cropped around the tracked athlete"""
    tiles, fps = [], None
    for vid in vids:
        path = f'data/videos/{vid}.mp4'
        w0, h0, fps = probe(path)
        pose = Pose(vid)
        f0 = best_window(pose, int(dur * fps), fps, start_after)
        m = (pose.frame >= f0) & (pose.frame < f0 + int(dur * fps)) & pose.ok
        bx = pose.box[m]
        x0, y0 = bx[:, 0].min(), bx[:, 1].min()
        x1, y1 = (bx[:, 0] + bx[:, 2]).max(), (bx[:, 1] + bx[:, 3]).max()
        ch = min(max((y1 - y0) * 1.25, (x1 - x0) * 1.25 * H / T, 360), h0); cw = min(ch * T / H, w0); ch = cw * H / T
        cx = np.clip((x0 + x1) / 2 - cw / 2, 0, w0 - cw); cy = np.clip((y0 + y1) / 2 - ch / 2, 0, h0 - ch)
        ims = []
        for im in frames(path, f0 / fps, dur, T, H, crop=f'{int(cw)}:{int(ch)}:{int(cx)}:{int(cy)}'):
            label(ImageDraw.Draw(im, 'RGBA'), (14, 12), {'Ball': '공', 'Clubs': '곤봉', 'Hoop': '후프', 'Ribbon': '리본'}[vid.split('_')[0]], 20)
            ims.append(im)
        tiles.append(ims)
    n = min(len(t) for t in tiles)
    wr = writer(f'{OUT}/{name}.mp4', 2 * T, 2 * H, fps)
    for k in range(n):
        g = Image.new('RGB', (2 * T, 2 * H))
        for i, t in enumerate(tiles):
            g.paste(t[k], ((i % 2) * T, (i // 2) * H))
        wr.stdin.write(g.tobytes())
    wr.stdin.close(); wr.wait()
    print(name, n, 'frames')


def posters():
    for f in sorted(os.listdir(OUT)):
        if f.endswith('.mp4'):
            t = '1.0' if 'same' in f else '3.0'      # same-code grid: the moment all four tiles show the code
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', t, '-i', f'{OUT}/{f}', '-frames:v', '1', '-q:v', '3',
                            f'{OUT}/{f[:-4]}.jpg'])


CODES = {s: np.load(f'data/codes_{s}.npy', allow_pickle=True).item() for s in ('rg', 'fisv')}

if __name__ == '__main__':
    # Fis-V #154 = 2014 Sochi Olympics ladies' short program (TES 39.03 / PCS 35.89 in the labels = Kim Yuna's
    # official score; SOCHI 2014 boards and the CCTV caption "冬奥会 花样滑冰 女单短节目" in the frames)
    clip_overlay('154', CODES['fisv'], 'fisv154_pose', dur=8.0)
    clip_overlay('154', None, 'fisv154_raw', dur=6.0, raw=True, start_after=60)
    clip_overlay('Ribbon_010', None, 'rg_ribbon010_raw', dur=6.0, raw=True)
    clip_overlay('Ribbon_010', CODES['rg'], 'rg_ribbon010_pose', dur=8.0, start_after=20)
    clip_same_code([('Ball_027', 87), ('Clubs_010', 15), ('Hoop_031', 116), ('Ribbon_013', 42)], 191, 'rg_same191')
    clip_grid_raw(['Ball_027', 'Clubs_010', 'Hoop_031', 'Ribbon_013'], 'rg_4app_raw')
    posters()

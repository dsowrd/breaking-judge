"""Bailando-style beat alignment on RG / Fis-V, per Video Swin clip (ours, Phase 6c: MLAVL + Bailando).

Bailando (Siyao et al., CVPR 2022) scores dance-music synchrony with the Beat Align Score: kinematic beats are the
local minima of the joint velocity, and each kinematic beat is scored by its distance to the nearest music beat
(exp(-d^2 / 2 sigma^2)). Here the same quantities are computed per MLAVL clip and fed to the model as an extra input
(ours/extra_inputs.py, --extra beat), so the audio-visual fusion sees whether the athlete moves on the music.

    $MLAVL_PY -I tools/aqa/aqa_beat.py --set rg [--shard k --nshards n]

Inputs: raw videos (audio track, /data/mlavl/videos/...), tracked poses {features}/pose_seq/{id}.npz (aqa_pose.py).
Output: {features}/beat/{id}.npz  b (T, 9) on the clip grid of the PAMFN Video Swin feature (T = its length):
  0 motion energy (mean joint speed, box heights / s)      1 kinematic beats per second
  2 music beats per second                                  3 Beat Align Score of the clip (Bailando, sigma 0.1 s)
  4 music beats followed by a kinematic beat within 0.2 s   5 music onset strength (video-normalised)
  6 corr(joint speed, onset strength) inside the clip       7 global tempo / 200 bpm      8 has_pose
"""
import argparse
import os
import subprocess

import numpy as np

FFMPEG = os.environ.get('FFMPEG', '/home/irl/anaconda3/bin/ffmpeg')
SETS = {'fisv': ('/data/mlavl/videos/fisv_raw/videos', '/data/mlavl/features/FISV'),
        'rg': ('/data/mlavl/videos/rg_raw/videos', '/data/mlavl/features/RG')}
SR, HOP = 22050, 512
SIGMA = 0.1


def vst_len(set_):
    feat = SETS[set_][1]
    files = ['FISV_rgb_VST.npy'] if set_ == 'fisv' else [f'{a}_rgb_VST.npy' for a in ('Ball', 'Clubs', 'Hoop', 'Ribbon')]
    out = {}
    for f in files:
        d = np.load(f'{feat}/{f}', allow_pickle=True).item()
        out.update({k: len(v) for k, v in d.items()})
    return out


def audio(path):
    raw = subprocess.run([FFMPEG, '-v', 'error', '-threads', '1', '-i', path, '-vn', '-ac', '1', '-ar', str(SR),
                          '-f', 's16le', '-'], capture_output=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768


def kinematic(z):
    """joint speed per pose sample and the kinematic beats (local minima of the smoothed speed), Bailando-style"""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import argrelextrema
    pos = z['rel'] + z['root'][:, None]                       # (F, 17, 2) in box heights
    hz = float(z['hz'])
    sp = np.zeros(len(pos))
    sp[1:] = np.linalg.norm(pos[1:] - pos[:-1], axis=-1).mean(1) * hz
    ok = z['ok'].astype(bool)
    ok2 = ok.copy(); ok2[1:] &= ok[:-1]
    sp = np.where(ok2, np.clip(sp, 0, 20), np.nan)
    fill = np.nan_to_num(sp, nan=np.nanmedian(sp) if np.isfinite(sp).any() else 0)
    sm = gaussian_filter1d(fill, 1.0)
    mins = argrelextrema(sm, np.less, order=1)[0]
    mins = mins[ok2[mins]]
    t = z['frame'] / float(z['fps'])
    return t, sm, ok2, t[mins]


def one(path, zpath, T):
    import librosa
    y = audio(path)
    b = np.zeros((T, 9), np.float32)
    if not os.path.exists(zpath) or len(y) < SR:
        return b
    z = np.load(zpath)
    dur = int(z['nframes']) / float(z['fps'])
    onset = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset, sr=SR, hop_length=HOP, units='time')
    tempo = float(np.atleast_1d(tempo)[0])
    ot = librosa.times_like(onset, sr=SR, hop_length=HOP)
    onset = onset / (onset.mean() + 1e-6)
    t, sp, ok, kb = kinematic(z)
    on_at_pose = np.interp(t, ot, onset)
    edges = np.arange(T + 1) * dur / T
    for c in range(T):
        lo, hi = edges[c], edges[c + 1]
        m = (t >= lo) & (t < hi) & ok
        if m.sum() < 3:
            continue
        L = hi - lo
        kbc = kb[(kb >= lo) & (kb < hi)]
        mb = beats[(beats >= lo) & (beats < hi)]
        bas = np.mean([np.exp(-np.min((beats - k) ** 2) / (2 * SIGMA ** 2)) for k in kbc]) if len(kbc) and len(beats) else 0
        hit = np.mean([np.any(np.abs(kb - x) < 0.2) for x in mb]) if len(mb) and len(kb) else 0
        a, o = sp[m], on_at_pose[m]
        r = np.corrcoef(a, o)[0, 1] if a.std() > 1e-6 and o.std() > 1e-6 else 0
        om = onset[(ot >= lo) & (ot < hi)]
        b[c] = [a.mean() / 2, len(kbc) / L / 4, len(mb) / L / 4, bas, hit,
                om.mean() if len(om) else 0, np.nan_to_num(r), tempo / 200, 1]
    return b


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--set', choices=list(SETS), required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1)
    a = p.parse_args()
    vids, feat = SETS[a.set]
    out = f'{feat}/beat'
    os.makedirs(out, exist_ok=True)
    lens = vst_len(a.set)
    names = sorted(lens)[a.shard::a.nshards]
    for k, n in enumerate(names):
        dst = f'{out}/{n}.npz'
        if os.path.exists(dst):
            continue
        path = next((f'{vids}/{n}.{e}' for e in ('mp4', 'mov', 'avi', 'mkv') if os.path.exists(f'{vids}/{n}.{e}')), None)
        b = one(path, f'{feat}/pose_seq/{n}.npz', lens[n]) if path else np.zeros((lens[n], 9), np.float32)
        np.savez(dst, b=b)
        if k % 50 == 0:
            print(f'[{k + 1}/{len(names)}] {n}: clips with beat features {b[:, -1].mean():.2f}', flush=True)


if __name__ == '__main__':
    main()

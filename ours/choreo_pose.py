"""Pose-code input for MLAVL on RG / Fis-V (ours, Phase 6b: MLAVL + Bailando with a pose memory).

tools/aqa/aqa_choreo.py encode writes {features}/choreo_pose/{name}.npz on the Video Swin clip grid:
  x (T, 2*CODE_DIM + 3) = [mean upper | lower code vector, upper code id, lower code id, has_pose]   (--choreo_src pose)
  c (T, 2*CODE_DIM + 3) = [continuous pose statistics, no VQ (padded to 2*CODE_DIM), 0, 0, has_pose] (--choreo_src posecont)
The datasets append it to the Video Swin feature (so random crops / padding stay aligned) and models/model.py splits
it off again. Ablations (--choreo_ablate): shuffle = clip order permuted within the video (fixed per video);
swap = the codes of another video (fixed derangement); upper / lower are handled in the model.
"""
import os
import zlib

import numpy as np

CODE_DIM, K = 64, 512
EXTRA = 2 * CODE_DIM + 3
_cache, _swap = {}, {}


def _load(d, name, key):
    if (d, name, key) not in _cache:
        _cache[(d, name, key)] = np.load(os.path.join(d, name + '.npz'))[key].astype(np.float32)
    return _cache[(d, name, key)]


def _swap_map(d):
    if d not in _swap:
        names = sorted(f[:-4] for f in os.listdir(d) if f.endswith('.npz'))
        p = [names[i] for i in np.random.default_rng(0).permutation(len(names))]
        _swap[d] = {p[i]: p[(i + 1) % len(p)] for i in range(len(p))}
    return _swap[d]


def attach(video_feat, name, video_path, src='pose', ablate='none'):
    d = os.path.join(os.path.dirname(video_path), 'choreo_pose')
    x = _load(d, _swap_map(d)[name] if ablate == 'swap' else name, 'c' if src == 'posecont' else 'x')
    T = len(video_feat)
    if len(x) != T:
        y = np.zeros((T, x.shape[1]), np.float32)
        y[:min(T, len(x))] = x[:T]
        x = y
    if ablate == 'shuffle':
        x = x[np.random.default_rng(zlib.crc32(name.encode())).permutation(T)]
    return np.concatenate([np.asarray(video_feat, np.float32), x], 1)

"""Pose-code input for MLAVL on RG / Fis-V (ours, Phase 6b: MLAVL + Bailando with a pose memory).

tools/aqa/aqa_choreo.py encode writes {features}/choreo_pose/{name}.npz: x (T, 2*CODE_DIM + 3) on the Video Swin clip
grid = [mean upper | lower code vector, upper code id, lower code id, has_pose]. With --choreo_src pose the datasets
append x to the Video Swin feature (so random crops / padding stay aligned) and models/model.py splits it off again.
"""
import os

import numpy as np

CODE_DIM, K = 64, 512
EXTRA = 2 * CODE_DIM + 3
_cache = {}


def attach(video_feat, name, video_path):
    d = os.path.join(os.path.dirname(video_path), 'choreo_pose')
    key = (d, name)
    if key not in _cache:
        _cache[key] = np.load(os.path.join(d, name + '.npz'))['x'].astype(np.float32)
    x = _cache[key]
    T = len(video_feat)
    if len(x) != T:
        y = np.zeros((T, x.shape[1]), np.float32)
        y[:min(T, len(x))] = x[:T]
        x = y
    return np.concatenate([np.asarray(video_feat, np.float32), x], 1)

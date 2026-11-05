"""Extra per-clip inputs for MLAVL on RG / Fis-V (ours, Phase 6c: MLAVL + Bailando).

Each source is a (T, d) array on the PAMFN Video Swin clip grid of a video. The datasets append the chosen sources to
the Video Swin feature (so random crops / padding stay aligned) and models/model.py splits them off again, projects
each to the hidden size and adds the sum to the visual tokens (as --choreo add does for the pose codes).

  pose      (129) Bailando choreographic memory: mean upper | lower code vector + has_pose   (tools/aqa/aqa_choreo.py)
  posecont  (109) the same pose clips without the codebook: mean / std / mean |change| of the 36 pose channels + has
  beat      (9)   Bailando beat alignment: joint speed, kinematic / music beats, Beat Align Score ... (tools/aqa/aqa_beat.py)
  flow      (1024) PAMFN I3D optical-flow feature (not from Bailando; a non-pose reference)

    --extra pose,beat
"""
import os

import numpy as np

DIMS = {'pose': 129, 'posecont': 109, 'beat': 9, 'flow': 1024}
HAS = {'pose': True, 'posecont': True, 'beat': True, 'flow': False}   # last column = has_pose mask
_npz, _flow = {}, {}


def _load(src, name, video_path):
    d = os.path.dirname(video_path)
    if src == 'flow':
        p = video_path.replace('_rgb_VST', '_flow_I3D')
        if p not in _flow:
            _flow[p] = np.load(p, allow_pickle=True).item()
        return np.asarray(_flow[p][name], np.float32)
    key = (src, name, d)
    if key not in _npz:
        if src == 'beat':
            x = np.load(f'{d}/beat/{name}.npz')['b']
        else:
            z = np.load(f'{d}/choreo_pose/{name}.npz')
            x = z['x'] if src == 'pose' else z['c']
            x = np.concatenate([x[:, :DIMS[src] - 1], x[:, -1:]], 1)
        _npz[key] = x.astype(np.float32)
    return _npz[key]


def attach(video_feat, name, video_path, srcs):
    T = len(video_feat)
    parts = [np.asarray(video_feat, np.float32)]
    for s in srcs:
        x = _load(s, name, video_path)
        if len(x) != T:
            y = np.zeros((T, x.shape[1]), np.float32)
            y[:min(T, len(x))] = x[:T]
            x = y
        parts.append(x)
    return np.concatenate(parts, 1)

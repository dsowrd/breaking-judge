"""Bailando-style choreographic memory for breaking judging (ours, Phase 6: MLAVL + Bailando).

Bailando (Siyao et al., CVPR 2022) learns a finite "dance vocabulary": separate VQ-VAEs for the upper and the lower
body quantise short pose windows into codebook entries; the lower half also carries the root trajectory. Here the
same memory is learned on 2D breaking poses and used as the visual input of the pairwise MLAVL judge instead of the
hand-made 114-d pose statistics (tools/battle/wdsf_pose.py), so the judge sees *which dance units* a dancer uses.

    $MLAVL_PY -I tools/battle/choreo_memory.py extract            # features/WDSF/pose_seq/{id}.npz (10 Hz, CPU)
    CUDA_VISIBLE_DEVICES=0 $MLAVL_PY -I tools/battle/choreo_memory.py train
    CUDA_VISIBLE_DEVICES=0 $MLAVL_PY -I tools/battle/choreo_memory.py encode   # features/WDSF/vq/{id}.npz

pose_seq: dancer chosen per frame by the BRACE-trained selector (estimate_pose.learned_select), joints relative to the
hip centre / box height, resampled to 10 Hz (nearest sample within 0.15 s). Codes cover 4 steps = 0.4 s.
vq/{id}.npz keeps the clips_pamfn grid: pose (T, 2*CODE_DIM) = mean quantised upper / lower code vectors of the codes
whose centre falls in the clip, has_pose (T,), plus the raw token sequences (up, low, code_t) for vocabulary analyses.
The VQ-VAE is trained without any judge label on the danced turns of all videos (transductive, label-free).
"""
import argparse
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

F = '/data/mlavl/features/WDSF'
D = '/data/mlavl/datasets/wdsf'
HZ, DOWN, WIN = 10.0, 4, 32                 # sample rate, steps per code, training window (3.2 s)
UPPER = list(range(0, 11))                  # COCO: nose, eyes, ears, shoulders, elbows, wrists
LOWER = list(range(11, 17))                 # hips, knees, ankles
CODE_DIM, K = 64, 512


# ----------------------------------------------------------------------------------------------- extraction
def _extract_one(vid):
    from estimate_pose import learned_select
    dst = f'{F}/pose_seq/{vid}.npz'
    if os.path.exists(dst):
        return vid, -1
    clf = pickle.load(open('/data/mlavl/features/BRACE/dancer_selector.pkl', 'rb'))
    d = dict(np.load(f'{F}/pose_est/{vid}.npz'))
    ap = np.load(f'{F}/appearance/{vid}.npz')
    c = np.load(f'{F}/clips_pamfn/{vid}.npz')
    fps = float(c['fps'])
    dur = (float(c['clip_start_frame'][-1]) + float(c['clip_len'])) / fps if len(c['clip_start_frame']) else 0.0
    n = int(dur * HZ)
    rel = np.zeros((n, 17, 2), np.float32)
    root = np.zeros((n, 2), np.float32)
    ok = np.zeros(n, bool)
    if len(d['frame']) and n:
        sel = learned_select(d, ap['box'], ap['has'], clf)
        idx = np.where(sel >= 0)[0]
        if len(idx):
            t_s = d['frame'][idx] / fps
            kp = d['kp'][idx, sel[idx], :, :2]
            sc = d['kp'][idx, sel[idx], :, 2]
            box = d['box'][idx, sel[idx]]
            h = np.maximum(box[:, 3], 1.0)
            hip = kp[:, 11:13].mean(1)
            bad_hip = sc[:, 11:13].min(1) < 0.2
            hip[bad_hip] = box[bad_hip, :2] + box[bad_hip, 2:4] / 2
            r = (kp - hip[:, None]) / h[:, None, None]
            grid = (np.arange(n) + 0.5) / HZ
            j = np.clip(np.searchsorted(t_s, grid), 1, len(t_s) - 1)
            j = np.where(np.abs(t_s[j - 1] - grid) < np.abs(t_s[j] - grid), j - 1, j)
            near = np.abs(t_s[j] - grid) <= 0.15
            rel[near], ok[near] = r[j[near]], True
            root[near] = (hip[j[near]] / h[j[near], None])          # position in box heights (only deltas are used)
    os.makedirs(f'{F}/pose_seq', exist_ok=True)
    np.savez(dst, rel=rel, root=root, ok=ok, hz=HZ)
    return vid, int(ok.sum())


def extract(a):
    ids = sorted(f[:-4] for f in os.listdir(f'{F}/pose_est') if f.endswith('.npz'))
    with Pool(a.workers) as p:
        for k, (vid, m) in enumerate(p.imap_unordered(_extract_one, ids)):
            print(f'[{k + 1}/{len(ids)}] {vid}: {m} valid steps', flush=True)


def body_parts(rel, root):
    """rel (n,17,2), root (n,2) -> upper (n,22), lower (n,14): lower = 6 joints + root velocity (Bailando)."""
    vel = np.zeros_like(root)
    vel[1:] = root[1:] - root[:-1]
    vel = np.clip(vel, -1, 1)
    up = rel[:, UPPER].reshape(len(rel), -1)
    low = np.concatenate([rel[:, LOWER].reshape(len(rel), -1), vel * 5], 1)
    return np.clip(up, -3, 3), np.clip(low, -3, 3)


def turn_windows():
    """Training windows: WIN steps fully inside a danced turn with >= 90% valid poses."""
    t = pd.read_csv(f'{D}/wdsf_turns.csv')
    U, L = [], []
    for vid, g in t.groupby('video_id'):
        p = f'{F}/pose_seq/{vid}.npz'
        if not os.path.exists(p):
            continue
        z = np.load(p)
        up, low = body_parts(z['rel'], z['root'])
        ok = z['ok']
        for r in g.itertuples():
            s, e = int(r.start_s * HZ), min(int(r.end_s * HZ), len(ok))
            for w in range(s, e - WIN, WIN // 4):
                if ok[w:w + WIN].mean() >= 0.9:
                    U.append(up[w:w + WIN]); L.append(low[w:w + WIN])
    return np.stack(U).astype(np.float32), np.stack(L).astype(np.float32)


# ----------------------------------------------------------------------------------------------- VQ-VAE
def build_vqvae(in_dim):
    import torch
    from torch import nn
    import torch.nn.functional as Fn

    class Res(nn.Module):
        def __init__(self, w, dil):
            super().__init__()
            self.b = nn.Sequential(nn.ReLU(), nn.Conv1d(w, w, 3, padding=dil, dilation=dil), nn.ReLU(), nn.Conv1d(w, w, 1))

        def forward(self, x):
            return x + self.b(x)

    class VQ(nn.Module):
        """EMA codebook (Jukebox / Bailando) with dead-code restart."""
        def __init__(self, k=K, d=CODE_DIM, decay=0.99):
            super().__init__()
            self.k, self.decay = k, decay
            self.register_buffer('emb', torch.randn(k, d) * 0.1)
            self.register_buffer('cnt', torch.ones(k))
            self.register_buffer('sum', self.emb.clone())

        def codes(self, z):                                  # z (N, d)
            return torch.cdist(z, self.emb).argmin(1)

        def forward(self, z):                                # z (B, d, T)
            B, d, T = z.shape
            flat = z.permute(0, 2, 1).reshape(-1, d)
            idx = self.codes(flat)
            q = self.emb[idx]
            if self.training:
                with torch.no_grad():
                    oh = Fn.one_hot(idx, self.k).float()
                    self.cnt.mul_(self.decay).add_(oh.sum(0), alpha=1 - self.decay)
                    self.sum.mul_(self.decay).add_(oh.t() @ flat, alpha=1 - self.decay)
                    n = self.cnt.sum()
                    c = (self.cnt + 1e-5) / (n + self.k * 1e-5) * n
                    self.emb.copy_(self.sum / c[:, None])
                    dead = self.cnt < 1e-2
                    if dead.any():
                        r = flat[torch.randint(len(flat), (int(dead.sum()),), device=flat.device)]
                        self.emb[dead], self.sum[dead], self.cnt[dead] = r, r, 1.0
            commit = Fn.mse_loss(flat, q.detach())
            q = flat + (q - flat).detach()
            usage = torch.bincount(idx, minlength=self.k).float()
            p = usage / usage.sum()
            perplex = torch.exp(-(p * torch.log(p + 1e-10)).sum())
            return q.reshape(B, T, d).permute(0, 2, 1), idx.reshape(B, T), commit, perplex

    class VQVAE(nn.Module):
        def __init__(self, in_dim, w=128):
            super().__init__()
            self.enc = nn.Sequential(nn.Conv1d(in_dim, w, 3, padding=1),
                                     nn.Conv1d(w, w, 4, stride=2, padding=1), Res(w, 1), Res(w, 3),
                                     nn.Conv1d(w, w, 4, stride=2, padding=1), Res(w, 1), Res(w, 3),
                                     nn.Conv1d(w, CODE_DIM, 3, padding=1))
            self.vq = VQ()
            self.dec = nn.Sequential(nn.Conv1d(CODE_DIM, w, 3, padding=1), Res(w, 3), Res(w, 1),
                                     nn.Upsample(scale_factor=2), nn.Conv1d(w, w, 3, padding=1), Res(w, 3), Res(w, 1),
                                     nn.Upsample(scale_factor=2), nn.Conv1d(w, w, 3, padding=1), nn.ReLU(),
                                     nn.Conv1d(w, in_dim, 3, padding=1))

        def forward(self, x):                                # x (B, T, in_dim)
            z = self.enc(x.permute(0, 2, 1))
            q, idx, commit, perplex = self.vq(z)
            y = self.dec(q).permute(0, 2, 1)
            return y, idx, commit, perplex, q

    return VQVAE(in_dim)


def train(a):
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(0)
    U, L = turn_windows()
    print(f'{len(U)} windows of {WIN} steps ({len(U) * WIN / HZ / 3600:.1f} h with 75% overlap)', flush=True)
    dev = torch.device('cuda')
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(U))
    nval = len(U) // 20
    for part, X in (('upper', U), ('lower', L)):
        Xt = torch.from_numpy(X[perm[nval:]]).to(dev)
        Xv = torch.from_numpy(X[perm[:nval]]).to(dev)
        m = build_vqvae(X.shape[2]).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=3e-4)
        for step in range(a.steps + 1):
            m.train()
            b = Xt[torch.randint(len(Xt), (a.batch,), device=dev)]
            y, _, commit, perplex, _ = m(b)
            rec = (y - b).abs().mean()
            vel = ((y[:, 1:] - y[:, :-1]) - (b[:, 1:] - b[:, :-1])).abs().mean()
            loss = rec + vel + 0.02 * commit
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 2000 == 0:
                m.eval()
                with torch.no_grad():
                    yv, iv, _, pv, _ = m(Xv)
                    rv = (yv - Xv).abs().mean().item()
                    base = (Xv - Xv.mean((0, 1))).abs().mean().item()
                print(f'{part} step {step}: train rec {rec.item():.4f} | val rec {rv:.4f} (mean-pose {base:.4f}) '
                      f'| val perplexity {pv.item():.0f} / {K}, codes used {len(torch.unique(iv))}', flush=True)
        torch.save(m.state_dict(), f'{F}/choreo_{part}.pt')


def encode(a):
    import torch
    torch.set_num_threads(4)
    dev = torch.device('cuda')
    ms = {}
    for part, dim in (('upper', len(UPPER) * 2), ('lower', len(LOWER) * 2 + 2)):
        m = build_vqvae(dim).to(dev)
        m.load_state_dict(torch.load(f'{F}/choreo_{part}.pt', map_location=dev))
        ms[part] = m.eval()
    os.makedirs(f'{F}/vq', exist_ok=True)
    ids = sorted(f[:-4] for f in os.listdir(f'{F}/pose_seq') if f.endswith('.npz'))
    for vid in ids:
        z = np.load(f'{F}/pose_seq/{vid}.npz')
        up, low = body_parts(z['rel'], z['root'])
        n = (len(up) // DOWN) * DOWN
        out, toks = [], []
        with torch.no_grad():
            for part, X in (('upper', up), ('lower', low)):
                x = torch.from_numpy(X[:n].astype(np.float32))[None].to(dev)
                _, idx, _, _, q = ms[part](x)
                out.append(q[0].t().cpu().numpy()); toks.append(idx[0].cpu().numpy())
        code_ok = z['ok'][:n].reshape(-1, DOWN).mean(1) >= 0.5
        code_t = (np.arange(n // DOWN) + 0.5) * DOWN / HZ
        c = np.load(f'{F}/clips_pamfn/{vid}.npz')
        fps, L = float(c['fps']), float(c['clip_len'])
        s0 = c['clip_start_frame'] / fps
        T = len(s0)
        feat = np.zeros((T, 2 * CODE_DIM), np.float32)
        has = np.zeros(T, np.int8)
        q = np.concatenate(out, 1)
        for t in range(T):
            mk = (code_t >= s0[t]) & (code_t < s0[t] + L / fps) & code_ok
            if mk.any():
                feat[t], has[t] = q[mk].mean(0), 1
        np.savez(f'{F}/vq/{vid}.npz', pose=feat, has_pose=has, up=toks[0], low=toks[1], code_t=code_t, code_ok=code_ok)
    print(len(ids), 'videos encoded')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['extract', 'train', 'encode'])
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--batch', type=int, default=256)
    a = p.parse_args()
    {'extract': extract, 'train': train, 'encode': encode}[a.cmd](a)

"""Pose-based choreographic memory on the paper datasets (ours, Phase 6b: MLAVL + Bailando on RG / Fis-V).

Stage 1 exactly as for breaking (tools/battle/choreo_memory.py: same VQ-VAE, upper / lower body codebooks, the lower
half carries the root velocity, K = 512, code = 4 pose samples), trained per dataset on the TRAINING split only.
Poses: tools/aqa/aqa_pose.py (25 fps / stride 3 = 8.33 Hz, so one code = 0.48 s).

    CUDA_VISIBLE_DEVICES=0 $MLAVL_PY -I tools/aqa/aqa_choreo.py train  --set fisv
    CUDA_VISIBLE_DEVICES=0 $MLAVL_PY -I tools/aqa/aqa_choreo.py encode --set fisv

encode -> {features}/choreo_pose/{name}.npz: x (T, 2*CODE_DIM + 3) on the Video Swin clip grid of that video
(T = len of the PAMFN feature; clip t = frames [t, t+1) * nframes / T): mean quantised upper | lower code vectors of
the codes whose centre falls in the clip, then the clip's most frequent upper and lower code id, then has_pose.
c: the same grid with continuous pose statistics instead of codes (mean / std / mean |change| of the 36 pose
channels, no VQ; --choreo_src posecont ablation). The trainer appends x (or c) to the Video Swin feature
(ours/choreo_pose.py) so crops / padding stay aligned.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'battle'))
import choreo_memory as CM  # noqa: E402

FEAT = {'fisv': '/data/mlavl/features/FISV', 'rg': '/data/mlavl/features/RG'}
WIN = 32


def train_names(set_):
    have = vst_len(set_)
    return [l.split()[0] for l in open(f'{FEAT[set_]}/train.txt').read().splitlines() if l.strip() and l.split()[0] in have]


def vst_len(set_):
    out = {}
    if set_ == 'fisv':
        d = np.load(f'{FEAT[set_]}/FISV_rgb_VST.npy', allow_pickle=True).item()
        out.update({k: len(v) for k, v in d.items()})
    else:
        for app in ('Ball', 'Clubs', 'Hoop', 'Ribbon'):
            d = np.load(f'{FEAT[set_]}/{app}_rgb_VST.npy', allow_pickle=True).item()
            out.update({k: len(v) for k, v in d.items()})
    return out


def windows(set_):
    U, L = [], []
    for n in train_names(set_):
        p = f'{FEAT[set_]}/pose_seq/{n}.npz'
        if not os.path.exists(p):
            continue
        z = np.load(p)
        up, low = CM.body_parts(z['rel'], z['root'])
        ok = z['ok']
        for w in range(0, len(ok) - WIN, WIN // 4):
            if ok[w:w + WIN].mean() >= 0.9:
                U.append(up[w:w + WIN]); L.append(low[w:w + WIN])
    return np.stack(U).astype(np.float32), np.stack(L).astype(np.float32)


def train(a):
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(0)
    U, L = windows(a.set)
    print(f'{a.set}: {len(U)} training windows of {WIN} steps', flush=True)
    dev = torch.device('cuda')
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(U))
    nval = len(U) // 20
    for part, X in (('upper', U), ('lower', L)):
        Xt = torch.from_numpy(X[perm[nval:]]).to(dev)
        Xv = torch.from_numpy(X[perm[:nval]]).to(dev)
        m = CM.build_vqvae(X.shape[2]).to(dev)
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
                      f'| val perplexity {pv.item():.0f} / {CM.K}, codes used {len(torch.unique(iv))}', flush=True)
        torch.save(m.state_dict(), f'{FEAT[a.set]}/choreo_pose_{part}.pt')


def encode(a):
    import torch
    torch.set_num_threads(4)
    dev = torch.device('cuda')
    ms = {}
    for part, dim in (('upper', len(CM.UPPER) * 2), ('lower', len(CM.LOWER) * 2 + 2)):
        m = CM.build_vqvae(dim).to(dev)
        m.load_state_dict(torch.load(f'{FEAT[a.set]}/choreo_pose_{part}.pt', map_location=dev))
        ms[part] = m.eval()
    out_dir = f'{FEAT[a.set]}/choreo_pose'
    os.makedirs(out_dir, exist_ok=True)
    lens = vst_len(a.set)
    D, cover = CM.CODE_DIM, []
    for name, T in sorted(lens.items()):
        p = f'{FEAT[a.set]}/pose_seq/{name}.npz'
        x = np.zeros((T, 2 * D + 3), np.float32)
        if os.path.exists(p):
            z = np.load(p)
            up, low = CM.body_parts(z['rel'], z['root'])
            n = (len(up) // 4) * 4
            if n >= 4:
                q, ids = [], []
                with torch.no_grad():
                    for part, X in (('upper', up), ('lower', low)):
                        xx = torch.from_numpy(X[:n].astype(np.float32))[None].to(dev)
                        _, idx, _, _, qq = ms[part](xx)
                        q.append(qq[0].t().cpu().numpy()); ids.append(idx[0].cpu().numpy())
                q = np.concatenate(q, 1)
                code_ok = z['ok'][:n].reshape(-1, 4).mean(1) >= 0.5
                centre = z['frame'][:n].reshape(-1, 4).mean(1)
                clip = np.minimum((centre * T / int(z['nframes'])).astype(int), T - 1)
                for t in range(T):
                    mk = (clip == t) & code_ok
                    if mk.any():
                        x[t, :2 * D] = q[mk].mean(0)
                        x[t, 2 * D] = np.bincount(ids[0][mk]).argmax()
                        x[t, 2 * D + 1] = np.bincount(ids[1][mk]).argmax()
                        x[t, 2 * D + 2] = 1
        c = np.zeros_like(x)                                   # continuous pose statistics (ablation: no VQ)
        if os.path.exists(p):
            sclip = np.minimum((z['frame'] * T / int(z['nframes'])).astype(int), T - 1)
            P = np.concatenate([up, low], 1)                    # (n, 36): joints + root velocity
            V = np.abs(np.diff(P, axis=0, prepend=P[:1]))
            for t in range(T):
                mk = (sclip == t) & z['ok']
                if mk.sum() >= 2:
                    c[t, :108] = np.concatenate([P[mk].mean(0), P[mk].std(0), V[mk].mean(0)])
                    c[t, -1] = 1
        np.savez(f'{out_dir}/{name}.npz', x=x, c=c)
        cover.append(x[:, -1].mean())
    print(f'{len(lens)} videos encoded, clips with pose {np.mean(cover):.3f} (videos without any: '
          f'{int(np.sum(np.array(cover) == 0))})')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['train', 'encode'])
    p.add_argument('--set', choices=list(FEAT), required=True)
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--batch', type=int, default=256)
    a = p.parse_args()
    {'train': train, 'encode': encode}[a.cmd](a)

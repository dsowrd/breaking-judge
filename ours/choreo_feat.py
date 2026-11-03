"""Bailando-style choreographic memory on the paper datasets' clip features (ours, Phase 6: MLAVL + Bailando).

RG and Fis-V ship only precomputed clip features (Video Swin 1024-d, AST 768-d), no videos, so the memory is learned
on the visual feature sequence instead of on poses. Stage 1 (as Bailando's VQ-VAE stage): a temporal-convolution
VQ-VAE reconstructs the standardised clip features of the TRAINING split only; each clip is assigned one of K codes,
i.e. a learned, finite vocabulary of performance units. Stage 2 (models/model.py, --choreo):
  graph  MAG^2 gets one extra node per code a video actually uses, connected only to the clips that use it (the
         per-sample action graph of the plan; the text action nodes stay as in the paper);
  add    the quantised code vector of each clip is projected and added to its visual token.
The memory is frozen in stage 2.

    CUDA_VISIBLE_DEVICES=1 $MLAVL_PY ours/choreo_feat.py --dataset rg     # -> ckpt/choreo_rg.pt (all 4 apparatus)
    CUDA_VISIBLE_DEVICES=1 $MLAVL_PY ours/choreo_feat.py --dataset fisv   # -> ckpt/choreo_fisv.pt
"""
import argparse
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as Fn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT = '/data/mlavl/features'


class Res(nn.Module):
    def __init__(self, w, dil):
        super().__init__()
        self.b = nn.Sequential(nn.ReLU(), nn.Conv1d(w, w, 3, padding=dil, dilation=dil), nn.ReLU(), nn.Conv1d(w, w, 1))

    def forward(self, x):
        return x + self.b(x)


class FeatVQVAE(nn.Module):
    """Encoder / EMA codebook / decoder over (B, T, in_dim); one code per clip (no temporal downsampling: a clip is
    already 1-2 s, Bailando's code length)."""
    def __init__(self, in_dim=1024, w=256, code_dim=64, k=128, decay=0.99):
        super().__init__()
        self.k, self.decay, self.code_dim = k, decay, code_dim
        self.enc = nn.Sequential(nn.Conv1d(in_dim, w, 1), Res(w, 1), Res(w, 2), Res(w, 4), nn.Conv1d(w, code_dim, 1))
        self.dec = nn.Sequential(nn.Conv1d(code_dim, w, 1), Res(w, 4), Res(w, 2), Res(w, 1), nn.ReLU(), nn.Conv1d(w, in_dim, 1))
        self.register_buffer('emb', torch.randn(k, code_dim) * 0.1)
        self.register_buffer('cnt', torch.ones(k))
        self.register_buffer('sum', self.emb.clone())
        self.register_buffer('mu', torch.zeros(in_dim))
        self.register_buffer('sd', torch.ones(in_dim))

    def encode(self, x):                                   # x raw (B, T, in_dim) -> z (B, T, d), idx (B, T)
        z = self.enc(((x - self.mu) / self.sd).transpose(1, 2)).transpose(1, 2)
        idx = torch.cdist(z.reshape(-1, self.code_dim), self.emb).argmin(1).reshape(z.shape[:2])
        return z, idx

    def forward(self, x):
        z, idx = self.encode(x)
        flat = z.reshape(-1, self.code_dim)
        q = self.emb[idx.reshape(-1)]
        if self.training:
            with torch.no_grad():
                oh = Fn.one_hot(idx.reshape(-1), self.k).float()
                self.cnt.mul_(self.decay).add_(oh.sum(0), alpha=1 - self.decay)
                self.sum.mul_(self.decay).add_(oh.t() @ flat, alpha=1 - self.decay)
                n = self.cnt.sum()
                c = (self.cnt + 1e-5) / (n + self.k * 1e-5) * n
                self.emb.copy_(self.sum / c[:, None])
                dead = self.cnt < 1e-2
                if dead.any():
                    r = flat[torch.randint(len(flat), (int(dead.sum()),), device=flat.device)].detach()
                    self.emb[dead], self.sum[dead], self.cnt[dead] = r, r, 1.0
        commit = Fn.mse_loss(flat, q.detach())
        qs = (flat + (q - flat).detach()).reshape(z.shape)
        y = self.dec(qs.transpose(1, 2)).transpose(1, 2)
        return y, idx, commit


class ChoreoMemory(nn.Module):
    """Frozen stage-1 memory for MLAVL: raw visual (B, T, in_dim) -> code ids (B, T) and code vectors (B, T, d)."""
    def __init__(self, path):
        super().__init__()
        ck = torch.load(path, map_location='cpu', weights_only=False)
        self.vq = FeatVQVAE(**ck['cfg'])
        self.vq.load_state_dict(ck['state'])
        for p in self.vq.parameters():
            p.requires_grad = False
        self.code_dim, self.k = self.vq.code_dim, self.vq.k

    def train(self, mode=True):                            # always eval
        return super().train(False)

    @torch.no_grad()
    def forward(self, x):
        _, idx = self.vq.encode(x)
        return idx, self.vq.emb[idx]


def load_split(dataset):
    """Training-split feature sequences only."""
    seqs = []
    if dataset == 'rg':
        names = [l.split()[0] for l in open(f'{FEAT}/RG/train.txt').read().splitlines()[1:] if l.strip()]
        for app in ('Ball', 'Clubs', 'Hoop', 'Ribbon'):
            d = np.load(f'{FEAT}/RG/{app}_rgb_VST.npy', allow_pickle=True).item()
            seqs += [np.asarray(d[n], np.float32) for n in names if n.startswith(app + '_')]
    else:
        names = [l.split()[0] for l in open(f'{FEAT}/FISV/train.txt').read().splitlines() if l.strip()]
        d = np.load(f'{FEAT}/FISV/FISV_rgb_VST.npy', allow_pickle=True).item()
        seqs = [np.asarray(d[n], np.float32) for n in names]
    return seqs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['rg', 'fisv'], required=True)
    p.add_argument('--k', type=int, default=128)
    p.add_argument('--code_dim', type=int, default=64)
    p.add_argument('--steps', type=int, default=15000)
    p.add_argument('--win', type=int, default=32)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--out', default=None)
    a = p.parse_args()
    torch.manual_seed(0); torch.set_num_threads(4)
    rng = np.random.default_rng(0)
    dev = torch.device('cuda')
    seqs = load_split(a.dataset)
    allx = np.concatenate(seqs)
    print(f'{a.dataset}: {len(seqs)} training videos, {len(allx)} clips', flush=True)
    perm = rng.permutation(len(seqs)); nval = max(len(seqs) // 10, 1)
    tr = [seqs[i] for i in perm[nval:]]; va = [seqs[i] for i in perm[:nval]]
    cfg = dict(in_dim=allx.shape[1], code_dim=a.code_dim, k=a.k)
    m = FeatVQVAE(**cfg).to(dev)
    m.mu.copy_(torch.from_numpy(allx.mean(0))); m.sd.copy_(torch.from_numpy(allx.std(0) + 1e-6))
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)
    for step in range(a.steps + 1):
        m.train()
        b = []
        for i in rng.integers(0, len(tr), a.batch):
            s = rng.integers(0, len(tr[i]) - a.win + 1); b.append(tr[i][s:s + a.win])
        x = torch.from_numpy(np.stack(b)).to(dev)
        y, _, commit = m(x)
        rec = Fn.mse_loss(y, (x - m.mu) / m.sd)
        loss = rec + 0.02 * commit
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1500 == 0:
            m.eval()
            with torch.no_grad():
                rv, used = [], []
                for s in va:
                    xv = torch.from_numpy(s)[None].to(dev)
                    yv, iv, _ = m(xv)
                    rv.append(Fn.mse_loss(yv, (xv - m.mu) / m.sd).item()); used.append(iv.flatten())
                u = torch.cat(used); pr = torch.bincount(u, minlength=a.k).float(); pr = pr / pr.sum()
                ppl = torch.exp(-(pr * torch.log(pr + 1e-10)).sum()).item()
            print(f'step {step}: train rec {rec.item():.3f} | val rec {np.mean(rv):.3f} (1.0 = mean feature) '
                  f'| val perplexity {ppl:.0f} / {a.k}', flush=True)
    out = a.out or os.path.join(ROOT, 'ckpt', f'choreo_{a.dataset}.pt')
    torch.save(dict(cfg=cfg, state=m.state_dict()), out)
    print('saved', out)


if __name__ == '__main__':
    main()

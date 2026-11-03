"""Comparative breaking judge on Bailando-style dance tokens (ours, Phase 6: MLAVL + Bailando).

Input: the upper / lower choreographic-memory tokens of each turn (tools/battle/choreo_memory.py, features/WDSF/vq/,
one code per 0.4 s). Output per round: d > 0 means the blue athlete wins the criterion. Same rounds, folds
(ours/wdsf/folds.json, 5 event-grouped folds) and prediction format as ours/train_wdsf_pair.py, so
ours/eval_pair.py evaluates both.

--model stats  vocabulary statistics per turn, the WDSF criteria read literally (Vocabulary = how many different
               dance units, Originality = how rare they are in the corpus and how fresh against the dancer's own
               earlier rounds and the opponent's), then logistic regression on blue - red with an intercept (which
               absorbs the second-dancer advantage). Coefficients are saved for interpretation.
--model seq    token sequence judge: code embeddings initialised from the VQ codebooks (the memory itself), a small
               Transformer encoder, masked mean pool -> turn score s; d = s_blue - s_red + b. Shared weights, BCE on
               the judges' winner. Epoch chosen on the largest training event (as train_wdsf_pair.py --inner_val).

    $MLAVL_PY ours/train_vocab_pair.py --model stats --target total --name vocab_stats_total
    CUDA_VISIBLE_DEVICES=0 $MLAVL_PY ours/train_vocab_pair.py --model seq --target total --seed 0 --name vseq_total_s0
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ours.train_wdsf import CRIT, D, F, FOLDS, event_of  # noqa: E402

K = 512


def load_rounds(target):
    js = pd.read_csv(f'{D}/wdsf_judge_scores.csv')
    if target != 'total':
        js = js[(js.system == '5crit') & js.has_criteria]
    rnd = js.groupby(['competition', 'battle_id', 'round'])[target].mean().rename('margin').reset_index()
    turns = pd.read_csv(os.environ.get('WDSF_TURNS', f'{D}/wdsf_turns.csv'))
    cache, toks = {}, {}
    for t in turns.itertuples():
        if t.video_id not in cache:
            cache[t.video_id] = np.load(f'{F}/vq/{t.video_id}.npz')
        z = cache[t.video_id]
        m = (z['code_t'] >= t.start_s) & (z['code_t'] <= t.end_s) & z['code_ok']
        toks[(t.competition, t.battle_id, t.round, t.athlete)] = (z['up'][m].astype(int), z['low'][m].astype(int),
                                                                   t.source, t.video_id)
    rows = []
    for r in rnd.itertuples():
        b = toks.get((r.competition, r.battle_id, r.round, 'blue'))
        rd = toks.get((r.competition, r.battle_id, r.round, 'red'))
        if b is None or rd is None or len(b[0]) < 20 or len(rd[0]) < 20:     # < 8 s of valid pose
            continue
        rows.append(dict(competition=r.competition, battle_id=r.battle_id, round=r.round, event=event_of(r.competition),
                         source=b[2], margin=float(r.margin), blue=b[:2], red=rd[:2]))
    return rows, toks


# ------------------------------------------------------------------------------------------- vocabulary stats
def turn_stats(up, low, logp_up, logp_low, logp_j, prev_self, prev_opp, opp_now):
    j = up * K + low
    f = {}
    for name, x, lp in (('up', up, logp_up), ('low', low, logp_low), ('joint', j, logp_j)):
        n = len(x)
        u, c = np.unique(x, return_counts=True)
        p = c / n
        big = set(zip(x[:-1], x[1:]))
        f[f'{name}_uniq_rate'] = len(u) / n
        f[f'{name}_entropy'] = float(-(p * np.log(p)).sum())
        f[f'{name}_rarity'] = float(np.mean([lp.get(v, lp['_min']) for v in x]))
        f[f'{name}_still'] = float(np.mean(x[1:] == x[:-1]))
        f[f'{name}_bigram_uniq'] = len(big) / max(n - 1, 1)
    us = set(j.tolist())
    f['fresh_self'] = len(us - prev_self) / len(us)                 # not used by this dancer in earlier rounds
    f['fresh_opp'] = len(us - prev_opp) / len(us)                   # not used by the opponent before this turn
    f['overlap_opp_now'] = len(us & opp_now) / len(us)
    f['log_len'] = float(np.log(len(up)))
    return f


def stats_table(rows, toks):
    cu = np.bincount(np.concatenate([v[0] for v in toks.values()]), minlength=K) + 1.0
    cl = np.bincount(np.concatenate([v[1] for v in toks.values()]), minlength=K) + 1.0
    cj = pd.Series(np.concatenate([v[0] * K + v[1] for v in toks.values()])).value_counts()
    lpu = {i: -np.log(cu[i] / cu.sum()) for i in range(K)}; lpu['_min'] = max(lpu.values())
    lpl = {i: -np.log(cl[i] / cl.sum()) for i in range(K)}; lpl['_min'] = max(lpl.values())
    lpj = (-np.log(cj / cj.sum())).to_dict(); lpj['_min'] = -np.log(1 / cj.sum())
    # sets of joint codes per (battle, athlete, round) for freshness
    used = {}
    for (comp, bid, rnd, ath), (u, l, _, _) in toks.items():
        used[(comp, bid, rnd, ath)] = set((u * K + l).tolist())
    X = []
    for r in rows:
        key = (r['competition'], r['battle_id'])
        fs = {}
        for side, opp in (('blue', 'red'), ('red', 'blue')):
            prev_self, prev_opp = set(), set()
            for k in range(1, r['round']):
                prev_self |= used.get(key + (k, side), set())
                prev_opp |= used.get(key + (k, opp), set())
            if side == 'blue':                                   # red danced first in this round
                prev_opp |= used.get(key + (r['round'], opp), set())
            u, l = r[side]
            fs[side] = turn_stats(u, l, lpu, lpl, lpj, prev_self, prev_opp, used.get(key + (r['round'], opp), set()))
        X.append({k: fs['blue'][k] - fs['red'][k] for k in fs['blue']})
    return pd.DataFrame(X)


def run_stats(a, rows, toks, assign):
    from sklearn.linear_model import LogisticRegression
    X = stats_table(rows, toks)
    y = np.array([r['margin'] for r in rows])
    ev = np.array([r['event'] for r in rows])
    fo = np.array([assign[e] for e in ev])
    pred = np.zeros(len(rows))
    coefs = []
    for f in range(5):
        tr, te = (fo != f) & (y != 0), fo == f
        mu, sd = X[tr].mean(), X[tr].std() + 1e-9
        m = LogisticRegression(C=a.C, max_iter=2000).fit(((X[tr] - mu) / sd).values, (y[tr] > 0).astype(int))
        pred[te] = m.decision_function(((X[te] - mu) / sd).values)
        coefs.append(pd.Series(m.coef_[0], index=X.columns))
    c = pd.concat(coefs, axis=1)
    c['mean'] = c.mean(1)
    c.sort_values('mean').to_csv(f'{a.out}/coefficients.csv')
    print(c['mean'].sort_values().round(3).to_string())
    return pred


# ------------------------------------------------------------------------------------------- token sequence judge
def run_seq(a, rows, assign):
    import torch
    from torch import nn
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device('cuda')
    mem = {p: torch.load(f'{F}/choreo_{p}.pt', map_location='cpu')['vq.emb'] for p in ('upper', 'lower')}
    MAXL = 256

    class Judge(nn.Module):
        def __init__(self, d=128):
            super().__init__()
            self.eu = nn.Embedding(K, 64); self.el = nn.Embedding(K, 64)
            self.eu.weight.data.copy_(mem['upper'] / mem['upper'].std()); self.el.weight.data.copy_(mem['lower'] / mem['lower'].std())
            self.pos = nn.Parameter(torch.zeros(MAXL, d))
            layer = nn.TransformerEncoderLayer(d, 4, 256, dropout=a.dropout, batch_first=True)
            self.tf = nn.TransformerEncoder(layer, 2)
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
            self.bias = nn.Parameter(torch.zeros(1))

        def score(self, u, l, mask):
            x = torch.cat([self.eu(u), self.el(l)], -1) + self.pos[:u.shape[1]]
            h = self.tf(x, src_key_padding_mask=~mask)
            h = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
            return self.head(h)[:, 0]

        def forward(self, b, r):
            return self.score(*b) - self.score(*r) + self.bias

    def batch(seqs, train):
        L = []
        for u, l in seqs:
            if train and len(u) > a.crop:
                s = rng.integers(0, len(u) - a.crop + 1); u, l = u[s:s + a.crop], l[s:s + a.crop]
            L.append((u[:MAXL], l[:MAXL]))
        n = max(len(u) for u, _ in L)
        U = np.zeros((len(L), n), int); Lo = np.zeros((len(L), n), int); M = np.zeros((len(L), n), bool)
        for i, (u, l) in enumerate(L):
            U[i, :len(u)], Lo[i, :len(u)], M[i, :len(u)] = u, l, True
        return tuple(torch.from_numpy(x).to(dev) for x in (U, Lo, M))

    def predict(m, idx):
        m.eval(); out = []
        with torch.no_grad():
            for s in range(0, len(idx), 64):
                ii = idx[s:s + 64]
                out.append(m(batch([rows[i]['blue'] for i in ii], False), batch([rows[i]['red'] for i in ii], False)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    y = np.array([r['margin'] for r in rows])
    ev = np.array([r['event'] for r in rows])
    fo = np.array([assign[e] for e in ev])
    pred = np.zeros(len(rows))
    E = [5, 10, 20, 30, 40]
    for f in range(5):
        tr_all = np.where((fo != f) & (y != 0))[0]
        val_ev = pd.Series(ev[tr_all]).value_counts().index[0]
        fit = tr_all[ev[tr_all] != val_ev]; val = tr_all[ev[tr_all] == val_ev]
        te = np.where(fo == f)[0]

        def train_for(idx, epochs, report=None):
            m = Judge().to(dev)
            opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-2)
            snaps = {}
            for ep in range(1, epochs + 1):
                m.train()
                for s in range(0, len(idx), a.batch):
                    ii = rng.permutation(idx)[:a.batch] if False else idx[rng.permutation(len(idx))[:a.batch]]
                    flip = rng.random(len(ii)) < 0.5
                    bb = [rows[i]['red'] if fl else rows[i]['blue'] for i, fl in zip(ii, flip)]
                    rr = [rows[i]['blue'] if fl else rows[i]['red'] for i, fl in zip(ii, flip)]
                    tgt = torch.tensor(((y[ii] > 0) ^ flip).astype(np.float32), device=dev)
                    sign = torch.tensor(np.where(flip, -1.0, 1.0).astype(np.float32), device=dev)
                    # flipped pairs: the order bias belongs to the true blue, so d_flipped = s_red - s_blue - b
                    d = m.score(*batch(bb, True)) - m.score(*batch(rr, True)) + sign * m.bias
                    loss = nn.functional.binary_cross_entropy_with_logits(d, tgt)
                    opt.zero_grad(); loss.backward(); opt.step()
                if report is not None and ep in E:
                    snaps[ep] = ((np.sign(predict(m, report)) == np.sign(y[report])).mean())
            return m, snaps

        _, val_acc = train_for(fit, max(E), report=val)
        best = max(E, key=lambda e: (round(val_acc[e], 4), -e))
        m, _ = train_for(tr_all, best)
        pred[te] = predict(m, te)
        acc = (np.sign(pred[te]) == np.sign(y[te]))[y[te] != 0].mean()
        print(f'fold {f}: val {val_ev} {dict((k, round(v, 3)) for k, v in val_acc.items())} -> epoch {best}, '
              f'test round acc {acc:.3f} (n={len(te)})', flush=True)
    return pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['stats', 'seq'], required=True)
    p.add_argument('--target', default='total', choices=['total'] + CRIT)
    p.add_argument('--name', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--C', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--crop', type=int, default=96)
    a = p.parse_args()
    a.out = os.path.join(ROOT, 'logs', 'wdsf', a.name)
    os.makedirs(a.out, exist_ok=True)
    rows, toks = load_rounds(a.target)
    assign = json.load(open(FOLDS))
    print(f'{a.model} {a.target}: {len(rows)} rounds', flush=True)
    pred = run_stats(a, rows, toks, assign) if a.model == 'stats' else run_seq(a, rows, assign)
    df = pd.DataFrame([dict(competition=r['competition'], battle_id=r['battle_id'], round=r['round'], event=r['event'],
                            source=r['source'], margin=r['margin']) for r in rows])
    df['pred'] = pred
    df.to_csv(f'{a.out}/predictions.csv', index=False)
    ok = df[df.margin != 0]
    print(f'ALL: round acc {(np.sign(ok.pred) == np.sign(ok.margin)).mean():.3f} (n={len(ok)}), '
          f'blue-wins rule {(ok.margin > 0).mean():.3f}')


if __name__ == '__main__':
    main()

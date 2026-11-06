"""Move-level units from the Bailando-style codebook, and their validity for Vocabulary / Originality (ours, Phase 7b).

Phase 7 (criteria_validity.py) found that single 0.4 s codes are pose fragments: within 16 s almost every code is new,
so code diversity / rarity cannot see whether a dancer repeats moves. Here a "move" is a short sequence of codes:
windows of L consecutive codes (hop L/2), described by the mean upper / lower code vectors (vq.emb of the trained
codebooks) of each half of the window (keeps coarse order), standardised, PCA 32, k-means into K move units. Fitted on
all turns without judge labels (as the VQ-VAE itself).

Per turn (the dancer's own turns):
  mv_distinct  expected number of distinct moves in N = 12 windows (rarefaction; removes the turn-length effect)
  mv_ent       entropy of the move distribution, mean over random subsamples of N windows
  mv_rare      mean -log p(move), p from all turns of *other* athletes (leave-athlete-out, add-1)
  mv_rare_bi   same for move transitions (repeats removed)
  mv_novel     share of the turn's moves the dancer did not use in earlier rounds of the same battle (rounds >= 2)

Pre-specified main setting L = 8 (3.2 s), K = 64; every other (L, K) in the grid is reported as sensitivity, none is
chosen on judge scores. Expected signs: all five positive for Vocabulary-specific (distinct, ent, novel) and
Originality-specific (rare, rare_bi).

    $MLAVL_PY -I tools/battle/move_units.py [L K ...]      # default grid 5 8 12 x 32 64 128
"""
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import criteria_validity as cv  # noqa: E402

F, D, KEY, C, TAG = cv.F, cv.D, cv.KEY, cv.C, cv.TAG
N = 12
MV = ['mv_distinct', 'mv_ent', 'mv_rare', 'mv_rare_bi', 'mv_novel']
HYP = {'mv_distinct': 'Vocabulary', 'mv_ent': 'Vocabulary', 'mv_novel': 'Vocabulary',
       'mv_rare': 'Originality', 'mv_rare_bi': 'Originality'}


def emb(part):
    return torch.load(f'{F}/choreo{TAG}_{part}.pt', map_location='cpu', weights_only=True)['vq.emb'].numpy()


def turn_windows(L):
    """per turn: window descriptors (n, 4*64) and window centre times, in dancing order"""
    eu, el = emb('upper'), emb('lower')
    turns = pd.read_csv(f'{D}/wdsf_turns.csv')
    out, cache = [], {}
    for t in turns.itertuples():
        if not os.path.exists(f'{F}/vq{TAG}/{t.video_id}.npz'):
            continue
        if t.video_id not in cache:
            cache[t.video_id] = dict(np.load(f'{F}/vq{TAG}/{t.video_id}.npz'))
        q = cache[t.video_id]
        m = np.where((q['code_t'] >= t.start_s) & (q['code_t'] < t.end_s))[0]
        X = []
        for s in range(0, len(m) - L + 1, L // 2):
            w = m[s:s + L]
            if not q['code_ok'][w].all():
                continue
            h = L // 2
            X.append(np.concatenate([eu[q['up'][w[:h]]].mean(0), eu[q['up'][w[h:]]].mean(0),
                                     el[q['low'][w[:h]]].mean(0), el[q['low'][w[h:]]].mean(0)]))
        out.append((t, np.array(X, np.float32).reshape(-1, 4 * eu.shape[1])))
    return out


def fit_moves(tw, K, seed=0):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    allx = np.concatenate([x for _, x in tw if len(x)])
    mu, sd = allx.mean(0), allx.std(0) + 1e-6
    pca = PCA(32, random_state=seed).fit((allx - mu) / sd)
    km = KMeans(K, n_init=4, random_state=seed).fit(pca.transform((allx - mu) / sd))
    return [(t, km.predict(pca.transform((x - mu) / sd)) if len(x) else np.zeros(0, int)) for t, x in tw]


def rarefied_distinct(seq, n=N):
    """expected distinct values in a random subsample of n items without replacement"""
    from math import comb
    T = len(seq)
    if T < n:
        return np.nan
    return float(sum(1 - comb(T - c, n) / comb(T, n) for c in Counter(seq).values()))


def measures(tm, K, seed=0):
    rng = np.random.default_rng(seed)
    cnt = {}
    for t, s in tm:
        r = s[np.r_[True, s[1:] != s[:-1]]] if len(s) else s
        c = cnt.setdefault(t.name, [Counter(), Counter()])
        c[0].update(s.tolist())
        c[1].update(zip(r[:-1].tolist(), r[1:].tolist()))
    tot = [sum((c[i] for c in cnt.values()), Counter()) for i in range(2)]
    rows = []
    for t, s in tm:
        own = cnt[t.name]
        rest = [tot[i] - own[i] for i in range(2)]
        n0, n1 = sum(rest[0].values()), sum(rest[1].values())
        row = dict(video_id=t.video_id, battle_id=t.battle_id, competition=t.competition, round=t.round,
                   order=t.order, athlete=t.athlete, name=t.name, source=t.source, n_win=len(s))
        if len(s) >= N:
            row['mv_distinct'] = rarefied_distinct(s.tolist())
            ents = []
            for _ in range(20):
                sub = rng.choice(s, N, replace=False)
                p = np.bincount(sub, minlength=K) / N
                p = p[p > 0]
                ents.append(-(p * np.log(p)).sum())
            row['mv_ent'] = float(np.mean(ents))
        if len(s):
            row['mv_rare'] = float(np.mean([-np.log((rest[0][c] + 1) / (n0 + K)) for c in s]))
            r = s[np.r_[True, s[1:] != s[:-1]]]
            bi = list(zip(r[:-1].tolist(), r[1:].tolist()))
            if bi:
                row['mv_rare_bi'] = float(np.mean([-np.log((rest[1][b] + 1) / (n1 + K * K)) for b in bi]))
        rows.append(row)
    df = pd.DataFrame(rows)
    # novelty vs the same dancer's earlier rounds in this battle
    seqs = {(t.battle_id, t.athlete, t.round, t.order): set(s.tolist()) for t, s in tm}
    nov = []
    for t, s in tm:
        prev = set().union(*[v for (b, a, r, o), v in seqs.items() if b == t.battle_id and a == t.athlete and r < t.round])
        nov.append(np.nan if (t.round < 2 or not len(s) or not prev) else float(np.mean([c not in prev for c in s])))
    df['mv_novel'] = nov
    return df


def validity(df, label=''):
    lab = cv.rounds()[KEY + C + ['overall'] + [c + '_spec' for c in C] + ['dur', 'energy']]
    per = df.groupby(KEY + ['athlete'])[MV + ['n_win']].mean().reset_index()
    b = per[per.athlete == 'blue'].set_index(KEY)
    r = per[per.athlete == 'red'].set_index(KEY)
    both = b.index.intersection(r.index)
    diff = (b.loc[both, MV] - r.loc[both, MV]).reset_index()
    d = diff.merge(lab, on=KEY)
    res = {}
    for m, c in HYP.items():
        dm = d.dropna(subset=[m])
        est, lo, hi, p = cv.boot(dm, m, c + '_spec', n=1000)
        others = {o: cv._spear(dm[m].values, dm[o + '_spec'].values) for o in C}
        best = max(others, key=others.get) == c
        res[m] = (est, lo, hi, p, best, len(dm), cv._spear(dm[m].values, dm['dur'].values))
        print(f'  {label} {m:11s} -> {c:11s} n={len(dm):3d} rho {est:+.3f} [{lo:+.3f}, {hi:+.3f}] p={p:.3f} '
              f'own-highest {best}  (rho with duration {res[m][6]:+.2f})  '
              + ' '.join(f'{o[:4]} {v:+.2f}' for o, v in others.items()), flush=True)
    return res


def signature(tm, K, n_neg=20000, seed=0):
    """Do move histograms identify the dancer? Cosine similarity of (dancer, battle) histograms: same dancer in different
    battles vs different dancers; AUC. No judge label. A move unit that means something should repeat within a dancer."""
    from scipy.stats import rankdata
    h = {}
    for t, s in tm:
        if len(s):
            h.setdefault((t.name, t.battle_id), np.zeros(K))[:] += np.bincount(s, minlength=K)
    keys = list(h)
    X = np.array([h[k] / max(np.linalg.norm(h[k]), 1e-9) for k in keys])
    names = np.array([k[0] for k in keys])
    battles = np.array([k[1] for k in keys])
    pos = [X[i] @ X[j] for i in range(len(keys)) for j in np.where((names == names[i]) & (battles != battles[i]))[0] if j > i]
    rng = np.random.default_rng(seed)
    neg = []
    while len(neg) < n_neg:
        i, j = rng.integers(len(keys), size=2)
        if names[i] != names[j]:
            neg.append(X[i] @ X[j])
    r = rankdata(np.r_[pos, neg])
    auc = (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    print(f'dancer signature: AUC {auc:.3f} ({len(pos)} same-dancer pairs, {len(neg)} different-dancer pairs)', flush=True)
    return auc


def main(grid):
    out = []
    for L in grid[0]:
        tw = turn_windows(L)
        for K in grid[1]:
            tm = fit_moves(tw, K)
            df = measures(tm, K)
            allm = np.concatenate([s for _, s in tm])
            sat = df.mv_distinct.median()
            print(f'\nL={L} ({L * 0.4:.1f} s) K={K}: windows {len(allm)}, used moves {len(np.unique(allm))}, '
                  f'median distinct per {N} windows {sat:.1f}, median windows / turn {df.n_win.median():.0f}', flush=True)
            if L == 8 and K == 64:
                df.to_csv(f'{D}/move_measures{TAG}_L8_K64.csv', index=False)
                signature(tm, K)
            for m, v in validity(df, f'L{L}K{K}').items():
                out.append(dict(L=L, K=K, measure=m, criterion=HYP[m], rho=v[0], lo=v[1], hi=v[2], p=v[3],
                                own_highest=v[4], n=v[5], rho_dur=v[6], median_distinct=sat))
    pd.DataFrame(out).to_csv(f'{D}/move_validity_grid{TAG}.csv', index=False)
    print('wrote', f'{D}/move_validity_grid{TAG}.csv')


if __name__ == '__main__':
    a = [int(x) for x in sys.argv[1:]]
    main(([a[0]], [a[1]]) if len(a) == 2 else ([5, 8, 12], [32, 64, 128]))

"""Benchmark table for breaking judging from motion (ours, Phase 7h). One table from existing out-of-fold predictions, so every method is
read on the same rounds, the same event folds and the same two tests:
  round     sign of the prediction vs the judges' margin; identity helps here
  within    order of two rounds of one battle; same dancers, so identity cannot help
References: blue rule; dancer strength (mean margin at other events, not from the video); judges half vs half (random split of the panel).
Battle bootstrap. No fitting here. Writes experiments/benchmark_wdsf.csv.

    $MLAVL_PY tools/battle/benchmark_table.py
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from ours.eval_pair import load  # noqa: E402

D = '/data/mlavl/datasets/wdsf'
L = 'logs/wdsf'
KEY = ['competition', 'battle_id', 'round']
K = KEY[:2]
TARGETS = ['total', 'Technique', 'Vocabulary', 'Originality', 'Execution', 'Musicality']
METHODS = [('Audio AST, linear', 'probe_pamfn_aonly_k8_{t}'), ('Video Swin, linear', 'probe_pamfn_vonly_k8_{t}'),
           ('Swin + AST, linear', 'probe_pamfn_k8_{t}'), ('Pose codes, linear', 'probe_vq2_ft_vonly_k8_{t}'),
           ('MLAVL pose codes + audio', 'vq2_ft_{t}_s*'), ('MLAVL pose codes only', 'na_vq2_ft_{t}_s*')]


def runs(pat, t):
    pat = pat.format(t=t)
    if pat.endswith('*'):
        r = [f'{L}/{pat[:-1]}{s}' for s in (0, 1, 2)]
        return [x for x in r if os.path.exists(f'{x}/predictions.csv')]
    return [f'{L}/{pat}'] if os.path.exists(f'{L}/{pat}/predictions.csv') else []


def within_table(df, col):
    out = []
    for key, g in df.groupby(K):
        for (_, a), (_, b) in itertools.combinations(g.iterrows(), 2):
            dm, dp = a.margin - b.margin, a[col] - b[col]
            if dm != 0 and dp != 0:
                out.append((key, float(np.sign(dm) == np.sign(dp))))
    return pd.DataFrame(out, columns=['b', 'ok'])


def ci(groups, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean() for _ in range(n)]
    return np.percentile(v, 2.5), np.percentile(v, 97.5), float(np.mean(np.array(v) <= .5))


def score(df, col):
    d = df[(df.margin != 0) & (df[col] != 0)]
    ok = (np.sign(d.margin) == np.sign(d[col])).astype(float)
    r = ci([v.values for _, v in ok.groupby([d.competition, d.battle_id])])
    res = dict(n=len(d), round=ok.mean(), r_lo=r[0], r_hi=r[1], r_p50=r[2])
    w = within_table(d, col)
    if len(w):    # dancer strength is constant within a battle, so it has no within-battle order
        q = ci([v.ok.values for _, v in w.groupby('b')])
        res.update(pairs=len(w), within=w.ok.mean(), w_lo=q[0], w_hi=q[1], w_p50=q[2])
    else:
        res.update(within=np.nan)
    return res


def strength(base, t, js):
    d = base[KEY + ['margin']].merge(js.drop_duplicates(KEY)[KEY + ['blue_id', 'red_id']], on=KEY)
    long = pd.concat([d[['competition', 'blue_id', 'margin']].rename(columns={'blue_id': 'a'}),
                      d[['competition', 'red_id']].assign(margin=-d.margin).rename(columns={'red_id': 'a'})])
    per = long.groupby(['a', 'competition']).margin.agg(['sum', 'count'])
    tot = per.groupby('a').sum()

    def other(a, c):
        if a not in tot.index:
            return np.nan
        su, n = tot.loc[a, 'sum'], tot.loc[a, 'count']
        if (a, c) in per.index:
            su, n = su - per.loc[(a, c), 'sum'], n - per.loc[(a, c), 'count']
        return su / n if n else np.nan
    d['pred'] = [other(b, c) - other(r, c) for b, r, c in zip(d.blue_id, d.red_id, d.competition)]
    return d.dropna(subset=['pred'])


def judges(base, t, js, it=20):
    j = js.merge(base[KEY], on=KEY)
    rng = np.random.default_rng(0)
    r, w = [], []
    for _ in range(it):
        rows = []
        for key, g in j.groupby(KEY):
            v = rng.permutation(g[t].values)
            n = len(v) // 2
            rows.append((*key, v[:n].mean(), v[n:].mean()))
        h = pd.DataFrame(rows, columns=KEY + ['pred', 'margin'])
        h = h[(h.margin != 0) & (h.pred != 0)]
        r.append((np.sign(h.pred) == np.sign(h.margin)).mean())
        w.append(within_table(h, 'pred').ok.mean())
    return dict(n=len(base), round=float(np.mean(r)), within=float(np.mean(w)))


if __name__ == '__main__':
    js = pd.read_csv(f'{D}/wdsf_judge_scores.csv')
    js = js[(js.system == '5crit') & js.has_criteria]
    rows = []
    for t in TARGETS:
        base = load([f'{L}/probe_pamfn_k8_{t}'])
        base = base[base.margin != 0].merge(js[KEY].drop_duplicates(), on=KEY).reset_index(drop=True)    # five-slider rounds only
        chk = js.groupby(KEY)[t].mean().reset_index().merge(base, on=KEY)
        print(f'\n== {t}: {len(base)} rounds; margin matches judge mean, corr {chk[t].corr(chk.margin):.4f}', flush=True)
        items = [('Blue rule', dict(n=len(base), round=(base.margin > 0).mean(), within=.5))]
        items.append(('Dancer strength (other events)', score(strength(base, t, js), 'pred')))
        for name, pat in METHODS:
            r = runs(pat, t)
            if r:
                s = score(load(r).merge(base[KEY], on=KEY), 'pred')
                s['seeds'] = len(r) if pat.endswith('*') else 0
                items.append((name, s))
        items.append(('Judges, half vs half', judges(base, t, js)))
        for name, s in items:
            rows.append(dict(target=t, method=name, **s))
            f = lambda x: '' if pd.isna(x) else f'{x:.3f}'
            print(f"{name:32s} round {f(s['round'])} [{f(s.get('r_lo', np.nan))}, {f(s.get('r_hi', np.nan))}]  "
                  f"within {f(s['within'])} [{f(s.get('w_lo', np.nan))}, {f(s.get('w_hi', np.nan))}] p {f(s.get('w_p50', np.nan))}  "
                  f"n {s['n']} pairs {s.get('pairs', '')} seeds {s.get('seeds', '')}", flush=True)
    pd.DataFrame(rows).to_csv('experiments/benchmark_wdsf.csv', index=False, float_format='%.4f')

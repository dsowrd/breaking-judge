"""Evaluation of the within-battle training (ours, Phase 7h; PLAN_weekend_1106.md, seventh addendum). Each line compares a run set with
a reference on the same pairs of rounds of one battle; also round accuracy against the blue rule.

    $MLAVL_PY tools/battle/within_train_eval.py
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
sys.path.insert(0, 'tools/battle')
from ours.eval_pair import load  # noqa: E402

K = ['competition', 'battle_id']
L = 'logs/wdsf'
USUAL = ''    # usual training with audio: logs/wdsf/vq2_ft_{target}_s{seed}


def runs(pre, f='vq2_ft', t='Technique'):
    return load([f'{L}/{pre}{"_" if pre else ""}{f}_{t}_s{s}' for s in (0, 1, 2)])


def pair_table(df):
    out = []
    for key, g in df.groupby(K):
        g = g.sort_values('round')
        for (_, a), (_, b) in itertools.combinations(g.iterrows(), 2):
            dm, dp = a.margin - b.margin, a.pred - b.pred
            if dm != 0:
                out.append((key, a['round'], b['round'], np.nan if dp == 0 else float(np.sign(dm) == np.sign(dp))))
    return pd.DataFrame(out, columns=['b', 'ra', 'rb', 'ok'])


def boot(groups, fn, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    return np.array([fn(pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])) for _ in range(n)])


def within(name, df, ref=None):
    p = pair_table(df).dropna()
    g = [v for _, v in p.groupby('b')]
    v = boot(g, lambda x: x.ok.mean())
    s = f'{name:34s} within {p.ok.mean():.3f} [{np.percentile(v, 2.5):.3f}, {np.percentile(v, 97.5):.3f}] p(<=50%) {(v <= .5).mean():.4f} pairs {len(p)}'
    acc = (np.sign(df.pred) == np.sign(df.margin)).mean()
    s += f' | round acc {acc:.3f} rule {(df.margin > 0).mean():.3f} n={len(df)}'
    print(s, flush=True)
    if ref is not None:
        q = pair_table(ref)
        m = pair_table(df).merge(q, on=['b', 'ra', 'rb'], suffixes=('', '_r')).dropna()
        g = [x for _, x in m.groupby('b')]
        d = boot(g, lambda x: x.ok.mean() - x.ok_r.mean())
        print(f'{"":34s} minus reference {m.ok.mean() - m.ok_r.mean():+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}] p(<=0) {(d <= 0).mean():.3f} on {len(m)} pairs', flush=True)


def strength_corr(df, t='Technique'):
    D = '/data/mlavl/datasets/wdsf'
    KEY = K + ['round']
    js = pd.read_csv(f'{D}/wdsf_judge_scores.csv')
    tr = js[(js.system == '5crit') & js.has_criteria].groupby(KEY + ['blue_id', 'red_id'])[t].mean().reset_index().merge(df[KEY], on=KEY)
    d = df.merge(tr, on=KEY)
    long = pd.concat([d[['competition', 'blue_id', t]].rename(columns={'blue_id': 'a'}),
                      d[['competition', 'red_id']].assign(**{t: -d[t]}).rename(columns={'red_id': 'a'})])
    # other events only, so strength holds nothing from this round's event
    sum_ = long.groupby(['a', 'competition'])[t].agg(['sum', 'count']).reset_index()
    tot = sum_.groupby('a')[['sum', 'count']].sum()
    own = sum_.set_index(['a', 'competition'])

    def other(a, c):
        if a not in tot.index:
            return np.nan
        su, n = tot.loc[a]
        if (a, c) in own.index:
            su, n = su - own.loc[(a, c), 'sum'], n - own.loc[(a, c), 'count']
        return su / n if n else np.nan
    d['strength'] = [other(b, c) - other(r_, c) for b, r_, c in zip(d.blue_id, d.red_id, d.competition)]
    x = d.dropna(subset=['strength'])
    return x.pred.rank().corr(x.strength.rank()), len(x), (np.sign(x.strength) == np.sign(x.margin)).mean()


if __name__ == '__main__':
    print('== cross-validation')
    for t in ('Technique', 'Execution'):
        within(f'no audio, within-trained {t}', runs('wt_na', t=t), runs('na', t=t))
        within(f'audio, within-trained {t}', runs('wt', t=t), runs(USUAL, t=t))
    within('no audio, mix Technique', runs('wm_na'), runs('na'))
    for pre in ('na', USUAL, 'wt_na', 'wt', 'wm_na'):
        try:
            print(pre or 'usual', 'corr with strength %+.3f (n=%d, strength acc %.3f)' % strength_corr(runs(pre)))
        except FileNotFoundError as e:
            print(pre, e)
    print('== linear')
    for t in ('Technique', 'Execution'):
        for n in ('pamfn_vonly', 'pamfn_aonly', 'vq2_ft_vonly'):
            within(f'{n} within-fit {t}', load([f'{L}/probe_{n}_within_k8_{t}']), load([f'{L}/probe_{n}_k8_{t}']))
    print('== holdouts, Technique')
    for x in ('hold', 'hold26'):
        within(f'{x} no audio within-trained', runs(f'{x}_wtna'), runs(f'{x}_na'))
        within(f'{x} audio within-trained', runs(f'{x}_wt'), runs(x))

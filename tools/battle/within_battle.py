"""Within-battle test (ours, Phase 7g; fixed in experiments/PLAN_weekend_1106.md, fourth addendum).

In one battle the two dancers are the same and the second-dancer rule is constant, so only the performance can change between
rounds. For each pair of rounds of one battle: does the prediction order them as the judges' margin does? Chance 50%.

    $MLAVL_PY tools/battle/within_battle.py logs/wdsf/vq2_ft_Technique_s0 logs/wdsf/vq2_ft_Technique_s1 logs/wdsf/vq2_ft_Technique_s2
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from ours.eval_pair import load  # noqa: E402

K = ['competition', 'battle_id']


def pairs(df, col='pred'):
    rows = []
    for key, g in df.groupby(K):
        for (_, a), (_, b) in itertools.combinations(g.iterrows(), 2):
            dm, dp = a.margin - b.margin, a[col] - b[col]
            if dm != 0 and dp != 0:
                rows.append((key, float(np.sign(dm) == np.sign(dp))))
    return pd.DataFrame(rows, columns=['b', 'ok'])


def test(df, col='pred', n=5000, seed=0):
    p = pairs(df, col)
    groups = [g.ok.values for _, g in p.groupby('b')]
    rng = np.random.default_rng(seed)
    v = np.array([np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean() for _ in range(n)])
    return len(p), len(groups), p.ok.mean(), np.percentile(v, 2.5), np.percentile(v, 97.5), (v <= .5).mean()


def show(name, df, col='pred'):
    r = test(df, col)
    print(f'{name:44s} pairs={r[0]:4d} battles={r[1]:3d} acc {r[2]:.3f} [{r[3]:.3f}, {r[4]:.3f}] p(<=50%) {r[5]:.4f}', flush=True)


if __name__ == '__main__':
    show(' + '.join(os.path.basename(x) for x in sys.argv[1:])[:44], load(sys.argv[1:]))

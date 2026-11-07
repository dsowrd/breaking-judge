"""Evaluation of the ninth addendum (experiments/PLAN_weekend_1106.md): identity reliance = accuracy(size) - accuracy(purge),
Swin + AST (pamfn) vs pose codes + audio (vq2_ft), Technique, 3-seed ensembles, battle bootstrap on the same rounds.

    $MLAVL_PY tools/battle/purge_eval.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
sys.path.insert(0, 'tools/battle')
from ours.eval_pair import load, report  # noqa: E402
from within_train_eval import within  # noqa: E402

L, K = 'logs/wdsf', ['competition', 'battle_id', 'round']


def ens(m, f):
    return load([f'{L}/ath{m}_{f}_Technique_s{s}' for s in (0, 1, 2)])


def boot(d, fn, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    g = [x.index.values for _, x in d.groupby(['competition', 'battle_id'])]
    return np.array([fn(d.loc[np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))])]) for _ in range(n)])


def main():
    runs = {(m, f): ens(m, f) for m in ('purge', 'size') for f in ('vq2_ft', 'pamfn')}
    d = None
    for (m, f), x in runs.items():
        x = x[x.margin != 0]
        c = x[K + ['margin', 'pred']].rename(columns={'pred': f'{m}_{f}'})
        d = c if d is None else d.merge(c.drop(columns='margin'), on=K)
    d = d.reset_index(drop=True)
    for c in [c for c in d.columns if c.startswith(('purge', 'size'))]:
        d['ok_' + c] = (np.sign(d[c]) == np.sign(d.margin)).astype(float)
    print(f'{len(d)} rounds on all four ensembles')
    for (m, f), x in runs.items():
        report(f'{m} {f}', x[x.margin != 0].reset_index(drop=True))
    rel = lambda x, f: x[f'ok_size_{f}'].mean() - x[f'ok_purge_{f}'].mean()
    for f in ('pamfn', 'vq2_ft'):
        v = boot(d, lambda x: rel(x, f))
        print(f'reliance {f:7s} {rel(d, f):+.3f} [{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}] p(<=0) {(v <= 0).mean():.4f}')
    v = boot(d, lambda x: rel(x, 'pamfn') - rel(x, 'vq2_ft'))
    print(f'MAIN reliance(Swin) - reliance(codes) {rel(d, "pamfn") - rel(d, "vq2_ft"):+.3f} '
          f'[{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}] p(<=0) {(v <= 0).mean():.4f}')
    v = boot(d, lambda x: x.ok_purge_vq2_ft.mean() - x.ok_purge_pamfn.mean())
    print(f'purge: codes - Swin {d.ok_purge_vq2_ft.mean() - d.ok_purge_pamfn.mean():+.3f} '
          f'[{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}] p(<=0) {(v <= 0).mean():.4f}')
    for (m, f), x in runs.items():
        within(f'{m} {f}', x[x.margin != 0].reset_index(drop=True))


if __name__ == '__main__':
    main()

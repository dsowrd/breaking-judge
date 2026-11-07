"""Pose-error test on the 2022 holdout (ours, Phase 7g; fixed in experiments/PLAN_weekend_1106.md).

Blue-minus-red measures vs Technique-specific (Technique minus the mean of the six sliders) on the rounds of the six-slider
events that have Technique. Both poses, Spearman with a battle bootstrap, then a joint rank regression of old + new pose.

    $MLAVL_PY -I tools/battle/holdout_measures.py
"""
import numpy as np
import pandas as pd

D = '/data/mlavl/datasets/wdsf'
KEY = ['competition', 'battle_id', 'round']
C6 = ['Technique', 'Variety', 'Performativity', 'Musicality', 'Creativity', 'Personality']
M = ['inverted', 'energy', 'spin']


def labels():
    js = pd.read_csv(f'{D}/wdsf_judge_scores.csv')
    js = js[js.system == '6crit']
    js = js[js.competition.isin(js[js.Technique != 0].competition.unique())]
    lab = js.groupby(KEY)[C6].mean()
    lab['spec'] = lab.Technique - lab[C6].mean(1)
    return lab


def diff(path, min_codes=20):
    m = pd.read_csv(path)
    m = m[m.n_codes >= min_codes]
    per = m.groupby(KEY + ['athlete'])[M].mean().reset_index()
    b, r = per[per.athlete == 'blue'].set_index(KEY), per[per.athlete == 'red'].set_index(KEY)
    i = b.index.intersection(r.index)
    return b.loc[i, M] - r.loc[i, M]


def main():
    old, new = diff(f'{D}/criteria_measures2.csv'), diff(f'{D}/criteria_measures2_ft.csv')
    lab = labels()
    i = old.index.intersection(new.index).intersection(lab.index)
    d = pd.concat([old.loc[i].add_suffix('_o'), new.loc[i].add_suffix('_n'), lab.loc[i]], axis=1).reset_index()
    d = d.dropna(subset=[m + s for m in M for s in ('_o', '_n')]).reset_index(drop=True)
    print(f'rounds {len(d)}, battles {d.battle_id.nunique()}, events {d.competition.nunique()}')
    rng = np.random.default_rng(0)
    g = [v.index.values for _, v in d.groupby(['competition', 'battle_id'])]
    S = [np.concatenate([g[j] for j in rng.integers(0, len(g), len(g))]) for _ in range(5000)]
    sp = lambda x, a, b: x[a].rank().corr(x[b].rank())

    def beta(x, cols):
        X = x[cols].rank().values
        X = (X - X.mean(0)) / X.std(0)
        Y = x['spec'].rank().values
        return np.linalg.lstsq(X, (Y - Y.mean()) / Y.std(), rcond=None)[0]

    for m in M:
        for s, nm in (('_o', 'old'), ('_n', 'fine-tuned')):
            v = np.array([sp(d.iloc[k], m + s, 'spec') for k in S])
            print(f'{m:9s} {nm:10s} -> Technique-specific rho {sp(d, m + s, "spec"):+.3f} '
                  f'[{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}] p(<=0) {(v <= 0).mean():.3f}')
        cols = [m + '_o', m + '_n']
        o = np.array([beta(d.iloc[k], cols) for k in S])
        b = beta(d, cols)
        print(f'{m:9s} joint: old {b[0]:+.3f} [{np.percentile(o[:, 0], 2.5):+.3f}, {np.percentile(o[:, 0], 97.5):+.3f}]  '
              f'new {b[1]:+.3f} [{np.percentile(o[:, 1], 2.5):+.3f}, {np.percentile(o[:, 1], 97.5):+.3f}]  '
              f'old-new {b[0] - b[1]:+.3f} [{np.percentile(o[:, 0] - o[:, 1], 2.5):+.3f}, {np.percentile(o[:, 0] - o[:, 1], 97.5):+.3f}]')


if __name__ == '__main__':
    main()

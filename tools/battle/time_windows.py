"""Do judges weigh the end of a turn? (ours, Phase 7g, sixth addendum in experiments/PLAN_weekend_1106.md)

Linear probe on video features pooled in part of each turn: first half, second half, last quarter. Same folds and settings as
probe_diff.py. Overall accuracy, within-battle test, second minus first half paired.

    $MLAVL_PY tools/battle/time_windows.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools/battle'))
from ours.train_wdsf import event_of  # noqa: E402
from within_battle import pairs, test  # noqa: E402

F = '/data/mlavl/features/WDSF'
D = '/data/mlavl/datasets/wdsf'
KEY = ['competition', 'battle_id', 'round']
WIN = {'first': (0, .5), 'second': (.5, 1), 'last25': (.75, 1), 'all': (0, 1)}


def turn_stats(feat):
    turns = pd.read_csv(f'{D}/wdsf_turns.csv')
    cache, out = {}, {}
    for t in turns.itertuples():
        if t.video_id not in cache:
            c = np.load(f'{F}/clips_pamfn/{t.video_id}.npz')
            centre = (c['clip_start_frame'] + float(c['clip_len']) / 2) / float(c['fps'])
            if feat == 'swin':
                v, ok = c['rgb'].astype(np.float32), np.ones(len(centre), bool)
            else:
                q = np.load(f'{F}/{feat}/{t.video_id}.npz')
                v, ok = q['pose'].astype(np.float32), q['has_pose'] > 0
            cache[t.video_id] = (v, centre, ok)
        v, centre, ok = cache[t.video_id]
        L = t.end_s - t.start_s
        for w, (a, b) in WIN.items():
            m = (centre >= t.start_s + a * L) & (centre < t.start_s + b * L) & ok
            if m.sum() >= 4:
                out[(t.competition, t.battle_id, t.round, t.athlete, w)] = np.concatenate([v[m].mean(0), v[m].std(0)])
    return out


def main():
    js = pd.read_csv(f'{D}/wdsf_judge_scores.csv')
    js = js[(js.system == '5crit') & js.has_criteria]
    lab = js.groupby(KEY)[['Technique', 'Execution']].mean()
    assign = json.load(open(os.path.join(ROOT, 'ours/wdsf/folds.json')))
    for feat in ['swin', 'vq2_ft']:
        S = turn_stats(feat)
        common = [k for k in lab.index if all((*k, a, w) in S for a in ('blue', 'red') for w in WIN)]
        for c in ['Technique', 'Execution']:
            res = {}
            keys = [k for k in common if lab.loc[k, c] != 0]
            y = lab.loc[keys, c].values
            ev = np.array([event_of(k[0]) for k in keys])
            for w in WIN:
                X = np.stack([S[(*k, 'blue', w)] - S[(*k, 'red', w)] for k in keys])
                pred = np.zeros(len(y))
                for f in sorted(set(assign.values())):
                    tr = np.array([assign[e] != f for e in ev])
                    m = make_pipeline(StandardScaler(), PCA(8, random_state=0), Ridge(alpha=10.0))
                    m.fit(np.concatenate([X[tr], -X[tr]]), np.concatenate([y[tr], -y[tr]]))
                    pred[~tr] = m.predict(X[~tr])
                d = pd.DataFrame(keys, columns=KEY).assign(margin=y, pred=pred)
                r = test(d)
                res[w] = d
                print(f'{feat:7s} {c:10s} {w:7s} n={len(d)} overall {(np.sign(pred) == np.sign(y)).mean():.3f} rule {(y > 0).mean():.3f} | '
                      f'within {r[2]:.3f} [{r[3]:.3f}, {r[4]:.3f}] p {r[5]:.4f}', flush=True)
            a, b = pairs(res['second']), pairs(res['first'])
            if len(a) == len(b):
                dd = a.assign(o=b.ok.values)
                g = [x[['ok', 'o']].values for _, x in dd.groupby('b')]
                rng = np.random.default_rng(0)
                v = []
                for _ in range(5000):
                    z = np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))])
                    v.append(z[:, 0].mean() - z[:, 1].mean())
                print(f'{feat:7s} {c:10s} second - first within {a.ok.mean() - b.ok.mean():+.3f} [{np.percentile(v, 2.5):+.3f}, '
                      f'{np.percentile(v, 97.5):+.3f}] p(<=0) {(np.array(v) <= 0).mean():.3f}', flush=True)
            else:
                print('pairs differ', len(a), len(b))


if __name__ == '__main__':
    main()

"""Where in time is the audio signal for a judged round? (ours, Phase 7g, fifth addendum in experiments/PLAN_weekend_1106.md)

Per turn, AST clip statistics (mean, std; clips_pamfn) and RMS loudness in dB in windows relative to the turn: 15 s before the start,
first half, second half, 15 s after the end. Blue minus red per round, antisymmetric ridge (PCA 8, the same event folds as probe_diff.py),
overall accuracy and the within-battle test. Loudness is used directly, no fitting.

    $MLAVL_PY -I tools/battle/audio_windows.py rms         # 1 s RMS per video from the mp4 audio, {F}/rms/{id}.npy (saved)
    $MLAVL_PY -I tools/battle/audio_windows.py eval
"""
import json
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools/battle'))
F = os.environ.get('WDSF_F', '/data/mlavl/features/WDSF')
D = os.environ.get('WDSF_D', '/data/mlavl/datasets/wdsf')
V = os.environ.get('WDSF_V', '/data/mlavl/videos/wdsf')
FFMPEG = os.environ.get('FFMPEG', '/home/irl/anaconda3/bin/ffmpeg')
KEY = ['competition', 'battle_id', 'round']
SR = 8000
WINS = ['before', 'first', 'second', 'after']


def _rms(vid):
    dst = f'{F}/rms/{vid}.npy'
    if os.path.exists(dst):
        return vid
    b = subprocess.run([FFMPEG, '-v', 'error', '-i', f'{V}/{vid}.mp4', '-vn', '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'],
                       capture_output=True).stdout
    x = np.frombuffer(b, np.float32)
    n = len(x) // SR
    r = np.sqrt((x[:n * SR].reshape(n, SR) ** 2).mean(1)) if n else np.zeros(0)
    np.save(dst, r.astype(np.float32))
    return vid


def rms():
    os.makedirs(f'{F}/rms', exist_ok=True)
    vids = sorted(pd.read_csv(f'{D}/wdsf_turns.csv').video_id.unique())
    with Pool(4) as p:
        for k, _ in enumerate(p.imap_unordered(_rms, vids)):
            if k % 50 == 0:
                print(k + 1, len(vids), flush=True)


def span(w, s, e, pad=15.0):
    return {'before': (s - pad, s), 'first': (s, (s + e) / 2), 'second': ((s + e) / 2, e), 'after': (e, e + pad)}[w]


def table():
    turns = pd.read_csv(f'{D}/wdsf_turns.csv')
    rows, cache = [], {}
    for t in turns.itertuples():
        if t.video_id not in cache:
            d = np.load(f'{F}/clips_pamfn/{t.video_id}.npz')
            centre = (d['clip_start_frame'] + float(d['clip_len']) / 2) / float(d['fps'])
            cache[t.video_id] = (d['audio'].astype(np.float32), centre, np.load(f'{F}/rms/{t.video_id}.npy'))
        a, centre, r = cache[t.video_id]
        row = dict(video_id=t.video_id, competition=t.competition, battle_id=t.battle_id, round=t.round, athlete=t.athlete, source=t.source)
        for w in WINS:
            lo, hi = span(w, t.start_s, t.end_s)
            m = (centre >= lo) & (centre < hi)
            row[f'{w}_ok'] = m.sum() >= 4
            row[f'{w}_a'] = np.concatenate([a[m].mean(0), a[m].std(0)]) if m.sum() >= 4 else None
            q = r[max(int(lo), 0):max(int(hi), 0)]
            row[f'{w}_db'] = float(20 * np.log10(np.sqrt((q ** 2).mean()) + 1e-5)) if len(q) >= 3 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from ours.eval_pair import boot
    from ours.train_wdsf import event_of
    from within_battle import test
    js = pd.read_csv(f'{D}/wdsf_judge_scores.csv')
    js = js[(js.system == '5crit') & js.has_criteria]
    C = ['Technique', 'Vocabulary', 'Originality', 'Execution', 'Musicality']
    lab = js.groupby(KEY)[C].mean().reset_index()
    t = table()
    bl, rd = t[t.athlete == 'blue'].set_index(KEY), t[t.athlete == 'red'].set_index(KEY)
    idx = bl.index.intersection(rd.index).intersection(lab.set_index(KEY).index)
    assign = json.load(open(os.path.join(ROOT, 'ours/wdsf/folds.json')))
    L = lab.set_index(KEY).loc[idx]
    ev = np.array([event_of(k[0]) for k in idx])
    print(f'{len(idx)} rounds')
    for c in ['Technique', 'Execution', 'Musicality']:
        y = L[c].values
        for w in WINS:
            ok = (bl.loc[idx, f'{w}_ok'].values & rd.loc[idx, f'{w}_ok'].values).astype(bool) & (y != 0)
            X = np.stack([bl.loc[k, f'{w}_a'] - rd.loc[k, f'{w}_a'] for k, o in zip(idx, ok) if o])
            yy, e = y[ok], ev[ok]
            pred = np.zeros(len(yy))
            for f in sorted(set(assign.values())):
                tr = np.array([assign[v] != f for v in e])
                m = make_pipeline(StandardScaler(), PCA(8, random_state=0), Ridge(alpha=10.0))
                m.fit(np.concatenate([X[tr], -X[tr]]), np.concatenate([yy[tr], -yy[tr]]))
                pred[~tr] = m.predict(X[~tr])
            d = pd.DataFrame(list(np.array(idx, dtype=object)[ok]), columns=KEY).assign(margin=yy, pred=pred)
            r = test(d)
            print(f'{c:10s} AST {w:7s} n={len(d):3d} overall {(np.sign(pred) == np.sign(yy)).mean():.3f} rule {(yy > 0).mean():.3f} | '
                  f'within pairs={r[0]} {r[2]:.3f} [{r[3]:.3f}, {r[4]:.3f}] p {r[5]:.4f}', flush=True)
            db = bl.loc[idx, f'{w}_db'].values[ok] - rd.loc[idx, f'{w}_db'].values[ok]
            keep = np.isfinite(db)
            e2 = d[keep].assign(pred=db[keep])
            r = test(e2)
            rho = pd.Series(db[keep]).rank().corr(pd.Series(yy[keep]).rank())
            print(f'{c:10s} dB  {w:7s} n={keep.sum():3d} louder-wins {(np.sign(db[keep]) == np.sign(yy[keep])).mean():.3f} rho {rho:+.3f} | '
                  f'within {r[2]:.3f} [{r[3]:.3f}, {r[4]:.3f}] p {r[5]:.4f}  mean blue-red {np.nanmean(db):+.2f} dB', flush=True)


if __name__ == '__main__':
    {'rms': rms, 'eval': evaluate}[sys.argv[1]]()

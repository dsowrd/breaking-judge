"""Do interpretable motion measures track the judges' *criterion-specific* scores? (ours, Phase 7: MLAVL + Bailando).

WDSF judges score each round on five criteria (Technique, Vocabulary, Originality, Execution, Musicality) as signed
sliders between the two dancers. The criteria share a large common part (overall preference), so a measure that only
tracks "the better dancer" would correlate with all five. The test here is discriminant: a measure built for one
criterion should agree with that criterion's *specific* part (criterion minus the mean of the five) more than with the
other criteria's specific parts.

Measures per turn (the dancer's own turns from wdsf_turns.csv), then blue minus red per round:
  Vocabulary   voc_up, voc_low   distinct upper / lower Bailando-style codes per 40 codes (16 s), contiguous windows
               voc_ent           entropy of the (upper, lower) code pairs per 40-code window
  Originality  rare_up, rare_low mean -log p(code), p from all turns of *other* athletes (leave-athlete-out, add-1)
               rare_bi           same for lower-code transitions (repeats removed)
  Musicality   bas               Bailando Beat Align Score: kinematic beats (local minima of joint speed) vs music
                                 beats (librosa), sigma 0.1 s
               hit               share of music beats followed by a kinematic beat within 0.1 s
               sync              corr(joint speed, onset strength) at 10 Hz
  Technique    inverted          share of pose samples with the nose below the hips (power moves, freezes)
               spin              90th percentile angular speed of the shoulder line (rad/s)
               energy            90th percentile joint speed (box heights / s)
  Execution    jerk              median joint jerk (lower = cleaner, so the expected sign is negative)
  Controls     dur, track        turn duration, share of samples with a tracked dancer

Codes and poses: tools/battle/choreo_memory.py (vq/, pose_seq/, 10 Hz, 1 code = 0.4 s); music beats are cached in
{F}/beats/{id}.npz. Nothing is fitted on judge scores.

    $MLAVL_PY -I tools/battle/criteria_validity.py beats      # librosa beats per video (CPU, nice)
    $MLAVL_PY -I tools/battle/criteria_validity.py measures   # -> {D}/criteria_measures.csv (one row per turn)
    $MLAVL_PY -I tools/battle/criteria_validity.py validity   # label reliability + measure x criterion matrix
"""
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

F = os.environ.get('WDSF_F', '/data/mlavl/features/WDSF')     # WDSF_F, WDSF_D, WDSF_V: another set, e.g. the 2026 holdout
D = os.environ.get('WDSF_D', '/data/mlavl/datasets/wdsf')
V = os.environ.get('WDSF_V', '/data/mlavl/videos/wdsf')
FFMPEG = os.environ.get('FFMPEG', '/home/irl/anaconda3/bin/ffmpeg')
TAG = os.environ.get('WDSF_POSE_TAG', '')     # '2' = YOLO11x-pose + track selection (pose_seq2, vq2, *2.csv)
C = ['Technique', 'Vocabulary', 'Originality', 'Execution', 'Musicality']
KEY = ['competition', 'battle_id', 'round']
SR, HOP, HZ, WIN, SIGMA = 22050, 512, 10.0, 40, 0.1
MEAS = ['voc_up', 'voc_low', 'voc_ent', 'rare_up', 'rare_low', 'rare_bi', 'bas', 'hit', 'sync',
        'inverted', 'spin', 'energy', 'jerk', 'dur', 'track']
HYP = {'voc_up': 'Vocabulary', 'voc_low': 'Vocabulary', 'voc_ent': 'Vocabulary', 'rare_up': 'Originality',
       'rare_low': 'Originality', 'rare_bi': 'Originality', 'bas': 'Musicality', 'hit': 'Musicality',
       'sync': 'Musicality', 'inverted': 'Technique', 'spin': 'Technique', 'energy': 'Technique', 'jerk': 'Execution'}


# ----------------------------------------------------------------------------------------------- music beats
def _beats(vid):
    import librosa
    dst = f'{F}/beats/{vid}.npz'
    if os.path.exists(dst):
        return vid
    raw = subprocess.run([FFMPEG, '-v', 'error', '-threads', '1', '-i', f'{V}/{vid}.mp4', '-vn', '-ac', '1',
                          '-ar', str(SR), '-f', 's16le', '-'], capture_output=True).stdout
    y = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    onset = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    _, beats = librosa.beat.beat_track(onset_envelope=onset, sr=SR, hop_length=HOP, units='time')
    np.savez(dst, beats=beats, onset=(onset / (onset.mean() + 1e-6)).astype(np.float32),
             onset_t=librosa.times_like(onset, sr=SR, hop_length=HOP).astype(np.float32))
    return vid


def beats():
    os.makedirs(f'{F}/beats', exist_ok=True)
    vids = sorted(pd.read_csv(f'{D}/wdsf_turns.csv').video_id.unique())
    with Pool(6) as p:
        for k, _ in enumerate(p.imap_unordered(_beats, vids)):
            if k % 50 == 0:
                print(f'beats {k + 1}/{len(vids)}', flush=True)


# ----------------------------------------------------------------------------------------------- measures
def _windows(x, w=WIN):
    """contiguous non-overlapping windows of w codes (the last partial window is dropped unless it is the only one)"""
    n = len(x) // w
    return [x[i * w:(i + 1) * w] for i in range(n)] if n else ([x] if len(x) >= w // 2 else [])


def _entropy(x):
    _, c = np.unique(x, return_counts=True, axis=0)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def _kinematic(rel, root, ok):
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import argrelextrema
    pos = rel + root[:, None]
    sp = np.zeros(len(pos))
    sp[1:] = np.linalg.norm(pos[1:] - pos[:-1], axis=-1).mean(1) * HZ
    ok2 = ok.copy()
    ok2[1:] &= ok[:-1]
    sp = np.where(ok2, np.clip(sp, 0, 20), np.nan)
    fill = np.nan_to_num(sp, nan=np.nanmedian(sp) if np.isfinite(sp).any() else 0)
    sm = gaussian_filter1d(fill, 1.0)
    mins = argrelextrema(sm, np.less, order=1)[0]
    return sm, ok2, mins[ok2[mins]]


def turn_measures(t, vq, ps, bt, freq):
    lo, hi = t.start_s, t.end_s
    m = (vq['code_t'] >= lo) & (vq['code_t'] < hi) & vq['code_ok']
    up, low = vq['up'][m], vq['low'][m]
    out = dict(dur=hi - lo, n_codes=int(m.sum()))
    if len(up) >= WIN // 2:
        wu, wl = _windows(up), _windows(low)
        out['voc_up'] = np.mean([len(np.unique(w)) / len(w) * WIN for w in wu])
        out['voc_low'] = np.mean([len(np.unique(w)) / len(w) * WIN for w in wl])
        out['voc_ent'] = np.mean([_entropy(np.stack([a, b], 1)) for a, b in zip(wu, wl)])
        fu, fl, fb = freq
        out['rare_up'] = float(np.mean([-np.log(fu(c)) for c in up]))
        out['rare_low'] = float(np.mean([-np.log(fl(c)) for c in low]))
        seq = low[np.r_[True, low[1:] != low[:-1]]]
        bi = list(zip(seq[:-1], seq[1:]))
        out['rare_bi'] = float(np.mean([-np.log(fb(b)) for b in bi])) if bi else np.nan
    tg = (np.arange(len(ps['ok'])) + 0.5) / HZ
    pm = (tg >= lo) & (tg < hi)
    rel, root, ok = ps['rel'][pm], ps['root'][pm], ps['ok'][pm]
    out['track'] = float(ok.mean()) if len(ok) else 0.0
    if ok.sum() >= 30:
        r = rel[ok]
        out['inverted'] = float((r[:, 0, 1] > 0).mean())                  # nose below hip centre (image y down)
        sh = r[:, 6] - r[:, 5]
        ang = np.unwrap(np.arctan2(sh[:, 1], sh[:, 0]))
        okc = ok.copy()
        okc[1:] &= ok[:-1]
        a_full = np.full(len(ok), np.nan)
        a_full[ok] = ang
        da = np.abs(np.diff(a_full)) * HZ
        da = da[np.isfinite(da)]
        out['spin'] = float(np.percentile(da, 90)) if len(da) else np.nan
        sm, ok2, kmin = _kinematic(rel, root, ok)
        out['energy'] = float(np.percentile(sm[ok2], 90)) if ok2.any() else np.nan
        pos = rel + root[:, None]
        j = np.diff(pos, n=3, axis=0) * HZ ** 3
        okj = ok[3:] & ok[2:-1] & ok[1:-2] & ok[:-3]
        out['jerk'] = float(np.median(np.linalg.norm(j[okj], axis=-1).mean(1))) if okj.any() else np.nan
        kb = tg[pm][kmin]
        mb = bt['beats'][(bt['beats'] >= lo) & (bt['beats'] < hi)]
        if len(kb) and len(mb):
            out['bas'] = float(np.mean([np.exp(-np.min((mb - k) ** 2) / (2 * SIGMA ** 2)) for k in kb]))
            out['hit'] = float(np.mean([np.any((kb - b >= 0) & (kb - b < 0.1)) for b in mb]))
        on = np.interp(tg[pm], bt['onset_t'], bt['onset'])
        if sm[ok2].std() > 1e-6 and on[ok2].std() > 1e-6:
            out['sync'] = float(np.corrcoef(sm[ok2], on[ok2])[0, 1])
    return out


def measures():
    turns = pd.read_csv(f'{D}/wdsf_turns.csv')
    vids = [v for v in turns.video_id.unique() if os.path.exists(f'{F}/vq{TAG}/{v}.npz')]
    turns = turns[turns.video_id.isin(vids)]
    vq = {v: dict(np.load(f'{F}/vq{TAG}/{v}.npz')) for v in vids}
    # code frequencies per athlete, for leave-athlete-out rarity
    from collections import Counter
    cnt = {}
    for t in turns.itertuples():
        q = vq[t.video_id]
        m = (q['code_t'] >= t.start_s) & (q['code_t'] < t.end_s) & q['code_ok']
        low = q['low'][m]
        seq = low[np.r_[True, low[1:] != low[:-1]]] if len(low) else low
        c = cnt.setdefault(t.name, [Counter(), Counter(), Counter()])
        c[0].update(q['up'][m].tolist())
        c[1].update(low.tolist())
        c[2].update(zip(seq[:-1].tolist(), seq[1:].tolist()))
    tot = [sum((c[i] for c in cnt.values()), Counter()) for i in range(3)]
    rows = []
    for name in turns.name.unique():
        own = cnt[name]
        rest = [tot[i] - own[i] for i in range(3)]
        n = [sum(r.values()) for r in rest]
        freq = (lambda c, r=rest[0], n=n[0]: (r[c] + 1) / (n + 512),
                lambda c, r=rest[1], n=n[1]: (r[c] + 1) / (n + 512),
                lambda b, r=rest[2], n=n[2]: (r[b] + 1) / (n + 512 * 512))
        for t in turns[turns.name == name].itertuples():
            ps = np.load(f'{F}/pose_seq{TAG}/{t.video_id}.npz')
            bt = np.load(f'{F}/beats/{t.video_id}.npz')
            rows.append(dict(video_id=t.video_id, competition=t.competition, battle_id=t.battle_id, round=t.round,
                             order=t.order, athlete=t.athlete, name=t.name, source=t.source,
                             **turn_measures(t, vq[t.video_id], ps, bt, freq)))
    df = pd.DataFrame(rows)
    df.to_csv(f'{D}/criteria_measures{TAG}.csv', index=False)
    print(df[MEAS].describe().T.round(3).to_string())


# ----------------------------------------------------------------------------------------------- validity
def rounds(min_codes=20):
    js = pd.read_csv(f'{D}/wdsf_judge_scores.csv')
    js = js[(js.system == '5crit') & js.has_criteria]
    lab = js.groupby(KEY)[C].mean()
    lab['overall'] = lab[C].mean(1)
    for c in C:
        lab[c + '_spec'] = lab[c] - lab['overall']
    m = pd.read_csv(f'{D}/criteria_measures{TAG}.csv')
    m = m[m.n_codes >= min_codes]
    per = m.groupby(KEY + ['athlete'])[MEAS].mean().reset_index()
    per['hand'] = m.groupby(KEY + ['athlete']).source.agg(lambda s: (s != 'auto').all()).values
    b = per[per.athlete == 'blue'].set_index(KEY)
    r = per[per.athlete == 'red'].set_index(KEY)
    both = b.index.intersection(r.index).intersection(lab.index)
    diff = b.loc[both, MEAS] - r.loc[both, MEAS]
    diff['hand'] = b.loc[both, 'hand'] & r.loc[both, 'hand']
    return diff.join(lab.loc[both]).reset_index()


def _spear(x, y):
    from scipy.stats import rankdata
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return np.nan
    return float(np.corrcoef(rankdata(x[ok]), rankdata(y[ok]))[0, 1])


def boot(df, x, y, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    groups = [g.index.values for _, g in df.groupby('battle_id')]
    est = _spear(df[x].values, df[y].values)
    bs = []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        d = df.loc[idx]
        bs.append(_spear(d[x].values, d[y].values))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    p = 2 * min(np.mean(np.array(bs) <= 0), np.mean(np.array(bs) >= 0))
    return est, lo, hi, p


def validity():
    df = rounds()
    print(f'rounds with both dancers measured: {len(df)} ({df.hand.sum()} hand-segmented), '
          f'battles {df.battle_id.nunique()}, events {df.competition.nunique()}')
    print('\nSpearman rho, blue-minus-red measure vs judge mean (rows: measure; cols: raw criterion | specific part)')
    targets = C + ['overall'] + [c + '_spec' for c in C]
    tab = pd.DataFrame({t: [_spear(df[m].values, df[t].values) for m in MEAS] for t in targets}, index=MEAS)
    print(tab.round(3).to_string())
    print('\nPre-specified hypotheses: measure vs its own criterion-specific part (battle bootstrap 95% CI, 2-sided p)')
    for m, c in HYP.items():
        e, lo, hi, p = boot(df, m, c + '_spec')
        others = [tab.loc[m, o + '_spec'] for o in C if o != c]
        sign = -1 if m == 'jerk' else 1
        best = all(sign * tab.loc[m, c + '_spec'] > sign * o for o in others)
        print(f'  {m:9s} -> {c:11s} rho {e:+.3f} [{lo:+.3f}, {hi:+.3f}] p={p:.3f}   '
              f'highest of the 5 specific parts: {best}')
    out = f'{D}/criteria_validity_table{TAG}.csv'
    tab.to_csv(out)
    print('\nwrote', out)


if __name__ == '__main__':
    {'beats': beats, 'measures': measures, 'validity': validity}[sys.argv[1]]()

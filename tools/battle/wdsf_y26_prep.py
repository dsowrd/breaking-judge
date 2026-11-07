"""Second holdout: WDSF World Championships Chengdu 2026, Youth, B-Boys and B-Girls (ours, Phase 7g).

These 32 battles have five sliders and videos on the same channel, but wdsf_match_videos.py rejects them because youth events
need "youth" in the title and these titles do not have it. Kept apart from the training data: own folders, own turns file.
A video matches when the title says "Chengdu 2026" and its two names equal the two athletes exactly (either order).

    $MLAVL_PY -I tools/battle/wdsf_y26_prep.py    # writes {OUT}/wdsf_battle_videos.csv, video_ids.txt, links the score table
"""
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdsf_match_videos import name_tokens, norm  # noqa: E402

D = '/data/mlavl/datasets/wdsf'
OUT = '/data/mlavl/datasets/wdsf_y26'
COMPS = ['World-Championships-chengdu-Youth-Breaking-1-vs-1-B-Boys-68021', 'World-Championships-chengdu-Youth-Breaking-1-vs-1-B-Girls-68208']


def main():
    os.makedirs(OUT, exist_ok=True)
    v = pd.read_csv(f'{D}/wdsf_battle_videos.csv')
    v = v[v.competition.isin(COMPS)].copy()
    y = pd.read_csv(f'{D}/yt/breakingforgold.tsv', sep='\t', header=None, names=['id', 'dur', 'title'], quoting=3)
    y = y[y.title.fillna('').str.contains('Chengdu 2026')].copy()
    pairs = {}
    for r in y.itertuples():
        head = re.split(r'[|►]', r.title)[0]
        parts = re.split(r'\bvs\.?\b', head, flags=re.I)
        if len(parts) == 2:
            pairs.setdefault(frozenset((norm(parts[0]), norm(parts[1]))), []).append(r)
    rows = []
    for b in v.itertuples():
        c = pairs.get(frozenset((name_tokens(b.blue), name_tokens(b.red))), [])
        if len(c) == 1:
            rows.append(dict(competition=b.competition, battle_id=b.battle_id, stage=b.stage, blue=b.blue, red=b.red, month=b.month,
                             system=b.system, has_criteria=b.has_criteria, n_rounds=b.n_rounds, n_candidates=1,
                             video_id=c[0].id, video_title=c[0].title, duration=c[0].dur))
        else:
            print('not matched', b.blue, b.red, b.stage, len(c))
    out = pd.DataFrame(rows)
    out.to_csv(f'{OUT}/wdsf_battle_videos.csv', index=False)
    open(f'{OUT}/video_ids.txt', 'w').write('\n'.join(out.video_id) + '\n')
    for f in ['wdsf_judge_scores.csv', 'competitions.tsv']:
        if not os.path.exists(f'{OUT}/{f}'):
            os.symlink(f'{D}/{f}', f'{OUT}/{f}')
    print(f'{len(out)} of {len(v)} battles matched, {out.n_rounds.sum()} rounds, {out.video_id.nunique()} videos, '
          f'duration {out.duration.min()}-{out.duration.max()} s')
    print(out.groupby('competition').size().to_string())
    print(out[['stage', 'blue', 'red', 'n_rounds', 'duration', 'video_title']].sample(8, random_state=0).to_string())


if __name__ == '__main__':
    main()

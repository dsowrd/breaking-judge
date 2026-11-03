"""Diagnostic (ours, Phase 5c): are the hand-labelled battles better only because their turns are exact?

Writes wdsf_turns_handcv.csv = wdsf_turns.csv with the turns of every hand-labelled battle replaced by the turns the
detector finds when its own event is held out (leave-one-event-out, as in `wdsf_turn_detector.py eval`). Training
the comparative MLAVL on this file (WDSF_TURNS=... ours/train_wdsf_pair.py) and scoring the same hand-labelled rounds
separates "the turns are exact" from "these battles are easier".

    CUDA_VISIBLE_DEVICES=0 $MLAVL_PY -I tools/battle/wdsf_turns_cv.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdsf_turn_detector import D, brace_samples, predict, segment, train, wdsf_data  # noqa: E402


def main():
    device = torch.device('cuda')
    W = wdsf_data()
    B = brace_samples()
    pred = {}
    for ev in sorted({d['event'] for d in W.values()}):
        tr = [(d['X'], d['y']) for d in W.values() if d['event'] != ev] * 3 + B     # 'joint' variant
        ms = [train(tr, device, epochs=60, seed=k) for k in range(3)]
        for vid, d in W.items():
            if d['event'] == ev:
                p = np.mean([predict(m, d['X'], device) for m in ms], 0)
                pred[vid], _ = segment(p, len(d['turns']), d['sec'], min_turn_s=15.0)
        print(ev, 'done', flush=True)
    T = pd.read_csv(f'{D}/wdsf_turns.csv')
    n_ok = 0
    for vid, turns in pred.items():
        idx = T.index[T.video_id == vid]
        if len(idx) != len(turns):
            print('skip (turn count differs)', vid, len(idx), len(turns))
            T = T.drop(idx)
            continue
        T.loc[idx, 'start_s'] = [round(s, 1) for s, _ in turns]
        T.loc[idx, 'end_s'] = [round(e, 1) for _, e in turns]
        n_ok += 1
    T.to_csv(f'{D}/wdsf_turns_handcv.csv', index=False)
    print(f'{n_ok} hand-labelled battles re-segmented by the held-out detector -> {D}/wdsf_turns_handcv.csv')


if __name__ == '__main__':
    main()

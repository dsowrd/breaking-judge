"""2022 holdout for the pairwise MLAVL (ours, Phase 7g; test fixed in experiments/PLAN_weekend_1106.md).

Train on all 5-criteria rounds of a target (epoch chosen on the largest training event, as in train_wdsf_pair.py) and
predict the rounds of the six-slider events that also have that slider. Those rounds took no part in choosing anything.

    CUDA_VISIBLE_DEVICES=0 $MLAVL_PY ours/wdsf_holdout.py --name hold_vq2_ft_s0 --feat vq2_ft --seed 0
    $MLAVL_PY ours/wdsf_holdout.py --eval vq2_ft vq2
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models import loss as mlavl_loss  # noqa: E402
from ours.eval_pair import boot, load  # noqa: E402
from ours.train_battle import aux_loss  # noqa: E402
from ours.train_wdsf import build_rows  # noqa: E402
from ours.train_wdsf_pair import (PairDataset, WithinDataset, build_model_dim, pairs, turn_score, within_acc,  # noqa: E402
                                  within_scale, within_step)
from types import SimpleNamespace  # noqa: E402

D = '/data/mlavl/datasets/wdsf'
OUT = 'logs/wdsf'
Y26 = ('/data/mlavl/datasets/wdsf_y26', '/data/mlavl/features/WDSF_Y26')
KEY = ['competition', 'battle_id', 'round']


def data(feat, target, holdout='2022'):
    stats = {}
    tr = [p for p in pairs(build_rows(target, feat=feat, stats=stats)) if p['margin'] != 0]
    kw = dict(system='6crit') if holdout == '2022' else dict(roots=Y26)
    hold = [p for p in pairs(build_rows(target, feat=feat, stats=stats, **kw)) if p['margin'] != 0]
    assert not {p['key'] for p in tr} & {p['key'] for p in hold}
    return tr, hold


def train(a):
    os.chdir(ROOT)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device('cuda')
    P, H = data(a.feat, a.target, a.holdout)
    if a.no_audio:
        for p in P + H:
            for s in 'br':
                p[s]['a'] = np.zeros_like(p[s]['a'])
    S = 3 * float(np.std([p['margin'] for p in P]))
    val_ev = pd.Series([p['event'] for p in P]).value_counts().index[0]
    fit = [p for p in P if p['event'] != val_ev]
    val = [p for p in P if p['event'] == val_ev]
    print(f'{len(P)} training rounds, {len(H)} holdout rounds, validation event {val_ev} ({len(val)})', flush=True)
    cfg = SimpleNamespace(text_feature=a.text_feature, n_text=a.n_text, clip_num=a.clip_num, dropout=a.dropout,
                          learn_text=False, action_list=a.action_list, init=None, pose=False)
    model = build_model_dim(cfg, device, P[0]['b']['v'].shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, eta_min=a.lr * 0.01)
    lf = mlavl_loss.LossFun(a.alpha, 1.0, alpha_ce=a.alpha_ce)
    if a.within == 'none':
        dl = DataLoader(PairDataset(fit, a.clip_num, S, a.seed), batch_size=a.batch, shuffle=True, num_workers=0, drop_last=True)
    else:
        dl = DataLoader(WithinDataset(fit, a.clip_num, S, within_scale(P), a.seed), batch_size=a.batch // 2, num_workers=0, drop_last=True)
    E, val_acc, pred = {5, 10, 20, 30, 40, a.epochs}, {}, {}
    margin = lambda p: turn_score(model, p['b'], a.clip_num, device) - turn_score(model, p['r'], a.clip_num, device)
    for ep in range(1, a.epochs + 1):
        model.train()
        for batch in dl:
            if a.within == 'none':
                vb, ab, vr, ar, t = batch
                B = vb.shape[0]
                o = model(torch.cat([vb, vr]).to(device), torch.cat([ab, ar]).to(device))
                d = o['output'][:B] - o['output'][B:]
                margin_loss = torch.mean((d - t.to(device)) ** 2)
            else:
                o, margin_loss = within_step(model, batch, a.within, device)
            loss = 10 * margin_loss + aux_loss(lf, o)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if ep in E:
            model.eval()
            pv = np.array([margin(p) for p in val])
            val_acc[ep] = (float((np.sign(pv) == np.sign([p['margin'] for p in val])).mean()) if a.within == 'none'
                           else within_acc(val, pv))
            pred[ep] = np.array([margin(p) * S for p in H])   # read only for the chosen epoch below
    best = max(val_acc, key=lambda e: (val_acc[e], -e))
    out = os.path.join(OUT, a.name)
    os.makedirs(out, exist_ok=True)
    pd.DataFrame([dict(competition=p['key'][0], battle_id=p['key'][1], round=p['key'][2], event=p['event'],
                       source=p['source'], margin=p['margin']) for p in H]).assign(pred=pred[best]).to_csv(
        os.path.join(out, 'predictions.csv'), index=False)
    json.dump(dict(val_event=val_ev, val_acc=val_acc, epoch=best, S=S, args=vars(a)),
              open(os.path.join(out, 'summary.json'), 'w'), indent=1)
    print(f'validation {val_acc} -> epoch {best}', flush=True)


def outside_training(df, feat, target, holdout):
    P, _ = data(feat, target, holdout)
    js = pd.read_csv(f'{D}/wdsf_judge_scores.csv').drop_duplicates(KEY)
    tk = pd.DataFrame([p['key'] for p in P], columns=KEY).merge(js, on=KEY)
    seen = set(tk.blue_id) | set(tk.red_id)
    d = df.merge(js[KEY + ['blue_id', 'red_id']], on=KEY, how='left')
    return df[~(d.blue_id.isin(seen) | d.red_id.isin(seen)).values].reset_index(drop=True)


def show(name, df):
    d = df.assign(ok=(np.sign(df.pred) == np.sign(df.margin)).astype(float), rule=(df.margin > 0).astype(float)).reset_index(drop=True)
    acc = boot(d, lambda x: x.ok.mean())
    diff = boot(d, lambda x: x.ok.mean() - x.rule.mean())
    lo, hi = np.percentile(acc, [2.5, 97.5])
    print(f'{name:28s} n={len(d):3d} battles={d.groupby(["competition", "battle_id"]).ngroups:3d} acc {d.ok.mean():.3f} '
          f'[{lo:.3f}, {hi:.3f}] rule {d.rule.mean():.3f} p(<=50%) {(acc <= .5).mean():.4f} p(<=rule) {(diff <= 0).mean():.4f}')


def evaluate(feats, target, holdout, pre=None):
    pre = pre or ('hold' if holdout == '2022' else 'hold26')
    os.chdir(ROOT)
    res = {}
    for F in feats:
        runs = [os.path.join(OUT, f'{pre}_{F}_{target}_s{s}') for s in (0, 1, 2)]
        for r in runs:
            d = load([r])
            print(os.path.basename(r), f'acc {(np.sign(d.pred) == np.sign(d.margin)).mean():.3f}',
                  'epoch', json.load(open(os.path.join(r, 'summary.json')))['epoch'])
        df = load(runs)
        res[F] = df
        show(f'{F} ens, all', df)
        un = outside_training(df, F, target, holdout)
        if len(un):
            show(f'{F} ens, not in training', un)
        for ev, g in df.groupby('event'):
            show('  ' + ev[:26], g)
    if len(feats) == 2:
        a, b = (res[f].set_index(KEY) for f in feats)
        idx = a.index.intersection(b.index)
        c = pd.DataFrame({'a': (np.sign(a.loc[idx].pred) == np.sign(a.loc[idx].margin)).astype(float).values,
                          'b': (np.sign(b.loc[idx].pred) == np.sign(b.loc[idx].margin)).astype(float).values}, index=idx).reset_index()
        v = boot(c, lambda x: x.a.mean() - x.b.mean())
        print(f'{feats[0]} - {feats[1]} n={len(c)} {c.a.mean() - c.b.mean():+.3f} [{np.percentile(v, 2.5):+.3f}, '
              f'{np.percentile(v, 97.5):+.3f}] p(<=0) {(v <= 0).mean():.3f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--eval', nargs='*')
    p.add_argument('--prefix', default=None)
    p.add_argument('--name')
    p.add_argument('--feat', default='vq2_ft')
    p.add_argument('--target', default='Technique')
    p.add_argument('--holdout', default='2022', choices=['2022', 'y26'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no_audio', action='store_true', help='zero audio input')
    p.add_argument('--within', default='none', choices=['none', 'only', 'mix'], help='train on differences between rounds of one battle')
    p.add_argument('--action_list', default='ours/vocab/action-label_BREAKING_draft0.csv')
    p.add_argument('--text_feature', default='ours/vocab/BREAKING_draft0_text_feature.npy')
    p.add_argument('--n_text', type=int, default=36)
    p.add_argument('--clip_num', type=int, default=32)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--alpha_ce', type=float, default=0.1)
    p.add_argument('--wd', type=float, default=1e-4)
    a = p.parse_args()
    if a.eval is not None:
        evaluate(a.eval, a.target, a.holdout, a.prefix)
    else:
        train(a)

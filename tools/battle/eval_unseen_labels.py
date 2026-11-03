import json, glob, os, numpy as np, pandas as pd

D = "/data/mlavl/datasets/wdsf"
old = json.load(open(f"{D}/wdsf_turn_labels.json"))
new = {}
for f in sorted(glob.glob("ours/wdsf/labels_raw_1103/*.json")):
    d = json.load(open(f))
    new[d["video_id"]] = {"turns": d["turns"], "note": d.get("note", "")}
chg = [v for v in old if old[v]["turns"] != new[v]["turns"]]
print("old", len(old), "changed", chg)
added = [v for v in new if v not in old]
print("added", len(added))
json.dump(new, open(f"{D}/wdsf_turn_labels_v41.json", "w"), ensure_ascii=False, indent=0)

T = pd.read_csv(f"{D}/wdsf_turns.csv")
rows = []
for v in added:
    truth = [t for t in new[v]["turns"] if t["start"] is not None]
    p = T[T.video_id == v].sort_values("start_s")
    pred = list(zip(p.start_s, p.end_s))
    if len(pred) != len(truth):
        print("count mismatch", v, len(pred), len(truth))
    for i, t in enumerate(truth):
        if i >= len(pred):
            rows.append(dict(video_id=v, turn=i, start_err=np.nan, end_err=np.nan, iou=0.0))
            continue
        ps, pe = pred[i]
        inter = max(0, min(pe, t["end"]) - max(ps, t["start"]))
        uni = max(pe, t["end"]) - min(ps, t["start"])
        rows.append(dict(video_id=v, turn=i, start_err=abs(ps - t["start"]), end_err=abs(pe - t["end"]),
                         iou=inter / uni if uni else 0))
df = pd.DataFrame(rows)
df.to_csv(f"{D}/turn_detector_eval_unseen19.csv", index=False)
cv = pd.read_csv(f"{D}/turn_detector_eval_joint_s3_min15.csv")
for name, x in [("LOEO 22 (old)", cv), ("unseen 19 (new)", df)]:
    print(f"{name}: {len(x)} turns, start err med {x.start_err.median():.1f}s <=5s {(x.start_err <= 5).mean():.2f} | "
          f"end err med {x.end_err.median():.1f}s | IoU mean {x.iou.mean():.3f} >=0.7 {(x.iou >= 0.7).mean():.2f} "
          f">=0.5 {(x.iou >= 0.5).mean():.2f} | videos all>=0.5 {(x.groupby('video_id').iou.min() >= 0.5).mean():.2f}")
print("worst new:", df.groupby("video_id").iou.mean().sort_values().head(6).round(2).to_dict())

key = ["competition", "battle_id", "round"]
bat = T[T.video_id.isin(added)][key].drop_duplicates()
hand = T[T.video_id.isin(list(old))][key].drop_duplicates()
for tgt in ["total", "Execution", "Technique"]:
    runs = [f"logs/wdsf/pose_{tgt}_s{s}" for s in range(3) if os.path.exists(f"logs/wdsf/pose_{tgt}_s{s}/predictions.csv")]
    P = pd.concat([pd.read_csv(r + "/predictions.csv") for r in runs]).groupby(key, as_index=False)[["pred", "margin"]].mean()
    P = P[P.margin != 0]
    P["ok"] = np.sign(P.pred) == np.sign(P.margin)
    m = P.merge(bat, on=key)
    h = P.merge(hand, on=key)
    o = P.merge(pd.concat([bat, hand]), on=key, how="left", indicator=True)
    o = o[o._merge == "left_only"]
    print(f"{tgt} ({len(runs)} seeds): new19 {m.ok.mean():.3f} (n={len(m)}) | old22 {h.ok.mean():.3f} (n={len(h)}) | "
          f"rest {o.ok.mean():.3f} (n={len(o)}) | all {P.ok.mean():.3f} (n={len(P)})")

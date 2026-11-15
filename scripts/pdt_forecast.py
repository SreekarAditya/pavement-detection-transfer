#!/usr/bin/env python3
"""Label-free forecast of each external collection's recall (rule: forecast_rule.md).

  fit       before the lock: fit F1 and F2 on development-fold outcomes (E1 object files, filtered to the
            development fold while reading) and E2 image properties; forecast every collection from its
            label-free image properties (session 2, ext_props). Writes forecasts.json.
  evaluate  after the lock (step 9): observed recall of YOLOv12s at confidence 0.25 on every collection from the
            sealed external predictions and the external labels; compares with the committed forecasts.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SEEDS = [0, 42, 123]
PRED = ["log_w", "log_h", "brightness_L", "contrast_L", "log_sharp", "edge_density", "grey_entropy", "knn10_cosine_dist"]
TARGETS = {"four_class": [0, 1, 2, 3], "pothole": [3]}


def features(df):
    x = pd.DataFrame(dict(log_w=np.log(df.src_w), log_h=np.log(df.src_h), brightness_L=df.brightness_L,
                          contrast_L=df.contrast_L, log_sharp=np.log(df.sharpness_lapvar + 1),
                          edge_density=df.edge_density, grey_entropy=df.grey_entropy,
                          knn10_cosine_dist=df.knn10_cosine_dist))
    return x


def fit(a):
    import statsmodels.api as sm
    dev = sorted(Path(a.folds, "fold_dev.txt").read_text().split())
    ip = pd.read_parquet(Path(a.e2) / "image_props.parquet")
    fd = pd.read_parquet(Path(a.e2) / "feature_distance.parquet")[["file", "knn10_cosine_dist"]]
    ip = ip.merge(fd, on="file")
    ip = ip[ip.file.isin(dev) & ip.read_ok].set_index("file")
    obj = []
    for s in SEEDS:
        o = pq.read_table(Path(a.e1) / "sealed" / f"objects__yolo12s_seed{s}.parquet",
                          columns=["file", "obj", "cls", "detected"], filters=[("file", "in", dev)]).to_pandas()
        assert set(o.file) <= set(dev)
        obj.append(o.assign(seed=s))
    O = pd.concat(obj)
    O = O.groupby(["file", "obj", "cls"]).detected.mean().reset_index()   # seed mean per object
    X_dev = features(ip)
    mu, sd = X_dev.mean(), X_dev.std(ddof=0)
    out = dict(rule_sha256=a.rule_sha, development_images=len(dev), standardization=dict(mean=mu.to_dict(), sd=sd.to_dict()),
               targets={})
    props = {c: pd.read_parquet(p) for c, p in
             ((p.stem, p) for p in sorted(Path(a.ext_props).glob("*.parquet")))}
    for t, cls in TARGETS.items():
        o = O[O.cls.isin(cls)]
        g = o.groupby("file").agg(n=("detected", "size"), det=("detected", "sum")).join(ip, how="inner")
        y = g.det / g.n
        f0 = float(g.det.sum() / g.n.sum())
        res = dict(F0_dev_recall=f0, training_images=int(len(g)), training_boxes=int(g.n.sum()), models={},
                   forecasts={})
        for name, cols in (("F1", PRED), ("F2", ["knn10_cosine_dist"])):
            Xs = ((features(g)[cols] - mu[cols]) / sd[cols])
            m = sm.GLM(y, sm.add_constant(Xs), family=sm.families.Binomial(), var_weights=g.n).fit()
            res["models"][name] = dict(coefficients=m.params.to_dict(), standard_errors=m.bse.to_dict(),
                                       dev_fitted_recall=float((m.fittedvalues * g.n).sum() / g.n.sum()))
            for c, P in props.items():
                Pe = P[P.read_ok]
                Xe = ((features(Pe)[cols] - mu[cols]) / sd[cols])
                p = m.predict(sm.add_constant(Xe, has_constant="add"))
                res["forecasts"].setdefault(c, {})[name] = float(p.mean())
        for c in props:
            res["forecasts"][c]["F0"] = f0
            res["forecasts"][c]["images"] = int(props[c].read_ok.sum())
        out["targets"][t] = res
        print(t, "F0", round(f0, 4), {c: {k: round(v, 4) for k, v in d.items() if k != "images"}
                                       for c, d in res["forecasts"].items()})
    json.dump(out, open(a.out, "w"), indent=1)


def evaluate(a):
    """Observed recall at confidence 0.25 (E1 matcher) of YOLOv12s per collection; comparison with forecasts."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pdt_core import box_iou
    from scipy.stats import spearmanr
    fc = json.load(open(a.forecasts))
    dec = json.load(open(Path(a.g4) / "g4_g6_decision.json"))["collections"]
    res = {}
    for t, cls in TARGETS.items():
        cols = [c for c in dec if (dec[c]["eligible_four_class"] if t == "four_class" else dec[c]["eligible_pothole"])]
        obs = {}
        for c in cols:
            L = pd.read_parquet(Path(a.g4) / f"labels_final_{c}.parquet"); L = L[L.cls.isin(cls)]
            rec = []
            for s in SEEDS:
                P = pd.read_parquet(Path(a.sealed_ext) / c / f"yolo12s_seed{s}.parquet")
                P = P[(P.conf >= 0.25) & P.cls.isin(cls)]
                Pg = dict(tuple(P.groupby("file"))); det = 0
                for f, lab in L.groupby("file"):
                    pr = Pg.get(f)
                    if pr is None:
                        continue
                    iou = box_iou(lab[["x1", "y1", "x2", "y2"]].to_numpy(), pr[["x1", "y1", "x2", "y2"]].to_numpy())
                    iou = iou * (lab.cls.to_numpy()[:, None] == pr.cls.to_numpy()[None, :])
                    m = np.array(np.nonzero(iou >= 0.5)).T
                    if len(m):
                        m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                        m = m[np.unique(m[:, 1], return_index=True)[1]]
                        m = m[np.unique(m[:, 0], return_index=True)[1]]
                        det += len(m)
                rec.append(det / len(L))
            obs[c] = float(np.mean(rec))
        f = fc["targets"][t]["forecasts"]
        r = dict(observed=obs, errors={}, mae={}, spearman={})
        for k in ("F0", "F1", "F2"):
            e = {c: f[c][k] - obs[c] for c in cols}
            r["errors"][k] = e; r["mae"][k] = float(np.mean(np.abs(list(e.values()))))
            if len(cols) >= 4 and k != "F0":
                r["spearman"][k] = float(spearmanr([f[c][k] for c in cols], [obs[c] for c in cols]).correlation)
        res[t] = r
    res["useful"] = all(min(res[t]["mae"]["F1"], res[t]["mae"]["F2"]) < res[t]["mae"]["F0"] for t in TARGETS)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "evaluate"])
    ap.add_argument("--e1"); ap.add_argument("--e2"); ap.add_argument("--folds"); ap.add_argument("--ext_props")
    ap.add_argument("--rule_sha", default="28416e0590318faa1d45eb3363e91a38dfc3d0c2215e66afd86cdbc910277e69")
    ap.add_argument("--forecasts"); ap.add_argument("--g4"); ap.add_argument("--sealed_ext")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fit(a) if a.cmd == "fit" else evaluate(a)

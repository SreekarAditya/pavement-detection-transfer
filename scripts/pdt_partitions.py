#!/usr/bin/env python3
"""Secondary items 9 to 11: what the RDD2022 partitions measure (protocol 3.3.9, Section 6). Run at step 9.

Inputs: E1 object files (one row per reference box and checkpoint: detected at confidence 0.25 one to one at
intersection over union 0.5; reachable above the 0.001 floor), E2 image and object properties, G3 folds and
clusters (main and 0.65). All 20 setup A checkpoints; clean holdout = development, S1 and S2 together.

 9. Shared misses and oracle ceiling, by partition and class.
10. Attribution of each partition's gap to measured properties: one binomial model (logit link), all partitions
    pooled, outcome YOLOv12s detection share over its three seeds (20-checkpoint mean as a sensitivity);
    predictors fixed by the protocol; explained part = mean predicted recall of the partition minus pooled recall;
    remainder = observed minus mean predicted.
11. Leakage exposure by partition: share of validation images linked to training (G3, 0.75 and 0.65) and recall at
    confidence 0.25 on linked versus clean validation images (YOLOv12s seeds and all 20 checkpoints).
Intervals: 2,000 cluster-bootstrap draws within partition strata (seed 2026); the model in item 10 is refitted in
every draw. Linked images (item 11) are resampled as single images within partition strata.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pdt_core as C  # noqa: E402

ARCH = ["yolov5s", "yolov8s", "yolov9s", "yolov10s", "yolo11s", "yolo12n", "yolo12s", "yolo12m"]
RUNS = [f"{a}_seed{s}" for a in ARCH for s in ([0, 42, 123] if a not in ("yolo12n", "yolo12m") else [0])]
NAMES = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}


def pct(x):
    return [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]


def load(a, files):
    O = []
    for r in RUNS:
        o = pd.read_parquet(Path(a.e1) / "sealed" / f"objects__{r}.parquet", columns=["file", "obj", "cls", "country", "detected", "reachable"])
        O.append(o[o.file.isin(files)].assign(run=r))
    O = pd.concat(O)
    W = O.pivot_table(index=["file", "obj", "cls", "country"], columns="run", values="detected", aggfunc="first").reset_index()
    R = O.pivot_table(index=["file", "obj"], columns="run", values="reachable", aggfunc="first").reset_index()
    return W, R


def boot_weights(cl):
    strata = C.majority_strata(cl, "cluster", "country")
    return C.unit_weights(cl.cluster.tolist(), strata, np.random.default_rng(C.SEED))


def design(df, ip, op):
    d = df.merge(op[["file", "obj", "area_input", "aspect", "cx_frac", "cy_frac"]], on=["file", "obj"], how="left") \
          .merge(ip[["file", "src_w", "src_h", "brightness_L", "contrast_L", "sharpness_lapvar", "edge_density",
                     "grey_entropy", "n_boxes"]], on="file", how="left")
    X = pd.DataFrame(dict(log_area=np.log(d.area_input), aspect=d.aspect, cx=d.cx_frac, cy=d.cy_frac,
                          src_w=d.src_w, src_h=d.src_h, L_mean=d.brightness_L, L_sd=d.contrast_L,
                          log_lapvar=np.log(d.sharpness_lapvar + 1), edge=d.edge_density, entropy=d.grey_entropy,
                          n_boxes=d.n_boxes))
    for c in (1, 2, 3):
        X[f"cls{c}"] = (d.cls == c).astype(float)
    return X.fillna(X.median())


def fit_predict(X, y, n_trials, mu=None, sd=None, w=None):
    import statsmodels.api as sm
    mu = X.mean() if mu is None else mu
    sd = X.std(ddof=0).replace(0, 1) if sd is None else sd
    Xs = sm.add_constant((X - mu) / sd, has_constant="add")
    vw = np.full(len(y), float(n_trials)) * (1.0 if w is None else np.asarray(w, float))
    m = sm.GLM(y, Xs, family=sm.families.Binomial(), var_weights=vw).fit()
    return m, m.predict(Xs), mu, sd


def main(a):
    out = {}
    g3 = Path(a.g3)
    cl = pd.read_csv(g3 / "main" / "clusters.csv"); cl065 = pd.read_csv(g3 / "sens065" / "clusters.csv")
    clean = sorted(cl.file)
    ip = pd.read_parquet(Path(a.e2) / "image_props.parquet"); op = pd.read_parquet(Path(a.e2) / "object_props.parquet")
    allval = sorted(ip.file)                             # all 5,757 validation images (E2)
    W, R = load(a, set(allval))
    det = W[RUNS].astype(float)
    Wc = W[W.file.isin(clean)].reset_index(drop=True)
    Rc = R.set_index(["file", "obj"]).loc[list(zip(Wc.file, Wc.obj))].reset_index()
    cl = cl.set_index("file").loc[sorted(set(Wc.file))].reset_index()
    BW = boot_weights(cl)                                # draws x images
    fidx = {f: i for i, f in enumerate(cl.file)}
    obj_img = Wc.file.map(fidx).to_numpy()
    wobj = BW[:, obj_img]                                # draws x objects

    # ---------------- item 9
    d = Wc[RUNS].astype(float).to_numpy()
    shared_miss = (d.sum(1) == 0).astype(float); oracle = (d.sum(1) > 0).astype(float)
    item9 = {}
    for part in sorted(Wc.country.unique()) + ["all"]:
        for c in [None, 0, 1, 2, 3]:
            m = np.ones(len(Wc), bool) if part == "all" else (Wc.country == part).to_numpy().copy()
            if c is not None:
                m &= (Wc.cls == c).to_numpy()
            if m.sum() == 0:
                continue
            ww = wobj[:, m]; n = ww.sum(1)
            best = {r: float(d[m, i].mean()) for i, r in enumerate(RUNS)}
            rb = max(best, key=best.get)
            key = f"{part}|{'all' if c is None else NAMES[c]}"
            item9[key] = dict(boxes=int(m.sum()), shared_miss=float(shared_miss[m].mean()),
                              shared_miss_ci=pct((ww * shared_miss[m]).sum(1) / n),
                              oracle=float(oracle[m].mean()), oracle_ci=pct((ww * oracle[m]).sum(1) / n),
                              best_single=best[rb], best_single_run=rb,
                              reachable_any=float(Rc[RUNS].to_numpy()[m].any(1).mean()))
    out["item9_shared_misses_oracle"] = item9

    # ---------------- item 10
    X = design(Wc[["file", "obj", "cls"]], ip, op)
    item10 = {}
    for tag, cols in (("yolo12s", [f"yolo12s_seed{s}" for s in (0, 42, 123)]), ("all20", RUNS)):
        y = Wc[cols].astype(float).mean(1).to_numpy(); nt = float(len(cols))
        m, p, mu, sd = fit_predict(X, y, nt)
        pooled = float(y.mean())
        res = dict(pooled_recall=pooled, coefficients=m.params.to_dict(), partitions={})
        boot = {part: [] for part in sorted(Wc.country.unique())}
        for b in range(a.draws):
            w = wobj[b].astype(float); keep = w > 0
            mb, pb, _, _ = fit_predict(X[keep], y[keep], nt, mu, sd, w[keep])
            pooled_b = np.average(y[keep], weights=w[keep])
            for part in boot:
                mk = (Wc.country == part).to_numpy()[keep]
                obs = np.average(y[keep][mk], weights=w[keep][mk]); pr = np.average(pb[mk], weights=w[keep][mk])
                boot[part].append((obs - pooled_b, pr - pooled_b, obs - pr))
        for part in boot:
            mk = (Wc.country == part).to_numpy()
            obs, pr = float(y[mk].mean()), float(p[mk].mean())
            bb = np.array(boot[part])
            res["partitions"][part] = dict(boxes=int(mk.sum()), observed=obs, predicted=pr, gap=obs - pooled,
                                           explained=pr - pooled, remainder=obs - pr,
                                           gap_ci=pct(bb[:, 0]), explained_ci=pct(bb[:, 1]), remainder_ci=pct(bb[:, 2]))
        item10[tag] = res
    out["item10_attribution"] = item10

    # ---------------- item 11
    item11 = {}
    # full clean holdout (line 79); `cl` was narrowed at line 86 to clean images with reference boxes (fixed after the lock)
    for tag, clean_set in (("threshold_0.75", set(clean)), ("threshold_0.65", set(cl065.file))):
        imgs = pd.DataFrame(dict(file=allval)); imgs["country"] = imgs.file.map(C_country)
        imgs["linked"] = ~imgs.file.isin(clean_set)
        r = {}
        rng = np.random.default_rng(C.SEED)
        for part, g in imgs.groupby("country"):
            Wp = W[W.country == part]
            lk = Wp.file.isin(set(g.file[g.linked]))
            for model, cols in (("yolo12s", [f"yolo12s_seed{s}" for s in (0, 42, 123)]), ("all20", RUNS)):
                y = Wp[cols].astype(float).mean(1).to_numpy()
                rl = float(y[lk.to_numpy()].mean()) if lk.any() else None
                rc = float(y[~lk.to_numpy()].mean()) if (~lk).any() else None
                r.setdefault(part, dict(images=int(len(g)), linked_share=float(g.linked.mean())))
                r[part][f"recall_linked_{model}"] = rl; r[part][f"recall_clean_{model}"] = rc
        item11[tag] = r
    out["item11_leakage"] = item11
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print("partitions done")


def C_country(f):
    for c in ["China_MotorBike", "China_Drone", "United_States", "Czech", "India", "Japan", "Norway"]:
        if f.startswith(c):
            return c
    return "UNKNOWN"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", required=True); ap.add_argument("--e2", required=True); ap.add_argument("--g3", required=True)
    ap.add_argument("--draws", type=int, default=C.B)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())

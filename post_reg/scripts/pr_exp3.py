#!/usr/bin/env python3
"""Experiments 3 and 4: pilot audit size counted in reference boxes, and the label-free forecast against pilot labels.

Experiment 3, for each collection, class set and checkpoint:
  operating threshold  params.exp3.threshold_rule = calibration_quota: the confidence giving k alerts per 100
                       images over the whole calibration part (label-free)
  true recall          recall at that threshold on the held-out part (one-to-one, same class, IoU 0.5)
  pilot draws          calibration blocks in random order until at least M reference boxes of the class set
                       (the block order is shared by all checkpoints: paired design)
  estimate (a)         ratio of matched to reference boxes over the sampled blocks, linearization interval
  estimate (b)         PPI++ (power-tuned prediction-powered ratio) with a consensus pseudo-reference: boxes on
                       which at least params.exp3.ppi.agreement_min_share of the other checkpoints agree (IoU 0.5,
                       same class), each at its own calibration quota threshold; population = all calibration blocks
  decisions            accept if the lower end of the params.exp3.decision_level interval is >= floor
Experiment 4: registered forecasts (YOLOv12s seed 0, confidence 0.25) against recall at 0.25 on the held-out part,
beside pilot estimates at the same threshold; shift points for the seven pothole collections (descriptive).
"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr_common as PC  # noqa: E402


# ------------------------------------------------------------------ per-image tables
def alerts_by_image(P, files, classes, t, merge=None):
    p = P[P.file.isin(set(files)) & P.cls.isin(classes) & (P.conf >= t)].copy()
    if merge:
        p["cls"] = p.cls.map(lambda c: merge.get(c, c))
    return dict(tuple(p.groupby("file")))


def consensus(boxes_by_ckpt, share, n_other):
    """boxes_by_ckpt: list of (ckpt_index, array Nx5 [cls, x1, y1, x2, y2]) for one image, audited checkpoint
    excluded. Keeps boxes on which >= share * n_other checkpoints agree; removes duplicates (IoU 0.5, same class),
    keeping the higher agreement. Returns K x 5 array."""
    C = sys.modules["pdt_core"]
    allb = [(k, b) for k, arr in boxes_by_ckpt for b in arr]
    if not allb:
        return np.zeros((0, 5))
    ck = np.array([k for k, _ in allb]); B = np.array([b for _, b in allb])
    iou = C.box_iou(B[:, 1:], B[:, 1:]) * (B[:, :1] == B[:, :1].T)
    ov = iou >= 0.5
    uk = np.unique(ck)
    agree = np.zeros(len(B), int)
    for k in uk:
        agree += ov[:, ck == k].any(1)
    keep = agree >= np.ceil(share * n_other - 1e-9)
    if not keep.any():
        return np.zeros((0, 5))
    idx = np.nonzero(keep)[0]
    idx = idx[np.argsort(-agree[idx], kind="stable")]
    chosen = []
    for i in idx:
        if not any(ov[i, j] for j in chosen):
            chosen.append(i)
    return B[chosen]


def image_table(R, coll, files, ckpts, audited, classes, merge, k, thr_rule_t=None, share=0.5, setup="A"):
    """Per image: a (matched true), x (true refs), ah (matched pseudo), xh (pseudo refs) for the audited detector
    at its operating threshold; pseudo-reference from the other checkpoints at their quota thresholds on files."""
    L = R.labels_external(coll)
    L = L[L.file.isin(set(files)) & L.cls.isin(classes)].copy()
    if merge:
        L["cls"] = L.cls.map(lambda c: merge.get(c, c))
    Lg = dict(tuple(L.groupby("file")))
    load = (lambda c: R.pred_external(c, coll)) if setup == "A" else (lambda c: R.pred_b(c, coll))
    preds = {c: load(c) for c in ckpts}
    thr = {}
    for c in ckpts:
        p = preds[c]; p = p[p.file.isin(set(files)) & p.cls.isin(classes)].conf.sort_values(ascending=False).to_numpy()
        thr[c] = PC.threshold_for_rate(p, len(files), k)
    t_aud = thr[audited] if thr_rule_t is None else thr_rule_t
    al = {c: alerts_by_image(preds[c], files, classes, (t_aud if c == audited else thr[c]), merge) for c in ckpts}
    others = [c for c in ckpts if c != audited]
    rows = []
    for f in files:
        lab = Lg.get(f)
        lb = lab[PC.BOX].to_numpy() if lab is not None else np.zeros((0, 4))
        lc = lab.cls.to_numpy() if lab is not None else np.zeros(0)
        g = al[audited].get(f)
        pb = g[PC.BOX].to_numpy() if g is not None else np.zeros((0, 4))
        pcl = g.cls.to_numpy() if g is not None else np.zeros(0)
        lm, _, _ = PC.greedy_match(lb, pb, lc, pcl)
        bo = []
        for j, c in enumerate(others):
            h = al[c].get(f)
            if h is not None:
                bo.append((j, np.column_stack([h.cls.to_numpy(), h[PC.BOX].to_numpy()])))
        ps = consensus(bo, share, len(others))
        hm, _, _ = PC.greedy_match(ps[:, 1:], pb, ps[:, 0], pcl) if len(ps) else (np.zeros(0, bool), None, None)
        rows.append((f, int(lm.sum()), len(lb), int(hm.sum()), len(ps), len(pb)))
    T = pd.DataFrame(rows, columns=["file", "a", "x", "ah", "xh", "alerts"])
    return T, t_aud


def true_recall(R, coll, files, ckpt, classes, merge, t, setup="A"):
    P = R.pred_external(ckpt, coll) if setup == "A" else R.pred_b(ckpt, coll)
    L = R.labels_external(coll)
    TR = PC.ThresholdRun(P, L, files, classes, merge=merge)
    a, m, x = TR.per_image_at(t)
    return (m.sum() / x.sum()) if x.sum() else np.nan, int(m.sum()), int(x.sum()), int(a.sum())


def pilot_draws(T, blocks, M, draws, seed, cidx, levels, decision_level, floors, true_r):
    """Draws of the pilot audit on block sums of T. Returns list of dicts."""
    T = T.assign(block=T.file.map(blocks))
    Bk = T.groupby("block")[["a", "x", "ah", "xh"]].sum()
    names = Bk.index.to_numpy()
    a, x, ah, xh = (Bk[c].to_numpy() for c in ("a", "x", "ah", "xh"))
    nimg = T.groupby("block").size().reindex(names).to_numpy()
    rng = np.random.default_rng(np.random.SeedSequence([seed, 4, cidx, M]))
    zs = {lv: norm.ppf(0.5 + lv / 2) for lv in levels}
    zd = norm.ppf(0.5 + decision_level / 2)
    out = []
    if x.sum() < M:
        return out
    for d in range(draws):
        perm = rng.permutation(len(names))
        cs = np.cumsum(x[perm])
        kk = int(np.searchsorted(cs, M)) + 1
        s = perm[:kk]
        r = dict(draw=d, M=M, blocks=kk, images=int(nimg[s].sum()), boxes=int(x[s].sum()), true_recall=true_r)
        est, lo_d, hi_d, se = PC.ratio_ci(a[s], x[s], zd)
        r.update(est_a=est, se_a=se, lo_dec_a=lo_d)
        for lv, z in zs.items():
            _, lo, hi, _ = PC.ratio_ci(a[s], x[s], z)
            r[f"lo_a_{int(lv*100)}"], r[f"hi_a_{int(lv*100)}"] = lo, hi
        th, lo_d, hi_d, se, lam = PC.ppi_ratio(a[s], x[s], ah[s], xh[s], ah, xh, zd)
        r.update(est_b=th, se_b=se, lo_dec_b=lo_d, lam=lam)
        for lv, z in zs.items():
            _, lo, hi, _, _ = PC.ppi_ratio(a[s], x[s], ah[s], xh[s], ah, xh, z)
            r[f"lo_b_{int(lv*100)}"], r[f"hi_b_{int(lv*100)}"] = lo, hi
        out.append(r)
    return out


# ------------------------------------------------------------------ jobs
def job3(args):
    root, rules, params, coll, cs, audited, setup = args
    PC.locked_scripts(root)
    R = PC.Root(root)
    cal, held = PC.read_split(rules, coll)
    classes, merge = PC.CLASS_SET_CLASSES[cs], PC.CLASS_SET_MERGE[cs]
    k = params["class_sets"][cs]["k"]
    ckpts = PC.RUNS_A if setup == "A" else PC.ARCH_B
    share = params["exp3"]["ppi"]["agreement_min_share"]
    T, t = image_table(R, coll, cal, ckpts, audited, classes, merge, k, share=share, setup=setup)
    tr, m, x, al = true_recall(R, coll, held, audited, classes, merge, t, setup)
    cidx = PC.EXT_ALL.index(coll)
    rows = []
    for M in params["exp3"]["m_grid"]:
        dr = pilot_draws(T, R.blocks(coll), M, params["sample_draws"], params["seed"], cidx, params["exp3"]["levels"],
                         params["exp3"]["decision_level"], params["exp3"]["floor_grid"], tr)
        if not dr:
            rows.append(dict(M=M, draw=-1, note="M larger than the calibration boxes"))
        rows += dr
    D = pd.DataFrame(rows)
    D.insert(0, "class_set", cs); D.insert(0, "collection", coll); D.insert(0, "checkpoint", audited)
    D.insert(0, "setup", setup)
    D["threshold"] = t; D["heldout_matched"] = m; D["heldout_refs"] = x
    T.insert(0, "checkpoint", audited); T.insert(0, "class_set", cs); T.insert(0, "collection", coll)
    return D, T


def floor_grid(fg):
    if len(fg) == 3 and fg[2] < fg[1] and fg[2] <= fg[1] - fg[0]:
        return list(np.round(np.arange(fg[0], fg[1] + 1e-9, fg[2]), 4))
    return list(fg)


def summarize3(D, params):
    m = params["exp3"]["margin"]; cov = params["exp3"]["coverage"]
    floors = floor_grid(params["exp3"]["floor_grid"])
    D = D[D.draw >= 0].copy() if "draw" in D.columns else pd.DataFrame(columns=["draw"])
    rows, dec = [], []
    for key, g in D.groupby(["setup", "checkpoint", "collection", "class_set", "M"]):
        tr = g.true_recall.iloc[0]
        r = dict(zip(["setup", "checkpoint", "collection", "class_set", "M"], key), true_recall=tr, draws=len(g),
                 median_images=float(g.images.median()), median_boxes=float(g.boxes.median()))
        for e in ("a", "b"):
            err = g[f"est_{e}"] - tr
            r[f"share_within_margin_{e}"] = float((err.abs() <= m).mean())
            for lv in params["exp3"]["levels"]:
                L_ = int(lv * 100)
                r[f"coverage{L_}_{e}"] = float(((g[f"lo_{e}_{L_}"] <= tr) & (tr <= g[f"hi_{e}_{L_}"])).mean())
                r[f"median_width{L_}_{e}"] = float((g[f"hi_{e}_{L_}"] - g[f"lo_{e}_{L_}"]).median())
        r["median_lambda"] = float(g.lam.median())
        se_srs = np.sqrt(tr * (1 - tr) / g.boxes.clip(lower=1))
        r["design_effect_a"] = float(((g.se_a / se_srs) ** 2).median())
        r["meets_margin_a"] = r["share_within_margin_a"] >= cov
        rows.append(r)
        for fl in floors:
            for e in ("a", "b"):
                acc = g[f"lo_dec_{e}"] >= fl
                truth = tr >= fl
                dec.append(dict(r | dict(floor=fl, estimate=e, accept_rate=float(acc.mean()),
                                         true_above_floor=bool(truth),
                                         wrong_accept=float((acc & (not truth)).mean()),
                                         wrong_reject=float(((~acc) & truth).mean()))))
    keys = ["setup", "checkpoint", "collection", "class_set"]
    S = pd.DataFrame(rows, columns=None if rows else keys + ["M", "meets_margin_a"])
    sm = []
    for key, g in S.groupby(keys):
        ok = g[g.meets_margin_a].M
        sm.append(dict(zip(["setup", "checkpoint", "collection", "class_set"], key),
                       smallest_M=int(ok.min()) if len(ok) else None))
    return S, pd.DataFrame(dec), pd.DataFrame(sm, columns=keys + ["smallest_M"])


def job4(args):
    root, rules, params, coll, cs = args
    PC.locked_scripts(root)
    R = PC.Root(root)
    cal, held = PC.read_split(rules, coll)
    classes, merge = PC.CLASS_SET_CLASSES[cs], PC.CLASS_SET_MERGE[cs]
    det, t = params["exp4"]["detector"], params["exp4"]["conf"]
    tr, m, x, _ = true_recall(R, coll, held, det, classes, merge, t)
    T, _ = image_table(R, coll, cal, PC.RUNS_A, det, classes, merge, params["class_sets"][cs]["k"], thr_rule_t=t,
                       share=params["exp3"]["ppi"]["agreement_min_share"])
    rows = []
    for M in params["exp3"]["m_grid"]:
        dr = pilot_draws(T, R.blocks(coll), M, params["sample_draws"], params["seed"], PC.EXT_ALL.index(coll),
                         [0.9], 0.9, [], tr)
        if dr:
            e = np.array([d["est_a"] for d in dr])
            rows.append(dict(collection=coll, class_set=cs, M=M, true_recall=tr,
                             median_abs_error=float(np.median(np.abs(e - tr))),
                             q90_abs_error=float(np.percentile(np.abs(e - tr), 90))))
    return tr, rows


def main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if not a.skip_gate:
        PC.verify_lock(a.root, a.manifest_sha)
        PC.rule_gate(a.rules, a.rules_sha, a.params_sha)
    params = PC.load_params(a.rules)
    PC.env_record(out)
    if a.part in ("all", "e3"):
        jobs = []
        for cs, spec in params["class_sets"].items():
            colls = PC.EXT_ALL if spec["networks"] == "all_seven" else spec["networks"]
            for coll in colls:
                for ck in ([params["selected"]] + [c for c in PC.RUNS_A if c != params["selected"]]):
                    jobs.append((a.root, a.rules, params, coll, cs, ck, "A"))
                if cs == "merged_crack":
                    for m in PC.ARCH_B:
                        jobs.append((a.root, a.rules, params, coll, cs, m, "B"))
        PC.log(f"experiment 3: {len(jobs)} jobs")
        res = PC.pmap(job3, jobs)
        D = pd.concat([r[0] for r in res], ignore_index=True)
        T = pd.concat([r[1] for r in res], ignore_index=True)
        D.to_parquet(out / "e3_draws.parquet"); T.to_parquet(out / "e3_image_tables.parquet")
        S, DEC, SM = summarize3(D, params)
        S.to_csv(out / "e3_summary.csv", index=False); DEC.to_parquet(out / "e3_decisions.parquet")
        SM.to_csv(out / "e3_smallest_M.csv", index=False)
        # checks (task 03, step 5): summary reproduced from the draws; split disjoint
        m_ = params["exp3"]["margin"]; L0 = int(params["exp3"]["levels"][0] * 100)
        Dd = D[D.draw >= 0] if "draw" in D.columns else D
        rep = []
        for r in S.itertuples():
            g = Dd[(Dd.setup == r.setup) & (Dd.checkpoint == r.checkpoint) & (Dd.collection == r.collection) &
                   (Dd.class_set == r.class_set) & (Dd.M == r.M)]
            tr = g.true_recall.iloc[0]
            w = float(((g.est_a - tr).abs() <= m_).mean())
            cv = float(((g[f"lo_a_{L0}"] <= tr) & (tr <= g[f"hi_a_{L0}"])).mean())
            rep.append(abs(w - r.share_within_margin_a) < 1e-12 and abs(cv - getattr(r, f"coverage{L0}_a")) < 1e-12)
        PC.locked_scripts(a.root)
        sp = PC.split_checks(a.rules, PC.Root(a.root), sorted(D.collection.unique()) if len(D) else [])
        PC.jdump(dict(summary_rows_reproduced=int(sum(rep)), summary_rows=len(rep), split=sp),
                 out / "e3_checks.json")
        if not all(rep) or not all(all(v.values()) for v in sp.values()):
            sys.exit("STOP: an experiment 3 check failed (see e3_checks.json)")
    if a.part in ("all", "e4"):
        R = PC.Root(a.root)
        F = __import__("json").load(open(Path(a.root) / "forecast" / "forecasts.json"))["targets"]
        rows, pil = [], []
        jobs = [(a.root, a.rules, params, c, cs) for cs, cl in (("four_class", PC.EXT_FOUR), ("pothole", PC.EXT_ALL))
                for c in cl]
        res = PC.pmap(job4, jobs)
        for (root, rules, params_, c, cs), (tr, pr) in zip(jobs, res):
            fc = F[cs]["forecasts"][c]
            for model in ("F0", "F1", "F2"):
                rows.append(dict(collection=c, target=cs, model=model, forecast=fc[model], heldout_recall_025=tr,
                                 abs_error=abs(fc[model] - tr)))
            pil += pr
        pd.DataFrame(rows).to_csv(out / "e4_errors.csv", index=False)
        pd.DataFrame(pil).to_csv(out / "e4_pilot_errors.csv", index=False)
        # shift points, seven pothole collections, descriptive
        PC.locked_scripts(a.root)
        dev = set(R.fold("dev"))
        fd = pd.read_parquet(Path(a.root) / "e2" / "feature_distance.parquet")
        dev_app = float(fd[fd.file.isin(dev)].knn10_cosine_dist.mean())
        LA = R.labels_internal()
        dev_dens = 100 * int(LA[LA.file.isin(dev) & (LA.cls == 3)].shape[0]) / len(dev)
        pts = []
        for c in PC.EXT_ALL:
            cal, _ = PC.read_split(a.rules, c)
            E = R.ext_props(c); E = E[E.file.isin(set(cal))]
            L = R.labels_external(c); L = L[L.file.isin(set(cal)) & (L.cls == 3)]
            pts.append(dict(collection=c, appearance_shift=float(E.knn10_cosine_dist.mean()) - dev_app,
                            pothole_density_shift=100 * len(L) / len(cal) - dev_dens))
        pd.DataFrame(pts).to_csv(out / "e4_shift_points.csv", index=False)
    PC.write_manifest(out)
    PC.log("experiments 3 and 4 done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--rules", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--part", choices=["all", "e3", "e4"], default="all")
    ap.add_argument("--manifest_sha", default="ef49673bdc6301009223fb7eecd0aa71be7f53eb6220395d30262b64b19228c7")
    ap.add_argument("--rules_sha", default=""); ap.add_argument("--params_sha", default="")
    ap.add_argument("--skip_gate", action="store_true", help="synthetic tests only")
    main(ap.parse_args())

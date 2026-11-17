#!/usr/bin/env python3
"""Benchmark, robustness and survey tables on the sealed predictions (task 04).

  codepath  recompute the registered setup A and setup B four-class primary with this code; compare with the
            locked scoring outputs (exact match required; uses 2,000 draws as registered)
  e7   setup B on merged cracks (k 136.69): selection on S1, ratio, leave-one-checkpoint-out
  e8   leakage effect on R: internal recalls from all validation images, linked images, clean 0.75, strict 0.65
  e9   four-zone assignment of clean-holdout boxes, chance baseline, fixed-checkpoint version
  e10  class-mix standardization at confidence 0.25 (and at the quota)
  e12  recall and R across alert quotas
  e13  per-class recall at the quota on each external collection
  e14  detection and shared misses against defect size at model input
  e16  confidence calibration (expected calibration error, risk-coverage)
  e19  image-level endpoint at the quota
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr_common as PC  # noqa: E402


def country_of(f):
    return next((c for c in PC.COUNTRIES if f.startswith(c)), "UNKNOWN")


class Ctx:
    def __init__(self, a, params):
        self.a, self.params = a, params
        self.C = PC.locked_scripts(a.root)
        self.R = PC.Root(a.root)
        self.B = params["bootstrap"]["draws"]
        self.seed = params["seed"]
        self.k4 = params["k"]["four_class"]
        self.clean = sorted(set(self.R.fold("dev")) | set(self.R.fold("S1")) | set(self.R.fold("S2")))

    def predA(self, run, pool):
        if pool in self.R_ext:
            return self.R.pred_external(run, pool)
        return self.R.pred_internal(run)

    R_ext = set(PC.EXT_ALL)


# ------------------------------------------------------------------ code path
def codepath(X, out):
    R, C = X.R, X.C
    import pdt_score as S
    pools = {"S1": PC.internal_pool(R, "S1", R.fold("S1")), "S2": PC.internal_pool(R, "S2", R.fold("S2"))}
    for c in PC.EXT_FOUR:
        pools[c] = PC.external_pool(R, c)
    pt, bt, _ = PC.panel(PC.RUNS_A, X.predA, pools, PC.FOUR, X.k4, 2000, X.seed)
    mine = PC.ratio_from_panel(pt, bt, PC.GROUPS_A, "S1", "S2", PC.EXT_FOUR)
    LB = S.db_labels(R.r / "g4c" / "db_test.parquet")
    pb = {}
    for f in ("S1", "S2"):
        files = sorted((R.r / "g4c" / "main" / f"fold_{f}.txt").read_text().split())
        cl = pd.read_csv(R.r / "g4c" / "main" / "clusters.csv").set_index("file").loc[files].reset_index()
        pb[f] = PC.Pool(f, files, cl.cluster.tolist(), C.majority_strata(cl, "cluster", "country"),
                        LB[LB.file.isin(set(files))])
    for c in PC.EXT_FOUR:
        pb[c] = pools[c]
    predB = lambda m, p: R.pred_b(m, "rdd2022_scoring" if p in ("S1", "S2") else p)  # noqa: E731
    ptb, btb, _ = PC.panel(PC.ARCH_B, predB, pb, PC.FOUR, X.k4, 2000, X.seed)
    mineB = PC.ratio_from_panel(ptb, btb, {m: [m] for m in PC.ARCH_B}, "S1", "S2", PC.EXT_FOUR)
    reg = json.load(open(Path(X.a.scoring) / "score_results.json"))
    rows = []
    for tag, m, r in (("A", mine, reg["A_four_class"]["primary"]), ("B", mineB, reg["B_four_class"]["primary"])):
        for key in ("R", "numerator", "denominator", "denominator_lower_95", "var_num", "var_den", "cov"):
            a_, b_ = m[key], r[key]
            both_nan = a_ is not None and b_ is not None and bool(np.isnan(float(a_)) and np.isnan(float(b_)))
            same = (a_ is None and b_ is None) or both_nan or (a_ is not None and b_ is not None
                                                               and float(a_) == float(b_))
            diff = (abs(float(a_) - float(b_)) if a_ is not None and b_ is not None and not both_nan else None)
            # exact equality: the registered file holds full-precision floats from the same core functions
            rows.append(dict(setup=tag, value=key, this_code=a_, registered=b_, abs_difference=diff, equal=bool(same)))
        rows.append(dict(setup=tag, value="outcome", this_code=m["outcome"], registered=r["outcome"],
                         equal=m["outcome"] == r["outcome"]))
    D = pd.DataFrame(rows); D.to_csv(out / "codepath_check.csv", index=False)
    if not D.equal.all():
        sys.exit("STOP: this code does not reproduce the registered primary; see codepath_check.csv")
    PC.log("code path reproduces the registered setup A and setup B primary")


# ------------------------------------------------------------------ e7
def e7(X, out):
    R, C = X.R, X.C
    import pdt_score as S
    LB = S.db_labels(R.r / "g4c" / "db_test.parquet")
    k = X.params["k"]["crack"]; merge = PC.CLASS_SET_MERGE["merged_crack"]
    pools = {}
    for f in ("S1", "S2"):
        files = sorted((R.r / "g4c" / "main" / f"fold_{f}.txt").read_text().split())
        cl = pd.read_csv(R.r / "g4c" / "main" / "clusters.csv").set_index("file").loc[files].reset_index()
        pools[f] = PC.Pool(f, files, cl.cluster.tolist(), C.majority_strata(cl, "cluster", "country"),
                           LB[LB.file.isin(set(files))])
    for c in PC.EXT_FOUR:
        pools[c] = PC.external_pool(R, c)
    predB = lambda m, p: R.pred_b(m, "rdd2022_scoring" if p in ("S1", "S2") else p)  # noqa: E731
    pt, bt, det = PC.panel(PC.ARCH_B, predB, pools, PC.CRACK, k, X.B, X.seed, merge=merge)
    g = {m: [m] for m in PC.ARCH_B}
    res = dict(primary=PC.ratio_from_panel(pt, bt, g, "S1", "S2", PC.EXT_FOUR))
    s1 = {m: pt[(m, "S1")] for m in PC.ARCH_B}
    res["selection_s1"] = dict(recall=s1, selected=max(s1, key=lambda m: (s1[m], m)))
    res["leave_one_out"] = {m: PC.ratio_from_panel(pt, bt, {x: [x] for x in PC.ARCH_B if x != m}, "S1", "S2",
                                                   PC.EXT_FOUR) for m in PC.ARCH_B}
    res["each_collection"] = {c: PC.ratio_from_panel(pt, bt, g, "S1", "S2", [c]) for c in PC.EXT_FOUR}
    PC.jdump(res, out / "e7_primary.json")
    pd.DataFrame(det).to_csv(out / "e7_recalls.csv", index=False)
    PC.boot_long(bt).to_parquet(out / "e7_bootstrap.parquet")
    pd.DataFrame([PC.flat_result(v, left_out=m) for m, v in res["leave_one_out"].items()] +
                 [PC.flat_result(v, collection_only=c) for c, v in res["each_collection"].items()]) \
        .to_csv(out / "e7_sensitivities.csv", index=False)
    PC.log(f"e7: {res['primary']['outcome']}")


# ------------------------------------------------------------------ e8
def halves(units, strata, seed):
    """Assign units to two halves at random within strata."""
    rng = np.random.default_rng(seed)
    h = {}
    for s in sorted(set(strata.values())):
        us = sorted(u for u in set(units) if strata[u] == s)
        perm = rng.permutation(len(us))
        for i, j in enumerate(perm):
            h[us[j]] = i % 2
    return h


def e8(X, out):
    R = X.R
    allv = sorted(R.image_props().file)
    clean = set(X.clean)
    linked = [f for f in allv if f not in clean]
    cl = R.clusters("main").set_index("file").cluster.to_dict()
    unit = {f: cl.get(f, f"img:{f}") for f in allv}
    strata = {unit[f]: country_of(f) for f in allv}
    variants = {}
    for name, files in (("all_validation", allv), ("linked_only", linked)):
        h = halves([unit[f] for f in files], {unit[f]: strata[unit[f]] for f in files}, X.seed)
        for i, tag in ((0, "S1"), (1, "S2")):
            fs = sorted(f for f in files if h[unit[f]] == i)
            LA = R.labels_internal()
            variants.setdefault(name, {})[tag] = PC.Pool(tag, fs, [unit[f] for f in fs],
                                                         {unit[f]: strata[unit[f]] for f in fs},
                                                         LA[LA.file.isin(set(fs))][["file", "cls"] + PC.BOX])
    variants["clean_075"] = {t: PC.internal_pool(R, t, R.fold(t, "main"), "main") for t in ("S1", "S2")}
    variants["strict_065"] = {t: PC.internal_pool(R, t, R.fold(t, "sens065"), "sens065") for t in ("S1", "S2")}
    ext = {c: PC.external_pool(R, c) for c in PC.EXT_FOUR}
    rows, out_json = [], {}
    for name, ip in variants.items():
        pools = {**ip, **ext}
        pt, bt, _ = PC.panel(PC.RUNS_A, X.predA, pools, PC.FOUR, X.k4, X.B, X.seed)
        r = PC.ratio_from_panel(pt, bt, PC.GROUPS_A, "S1", "S2", PC.EXT_FOUR)
        out_json[name] = r
        rows.append(dict(variant=name, R=r["R"], fieller=r["fieller"], gate=r["repeatability_gate"], outcome=r["outcome"]))
    pd.DataFrame(rows).to_csv(out / "e8_ratios.csv", index=False)
    PC.jdump(out_json, out / "e8_ratios.json")
    # gain on linked images against model size (point recall at the quota on each image set)
    import pdt_score as S
    pools = {"linked": PC.image_pool(R, "linked", linked, country_of), "clean": PC.image_pool(R, "clean", X.clean, country_of)}
    pt, bt, _ = PC.panel(PC.RUNS_A, X.predA, pools, PC.FOUR, X.k4, X.B, X.seed)
    g = [dict(architecture=a, gigaflops=S.OPS_A[a],
              recall_linked=float(np.mean([pt[(r, "linked")] for r in PC.GROUPS_A[a]])),
              recall_clean=float(np.mean([pt[(r, "clean")] for r in PC.GROUPS_A[a]]))) for a in PC.ARCH_A]
    G = pd.DataFrame(g); G["gain"] = G.recall_linked - G.recall_clean
    G.to_csv(out / "e8_gain_by_size.csv", index=False)
    PC.log("e8 done")


# ------------------------------------------------------------------ e9
def e9(X, out):
    R = X.R
    p9 = X.params["exp9"]
    op, fl = p9["operating"], p9["floor_conf"]   # operating: "quota" (k alerts per 100 images) or a confidence
    LA = R.labels_internal()
    L = LA[LA.file.isin(set(X.clean))].reset_index(drop=True)
    sizes = R.image_props().set_index("file")[["src_w", "src_h"]]
    found_op, found_fl = {}, {}
    rng = np.random.default_rng(np.random.SeedSequence([X.seed, 9]))
    chance_op = {r: [] for r in PC.RUNS_A}; chance_fl = {r: [] for r in PC.RUNS_A}
    for r in PC.RUNS_A:
        P = R.pred_internal(r); P = P[P.file.isin(set(X.clean)) & P.cls.isin(PC.FOUR) & (P.conf >= fl)]
        P = P.reset_index(drop=True)
        if op == "quota":   # each checkpoint's alerts at the four-class quota over the clean holdout, locked order
            order = np.lexsort((P.y2.to_numpy(), P.x2.to_numpy(), P.y1.to_numpy(), P.x1.to_numpy(),
                                P.file.to_numpy(), -P.conf.to_numpy()))
            opm = np.zeros(len(P), bool)
            opm[order[:X.C.quota_of(X.params["class_sets"]["four_class"]["k"], len(X.clean))]] = True
        else:
            opm = (P.conf >= float(op)).to_numpy()
        found_fl[r] = PC.match_flags(P, L)[0]
        found_op[r] = PC.match_flags(P[opm], L)[0]
        for _ in range(p9["chance_repeats"]):
            Q = P.copy()
            wh = sizes.loc[Q.file].to_numpy()
            bw, bh = (Q.x2 - Q.x1).to_numpy(), (Q.y2 - Q.y1).to_numpy()
            nx = rng.uniform(0, np.maximum(wh[:, 0] - bw, 0)); ny = rng.uniform(0, np.maximum(wh[:, 1] - bh, 0))
            Q["x1"], Q["y1"], Q["x2"], Q["y2"] = nx, ny, nx + bw, ny + bh
            chance_fl[r].append(PC.match_flags(Q, L)[0])
            chance_op[r].append(PC.match_flags(Q[opm], L)[0])
    rec = {r: found_op[r].mean() for r in PC.RUNS_A}
    best = max(rec, key=lambda r: (rec[r], r))

    def zones(fop, ffl, ref):
        z = np.full(len(L), "unreachable", object)
        anyfl = np.any([ffl[r] for r in PC.RUNS_A], 0)
        anyop = np.any([fop[r] for r in PC.RUNS_A], 0)
        z[anyfl] = "low_confidence_only"
        z[anyop] = "found_by_another"
        z[fop[ref]] = "found_by_reference"
        return z
    Z = L[["file", "obj", "cls", "country"]].copy()
    Z["zone_best"] = zones(found_op, found_fl, best)
    Z["zone_fixed"] = zones(found_op, found_fl, "yolo12s_seed0")
    for i in range(p9["chance_repeats"]):
        Z[f"chance_{i}"] = zones({r: chance_op[r][i] for r in PC.RUNS_A}, {r: chance_fl[r][i] for r in PC.RUNS_A}, best)
    Z.to_parquet(out / "e9_boxes.parquet")
    cl = R.clusters("main").set_index("file").cluster
    Z["unit"] = Z.file.map(cl)
    units = sorted(Z.unit.unique()); ui = {u: i for i, u in enumerate(units)}
    strata = Z.groupby("unit").country.first().to_dict()
    W = X.C.unit_weights(units, strata, np.random.default_rng(X.seed), B=X.B)  # B x units
    uidx = Z.unit.map(ui).to_numpy()
    rows = []
    for col in ["zone_best", "zone_fixed"]:
        for zname in ["found_by_reference", "found_by_another", "low_confidence_only", "unreachable"]:
            ind = (Z[col] == zname).to_numpy().astype(float)
            for grp, mask in [("all", np.ones(len(Z), bool))] + [(f"country:{c}", (Z.country == c).to_numpy())
                                                                  for c in PC.COUNTRIES] + \
                    [(f"class:{c}", (Z.cls == c).to_numpy()) for c in PC.FOUR]:
                if not mask.any():
                    continue
                num = np.bincount(uidx[mask], weights=ind[mask], minlength=len(units))
                den = np.bincount(uidx[mask], minlength=len(units)).astype(float)
                bs = (W @ num) / np.maximum(W @ den, 1)
                ch = np.mean([(Z[f"chance_{i}"][mask] == zname).mean() for i in range(p9["chance_repeats"])]) \
                    if col == "zone_best" else None
                rows.append(dict(version=col, zone=zname, group=grp, boxes=int(mask.sum()),
                                 share=float(ind[mask].mean()), ci95=PC.pct(bs), chance_share=ch))
    pd.DataFrame(rows).to_csv(out / "e9_summary.csv", index=False)
    PC.jdump(dict(best_checkpoint=best, operating=op, recall_at_operating_point=rec), out / "e9_best.json")
    PC.log(f"e9 done (best {best})")


# ------------------------------------------------------------------ e10
def per_class_counts(P, L, files, classes):
    lm, _, _ = PC.match_flags(P[P.file.isin(set(files))], L[L.file.isin(set(files))])
    Lf = L[L.file.isin(set(files))].reset_index(drop=True)
    return Lf.assign(hit=lm)


def e10(X, out):
    R = X.R
    conf = X.params["exp10"]["conf"]
    LA = R.labels_internal()
    Lb = LA[LA.file.isin(set(X.clean))][["file", "cls"] + PC.BOX]
    cl = R.clusters("main").set_index("file").cluster.to_dict()
    rows, boots = [], {}
    for mode in ("conf", "quota"):
        for det in [X.params["selected"]]:
            if mode == "conf":
                Pb = R.pred_internal(det); Pb = Pb[Pb.conf >= conf]
            else:
                Pb = PC.top_quota(R.pred_internal(det), X.clean, PC.FOUR, X.k4)
            HB = per_class_counts(Pb[Pb.cls.isin(PC.FOUR)], Lb, X.clean, PC.FOUR)
            HB["unit"] = HB.file.map(cl)
            for c in PC.EXT_FOUR:
                files = R.pool(c); B = R.blocks(c)
                Pe = R.pred_external(det, c)
                Pe = Pe[Pe.conf >= conf] if mode == "conf" else PC.top_quota(Pe, files, PC.FOUR, X.k4)
                HE = per_class_counts(Pe[Pe.cls.isin(PC.FOUR)], R.labels_external(c), files, PC.FOUR)
                HE["unit"] = HE.file.map(B)
                pt = mix_decomp(HB, HE)
                bs = []
                rng = np.random.default_rng(np.random.SeedSequence([X.seed, 10, PC.EXT_ALL.index(c)]))
                ub, ue = HB.unit.unique(), HE.unit.unique()
                gb, ge = HB.groupby("unit").indices, HE.groupby("unit").indices
                for _ in range(X.B):
                    ib = np.concatenate([gb[u] for u in rng.choice(ub, len(ub))])
                    ie = np.concatenate([ge[u] for u in rng.choice(ue, len(ue))])
                    bs.append(mix_decomp(HB.iloc[ib], HE.iloc[ie]))
                bs = pd.DataFrame(bs)
                boots[(mode, c)] = bs.assign(mode=mode, detector=det, collection=c, draw=np.arange(len(bs)))
                r = dict(mode=mode, detector=det, collection=c, **pt)
                for k_ in ("explained_share_benchmark_reweighted", "explained_share_external_standardized",
                           "explained_share_mean"):
                    r[f"{k_}_ci95"] = PC.pct(bs[k_])
                rows.append(r)
    pd.DataFrame(rows).to_csv(out / "e10_standardized.csv", index=False)
    pd.concat(boots.values(), ignore_index=True).to_parquet(out / "e10_bootstrap.parquet")
    PC.log("e10 done")


def mix_decomp(HB, HE):
    rb = HB.groupby("cls").hit.mean().reindex(PC.FOUR)
    re_ = HE.groupby("cls").hit.mean().reindex(PC.FOUR)
    wb = HB.cls.value_counts(normalize=True).reindex(PC.FOUR).fillna(0)
    we = HE.cls.value_counts(normalize=True).reindex(PC.FOUR).fillna(0)
    RB, RE = float(HB.hit.mean()), float(HE.hit.mean())
    RB_e = float((we * rb.fillna(0)).sum())
    RE_std = float((wb * re_.fillna(0)).sum())
    gap = RB - RE
    s1 = (RB - RB_e) / gap if gap else np.nan
    s2 = (RE_std - RE) / gap if gap else np.nan
    return dict(benchmark_recall=RB, external_recall=RE, benchmark_at_external_mix=RB_e,
                external_at_benchmark_mix=RE_std, gap=gap, explained_share_benchmark_reweighted=s1,
                explained_share_external_standardized=s2, explained_share_mean=(s1 + s2) / 2)


# ------------------------------------------------------------------ e12
def e12(X, out):
    R = X.R
    pools = {"S1": PC.internal_pool(R, "S1", R.fold("S1")), "S2": PC.internal_pool(R, "S2", R.fold("S2"))}
    for c in PC.EXT_FOUR:
        pools[c] = PC.external_pool(R, c)
    rows, rr = [], []
    for k in X.params["exp12"]["quotas"]:
        try:
            pt, bt, det = PC.panel(PC.RUNS_A, X.predA, pools, PC.FOUR, k, X.B, X.seed)
        except RuntimeError as e:
            rr.append(dict(k=k, note=f"not enough predictions: {e}")); continue
        for d in det:
            rows.append(dict(k=k, **d))
        r = PC.ratio_from_panel(pt, bt, PC.GROUPS_A, "S1", "S2", PC.EXT_FOUR)
        rr.append(dict(k=k, R=r["R"], fieller=r["fieller"], gate=r["repeatability_gate"], outcome=r["outcome"]))
    pd.DataFrame(rows).to_csv(out / "e12_curve.csv", index=False)
    pd.DataFrame(rr).to_csv(out / "e12_R.csv", index=False)
    PC.log("e12 done")


# ------------------------------------------------------------------ e13 / e14 / e19
def e13_14_19(X, out):
    R = X.R
    p14 = X.params["exp14"]["bins"]
    rows13, rows14, rows19, boxes14 = [], [], [], []
    sets = {"clean_holdout": (X.clean, None)}
    for c in PC.EXT_ALL:
        sets[c] = (R.pool(c), c)
    for name, (files, coll) in sets.items():
        classes = PC.FOUR if (coll is None or coll in PC.EXT_FOUR) else PC.POTHOLE
        k = X.k4 if classes == PC.FOUR else X.params["k"]["pothole"]
        if coll is None:
            LA = R.labels_internal(); L = LA[LA.file.isin(set(files)) & LA.cls.isin(classes)].reset_index(drop=True)
            sz = R.image_props().set_index("file")[["src_w", "src_h"]]
            unit = R.clusters("main").set_index("file").cluster.to_dict()
        else:
            L = R.labels_external(coll); L = L[L.cls.isin(classes)].reset_index(drop=True)
            sz = R.ext_props(coll).set_index("file")[["src_w", "src_h"]]
            unit = R.blocks(coll)
        hits = {}
        for r in PC.RUNS_A:
            P = R.pred_internal(r) if coll is None else R.pred_external(r, coll)
            Q = PC.top_quota(P, files, classes, k)
            lm, pm, _ = PC.match_flags(Q, L)
            hits[r] = lm
            if r == X.params["selected"]:
                img_alert = set(Q.file)
                has_ref = set(L.file)
                pos = [f for f in files if f in has_ref]; neg = [f for f in files if f not in has_ref]
                rows19.append(dict(set=name, images_with_damage=len(pos), images_without=len(neg),
                                   image_recall=np.mean([f in img_alert for f in pos]) if pos else None,
                                   images_without_alerted=np.mean([f in img_alert for f in neg]) if neg else None))
        sel = hits[X.params["selected"]]
        shared = ~np.any([hits[r] for r in PC.RUNS_A], 0)
        if coll is not None:
            for c in classes:
                m = (L.cls == c).to_numpy()
                if not m.any():
                    continue
                u = L.file.map(unit).to_numpy()[m]
                rec = [hits[r][m].mean() for r in PC.RUNS_A]
                bs = boot_share(sel[m].astype(float), u, X.B, X.seed)
                rows13.append(dict(collection=coll, cls=c, boxes=int(m.sum()), recall_selected=float(sel[m].mean()),
                                   ci95=PC.pct(bs), recall_mean_20=float(np.mean(rec))))
        s = 640.0 / sz.loc[L.file].max(axis=1).to_numpy()
        w = (L.x2 - L.x1).to_numpy() * s; h = (L.y2 - L.y1).to_numpy() * s
        boxes14.append(pd.DataFrame(dict(set=name, file=L.file.to_numpy(), cls=L.cls.to_numpy(), width=w,
                                         shortest_side=np.minimum(w, h), sqrt_area=np.sqrt(w * h),
                                         found_selected=sel, missed_by_all=shared)))
        for measure, v in (("width", w), ("shortest_side", np.minimum(w, h)), ("sqrt_area", np.sqrt(w * h))):
            b = np.digitize(v, p14) - 1
            for i in range(len(p14) - 1):
                for c in classes:
                    m = (b == i) & (L.cls == c).to_numpy()
                    if m.any():
                        rows14.append(dict(set=name, measure=measure, bin_lo=p14[i], bin_hi=p14[i + 1], cls=c,
                                           boxes=int(m.sum()), recall_selected=float(sel[m].mean()),
                                           shared_miss_share=float(shared[m].mean())))
    pd.DataFrame(rows13).to_csv(out / "e13_by_class.csv", index=False)
    pd.DataFrame(rows14).to_csv(out / "e14_bins.csv", index=False)
    pd.concat(boxes14, ignore_index=True).to_parquet(out / "e14_boxes.parquet")
    if X.params.get("exp19", {}).get("include", True):
        pd.DataFrame(rows19).to_csv(out / "e19_image_level.csv", index=False)
    PC.log("e13, e14, e19 done")


def boot_share(ind, units, B, seed):
    uu, inv = np.unique(units, return_inverse=True)
    num = np.bincount(inv, weights=ind); den = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uu), size=(B, len(uu)))
    return num[idx].sum(1) / np.maximum(den[idx].sum(1), 1)


# ------------------------------------------------------------------ e16
def e16(X, out):
    R = X.R
    nb, mc = X.params["exp16"]["bins"], X.params["exp16"]["min_conf"]
    det = X.params["selected"]
    rows, rc = [], []
    sets = {"clean_holdout": None, **{c: c for c in PC.EXT_ALL}}
    for name, coll in sets.items():
        classes = PC.FOUR if (coll is None or coll in PC.EXT_FOUR) else PC.POTHOLE
        if coll is None:
            P = R.pred_internal(det); P = P[P.file.isin(set(X.clean))]
            LA = R.labels_internal(); L = LA[LA.file.isin(set(X.clean))]
        else:
            P = R.pred_external(det, coll); L = R.labels_external(coll)
        P = P[P.cls.isin(classes) & (P.conf >= mc)].sort_values("conf", ascending=False).reset_index(drop=True)
        L = L[L.cls.isin(classes)]
        _, pm, _ = PC.match_flags(P, L)
        conf = P.conf.to_numpy()
        edges = np.linspace(0, 1, nb + 1)
        b = np.clip(np.digitize(conf, edges) - 1, 0, nb - 1)
        ece = 0.0
        for i in range(nb):
            m = b == i
            if m.any():
                ece += m.mean() * abs(pm[m].mean() - conf[m].mean())
                rows.append(dict(set=name, bin=i, lo=edges[i], hi=edges[i + 1], n=int(m.sum()),
                                 mean_conf=float(conf[m].mean()), share_matched=float(pm[m].mean())))
        rows.append(dict(set=name, bin="ECE", n=len(conf), mean_conf=None, share_matched=float(ece)))
        cum = np.cumsum(pm) / np.arange(1, len(pm) + 1)
        for cov in np.linspace(0.01, 1, 100):
            j = max(int(np.ceil(cov * len(pm))) - 1, 0)
            if len(pm):
                rc.append(dict(set=name, coverage=float(cov), risk=float(1 - cum[j]), conf_at=float(conf[j])))
    pd.DataFrame(rows).to_csv(out / "e16_calibration.csv", index=False)
    pd.DataFrame(rc).to_csv(out / "e16_risk_coverage.csv", index=False)
    PC.log("e16 done")


PARTS = {"codepath": codepath, "e7": e7, "e8": e8, "e9": e9, "e10": e10, "e12": e12, "e13_14_19": e13_14_19,
         "e16": e16}


def main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if not a.skip_gate:
        PC.verify_lock(a.root, a.manifest_sha)
        PC.rule_gate(a.rules, a.rules_sha, a.params_sha)
    params = PC.load_params(a.rules)
    PC.env_record(out)
    X = Ctx(a, params)
    for p in a.parts:
        PARTS[p](X, out)
    PC.write_manifest(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--rules", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--scoring", default="")
    ap.add_argument("--parts", nargs="+", choices=list(PARTS), default=list(PARTS))
    ap.add_argument("--manifest_sha", default="ef49673bdc6301009223fb7eecd0aa71be7f53eb6220395d30262b64b19228c7")
    ap.add_argument("--rules_sha", default=""); ap.add_argument("--params_sha", default="")
    ap.add_argument("--skip_gate", action="store_true", help="synthetic tests only")
    main(ap.parse_args())

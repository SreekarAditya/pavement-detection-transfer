#!/usr/bin/env python3
"""Scoring (protocol 3.3.9, Section 9, steps 8 and 9). Run only after the lock (gate G8).

  --stage s1   step 8: open S1 only; record the selected architecture of setup A and setup B (Section 6, item 1).
  --stage all  step 9: open S2 and the external scores; primary analysis, prespecified sensitivities, pothole
               analysis (Section 6, item 6), setup B replication, secondary items 1, 3, 4, 5 and 7.

Before anything is read, every file listed in lock_manifest.json is hashed and compared; any missing file or
mismatch stops the run, and the SHA-256 of lock_manifest.json itself must equal --manifest_sha (the value in
the public deposit).

Layout under --root (as packed at the lock):
  e1/labels.parquet, e1/sealed/predictions/<run>.parquet          setup A internal predictions (E1)
  g3/main/{fold_dev,fold_S1,fold_S2}.txt, clusters.csv           setup A folds (G3), g3/sens065/ (strict)
  g4c/main/..., g4c/sens065/..., g4c/db_test.parquet             setup B folds and DetectionBench labels
  s2/g4/{pool_final_<c>.txt, labels_final_<c>.parquet, blocks.csv, g4_g6_decision.json}
  s2/s2a/sealed_ext/<c>/<run>.parquet                             setup A external predictions
  s2/s2b/sealed_b/<pool>/<model>.parquet                          setup B predictions
"""
import argparse, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pdt_core as C  # noqa: E402

K_ALL, K_POT = 152.49, 15.80          # G7 (protocol Section 4)
ARCH_A = ["yolov5s", "yolov8s", "yolov9s", "yolov10s", "yolo11s", "yolo12n", "yolo12s", "yolo12m"]
SEEDS_A = {a: ([0, 42, 123] if a not in ("yolo12n", "yolo12m") else [0]) for a in ARCH_A}
OPS_A = {"yolov5s": 24.0, "yolov8s": 28.6, "yolov9s": 26.7, "yolov10s": 21.6, "yolo11s": 21.5, "yolo12n": 6.5,
         "yolo12s": 21.4, "yolo12m": 67.5}                            # G0 freeze
DEFAULT_A = "yolo12m"
ARCH_B = ["yolov8n", "yolov8s", "yolo26s", "yolo26m", "rfdetr_nano", "rfdetr_small"]
OPS_B = {"yolov8n": 8.7, "yolov8s": 28.6, "yolo26s": 20.7, "yolo26m": 68.2, "rfdetr_nano": 31.9, "rfdetr_small": 59.8}
DEFAULT_B = "rfdetr_small"
COUNTRIES = ["China_MotorBike", "China_Drone", "United_States", "Czech", "India", "Japan", "Norway"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def country_of(n):
    return next((c for c in COUNTRIES if n.startswith(c)), "UNKNOWN")


def verify(root, manifest_sha):
    mp = Path(root) / "lock_manifest.json"
    got = sha(mp)
    if got != manifest_sha:
        sys.exit(f"STOP: lock_manifest.json SHA-256 {got} differs from the published value")
    man = json.load(open(mp))
    bad = [f for f, h in man.items() if not (Path(root) / f).exists() or sha(Path(root) / f) != h]
    if bad:
        sys.exit(f"STOP: {len(bad)} locked files missing or changed, e.g. {bad[:5]}")
    log(f"lock verified: {len(man)} files")


# ------------------------------------------------------------------ pools
class Pool:
    def __init__(self, name, files, units, strata, L):
        self.name, self.files, self.units, self.strata, self.L = name, list(files), list(units), strata, L


def internal_pools(fold_dir, labels, folds):
    cl = pd.read_csv(Path(fold_dir) / "clusters.csv")
    out = {}
    for f in folds:
        files = sorted(Path(fold_dir, f"fold_{f}.txt").read_text().split())
        c = cl.set_index("file").loc[files].reset_index()
        strata = C.majority_strata(c, "cluster", "country")
        out[f] = Pool(f, files, c.cluster.tolist(), strata, labels[labels.file.isin(files)])
    return out


def external_pools(g4, slugs):
    blocks = pd.read_csv(Path(g4) / "blocks.csv")
    out = {}
    for s in slugs:
        files = sorted(Path(g4, f"pool_final_{s}.txt").read_text().split())
        b = blocks[blocks.collection == s].set_index("file").loc[files].block.tolist()
        L = pd.read_parquet(Path(g4) / f"labels_final_{s}.parquet")
        L = L[L.cls >= 0][["file", "cls", "x1", "y1", "x2", "y2"]]
        out[s] = Pool(s, files, b, {u: s for u in set(b)}, L)
    return out


def db_labels(path):
    """DetectionBench labels: COCO-style [x, y, w, h] in pixels, categories 0..3; geometry rule as for externals."""
    d = pd.read_parquet(path)
    rows = []
    for f, o in zip(d.file_name, d.objects):
        seen = set()
        for c, (x, y, w, h) in zip(o["categories"], o["bbox"]):
            k = (int(c), float(x), float(y), float(w), float(h))
            if w < 1 or h < 1 or k in seen:
                continue
            seen.add(k); rows.append((f, int(c), float(x), float(y), float(x + w), float(y + h)))
    return pd.DataFrame(rows, columns=["file", "cls", "x1", "y1", "x2", "y2"])


def draws(pools, order):
    rng = np.random.default_rng(C.SEED)
    return {p: C.unit_weights(pools[p].units, pools[p].strata, rng) for p in order}


# ------------------------------------------------------------------ recall at the quota for a panel
class Panel:
    """Point and bootstrap recall per run and pool. preds(run, pool) -> DataFrame of predictions."""

    def __init__(self, runs, preds, pools, order, classes, k, density=False, merge=None, k_dev=None):
        self.runs, self.pools, self.order = runs, pools, order
        W = draws(pools, order)
        self.point, self.boot, self.detail = {}, {}, {}
        for p in order:
            pool = pools[p]; n = len(pool.files)
            hint = k
            if density:
                rho = 100 * int(pool.L.cls.isin(classes).sum()) / n
                hint = k * max(1.0, rho / k_dev) * 1.5
            for r in runs:
                R = C.QuotaRun(preds(r, p), pool.L, pool.files, classes, merge=merge, k_hint=hint)
                one = np.ones(n, np.int64)

                def q(w):
                    if density:
                        refs = int((w * R.nref).sum())
                        return int(np.floor(k / k_dev * refs))
                    return C.quota_of(k, int(w.sum()))
                m, rf, used = R.evaluate(one, q(one))
                self.point[(r, p)] = m / rf if rf else np.nan
                self.detail[(r, p)] = dict(run=r, pool=p, images=n, quota=q(one), used=used, underfill=q(one) - used,
                                           matched=m, refs=rf, refs_per_100=100 * rf / n, recall=m / rf if rf else None)
                self.boot[(r, p)] = np.array([(lambda mm, rr: mm / rr if rr else np.nan)(*R.evaluate(W[p][b], q(W[p][b]))[:2])
                                              for b in range(C.B)])
            log(f"  {p}: {len(runs)} runs scored")

    def arch(self, groups, p, boot=False):
        """groups: {arch: [runs]} -> (n_arch,) point or (B, n_arch) bootstrap seed means."""
        src = self.boot if boot else self.point
        v = [np.mean([src[(r, p)] for r in groups[a]], axis=0) for a in groups]
        return np.stack(v, axis=-1)


def ratio(panel, groups, s1, s2, ext):
    x1, x2 = panel.arch(groups, s1), panel.arch(groups, s2)
    x1b, x2b = panel.arch(groups, s1, True), panel.arch(groups, s2, True)
    z = np.mean([panel.arch(groups, e) for e in ext], 0)
    zb = np.mean([panel.arch(groups, e, True) for e in ext], 0)
    res = C.ratio_result(x1, x2, z, x1b, x2b, zb)
    res["per_arch"] = {a: dict(x1=float(x1[i]), x2=float(x2[i]), z=float(z[i]),
                               **{f"z_{e}": float(panel.arch(groups, e)[i]) for e in ext})
                       for i, a in enumerate(groups)}
    res["collections"] = list(ext)
    return res


def analysis(panel, groups, ext, seed0=None):
    out = dict(primary=ratio(panel, groups, "S1", "S2", ext))
    out["each_collection"] = {e: ratio(panel, groups, "S1", "S2", [e]) for e in ext}
    out["leave_one_architecture_out"] = {a: ratio(panel, {b: groups[b] for b in groups if b != a}, "S1", "S2", ext)
                                         for a in groups}
    if seed0:
        out["seed0_only"] = ratio(panel, seed0, "S1", "S2", ext)
    return out


def strip(d):
    """Keep JSON small: drop per-arch detail from nested sensitivity entries."""
    return {k: ({kk: vv for kk, vv in v.items() if kk != "per_arch"} if isinstance(v, dict) and "outcome" in v else
                (strip(v) if isinstance(v, dict) else v)) for k, v in d.items()}


# ------------------------------------------------------------------ main
def main(a):
    root, out = Path(a.root), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    verify(root, a.manifest_sha)
    if a.stage == "verify":
        return
    e1 = root / "e1"
    LA = pd.read_parquet(e1 / "labels.parquet")[["file", "cls", "x1", "y1", "x2", "y2"]]
    runsA = [f"{x}_seed{s}" for x in ARCH_A for s in SEEDS_A[x]]
    gA = {x: [f"{x}_seed{s}" for s in SEEDS_A[x]] for x in ARCH_A}
    g0 = {x: [f"{x}_seed0"] for x in ARCH_A}
    gB = {m: [m] for m in ARCH_B}
    LB = db_labels(root / "g4c" / "db_test.parquet")
    folds = ["S1"] if a.stage == "s1" else ["S1", "S2"]

    def predA(run, pool):
        if pool in ("S1", "S2"):
            return pq.read_table(e1 / "sealed" / "predictions" / f"{run}.parquet",
                                 columns=["file", "cls", "conf", "x1", "y1", "x2", "y2"]).to_pandas()
        return pd.read_parquet(root / "s2" / "s2a" / "sealed_ext" / pool / f"{run}.parquet")

    def predB(model, pool):
        p = "rdd2022_scoring" if pool in ("S1", "S2") else pool
        return pd.read_parquet(root / "s2" / "s2b" / "sealed_b" / p / f"{model}.parquet")

    if a.stage == "s1":
        res = {}
        for tag, runs, groups, pred, fdir, L, default, ops in [
                ("A", runsA, gA, predA, root / "g3" / "main", LA, DEFAULT_A, OPS_A),
                ("B", ARCH_B, gB, predB, root / "g4c" / "main", LB, DEFAULT_B, OPS_B)]:
            pools = internal_pools(fdir, L, ["S1"])
            P = Panel(runs, pred, pools, ["S1"], [0, 1, 2, 3], K_ALL)
            x = P.arch(groups, "S1")
            best = sorted(groups, key=lambda m: (-x[list(groups).index(m)], ops[m], m))[0]
            res[tag] = dict(s1_recall={m: float(x[i]) for i, m in enumerate(groups)}, selected=best, default=default,
                            same=best == default)
            pd.DataFrame(P.detail.values()).to_csv(out / f"s1_detail_{tag}.csv", index=False)
            log(f"setup {tag}: selected {best}, default {default}")
        json.dump(res, open(out / "selection_s1.json", "w"), indent=1)
        return

    sel = json.load(open(Path(a.selection)))
    dec = json.load(open(root / "s2" / "g4" / "g4_g6_decision.json"))
    c = dec["collections"]
    ext4 = sorted(s for s in c if c[s]["eligible_four_class"])
    extP = sorted(s for s in c if c[s]["eligible_pothole"])
    in_force = dec["primary_in_force"]
    results = dict(primary_in_force=in_force, collections_four_class=ext4, collections_pothole=extP)

    def full(tag, runs, groups, pred, fdir, L, ext, classes, k, seed0=None, strict_dir=None, k_dev=None):
        log(f"{tag}: pools S1, S2, {ext}")
        pools = {**internal_pools(fdir, L, ["S1", "S2"]), **external_pools(root / "s2" / "g4", ext)}
        order = ["S1", "S2"] + list(ext)
        P = Panel(runs, pred, pools, order, classes, k)
        r = analysis(P, groups, ext, seed0)
        log(f"  primary outcome: {r['primary']['outcome']} (R {r['primary']['R']}, set {r['primary']['fieller']})")
        Pd = Panel(runs, pred, pools, order, classes, k, density=True, k_dev=k_dev)
        r["density_matched"] = ratio(Pd, groups, "S1", "S2", ext)
        if strict_dir is not None:
            ps = {**internal_pools(strict_dir, L, ["S1", "S2"]), **{e: pools[e] for e in ext}}
            Ps = Panel(runs, pred, ps, order, classes, k)
            r["strict_leakage_065"] = ratio(Ps, groups, "S1", "S2", ext)
        pd.DataFrame(P.detail.values()).to_csv(out / f"detail_{tag}.csv", index=False)
        np.savez_compressed(out / f"boot_{tag}.npz", **{f"{rr}|{pp}": v for (rr, pp), v in P.boot.items()})
        return r, P

    # setup A: four-class (primary unless a fallback is in force) and pothole
    if ext4:
        results["A_four_class"], PA = full("A_four_class", runsA, gA, predA, root / "g3" / "main", LA, ext4,
                                           [0, 1, 2, 3], K_ALL, g0, root / "g3" / "sens065", K_ALL)
    if extP:
        results["A_pothole"], PP = full("A_pothole", runsA, gA, predA, root / "g3" / "main", LA, extP, [3], K_POT, g0,
                                        root / "g3" / "sens065", K_POT)
    # setup B replication (G4c passed: its scoring set exists in the lock)
    if ext4:
        results["B_four_class"], PB = full("B_four_class", ARCH_B, gB, predB, root / "g4c" / "main", LB, ext4,
                                           [0, 1, 2, 3], K_ALL, None, root / "g4c" / "sens065", K_ALL)
    if extP:
        results["B_pothole"], _ = full("B_pothole", ARCH_B, gB, predB, root / "g4c" / "main", LB, extP, [3], K_POT,
                                       None, root / "g4c" / "sens065", K_POT)

    # secondary 1: selected versus default, paired external difference with block-bootstrap interval
    sec = {}
    for tag, P, groups, ext in [("A", PA if ext4 else None, gA, ext4), ("B", PB if ext4 else None, gB, ext4)]:
        if P is None:
            continue
        s, d = sel[tag]["selected"], sel[tag]["default"]
        rows = {}
        for e in ext + ["mean"]:
            es = ext if e == "mean" else [e]
            pt = np.mean([np.mean([P.point[(r, x)] for r in groups[s]]) - np.mean([P.point[(r, x)] for r in groups[d]])
                          for x in es])
            bt = np.mean([np.mean([P.boot[(r, x)] for r in groups[s]], 0) - np.mean([P.boot[(r, x)] for r in groups[d]], 0)
                          for x in es], 0)
            rows[e] = dict(difference=float(pt), ci95=[float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))])
        sec[f"selected_vs_default_{tag}"] = dict(selected=s, default=d, identical=s == d, by_collection=rows)

    # secondary 7: Spearman rank correlation between benchmark recall and each collection's recall (setup A)
    if ext4:
        from scipy.stats import spearmanr
        xb = (PA.arch(gA, "S1") + PA.arch(gA, "S2")) / 2
        sec["spearman_A"] = {e: float(spearmanr(xb, PA.arch(gA, e)).correlation) for e in ext4}

    # secondary 3: alert rate on each collection at the confidence threshold giving k alerts per 100 dev images
    dev = sorted((root / "g3" / "main" / "fold_dev.txt").read_text().split())
    rates = {}
    for r in runsA:
        Pd = pq.read_table(e1 / "sealed" / "predictions" / f"{r}.parquet", columns=["file", "cls", "conf"],
                           filters=[("file", "in", dev)]).to_pandas()
        qd = C.quota_of(K_ALL, len(dev))
        t = float(np.sort(Pd.conf.to_numpy())[::-1][qd - 1])
        rates[r] = {"threshold": t}
        for e in ext4:
            pe = predA(r, e); n = len(Path(root / "s2" / "g4" / f"pool_final_{e}.txt").read_text().split())
            rates[r][e] = float(100 * (pe.conf >= t).sum() / n)
    sec["dev_threshold_alert_rate_A"] = rates

    # secondary 4: source-balanced benchmark recall (partitions weighted equally), S1 and S2 pooled, at the quota
    if ext4:
        sb = {}
        for x in ARCH_A:
            vals = []
            for r in gA[x]:
                per = {}
                for f in ("S1", "S2"):
                    pool = PA.pools[f]
                    R = C.QuotaRun(predA(r, f), pool.L, pool.files, [0, 1, 2, 3], k_hint=K_ALL)
                    n = len(pool.files); one = np.ones(n, np.int64)
                    # per-image matched at the point quota
                    pw = one[R.img]; cw = np.cumsum(pw); full_ = cw <= C.quota_of(K_ALL, n)
                    j = np.bincount(R.img[full_], minlength=n); m = R.M[R.off[:-1] + j]
                    for i, fl in enumerate(pool.files):
                        cc = country_of(fl); per.setdefault(cc, [0, 0])
                        per[cc][0] += int(m[i]); per[cc][1] += int(R.nref[i])
                vals.append(np.mean([v[0] / v[1] for v in per.values() if v[1]]))
            sb[x] = float(np.mean(vals))
        sec["source_balanced_benchmark_recall_A"] = sb

    # secondary 5: merged-crack endpoint on collections mapping all three crack classes
    crack_ext = [e for e in ext4]
    if crack_ext:
        Ld = LA[LA.file.isin(dev)]
        k_crack = round(100 * int(Ld.cls.isin([0, 1, 2]).sum()) / len(dev), 2)
        pools = {**internal_pools(root / "g3" / "main", LA, ["S1", "S2"]), **external_pools(root / "s2" / "g4", crack_ext)}
        Pc = Panel(runsA, predA, pools, ["S1", "S2"] + crack_ext, [0, 1, 2], k_crack, merge={0: 0, 1: 0, 2: 0})
        sec["merged_crack_A"] = dict(k_crack=k_crack, result=ratio(Pc, gA, "S1", "S2", crack_ext))
    sec["external_spacing_workflow"] = "not run: no eligible collection releases verified frame-spacing metadata (G5)"
    results["secondary"] = sec
    json.dump(results, open(out / "score_results.json", "w"), indent=1, default=float)
    json.dump(strip(results), open(out / "score_summary.json", "w"), indent=1, default=float)
    log("scoring done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--manifest_sha", required=True)
    ap.add_argument("--stage", choices=["verify", "s1", "all"], required=True)
    ap.add_argument("--selection", default="", help="selection_s1.json from step 8 (stage all)")
    ap.add_argument("--out", required=True)
    main(ap.parse_args())

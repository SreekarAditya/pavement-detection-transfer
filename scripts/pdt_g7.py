#!/usr/bin/env python3
"""Gate G7: alert rate k and precision projection, development fold only (protocol 3.3.8, Section 9, step 4).

Rule fixed in G7_rule.md before any outcome was opened. This script
  1. verifies every sealed prediction file it reads against e1/sealed_manifest.json;
  2. reads only development-fold rows of class pothole (filtered while reading; S1 and S2 rows are never loaded);
  3. fixes k = development-fold pothole reference boxes per 100 images (label counts only);
  4. per run: quota floor(k N / 100), ranking by confidence, tie-break file name, box coordinates, copy index;
     one-to-one greedy matching at intersection over union 0.5 in original pixels (Ultralytics' rule);
  5. 2,000 cluster-bootstrap draws within partition strata (seed 2026), quota and matching reapplied per draw;
  6. precision projection (rule items 4 to 8).

Fast path: within one image copy, the selected predictions are always a prefix of that image's own order, so the
matched count for every prefix is computed once per image. A direct implementation (explicit copies, global sort,
selection, matching) is run on several draws and must agree exactly (--selftest, run by default).
"""
import argparse, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

POTHOLE = 3
CLS = [POTHOLE]  # target classes; set from --classes
ARCHS = ["yolov5s", "yolov8s", "yolov9s", "yolov10s", "yolo11s", "yolo12n", "yolo12s", "yolo12m"]
SEEDS = {a: ([0, 42, 123] if a not in ("yolo12n", "yolo12m") else [0]) for a in ARCHS}
B = 2000
SEED = 2026


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ matching (copy of Ultralytics' rule, threshold 0.5)
def box_iou(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area_a = (a[:, 2:] - a[:, :2]).prod(1)
    area_b = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-7)


def n_matched(iou):
    """iou: refs x selected predictions. Ultralytics' greedy one-to-one matching at 0.5; returns matched pairs."""
    m = np.array(np.nonzero(iou >= 0.5)).T
    if m.shape[0] == 0:
        return 0
    if m.shape[0] > 1:
        m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
        m = m[np.unique(m[:, 1], return_index=True)[1]]
        m = m[np.unique(m[:, 0], return_index=True)[1]]
    return int(m.shape[0])


# ------------------------------------------------------------------ data
def load_dev(e1, folds):
    dev = sorted(set(Path(folds, "fold_dev.txt").read_text().split()))
    C = pd.read_csv(Path(folds, "clusters.csv"))
    C = C[C.fold == "dev"]
    assert set(C.file) == set(dev), "clusters.csv and fold_dev.txt disagree"
    L = pd.read_parquet(Path(e1, "labels.parquet"))
    Lall = L
    L = L[L.cls.isin(CLS) & L.file.isin(dev)].reset_index(drop=True)
    return dev, C, L, Lall


def strata_of(C):
    """Cluster stratum: partition of most of its images, ties alphabetical (same rule as fold assignment)."""
    cnt = C.groupby(["cluster", "country"]).size().reset_index(name="n")
    cnt = cnt.sort_values(["cluster", "n", "country"], ascending=[True, False, True])
    return cnt.drop_duplicates("cluster").set_index("cluster").country


def bootstrap_weights(C, files):
    """B x n_images matrix of image copy counts; same draws for every run."""
    major = strata_of(C)
    clusters = sorted(major.index)
    cidx = {c: i for i, c in enumerate(clusters)}
    img_c = np.array([cidx[c] for c in C.set_index("file").loc[files].cluster])
    rng = np.random.default_rng(SEED)
    Wc = np.zeros((B, len(clusters)), np.int32)
    for s in sorted(major.unique()):
        ids = np.array([cidx[c] for c in sorted(major[major == s].index)])
        draw = rng.integers(0, len(ids), size=(B, len(ids)))
        for b in range(B):
            np.add.at(Wc[b], ids[draw[b]], 1)
    return Wc[:, img_c]


def read_run(e1, run, man, dev):
    rel = f"sealed/predictions/{run}.parquet"
    p = Path(e1, rel)
    if man.get(rel) != sha(p):
        sys.exit(f"STOP: hash mismatch or missing manifest entry for {rel}")
    t = pq.read_table(p, columns=["file", "cls", "conf", "x1", "y1", "x2", "y2"],
                      filters=[("file", "in", dev), ("cls", "in", CLS)])
    P = t.to_pandas()
    assert set(P.file) <= set(dev) and P.cls.isin(CLS).all()
    return P


class Run:
    """Pre-computation for one run on the development pool."""

    def __init__(self, P, L, files):
        fi = {f: i for i, f in enumerate(files)}
        P = P.copy()
        P["img"] = P.file.map(fi)
        # global base order: confidence descending, then file name, then box coordinates (x1, y1, x2, y2)
        order = np.lexsort((P.y2.to_numpy(), P.x2.to_numpy(), P.y1.to_numpy(), P.x1.to_numpy(),
                            P.file.to_numpy(), -P.conf.to_numpy()))
        P = P.iloc[order].reset_index(drop=True)
        self.P = P
        self.img = P.img.to_numpy()
        n = len(files)
        self.nref = np.zeros(n, np.int64)
        Lg = dict(tuple(L.groupby("file")))
        # per image: matched count for each prefix length of the image's own order (which is the base order)
        self.prefix = [np.zeros(1, np.int64) for _ in range(n)]
        Pg = dict(tuple(P.groupby("img", sort=False)))
        for f, i in fi.items():
            lab = Lg.get(f)
            nr = 0 if lab is None else len(lab)
            self.nref[i] = nr
            prd = Pg.get(i)
            npd = 0 if prd is None else len(prd)
            m = np.zeros(npd + 1, np.int64)
            if nr and npd:
                iou = box_iou(lab[["x1", "y1", "x2", "y2"]].to_numpy(), prd[["x1", "y1", "x2", "y2"]].to_numpy())
                iou = iou * (lab.cls.to_numpy()[:, None] == prd.cls.to_numpy()[None, :])
                for j in range(1, npd + 1):
                    m[j] = n_matched(iou[:, :j])
            self.prefix[i] = m
        self.pmax = max(len(m) for m in self.prefix)
        self.M = np.zeros((n, self.pmax), np.int64)  # M[i, j] = matched with j selected (padded with last value)
        for i, m in enumerate(self.prefix):
            self.M[i, :len(m)] = m
            self.M[i, len(m):] = m[-1]

    def evaluate(self, w, k):
        """w: copies per image. Returns dict with quota, used, refs, matched, recall."""
        N = int(w.sum())
        Q = int(np.floor(k * N / 100.0))
        pw = w[self.img]
        cw = np.cumsum(pw)
        total = int(cw[-1]) if len(cw) else 0
        full = cw <= Q
        jfull = np.bincount(self.img[full], minlength=len(w))
        matched = int((w * self.M[np.arange(len(w)), jfull]).sum())
        used = min(Q, total)
        if total > Q:
            b = int(np.argmax(cw > Q))
            partial = Q - (int(cw[b - 1]) if b > 0 else 0)
            if partial > 0:
                i = self.img[b]
                matched += partial * int(self.M[i, jfull[i] + 1] - self.M[i, jfull[i]])
        refs = int((w * self.nref).sum())
        return dict(N=N, quota=Q, used=used, underfill=Q - used, refs=refs, matched=matched,
                    recall=matched / refs if refs else np.nan)


def direct(P, L, files, w, k):
    """Slow reference: explicit copies, global sort with copy index, selection, per-copy matching."""
    fi = {f: i for i, f in enumerate(files)}
    rows = []
    for c in range(int(w.max()) if len(w) else 0):
        sel = [f for f in files if w[fi[f]] > c]
        X = P[P.file.isin(sel)].copy(); X["copy"] = c
        rows.append(X)
    X = pd.concat(rows) if rows else P.iloc[:0].assign(copy=0)
    X = X.sort_values(["conf", "file", "x1", "y1", "x2", "y2", "copy"],
                      ascending=[False, True, True, True, True, True, True])
    N = int(w.sum()); Q = int(np.floor(k * N / 100.0))
    S = X.iloc[:Q]
    Lg = dict(tuple(L.groupby("file")))
    matched = 0
    for (f, c), g in S.groupby(["file", "copy"]):
        lab = Lg.get(f)
        if lab is None:
            continue
        iou = box_iou(lab[["x1", "y1", "x2", "y2"]].to_numpy(), g[["x1", "y1", "x2", "y2"]].to_numpy())
        matched += n_matched(iou * (lab.cls.to_numpy()[:, None] == g.cls.to_numpy()[None, :]))
    refs = int(sum(w[fi[f]] * len(g) for f, g in Lg.items()))
    return matched, refs, Q


# ------------------------------------------------------------------ projection
def cov_across(a, b):
    """Sample covariance across architectures (last axis), ddof 1."""
    a = a - a.mean(-1, keepdims=True); b = b - b.mean(-1, keepdims=True)
    return (a * b).sum(-1) / (a.shape[-1] - 1)


def fieller(a, b, v11, v22, v12, z=1.96):
    A = b * b - z * z * v22
    Bq = a * b - z * z * v12
    Cq = a * a - z * z * v11
    if A <= 0:
        return None  # unbounded or disjoint
    disc = Bq * Bq - A * Cq
    if disc < 0:
        return None
    r = np.sqrt(disc)
    return (Bq - r) / A, (Bq + r) / A


def category(gate, fs):
    if not gate or fs is None:
        return "inconclusive"
    L, U = fs
    if L <= 0 and U >= 1:
        return "inconclusive"
    if U < 0:
        return "negative_slope"
    if L <= 0 <= U and U < 1:
        return "no_detectable_retention"
    if 0 < L and U < 1:
        return "partial_retention"
    if 0 < L <= 1 <= U:
        return "compatible_with_full_retention"
    if L > 1:
        return "amplification"
    return "unclassified"


CATS = ["inconclusive", "negative_slope", "no_detectable_retention", "partial_retention",
        "compatible_with_full_retention", "amplification"]


def project(x_dev, Sig, n_dev, n_s1, n_s2, sizes, Rs, n_coll=2, studies=2000, reps=400, seed=SEED):
    n = len(x_dev)
    H = np.eye(n) - 1.0 / n
    s2 = x_dev.var(ddof=1)
    noise = np.trace(H @ Sig @ H) / (n - 1)
    tau2 = max(s2 - noise, 0.0)
    mu = x_dev.mean() + (np.sqrt(tau2 / s2) if s2 > 0 else 0.0) * (x_dev - x_dev.mean())
    S1 = Sig * n_dev / n_s1; S2 = Sig * n_dev / n_s2
    rng = np.random.default_rng(seed)
    out = []
    jitter = 1e-12 * np.eye(n)
    c1, c2 = np.linalg.cholesky(S1 + jitter), np.linalg.cholesky(S2 + jitter)
    # internal-only gate share
    gate_int = 0
    for _ in range(studies):
        x1 = mu + c1 @ rng.standard_normal(n); x2 = mu + c2 @ rng.standard_normal(n)
        e1 = x1 + (c1 @ rng.standard_normal((n, reps))).T; e2 = x2 + (c2 @ rng.standard_normal((n, reps))).T
        gate_int += np.percentile(cov_across(e1, e2), 5) > 0
    gate_int /= studies
    for size in sizes:
        Se = Sig * n_dev / size; ce = np.linalg.cholesky(Se + jitter)
        for R in Rs:
            for extra in (False, True):
                cnt = dict.fromkeys(CATS, 0); gates = 0
                for _ in range(studies):
                    x1 = mu + c1 @ rng.standard_normal(n); x2 = mu + c2 @ rng.standard_normal(n)
                    zs = []
                    for _s in range(n_coll):
                        eta = np.sqrt(tau2) * rng.standard_normal(n) if extra else 0.0
                        zs.append(0.5 + R * (mu - mu.mean()) + eta + ce @ rng.standard_normal(n))
                    zs = np.array(zs)
                    a = cov_across((x1 + x2) / 2, zs.mean(0)); b = cov_across(x1, x2)
                    e1 = x1 + (c1 @ rng.standard_normal((n, reps))).T
                    e2 = x2 + (c2 @ rng.standard_normal((n, reps))).T
                    ez = np.mean([zs[s] + (ce @ rng.standard_normal((n, reps))).T for s in range(n_coll)], axis=0)
                    ab = cov_across((e1 + e2) / 2, ez); bb = cov_across(e1, e2)
                    V = np.cov(np.vstack([ab, bb]))
                    gate = np.percentile(bb, 5) > 0
                    gates += gate
                    cnt[category(gate, fieller(a, b, V[0, 0], V[1, 1], V[0, 1]))] += 1
                out.append(dict(boxes_per_collection=size, R=R, extra_noise=extra, gate_pass=gates / studies,
                                **{c: cnt[c] / studies for c in CATS}))
    return pd.DataFrame(out), dict(sample_var=s2, sampling_part=noise, spread_tau2=tau2,
                                   spread_sd=float(np.sqrt(tau2)), mu=mu.tolist(), gate_internal_only=gate_int)


# ------------------------------------------------------------------ main
def main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    man = json.load(open(Path(a.e1, "sealed_manifest.json")))
    global CLS
    CLS = [POTHOLE] if a.classes == "pothole" else [0, 1, 2, 3]
    dev, C, L, Lall = load_dev(a.e1, a.folds)
    nfold = {f: int((Lall.cls.isin(CLS) & Lall.file.isin(set(Path(a.folds, f"fold_{f}.txt").read_text().split()))).sum())
             for f in ["S1", "S2"]}  # label counts only
    files = sorted(dev)
    n_dev_pot = len(L)
    k = round(100.0 * n_dev_pot / len(files), 2)
    log(f"development fold: {len(files)} images, {n_dev_pot} target boxes (classes {CLS}), k = {k}")
    if a.k_expected is not None:
        assert abs(k - a.k_expected) < 1e-9, f"k {k} differs from the recorded value {a.k_expected}"
    W = bootstrap_weights(C, files)
    rows, boot = [], {}
    for arch in ARCHS:
        for s in SEEDS[arch]:
            run = f"{arch}_seed{s}"
            P = read_run(a.e1, run, man, files)
            R = Run(P, L, files)
            pt = R.evaluate(np.ones(len(files), np.int64), k)
            if a.selftest and run in ("yolo12s_seed0", "yolov5s_seed0"):
                for b in [0, 1, 2]:
                    got = R.evaluate(W[b], k)
                    m, r, Q = direct(P, L, files, W[b], k)
                    assert (got["matched"], got["refs"], got["quota"]) == (m, r, Q), (run, b, got, m, r, Q)
                m, r, Q = direct(P, L, files, np.ones(len(files), np.int64), k)
                assert (pt["matched"], pt["refs"], pt["quota"]) == (m, r, Q)
                log(f"  selftest passed for {run} (point and 3 bootstrap draws)")
            boot[run] = np.array([R.evaluate(W[b], k)["recall"] for b in range(B)])
            rows.append(dict(run=run, arch=arch, seed=s, n_pothole_pred=len(P), **pt))
            log(f"  {run}: quota {pt['quota']}, used {pt['used']}, matched {pt['matched']}/{pt['refs']}")
    D = pd.DataFrame(rows); D.to_csv(out / "g7_dev_runs.csv", index=False)
    xa = np.array([D[D.arch == ar].recall.mean() for ar in ARCHS])
    Bm = np.column_stack([np.mean([boot[f"{ar}_seed{s}"] for s in SEEDS[ar]], axis=0) for ar in ARCHS])
    pd.DataFrame(Bm, columns=ARCHS).to_csv(out / "g7_dev_bootstrap_arch.csv", index=False)
    Sig = np.cov(Bm, rowvar=False)
    A = pd.DataFrame(dict(arch=ARCHS, dev_recall=xa, boot_sd=np.sqrt(np.diag(Sig)),
                          seeds=[len(SEEDS[ar]) for ar in ARCHS]))
    A.to_csv(out / "g7_dev_arch.csv", index=False)
    log("projection")
    sizes = [150, 500, 1500] if a.classes == "pothole" else [500, 1500, 5000]
    Pj, info = project(xa, Sig, n_dev_pot, nfold["S1"], nfold["S2"], sizes, [0.0, 0.5, 1.0])
    Pj.to_csv(out / "g7_projection.csv", index=False)
    summ = dict(k=k, dev_images=len(files), dev_boxes=n_dev_pot, quota_dev=int(D.quota.iloc[0]),
                bootstrap_draws=B, seed=SEED, classes=CLS, s1_boxes=nfold["S1"], s2_boxes=nfold["S2"],
                mean_boot_corr=float(np.mean(np.corrcoef(Bm, rowvar=False)[np.triu_indices(8, 1)])), **info,
                inputs={"labels.parquet": sha(Path(a.e1, "labels.parquet")),
                        "fold_dev.txt": sha(Path(a.folds, "fold_dev.txt")),
                        "clusters.csv": sha(Path(a.folds, "clusters.csv")),
                        "sealed_manifest.json": sha(Path(a.e1, "sealed_manifest.json"))})
    json.dump(summ, open(out / "g7_summary.json", "w"), indent=1)
    log("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", required=True); ap.add_argument("--folds", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--classes", choices=["pothole", "all"], default="pothole")
    ap.add_argument("--k_expected", type=float, default=None)
    ap.add_argument("--no_selftest", dest="selftest", action="store_false")
    main(ap.parse_args())

"""Scoring core: recall at an alert quota with deterministic tie-break, cluster or block bootstrap, Fieller set.
Protocol 3.3.9, Sections 2, 4 and 5. Used by pdt_score.py (primary, sensitivities, pothole, setup B, secondary).

Quota. A detector takes up to floor(k N / 100) of its eligible predictions in the pool, ranked by confidence; ties by
image file name, then box coordinates (x1, y1, x2, y2), then bootstrap copy index. Matching: Ultralytics' greedy
one-to-one rule at intersection over union 0.5, same class, in original pixels, within each image copy.

Exactness of the fast path. Within one image copy the selected predictions are a prefix of that image's own order,
so the matched count is precomputed for every prefix length. Predictions that overlap no reference box of their
class at 0.5 cannot change the matching, so the count changes only where such a prediction is added. Predictions
far below the point-estimate cutoff are dropped from the ranking (kept: three times the point quota); every
evaluation asserts that the kept predictions still cover the quota, so the result is exact.
"""
import numpy as np
import pandas as pd

SEED, B = 2026, 2000
COLS = ["x1", "y1", "x2", "y2"]


def box_iou(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    return inter / ((a[:, 2:] - a[:, :2]).prod(1)[:, None] + (b[:, 2:] - b[:, :2]).prod(1)[None, :] - inter + 1e-7)


def n_matched(iou):
    m = np.array(np.nonzero(iou >= 0.5)).T
    if m.shape[0] == 0:
        return 0
    if m.shape[0] > 1:
        m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
        m = m[np.unique(m[:, 1], return_index=True)[1]]
        m = m[np.unique(m[:, 0], return_index=True)[1]]
    return int(m.shape[0])


class QuotaRun:
    """One detector on one pool.
    P: predictions (file, cls, conf, x1, y1, x2, y2); L: reference boxes (file, cls, x1, y1, x2, y2);
    files: pool images (defines N); classes: target classes; merge: optional dict mapping classes before matching."""

    def __init__(self, P, L, files, classes, merge=None, keep_factor=3.0, k_hint=None):
        self.files = list(files); fi = {f: i for i, f in enumerate(self.files)}
        P = P[P.file.isin(fi) & P.cls.isin(classes)].copy()
        L = L[L.file.isin(fi) & L.cls.isin(classes)].copy()
        if merge:
            P["mcls"] = P.cls.map(lambda c: merge.get(c, c)); L["mcls"] = L.cls.map(lambda c: merge.get(c, c))
        else:
            P["mcls"] = P.cls; L["mcls"] = L.cls
        P["img"] = P.file.map(fi)
        order = np.lexsort((P.y2.to_numpy(), P.x2.to_numpy(), P.y1.to_numpy(), P.x1.to_numpy(),
                            P.file.to_numpy(), -P.conf.to_numpy()))
        P = P.iloc[order].reset_index(drop=True)
        n = len(self.files)
        self.n_all = len(P)
        if k_hint is not None and len(P):
            qmax = int(np.ceil(keep_factor * k_hint * n / 100.0))
            P = P.iloc[:max(qmax, 1)].reset_index(drop=True)
        self.truncated = k_hint is not None
        self.P = P
        self.img = P.img.to_numpy()
        self.nref = np.bincount(L.file.map(fi).to_numpy(), minlength=n).astype(np.int64)
        Lg = dict(tuple(L.groupby("file")))
        self.off = np.zeros(n + 1, np.int64)
        lens = np.bincount(self.img, minlength=n) + 1
        self.off[1:] = np.cumsum(lens)
        M = np.zeros(self.off[-1], np.int64)
        for i, g in P.groupby("img", sort=False):
            f = self.files[i]; lab = Lg.get(f)
            if lab is None:
                continue
            iou = box_iou(lab[COLS].to_numpy(), g[COLS].to_numpy())
            iou = iou * (lab.mcls.to_numpy()[:, None] == g.mcls.to_numpy()[None, :])
            cand = np.nonzero((iou >= 0.5).any(0))[0]
            m = np.zeros(len(g) + 1, np.int64)
            last = 0
            for c in cand:
                m[last + 1:c + 1] = m[last]
                m[c + 1] = n_matched(iou[:, cand[cand <= c]])
                last = c + 1
            m[last + 1:] = m[last]
            M[self.off[i]:self.off[i] + len(m)] = m
        self.M = M
        self.npred = lens - 1

    def evaluate(self, w, quota):
        """w: copies per image (int array, length N). quota: number of alerts. Returns (matched, refs, used)."""
        pw = w[self.img]
        cw = np.cumsum(pw)
        total = int(cw[-1]) if len(cw) else 0
        if self.truncated and total <= quota and len(self.P) < self.n_all:
            raise RuntimeError("kept predictions do not cover the quota; rerun without truncation")
        full = cw <= quota
        jfull = np.bincount(self.img[full], minlength=len(w))
        matched = int((w * self.M[self.off[:-1] + jfull]).sum())
        if total > quota:
            b = int(np.argmax(cw > quota))
            partial = quota - (int(cw[b - 1]) if b > 0 else 0)
            if partial > 0:
                i = self.img[b]
                matched += partial * int(self.M[self.off[i] + jfull[i] + 1] - self.M[self.off[i] + jfull[i]])
        refs = int((w * self.nref).sum())
        return matched, refs, min(quota, total)


def quota_of(k, N):
    return int(np.floor(k * N / 100.0))


# ------------------------------------------------------------------ bootstrap units
def unit_weights(units, strata, rng, B=B):
    """units: unit id per image; strata: stratum per unit (dict). Returns B x n_images copy counts. Units are drawn
    with replacement within strata (strata in sorted order, units in sorted order)."""
    uid = sorted(set(units)); ui = {u: i for i, u in enumerate(uid)}
    img_u = np.array([ui[u] for u in units])
    W = np.zeros((B, len(uid)), np.int32)
    for s in sorted(set(strata[u] for u in uid)):
        ids = np.array([ui[u] for u in uid if strata[u] == s])
        d = rng.integers(0, len(ids), size=(B, len(ids)))
        for b in range(B):
            np.add.at(W[b], ids[d[b]], 1)
    return W[:, img_u]


def majority_strata(df, unit_col, part_col):
    c = df.groupby([unit_col, part_col]).size().reset_index(name="n")
    c = c.sort_values([unit_col, "n", part_col], ascending=[True, False, True]).drop_duplicates(unit_col)
    return dict(zip(c[unit_col], c[part_col]))


# ------------------------------------------------------------------ ratio
def cov_across(a, b):
    a = a - a.mean(-1, keepdims=True); b = b - b.mean(-1, keepdims=True)
    return (a * b).sum(-1) / (a.shape[-1] - 1)


def fieller(a, b, v11, v22, v12, z=1.96):
    A = b * b - z * z * v22
    Bq = a * b - z * z * v12
    Cq = a * a - z * z * v11
    if A <= 0:
        return None
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
        return "negative slope"
    if L <= 0 <= U and U < 1:
        return "no detectable retention"
    if 0 < L and U < 1:
        return "partial retention"
    if 0 < L <= 1 <= U:
        return "compatible with full retention"
    if L > 1:
        return "amplification"
    return "unclassified"


def ratio_result(x1, x2, z, x1b, x2b, zb):
    """x1, x2, z: point values per architecture (n,); *b: bootstrap draws (B, n)."""
    num, den = cov_across((x1 + x2) / 2, z), cov_across(x1, x2)
    nb, db = cov_across((x1b + x2b) / 2, zb), cov_across(x1b, x2b)
    V = np.cov(np.vstack([nb, db]))
    gate_lb = float(np.percentile(db, 5))
    fs = fieller(num, den, V[0, 0], V[1, 1], V[0, 1])
    return dict(R=float(num / den) if den != 0 else None, numerator=float(num), denominator=float(den),
                denominator_lower_95=gate_lb, repeatability_gate=bool(gate_lb > 0),
                fieller=None if fs is None else [float(fs[0]), float(fs[1])],
                fieller_shape="bounded" if fs is not None else "unbounded or disjoint",
                var_num=float(V[0, 0]), var_den=float(V[1, 1]), cov=float(V[0, 1]),
                outcome=category(gate_lb > 0, fs))

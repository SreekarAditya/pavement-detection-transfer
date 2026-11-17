"""Shared helpers for the analyses after registration.

Reuses the locked definitions in scripts/pdt_core.py (quota with deterministic tie-break, one-to-one matching at
intersection over union 0.5, cluster and block bootstrap, Fieller set). Nothing here changes a locked file.

Conventions
  root      unpacked lock archive plus private inputs (verified with pdt_score.verify before use)
  params    post_reg/rules/params.yaml (read through load_params; every number an analysis needs comes from it)
  pools     internal: g3/main folds; external: s2/g4/pool_final_<c>.txt with blocks from s2/g4/blocks.csv
  split     per external collection, calibration and held-out lists in post_reg/rules/split/<c>_{calibration,heldout}.txt
"""
import hashlib, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ARCH_A = ["yolov5s", "yolov8s", "yolov9s", "yolov10s", "yolo11s", "yolo12n", "yolo12s", "yolo12m"]
SEEDS_A = {a: ([0, 42, 123] if a not in ("yolo12n", "yolo12m") else [0]) for a in ARCH_A}
RUNS_A = [f"{a}_seed{s}" for a in ARCH_A for s in SEEDS_A[a]]
GROUPS_A = {a: [f"{a}_seed{s}" for s in SEEDS_A[a]] for a in ARCH_A}
ARCH_B = ["yolov8n", "yolov8s", "yolo26s", "yolo26m", "rfdetr_nano", "rfdetr_small"]
SELECTED = "yolo12m_seed0"
FOUR, CRACK, POTHOLE = [0, 1, 2, 3], [0, 1, 2], [3]
CLASS_SET_CLASSES = {"four_class": FOUR, "merged_crack": CRACK, "pothole": POTHOLE}
CLASS_SET_MERGE = {"four_class": None, "merged_crack": {0: 0, 1: 0, 2: 0}, "pothole": None}
EXT_ALL = ["ecupdd", "hrp4k", "ird", "kent", "pdd2026", "rome", "svrdd"]
EXT_FOUR = ["ecupdd", "ird", "svrdd"]
COUNTRIES = ["China_MotorBike", "China_Drone", "United_States", "Czech", "India", "Japan", "Norway"]
PRED_COLS = ["file", "cls", "conf", "x1", "y1", "x2", "y2"]
BOX = ["x1", "y1", "x2", "y2"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def locked_scripts(root):
    p = str(Path(root) / "scripts")
    if p not in sys.path:
        sys.path.insert(0, p)
    import pdt_core  # noqa: F401
    return sys.modules["pdt_core"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ gates
def rule_gate(rules_dir, want_rules_sha, want_params_sha):
    """Stop unless RULES.md and params.yaml exist and match the hashes written into the task header."""
    r, p = Path(rules_dir) / "RULES.md", Path(rules_dir) / "params.yaml"
    for f, w in ((r, want_rules_sha), (p, want_params_sha)):
        if not f.exists():
            sys.exit(f"STOP: {f} missing")
        got = sha(f)
        if got != w:
            sys.exit(f"STOP: {f.name} SHA-256 {got} differs from the value in the task header ({w})")
    log("rule gate passed")


def load_params(rules_dir):
    import yaml
    P = yaml.safe_load(open(Path(rules_dir) / "params.yaml"))
    bad = [k for k, v in _flatten(P).items() if v is None or v == "?"]
    if bad:
        sys.exit(f"STOP: params.yaml has undefined values: {bad[:10]}")
    return P


def _flatten(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


def verify_lock(root, manifest_sha):
    locked_scripts(root)
    import pdt_score  # noqa: E402
    pdt_score.verify(root, manifest_sha)


# ------------------------------------------------------------------ inputs
class Root:
    def __init__(self, root):
        self.r = Path(root)
        self._cache = {}

    # internal
    def labels_internal(self):
        if "LA" not in self._cache:
            self._cache["LA"] = pd.read_parquet(self.r / "e1" / "labels.parquet")
        return self._cache["LA"]

    def fold(self, name, kind="main"):
        return sorted((self.r / "g3" / kind / f"fold_{name}.txt").read_text().split())

    def clusters(self, kind="main"):
        return pd.read_csv(self.r / "g3" / kind / "clusters.csv")

    def pred_internal(self, run):
        return pd.read_parquet(self.r / "e1" / "sealed" / "predictions" / f"{run}.parquet", columns=PRED_COLS)

    # external
    def pool(self, c):
        return sorted((self.r / "s2" / "g4" / f"pool_final_{c}.txt").read_text().split())

    def blocks(self, c):
        b = pd.read_csv(self.r / "s2" / "g4" / "blocks.csv")
        b = b[b.collection == c]
        return dict(zip(b.file, b.block))

    def labels_external(self, c):
        L = pd.read_parquet(self.r / "s2" / "g4" / f"labels_final_{c}.parquet")
        return L[L.cls >= 0][["file", "cls"] + BOX].reset_index(drop=True)

    def labels_external_subset(self, c, files):
        """Labels of the given images only, filtered while reading (the other rows are never loaded)."""
        L = pd.read_parquet(self.r / "s2" / "g4" / f"labels_final_{c}.parquet",
                            filters=[("file", "in", sorted(set(files)))])
        return L[L.cls >= 0][["file", "cls"] + BOX].reset_index(drop=True)

    def pred_external(self, run, c):
        return pd.read_parquet(self.r / "s2" / "s2a" / "sealed_ext" / c / f"{run}.parquet", columns=PRED_COLS)

    def pred_b(self, model, pool):
        return pd.read_parquet(self.r / "s2" / "s2b" / "sealed_b" / pool / f"{model}.parquet", columns=PRED_COLS)

    def ext_props(self, c):
        return pd.read_parquet(self.r / "s2" / "ext_props" / f"{c}.parquet")

    def image_props(self):
        return pd.read_parquet(self.r / "e2" / "image_props.parquet")


def forbid_labels():
    """Make every label reader of Root raise in this process (for label-free steps)."""
    def no(*a, **k):
        raise RuntimeError("a label table was requested in a label-free step")
    for name in ("labels_internal", "labels_external", "labels_external_subset"):
        setattr(Root, name, no)


def split_checks(rules_dir, R, colls):
    """Calibration and held-out parts: disjoint, together the whole pool, and whole blocks on each side."""
    out = {}
    for c in colls:
        cal, held = read_split(rules_dir, c)
        B = R.blocks(c)
        out[c] = dict(disjoint=not (set(cal) & set(held)), cover_pool=sorted(set(cal) | set(held)) == R.pool(c),
                      whole_blocks=not ({B[f] for f in cal} & {B[f] for f in held}))
    return out


def read_split(rules_dir, c):
    d = Path(rules_dir) / "split"
    return (sorted((d / f"{c}_calibration.txt").read_text().split()),
            sorted((d / f"{c}_heldout.txt").read_text().split()))


def make_split(files, blocks, share, seed, balance="images"):
    """Calibration and held-out parts made of whole blocks. Blocks are put in one random order (seed); with
    balance="images" they go to the calibration part in that order until it holds at least share of the images
    (so a very large block cannot unbalance the image counts); balance="blocks" takes share of the blocks.
    Returns (calibration files, held-out files)."""
    ub = sorted({blocks[f] for f in files})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ub))
    if balance == "blocks":
        cal_b = {ub[i] for i in perm[:int(round(share * len(ub)))]}
    elif balance == "images":
        size = {b: 0 for b in ub}
        for f in files:
            size[blocks[f]] += 1
        target, got, cal_b = share * len(files), 0, set()
        for i in perm:
            if got >= target:
                break
            cal_b.add(ub[i]); got += size[ub[i]]
    else:
        raise ValueError(balance)
    cal = sorted(f for f in files if blocks[f] in cal_b)
    held = sorted(f for f in files if blocks[f] not in cal_b)
    return cal, held


def recall_floor(floor, cs):
    """The recall floor of a class set (a number, or a mapping class set -> number)."""
    return float(floor[cs]) if isinstance(floor, dict) else float(floor)


# ------------------------------------------------------------------ matching
def greedy_match(lab, pred, lab_cls, pred_cls, thr=0.5):
    """Ultralytics' one-to-one rule (as pdt_core.n_matched): pairs at IoU >= thr and same class, taken in
    decreasing IoU, each label and prediction used once. Returns (label_matched bool, pred_matched bool,
    pred_iou_best_same_class)."""
    C = sys.modules.get("pdt_core")
    nl, npd = len(lab), len(pred)
    lm, pm = np.zeros(nl, bool), np.zeros(npd, bool)
    if nl == 0 or npd == 0:
        return lm, pm, np.zeros(npd)
    iou = C.box_iou(np.asarray(lab, float), np.asarray(pred, float))
    same = np.asarray(lab_cls)[:, None] == np.asarray(pred_cls)[None, :]
    iou_s = iou * same
    m = np.array(np.nonzero(iou_s >= thr)).T
    if m.shape[0]:
        if m.shape[0] > 1:
            m = m[iou_s[m[:, 0], m[:, 1]].argsort()[::-1]]
            m = m[np.unique(m[:, 1], return_index=True)[1]]
            m = m[np.unique(m[:, 0], return_index=True)[1]]
        lm[m[:, 0]] = True
        pm[m[:, 1]] = True
    return lm, pm, iou_s.max(0)


class ThresholdRun:
    """One detector on one set of images and class set, for thresholds and quotas without bootstrap.
    Wraps pdt_core.QuotaRun (no truncation): predictions ordered by confidence with the locked tie-break; for any
    confidence threshold t the alerts of each image are a prefix of that image's order, so matched counts per
    image come from the locked prefix table M."""

    def __init__(self, P, L, files, classes, merge=None):
        C = sys.modules["pdt_core"]
        files = list(files)
        if len(files) and not (L.file.isin(set(files)) & L.cls.isin(classes)).any():
            # no reference box on these images: the locked constructor cannot count an empty label table, so one
            # box outside every image is passed and the counts are then set to zero (nothing can match it)
            dummy = pd.DataFrame([dict(file=files[0], cls=list(classes)[0], x1=-10.0, y1=-10.0, x2=-9.0, y2=-9.0)])
            self.R = C.QuotaRun(P, dummy, files, classes, merge=merge, k_hint=None)
            self.R.nref[:] = 0; self.R.M[:] = 0
        else:
            self.R = C.QuotaRun(P, L, files, classes, merge=merge, k_hint=None)
        self.files = self.R.files
        self.conf = self.R.P.conf.to_numpy()
        self.n = len(self.files)

    def per_image_at(self, t):
        """(alerts, matched, refs) per image at confidence >= t."""
        sel = self.conf >= t
        a = np.bincount(self.R.img[sel], minlength=self.n)
        m = self.R.M[self.R.off[:-1] + a]
        return a, m, self.R.nref

    def per_image_quota(self, q):
        """(alerts, matched, refs) per image when the top q alerts over these images are taken (locked order)."""
        q = min(q, len(self.conf))
        a = np.bincount(self.R.img[:q], minlength=self.n)
        m = self.R.M[self.R.off[:-1] + a]
        return a, m, self.R.nref


def threshold_for_rate(conf_sorted_desc, n_images, k):
    """Confidence threshold whose alert count on these images is floor(k n / 100): the confidence of that
    alert in the locked order. Returns +inf when the quota is 0 and the lowest confidence when it underfills."""
    q = int(np.floor(k * n_images / 100.0))
    if q <= 0:
        return np.inf
    if q > len(conf_sorted_desc):
        return float(conf_sorted_desc[-1]) if len(conf_sorted_desc) else np.inf
    return float(conf_sorted_desc[q - 1])


# ------------------------------------------------------------------ intervals
def ratio_ci(y, x, z):
    """Ratio estimate sum(y)/sum(x) over cluster units with the linearization variance; returns (r, lo, hi, se)."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    n = len(x)
    X = x.sum()
    if X <= 0:
        return np.nan, np.nan, np.nan, np.nan
    r = y.sum() / X
    if n < 2:
        return r, np.nan, np.nan, np.nan
    e = y - r * x
    se = np.sqrt(n / (n - 1) * (e * e).sum()) / X
    return r, r - z * se, r + z * se, se


def ppi_ratio(a_l, x_l, ah_l, xh_l, ah_u, xh_u, z, tune=True):
    """Prediction-powered ratio estimate (PPI++ with power tuning), cluster units.
    Labeled units: true matched a, true references x, and their pseudo counterparts ah, xh.
    Unlabeled units (all units of the population, labeled ones included): ah_u, xh_u.
    theta = mu_a / mu_x with mu_a(lam) = lam * mean_u(ah) + mean_l(a - lam ah) (same for x); delta-method variance;
    lam minimizes the variance and is clipped to [0, 1]. lam = 0 gives the labeled-sample ratio estimate.
    Returns (theta, lo, hi, se, lam)."""
    a_l, x_l, ah_l, xh_l = (np.asarray(v, float) for v in (a_l, x_l, ah_l, xh_l))
    ah_u, xh_u = np.asarray(ah_u, float), np.asarray(xh_u, float)
    n, N = len(a_l), len(ah_u)
    if n < 2 or x_l.sum() <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    Y = np.vstack([a_l, x_l])           # 2 x n
    Yh = np.vstack([ah_l, xh_l])        # 2 x n
    Yu = np.vstack([ah_u, xh_u])        # 2 x N
    th0 = a_l.mean() / x_l.mean()
    g = np.array([1.0 / x_l.mean(), -th0 / x_l.mean()])
    Shh = np.cov(Yh); Suu = np.cov(Yu)
    Syh = np.cov(np.vstack([Y, Yh]))[:2, 2:]
    if tune:
        den = (g @ Shh @ g) / n + (g @ Suu @ g) / N
        num = (g @ ((Syh + Syh.T) / 2) @ g) / n
        lam = float(np.clip(num / den, 0.0, 1.0)) if den > 0 else 0.0
    else:
        lam = 1.0
    mu = lam * Yu.mean(1) + (Y - lam * Yh).mean(1)
    if mu[1] <= 0:
        return np.nan, np.nan, np.nan, np.nan, lam
    th = mu[0] / mu[1]
    g = np.array([1.0 / mu[1], -th / mu[1]])
    Vl = np.cov(Y - lam * Yh) / n
    Vu = (lam ** 2) * Suu / N
    se = float(np.sqrt(max(g @ (Vl + Vu) @ g, 0.0)))
    return th, th - z * se, th + z * se, se, lam


def pct(x, lo=2.5, hi=97.5):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return [None, None]
    return [float(np.percentile(x, lo)), float(np.percentile(x, hi))]


# ------------------------------------------------------------------ outputs
def write_manifest(out):
    out = Path(out)
    lines = [f"{sha(p)}  {p.relative_to(out).as_posix()}" for p in sorted(out.rglob("*"))
             if p.is_file() and p.name != "MANIFEST.sha256"]
    (out / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")


def env_record(out):
    import platform
    rec = dict(python=platform.python_version(), host=platform.node(), cpus=os.cpu_count())
    for m in ("numpy", "pandas", "pyarrow", "scipy", "statsmodels", "ultralytics", "torch"):
        try:
            rec[m] = __import__(m).__version__
        except Exception:
            rec[m] = None
    json.dump(rec, open(Path(out) / "env.json", "w"), indent=1)
    return rec


def jdump(obj, path):
    json.dump(obj, open(path, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def pmap(fn, items, workers=None):
    """Process pool map with a fixed order of results; workers default to all CPUs."""
    from concurrent.futures import ProcessPoolExecutor
    items = list(items)
    w = workers or int(os.environ.get("PR_WORKERS", os.cpu_count() or 1))
    if w <= 1 or len(items) <= 1:
        return [fn(i) for i in items]
    with ProcessPoolExecutor(max_workers=w) as ex:
        return list(ex.map(fn, items))


# ------------------------------------------------------------------ panel of detectors at a quota (bootstrap)
class Pool:
    def __init__(self, name, files, units, strata, L):
        self.name, self.files, self.units, self.strata, self.L = name, list(files), list(units), strata, L


def internal_pool(R, name, files, kind="main"):
    C = sys.modules["pdt_core"]
    cl = R.clusters(kind).set_index("file")
    files = sorted(files)
    c = cl.loc[files].reset_index()
    strata = C.majority_strata(c, "cluster", "country")
    LA = R.labels_internal()
    return Pool(name, files, c.cluster.tolist(), strata, LA[LA.file.isin(set(files))][["file", "cls"] + BOX])


def image_pool(R, name, files, country_of):
    """Pool whose bootstrap units are single images within partition strata (for images without clusters)."""
    files = sorted(files)
    LA = R.labels_internal()
    return Pool(name, files, files, {f: country_of(f) for f in files}, LA[LA.file.isin(set(files))][["file", "cls"] + BOX])


def external_pool(R, c, files=None):
    files = sorted(files) if files is not None else R.pool(c)
    B = R.blocks(c)
    b = [B[f] for f in files]
    L = R.labels_external(c)
    return Pool(c, files, b, {u: c for u in set(b)}, L[L.file.isin(set(files))])


def panel(runs, pred, pools, classes, k, B, seed=2026, merge=None):
    """Recall at the quota per run and pool: point values and B bootstrap draws (units resampled within strata,
    the quota reapplied in every draw, as pdt_score.Panel). pred(run, pool_name) -> predictions.
    Returns (point {(run, pool): recall}, boot {(run, pool): array B}, detail rows)."""
    C = sys.modules["pdt_core"]
    rng = np.random.default_rng(seed)
    W = {p: C.unit_weights(pools[p].units, pools[p].strata, rng, B=B) for p in pools}
    point, boot, det = {}, {}, []
    for p, pool in pools.items():
        n = len(pool.files)
        for r in runs:
            Rq = C.QuotaRun(pred(r, p), pool.L, pool.files, classes, merge=merge, k_hint=k)
            one = np.ones(n, np.int64)
            m, rf, used = Rq.evaluate(one, C.quota_of(k, n))
            point[(r, p)] = m / rf if rf else np.nan
            det.append(dict(run=r, pool=p, images=n, quota=C.quota_of(k, n), used=used, matched=m, refs=rf,
                            recall=point[(r, p)]))
            bb = np.empty(B)
            for i in range(B):
                w = W[p][i]
                mm, rr, _ = Rq.evaluate(w, C.quota_of(k, int(w.sum())))
                bb[i] = mm / rr if rr else np.nan
            boot[(r, p)] = bb
    return point, boot, det


def arch_values(src, groups, p):
    return np.stack([np.mean([src[(r, p)] for r in groups[a]], axis=0) for a in groups], axis=-1)


def ratio_from_panel(point, boot, groups, s1, s2, ext):
    C = sys.modules["pdt_core"]
    x1, x2 = arch_values(point, groups, s1), arch_values(point, groups, s2)
    x1b, x2b = arch_values(boot, groups, s1), arch_values(boot, groups, s2)
    z = np.mean([arch_values(point, groups, e) for e in ext], 0)
    zb = np.mean([arch_values(boot, groups, e) for e in ext], 0)
    res = C.ratio_result(x1, x2, z, x1b, x2b, zb)
    res["per_arch"] = {a: dict(x1=float(x1[i]), x2=float(x2[i]), z=float(z[i])) for i, a in enumerate(groups)}
    return res


def boot_long(bt, **cols):
    """Bootstrap draws of a panel ({(run, pool): array}) as a long table: run, pool, draw, recall (+ given columns)."""
    parts = [pd.DataFrame(dict(run=r, pool=p, draw=np.arange(len(v)), recall=np.asarray(v, float)))
             for (r, p), v in bt.items()]
    D = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["run", "pool", "draw", "recall"])
    for k_, v in cols.items():
        D[k_] = v
    return D


def flat_result(res, **cols):
    """One row of scalars from a ratio result (lists and dicts as text)."""
    return dict(cols, **{k_: (v if np.isscalar(v) or v is None else json.dumps(v, default=str)) for k_, v in res.items()})


def top_quota(P, files, classes, k):
    """The alerts at the quota: top floor(k N / 100) predictions of the classes over these files, locked order."""
    C = sys.modules["pdt_core"]
    p = P[P.file.isin(set(files)) & P.cls.isin(classes)]
    order = np.lexsort((p.y2.to_numpy(), p.x2.to_numpy(), p.y1.to_numpy(), p.x1.to_numpy(), p.file.to_numpy(),
                        -p.conf.to_numpy()))
    return p.iloc[order[:C.quota_of(k, len(files))]].reset_index(drop=True)


def match_flags(P, L, merge=None):
    """Greedy one-to-one matching per image. Returns (label_matched aligned with L.index order, pred_matched
    aligned with P, pred best same-class IoU). P and L restricted beforehand to the files and classes wanted."""
    L = L.reset_index(drop=True); P = P.reset_index(drop=True)
    lc = L.cls.map(lambda c: merge.get(c, c)).to_numpy() if merge else L.cls.to_numpy()
    pc = P.cls.map(lambda c: merge.get(c, c)).to_numpy() if merge else P.cls.to_numpy()
    lm = np.zeros(len(L), bool); pm = np.zeros(len(P), bool); piou = np.zeros(len(P))
    lg = L.groupby("file").indices; pg = P.groupby("file").indices
    lb, pb = L[BOX].to_numpy(float), P[BOX].to_numpy(float)
    for f, pi in pg.items():
        li = lg.get(f)
        if li is None:
            continue
        a, b, c = greedy_match(lb[li], pb[pi], lc[li], pc[pi])
        lm[li] = a; pm[pi] = b; piou[pi] = c
    return lm, pm, piou

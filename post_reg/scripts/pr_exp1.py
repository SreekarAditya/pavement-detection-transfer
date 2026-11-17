#!/usr/bin/env python3
"""Experiment 1 (primary): the whole procedure against floor and ceiling (task 08), and its constants.

  constants  n and M per collection and class set = the maximum, over the OTHER collections of the class set, of
             the smallest grid value that met the criterion there (selected detector, block samples); if some
             other collection met it at no grid value, the largest grid value is used and flagged. "unseen" holds
             the maximum over all collections (for experiment 18). The recall floor comes from params.
  chain      per collection, class set and draw: threshold from n unlabeled calibration images; stated recall from
             a pilot of M calibration boxes at that threshold (ratio estimate); accept or adapt against the floor;
             for "adapt", the adapted model of task 07 (size params.exp1.adapt_size, draw = draw mod 3) with its
             recalibrated threshold and audit estimate. Floors: development-fold threshold and confidence 0.25,
             each with benchmark recall (S1 and S2) as the stated recall. Ceiling: threshold giving k per 100 over
             the whole collection, true recall on all labels. Measures on the held-out part.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr_common as PC  # noqa: E402
from pr_exp2 import block_samples  # noqa: E402


def constants(a, params):
    S2 = pd.read_csv(Path(a.e2) / "e2_smallest_n.csv"); S3 = pd.read_csv(Path(a.e3) / "e3_smallest_M.csv")
    sel = params["selected"]
    S2 = S2[(S2.setup == "A") & (S2.checkpoint == sel) & (S2["mode"] == "block")]
    S3 = S3[(S3.setup == "A") & (S3.checkpoint == sel)]
    ng, mg = params["exp2"]["n_grid"], params["exp3"]["m_grid"]
    out = {"recall_floor": params["exp1"]["recall_floor"], "rule": "max over other collections", "values": {}}
    for cs, spec in params["class_sets"].items():
        colls = PC.EXT_ALL if spec["networks"] == "all_seven" else spec["networks"]
        sn = dict(zip(S2[S2.class_set == cs].collection, S2[S2.class_set == cs].smallest_n))
        sm = dict(zip(S3[S3.class_set == cs].collection, S3[S3.class_set == cs].smallest_M))
        for target in colls + ["unseen"]:
            others = [c for c in colls if c != target]
            vn = [sn.get(c) for c in others]; vm = [sm.get(c) for c in others]
            fn = any(v is None or pd.isna(v) for v in vn); fm = any(v is None or pd.isna(v) for v in vm)
            n = max(ng) if fn else int(max(vn)); M = max(mg) if fm else int(max(vm))
            ncal = len(PC.read_split(a.rules, target)[0]) if target != "unseen" else None
            out["values"].setdefault(cs, {})[target] = dict(
                n=n, M=M, n_flag_not_met_elsewhere=fn, M_flag_not_met_elsewhere=fm, from_collections=others,
                calibration_images=ncal,
                # Section E1: a constant larger than the calibration part uses the whole part and is flagged
                # (M against calibration boxes is flagged per draw as pilot_shortfall: counting boxes needs labels)
                n_flag_exceeds_calibration=bool(ncal is not None and n > ncal))
    import yaml
    yaml.safe_dump(out, open(Path(a.out) / "exp1_constants.yaml", "w"), sort_keys=False)
    PC.log("constants written (to be approved and tagged before experiment 1 runs)")


class Net:
    """Everything one chain needs for one detector on one collection and class set."""

    def __init__(self, R, coll, cs, det, rules, params, procedure_only=False):
        self.coll, self.cs, self.det = coll, cs, det
        self.classes, self.merge = PC.CLASS_SET_CLASSES[cs], PC.CLASS_SET_MERGE[cs]
        self.k = params["class_sets"][cs]["k"]
        self.cal, self.held = PC.read_split(rules, coll)
        self.B = R.blocks(coll)
        P = R.pred_external(det, coll); L = R.labels_external(coll)
        self.Tc = PC.ThresholdRun(P, L, self.cal, self.classes, self.merge)
        self.Th = self.Ta = None  # held-out and whole-collection runs need the held-out labels
        if not procedure_only:
            self.Th = PC.ThresholdRun(P, L, self.held, self.classes, self.merge)
            self.Ta = PC.ThresholdRun(P, L, R.pool(coll), self.classes, self.merge)
        self.cal_conf = [np.sort(c)[::-1] for c in self._split_conf(self.Tc)]
        self.cal_block = np.array([self.B[f] for f in self.cal])

    @staticmethod
    def _split_conf(T):
        out = [[] for _ in range(T.n)]
        for i, c in zip(T.R.img, T.conf):
            out[i].append(c)
        return [np.array(o) for o in out]

    def heldout(self, t):
        a, m, x = self.Th.per_image_at(t)
        return 100.0 * a.sum() / self.Th.n, (m.sum() / x.sum() if x.sum() else np.nan)


def pool_cap(T, t, k):
    sel = T.conf >= t
    q = int(np.floor(k * T.n / 100.0))
    idx = np.nonzero(sel)[0][:q]
    a = np.bincount(T.R.img[idx], minlength=T.n)
    m = T.R.M[T.R.off[:-1] + a]
    return m.sum() / T.R.nref.sum()


def batch_cap(T, t, k, Bsz, blocks_of_img, rng):
    """Images in random block order, batches of Bsz images; in each batch the alerts >= t in the locked order up
    to floor(k Bsz / 100); unused capacity is not carried over."""
    ub = np.unique(blocks_of_img)
    order_b = rng.permutation(ub)
    pos = {b: i for i, b in enumerate(order_b)}
    img_order = np.argsort([pos[b] for b in blocks_of_img], kind="stable")
    rank = np.empty(T.n, int); rank[img_order] = np.arange(T.n)
    batch = rank // Bsz
    size = np.bincount(batch)
    sel = np.nonzero(T.conf >= t)[0]
    b_of = batch[T.R.img[sel]]
    a = np.zeros(T.n, int)
    for b in np.unique(b_of):
        take = sel[b_of == b][:int(np.floor(k * size[b] / 100.0))]
        a += np.bincount(T.R.img[take], minlength=T.n)
    m = T.R.M[T.R.off[:-1] + a]
    return m.sum() / T.R.nref.sum()


def benchmark_floors_cs(R, det, cs, k, conf025):
    classes, merge = PC.CLASS_SET_CLASSES[cs], PC.CLASS_SET_MERGE[cs]
    P = R.pred_internal(det); LA = R.labels_internal()[["file", "cls"] + PC.BOX]
    dev = R.fold("dev")
    pc = P[P.file.isin(set(dev)) & P.cls.isin(classes)].conf.sort_values(ascending=False).to_numpy()
    t_dev = PC.threshold_for_rate(pc, len(dev), k)
    s12 = sorted(set(R.fold("S1")) | set(R.fold("S2")))
    T = PC.ThresholdRun(P, LA, s12, classes, merge)
    out = {}
    for name, t in (("dev_threshold", t_dev), ("default_conf_0_25", conf025)):
        a, m, x = T.per_image_at(t)
        out[name] = (t, m.sum() / x.sum())
    return out


class CalibrationOnly:
    """A root whose external labels are those of the calibration part only (for the held-out-label check)."""

    def __init__(self, R, rules):
        self.R, self.rules = R, rules

    def labels_external(self, c):
        cal, _ = PC.read_split(self.rules, c)
        L = self.R.labels_external(c)
        return L[L.file.isin(set(cal))].reset_index(drop=True)

    def __getattr__(self, name):
        return getattr(self.R, name)


CHECK_COLS = ["draw", "threshold", "stated_recall", "stated_lo", "stated_hi", "decision", "pilot_boxes"]


def job(args):
    root, rules, params, cons, coll, cs, det, adapted = args
    PC.locked_scripts(root); R = PC.Root(root)
    v = cons["values"][cs][coll]
    cargs = (coll, cs, det, rules, params, v["n"], v["M"], PC.recall_floor(cons["recall_floor"], cs), PC.EXT_ALL.index(coll),
             params["sample_draws"])
    Dd = run_chain(R, *cargs, adapted)
    boot = Dd.attrs.get("boot")
    if det == params["selected"]:
        # the procedure arm recomputed without any held-out label must give the same thresholds, estimates and
        # decisions (task 08, step 7)
        C0 = run_chain(CalibrationOnly(R, rules), *cargs, procedure_only=True)
        Pfull = Dd[Dd.arm == "procedure"][CHECK_COLS].reset_index(drop=True)
        Pcal = C0[CHECK_COLS].reset_index(drop=True)
        pd.testing.assert_frame_equal(Pcal, Pfull, check_dtype=False)
        Dd["check_procedure_without_heldout_labels"] = np.where(Dd.arm == "procedure", "same", "")
    Dd.attrs = {}
    return Dd, boot


def run_chain(R, coll, cs, det, rules, params, n, M, floor, cidx, D, adapted=None, procedure_only=False, floors=True):
    """The chain for one detector, network and class set; R supplies predictions, labels, pool and blocks, and
    rules/split holds the calibration and held-out lists. cidx keys the random streams; D is the number of draws.
    procedure_only: threshold, pilot estimate and decision only, from calibration labels (no floors, ceiling or
    held-out measures); the decisions are identical to those of the full run. floors=False: no benchmark floors
    (checkpoints without setup A's internal predictions)."""
    N = Net(R, coll, cs, det, rules, params, procedure_only)
    zd = norm.ppf(0.5 + params["exp3"]["decision_level"] / 2)
    rows = []
    csi = list(PC.CLASS_SET_CLASSES).index(cs)
    Bs = [params["exp1"]["batch_B"]] + params["exp1"]["batch_B_sensitivity"]
    def rng_cap(key):  # random block order of the batch-level cap: one stream per arm and draw (Section E1)
        return np.random.default_rng(np.random.SeedSequence([params["seed"], 11, cidx, csi, key]))
    hb = np.array([N.B[f] for f in N.held])
    fl = {} if (procedure_only or not floors) else benchmark_floors_cs(R, det, cs, N.k, params["exp1"]["default_conf"])
    for ai, (arm, (t, stated)) in enumerate(fl.items()):
        rate, tr = N.heldout(t)
        rows.append(dict(arm=arm, draw=-1, threshold=t, stated_recall=stated, true_recall=tr, heldout_rate=rate,
                         rate_rel_error=rate / N.k - 1, recall_error=stated - tr,
                         pool_cap_recall=pool_cap(N.Th, t, N.k),
                         **{f"batch_cap_recall_B{b}": batch_cap(N.Th, t, N.k, b, hb, rng_cap(10 ** 6 + ai))
                            for b in Bs}))
    if not procedure_only:
        # ceiling: threshold giving k per 100 over the whole collection; true recall on all labels
        t_ceil = PC.threshold_for_rate(np.sort(N.Ta.conf)[::-1], N.Ta.n, N.k)
        a_, m_, x_ = N.Ta.per_image_at(t_ceil)
        ceil_true = m_.sum() / x_.sum()
        rate, tr = N.heldout(t_ceil)
        rows.append(dict(arm="ceiling", draw=-1, threshold=t_ceil, stated_recall=ceil_true, true_recall=tr, heldout_rate=rate,
                         rate_rel_error=rate / N.k - 1, recall_error=ceil_true - tr,
                         pool_cap_recall=pool_cap(N.Th, t_ceil, N.k),
                         **{f"batch_cap_recall_B{b}": batch_cap(N.Th, t_ceil, N.k, b, hb, rng_cap(10 ** 6 + 99)) for b in Bs}))
    # procedure draws: threshold samples (experiment 2 stream) and pilot samples (independent stream)
    tsamp = block_samples(N.cal, N.B, n, D, params["seed"], cidx)
    a_cal_cache = {}
    ub = np.array(sorted(set(N.cal_block))); bi = {b: i for i, b in enumerate(ub)}
    img_b = np.array([bi[b] for b in N.cal_block])
    rng = np.random.default_rng(np.random.SeedSequence([params["seed"], 1, cidx, csi, M]))
    for d, idx in enumerate(tsamp):
        arr = np.concatenate([N.cal_conf[i] for i in idx]) if len(idx) else np.zeros(0)
        t = PC.threshold_for_rate(np.sort(arr)[::-1], len(idx), N.k)
        if t not in a_cal_cache:
            a_cal_cache[t] = N.Tc.per_image_at(t)
        _, m, x = a_cal_cache[t]
        mb = np.bincount(img_b, weights=m, minlength=len(ub)); xb = np.bincount(img_b, weights=x, minlength=len(ub))
        perm = rng.permutation(len(ub))
        kk = min(int(np.searchsorted(np.cumsum(xb[perm]), M)) + 1, len(ub))
        s = perm[:kk]
        est, lo, hi, _ = PC.ratio_ci(mb[s], xb[s], zd)
        accept = bool(lo >= floor) if np.isfinite(lo) else False
        row = dict(arm="procedure", draw=d, threshold=t, stated_recall=est, stated_lo=lo, stated_hi=hi,
                   decision="accept" if accept else "adapt",
                   pilot_boxes=int(xb[s].sum()), pilot_blocks=int(kk), pilot_shortfall=bool(xb[s].sum() < M),
                   threshold_images=int(len(idx)), threshold_shortfall=bool(len(idx) < n))
        if not procedure_only:
            rate, tr = N.heldout(t)
            row.update(true_recall=tr, heldout_rate=rate, rate_rel_error=rate / N.k - 1, recall_error=est - tr,
                       pool_cap_recall=pool_cap(N.Th, t, N.k),
                       **{f"batch_cap_recall_B{b}": batch_cap(N.Th, t, N.k, b, hb, rng_cap(d)) for b in Bs})
        if not accept and adapted is not None and cs == "four_class":
            ar = adapted.get(d % 3)
            if ar is not None:
                ast = ar.get("audit_estimate")
                row.update(adapted_accept=bool(ast is not None and np.isfinite(ast) and ast >= floor),
                           adapted_run=ar["run_id"], adapted_recall_at_quota=ar["recall_adapted"],
                           adapted_threshold=ar["recalibrated_threshold"],
                           adapted_recall_at_threshold=ar["recall_at_recalibrated"],
                           adapted_stated_recall=ar.get("audit_estimate"),
                           adapted_rate=ar["alerts_per_100_at_recalibrated"])
        rows.append(row)
    Dd = pd.DataFrame(rows)
    Dd.insert(0, "class_set", cs); Dd.insert(0, "collection", coll); Dd.insert(0, "detector", det)
    Dd["n"] = n; Dd["M"] = M; Dd["recall_floor"] = floor
    if not procedure_only:
        Dd.attrs["boot"] = boot_shares(N, Dd[Dd.arm == "procedure"], params, cidx, csi).assign(
            detector=det, collection=coll, class_set=cs)
    return Dd


def boot_shares(N, P, params, cidx, csi):
    """Uncertainty from the finite held-out part: held-out blocks resampled with replacement (bootstrap draws and
    seed of params), true recall and alert volume recomputed at every draw's threshold, and the two validation
    shares recomputed. One row per bootstrap draw."""
    crit, marg = params["exp2"]["criterion_rel"], params["exp3"]["margin"]
    hb = np.array([N.B[f] for f in N.held]); ub, bi = np.unique(hb, return_inverse=True)
    nimg = np.bincount(bi, minlength=len(ub)).astype(float)
    ts = np.unique(P.threshold.to_numpy())
    A = np.zeros((len(ts), len(ub))); Mm = np.zeros_like(A); X = np.zeros_like(A)
    for j, t in enumerate(ts):
        a, m, x = N.Th.per_image_at(t)
        A[j] = np.bincount(bi, a, len(ub)); Mm[j] = np.bincount(bi, m, len(ub)); X[j] = np.bincount(bi, x, len(ub))
    Bn = params["bootstrap"]["draws"]
    rng = np.random.default_rng(np.random.SeedSequence([params["seed"], 12, cidx, csi]))
    W = np.stack([np.bincount(rng.integers(0, len(ub), len(ub)), minlength=len(ub)) for _ in range(Bn)]).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = 100.0 * (W @ A.T) / (W @ nimg)[:, None]          # bootstrap x thresholds
        rec = (W @ Mm.T) / (W @ X.T)
    ti = np.searchsorted(ts, P.threshold.to_numpy())
    est = P.stated_recall.to_numpy()[None, :]
    rate_ok = np.abs(rate[:, ti] / N.k - 1) <= crit
    rec_ok = np.abs(est - rec[:, ti]) <= marg
    share = params["exp2"]["criterion_share"]
    sr, sc = rate_ok.mean(1), rec_ok.mean(1)
    return pd.DataFrame(dict(boot=np.arange(Bn), share_rate_within=sr, share_recall_within=sc,
                             validated=(sr >= share) & (sc >= share)))


def audit_estimates(a, params, e5):
    """Post-adaptation audit estimate of recall for each adapted run: ratio estimate on the audit images at the
    recalibrated threshold (calibration labels only)."""
    R = PC.Root(a.root)
    out = {}
    for r in e5.itertuples():
        smp = json.load(open(Path(a.samples) / f"{r.collection}_draw{r.draw}.json"))
        P = pd.read_parquet(Path(a.pred5) / f"adapted_{r.run_id}" / f"{r.collection}_calibration.parquet",
                            columns=PC.PRED_COLS)
        aud = smp["audit_images"]
        if not aud:
            out[r.run_id] = np.nan; continue
        if not r.recalibration_possible:
            out[r.run_id] = np.nan; continue
        T = PC.ThresholdRun(P, R.labels_external(r.collection), aud, PC.FOUR)
        _, m, x = T.per_image_at(r.recalibrated_threshold)
        out[r.run_id] = m.sum() / x.sum() if x.sum() else np.nan
    return out


def chain(a, params):
    import yaml
    cons = yaml.safe_load(open(a.constants))
    PC.locked_scripts(a.root)
    adapted_by_coll = {}
    if a.e5:
        e5 = pd.read_csv(Path(a.e5) / "e5_scores.csv")
        e5 = e5[(e5.kind == "size_curve") & (e5.checkpoint == params["selected"]) &
                (e5["size"].astype(str) == str(params["exp1"]["adapt_size"]))]
        est = audit_estimates(a, params, e5)
        for r in e5.itertuples():
            adapted_by_coll.setdefault(r.collection, {})[int(r.draw)] = dict(r._asdict(), audit_estimate=est[r.run_id])
    jobs = []
    dets = [params["selected"]] + ([c for c in PC.RUNS_A if c != params["selected"]] if a.sensitivity else [])
    for cs in params["exp1"]["class_sets"]:
        spec = params["class_sets"][cs]
        colls = PC.EXT_ALL if spec["networks"] == "all_seven" else spec["networks"]
        for c in colls:
            for det in dets:
                jobs.append((a.root, a.rules, params, cons, c, cs, det,
                             adapted_by_coll.get(c) if det == params["selected"] else None))
    PC.log(f"experiment 1: {len(jobs)} jobs")
    res = PC.pmap(job, jobs)
    D = pd.concat([r[0] for r in res], ignore_index=True)
    BT = pd.concat([r[1] for r in res if r[1] is not None], ignore_index=True)
    D.to_parquet(Path(a.out) / "e1_draws.parquet"); BT.to_parquet(Path(a.out) / "e1_bootstrap.parquet")
    crit, marg, share = params["exp2"]["criterion_rel"], params["exp3"]["margin"], params["exp2"]["criterion_share"]
    rows = []
    for key, g in D.groupby(["detector", "collection", "class_set", "arm"]):
        r = dict(zip(["detector", "collection", "class_set", "arm"], key), draws=len(g),
                 share_rate_within=float((g.rate_rel_error.abs() <= crit).mean()),
                 share_recall_within=float((g.recall_error.abs() <= marg).mean()),
                 median_rate_rel_error=float(g.rate_rel_error.median()), median_recall_error=float(g.recall_error.median()),
                 median_pool_cap_recall=float(g.pool_cap_recall.median()))
        if key[3] == "procedure":
            r["share_adapt"] = float((g.decision == "adapt").mean())
            r["validated"] = bool(r["share_rate_within"] >= share and r["share_recall_within"] >= share)
            b = BT[(BT.detector == key[0]) & (BT.collection == key[1]) & (BT.class_set == key[2])]
            r["share_rate_within_ci95"] = PC.pct(b.share_rate_within)
            r["share_recall_within_ci95"] = PC.pct(b.share_recall_within)
            r["bootstrap_share_validated"] = float(b.validated.mean())
        rows.append(r)
    S = pd.DataFrame(rows); S.to_csv(Path(a.out) / "e1_summary.csv", index=False)
    # check: summary values reproduced from the detail table (task 08, step 7)
    chk = []
    for r in S.itertuples():
        g = D[(D.detector == r.detector) & (D.collection == r.collection) & (D.class_set == r.class_set) & (D.arm == r.arm)]
        chk.append(abs(float((g.rate_rel_error.abs() <= crit).mean()) - r.share_rate_within) < 1e-12 and
                   abs(float((g.recall_error.abs() <= marg).mean()) - r.share_recall_within) < 1e-12)
    PC.jdump(dict(summary_rows_reproduced=int(sum(chk)), summary_rows=len(chk),
                  procedure_without_heldout_labels="same for every selected-detector job"),
             Path(a.out) / "e1_checks.json")
    if not all(chk):
        sys.exit("STOP: a summary value does not follow from the detail table")
    v = S[(S.arm == "procedure") & (S.detector == params["selected"])]
    PC.jdump(dict(validated_networks=int(v.validated.sum()), networks=len(v),
                  worst=v.sort_values(["share_recall_within", "share_rate_within"]).head(1).to_dict("records")),
             Path(a.out) / "e1_criteria.json")
    PC.log("experiment 1 done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["constants", "chain"])
    ap.add_argument("--root", required=True); ap.add_argument("--rules", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--e2", default=""); ap.add_argument("--e3", default="")
    ap.add_argument("--constants", default=""); ap.add_argument("--e5", default="")
    ap.add_argument("--samples", default=""); ap.add_argument("--pred5", default="")
    ap.add_argument("--sensitivity", action="store_true", help="also run the other 19 checkpoints")
    ap.add_argument("--manifest_sha", default="ef49673bdc6301009223fb7eecd0aa71be7f53eb6220395d30262b64b19228c7")
    ap.add_argument("--rules_sha", default=""); ap.add_argument("--params_sha", default="")
    ap.add_argument("--constants_sha", default="")
    ap.add_argument("--skip_gate", action="store_true", help="synthetic tests only")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if not a.skip_gate:
        PC.verify_lock(a.root, a.manifest_sha); PC.rule_gate(a.rules, a.rules_sha, a.params_sha)
        if a.cmd == "chain" and PC.sha(a.constants) != a.constants_sha:
            sys.exit("STOP: exp1_constants.yaml differs from the approved hash")
    params = PC.load_params(a.rules)
    PC.locked_scripts(a.root)
    {"constants": constants, "chain": chain}[a.cmd](a, params)
    PC.write_manifest(a.out)

#!/usr/bin/env python3
"""Experiment 2: threshold from an unlabeled calibration sample. Uses no label.

For each checkpoint, collection and class set: draw n calibration images by whole blocks (blocks in random order
until at least n images), take the confidence threshold whose alerts per 100 sampled images equal k (the confidence
of the floor(k n / 100)-th alert in the locked order), apply it unchanged to the held-out part, and record the
held-out alerts per 100 images. The block samples are shared by all checkpoints and class sets (paired design).
Sensitivities: a contiguous sample in file-name order for the collections params.exp2.contiguous lists (order
unverified), and the six setup B checkpoints on merged cracks.
"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr_common as PC  # noqa: E402


def block_samples(files, blocks, n, draws, seed, cidx):
    """List of index arrays (positions in files), one per draw: whole blocks in random order until >= n images."""
    ub = sorted({blocks[f] for f in files}); bi = {b: i for i, b in enumerate(ub)}
    members = [[] for _ in ub]
    for j, f in enumerate(files):
        members[bi[blocks[f]]].append(j)
    members = [np.array(m) for m in members]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2, cidx, n]))
    out = []
    for _ in range(draws):
        perm = rng.permutation(len(ub))
        sizes = np.cumsum([len(members[p]) for p in perm])
        k = int(np.searchsorted(sizes, n)) + 1
        out.append(np.concatenate([members[p] for p in perm[:k]]))
    return out


def contiguous_samples(files, n, draws, seed, cidx):
    order = np.argsort(np.array(files))          # file-name order (natural for zero-padded numbers)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3, cidx, n]))
    if n >= len(files):
        return [order.copy() for _ in range(draws)]
    starts = rng.integers(0, len(files) - n + 1, size=draws)
    return [order[s:s + n] for s in starts]


def per_image_conf(P, files, classes):
    P = P[P.cls.isin(classes)]
    g = dict(tuple(P.groupby("file").conf))
    return [np.sort(g[f].to_numpy())[::-1] if f in g else np.zeros(0) for f in files]


def job(args):
    root, rules, params, ckpt, setup, coll, cs, cidx = args
    PC.locked_scripts(root)
    PC.forbid_labels()  # experiment 2 is label-free
    R = PC.Root(root)
    cal, held = PC.read_split(rules, coll)
    B = R.blocks(coll)
    classes = PC.CLASS_SET_CLASSES[cs]
    k = params["class_sets"][cs]["k"]
    P = R.pred_external(ckpt, coll) if setup == "A" else R.pred_b(ckpt, coll)
    cal_conf = per_image_conf(P, cal, classes)
    hc = P[P.file.isin(set(held)) & P.cls.isin(classes)].conf.to_numpy()
    hc = np.sort(hc)
    nh = len(held)
    rows = []
    D = params["sample_draws"]
    modes = [("block", None)]
    if coll in params["exp2"].get("contiguous", []):
        modes.append(("contiguous", None))
    for n in params["exp2"]["n_grid"]:
        if n > len(cal):
            rows.append(dict(checkpoint=ckpt, setup=setup, collection=coll, class_set=cs, n=n, mode="block", draw=-1,
                             images=len(cal), threshold=np.nan, heldout_rate=np.nan, rel_error=np.nan,
                             note="n larger than the calibration part"))
            continue
        for mode, _ in modes:
            samples = (block_samples(cal, B, n, D, params["seed"], cidx) if mode == "block"
                       else contiguous_samples(cal, n, D, params["seed"], cidx))
            for d, idx in enumerate(samples):
                arr = np.concatenate([cal_conf[i] for i in idx]) if len(idx) else np.zeros(0)
                q = int(np.floor(k * len(idx) / 100.0))
                if q <= 0:
                    t = np.inf
                elif q > len(arr):
                    t = float(arr.min()) if len(arr) else np.inf
                else:
                    t = float(-np.partition(-arr, q - 1)[q - 1])
                alerts = len(hc) - int(np.searchsorted(hc, t, side="left")) if np.isfinite(t) else 0
                rate = 100.0 * alerts / nh
                rows.append(dict(checkpoint=ckpt, setup=setup, collection=coll, class_set=cs, n=n, mode=mode, draw=d,
                                 images=len(idx), threshold=t, heldout_rate=rate, rel_error=rate / k - 1, note=""))
    return pd.DataFrame(rows)


def main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if not a.skip_gate:
        PC.verify_lock(a.root, a.manifest_sha)
        PC.rule_gate(a.rules, a.rules_sha, a.params_sha)
    params = PC.load_params(a.rules)
    PC.env_record(out)
    jobs = []
    for cs, spec in params["class_sets"].items():
        colls = PC.EXT_ALL if spec["networks"] == "all_seven" else spec["networks"]
        for coll in colls:
            cidx = PC.EXT_ALL.index(coll)
            for ck in PC.RUNS_A:
                jobs.append((a.root, a.rules, params, ck, "A", coll, cs, cidx))
            if cs == "merged_crack":
                for m in PC.ARCH_B:
                    jobs.append((a.root, a.rules, params, m, "B", coll, cs, cidx))
    PC.log(f"experiment 2: {len(jobs)} jobs")
    D = pd.concat(PC.pmap(job, jobs), ignore_index=True)
    D.to_parquet(out / "e2_draws.parquet")
    crit, share = params["exp2"]["criterion_rel"], params["exp2"]["criterion_share"]
    ok = D[D.draw >= 0].copy()
    ok["within"] = ok.rel_error.abs() <= crit
    S = ok.groupby(["setup", "checkpoint", "collection", "class_set", "mode", "n"]).agg(
        draws=("draw", "size"), share_within=("within", "mean"), median_rel_error=("rel_error", "median"),
        q05_rel_error=("rel_error", lambda x: np.percentile(x, 5)),
        q95_rel_error=("rel_error", lambda x: np.percentile(x, 95)),
        median_images=("images", "median")).reset_index()
    S["meets"] = S.share_within >= share
    S.to_csv(out / "e2_summary.csv", index=False)
    rows = []
    for key, g in S.groupby(["setup", "checkpoint", "collection", "class_set", "mode"]):
        m = g[g.meets].n
        rows.append(dict(zip(["setup", "checkpoint", "collection", "class_set", "mode"], key),
                         smallest_n=int(m.min()) if len(m) else None))
    pd.DataFrame(rows, columns=["setup", "checkpoint", "collection", "class_set", "mode", "smallest_n"]) \
        .to_csv(out / "e2_smallest_n.csv", index=False)
    # checks (task 03, step 5): summary reproduced from the draws; split disjoint; labels never read in the jobs
    rep = []
    for r in S.itertuples():
        g = ok[(ok.setup == r.setup) & (ok.checkpoint == r.checkpoint) & (ok.collection == r.collection) &
               (ok.class_set == r.class_set) & (ok["mode"] == r.mode) & (ok.n == r.n)]
        rep.append(abs(float((g.rel_error.abs() <= crit).mean()) - r.share_within) < 1e-12 and len(g) == r.draws)
    PC.locked_scripts(a.root)
    sp = PC.split_checks(a.rules, PC.Root(a.root), sorted(D.collection.unique()))
    PC.jdump(dict(summary_rows_reproduced=int(sum(rep)), summary_rows=len(rep), split=sp,
                  labels="every label reader disabled in every job"), out / "e2_checks.json")
    if not all(rep) or not all(all(v.values()) for v in sp.values()):
        sys.exit("STOP: an experiment 2 check failed (see e2_checks.json)")
    PC.write_manifest(out)
    PC.log("experiment 2 done")


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--rules", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest_sha", default="ef49673bdc6301009223fb7eecd0aa71be7f53eb6220395d30262b64b19228c7")
    ap.add_argument("--rules_sha", default=""); ap.add_argument("--params_sha", default="")
    ap.add_argument("--skip_gate", action="store_true", help="synthetic tests only")
    return ap


if __name__ == "__main__":
    main(cli().parse_args())

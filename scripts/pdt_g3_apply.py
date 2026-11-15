#!/usr/bin/env python3
"""Gate G3, laptop part 2: apply the near-duplicate rule, build clusters and assign folds (protocol Section 3).

Label-free: reads C0 outputs, the G3 answers and the validation file list; no labels, predictions or outcomes.

1. Threshold decision from the person's answers (g3_answers.json), by group (bins with fewer than 5 pairs merged
   into the next higher bin). "Can't tell" answers do not count. If more than half of the decided pairs in the
   group containing 0.75 to 0.80 are different locations, the threshold is raised to the lowest of 0.80, 0.85 or
   0.90 whose group has at most half different-location pairs among its decided pairs; if none passes, the
   descriptor criterion is dropped (pHash only). Never lowered. If C0 had no SSCD descriptors, pHash only.
2. Links: two images are linked if pHash distance <= 4 or SSCD cosine >= the threshold.
3. Clean holdout: validation images with no link to any training image.
4. Clusters: connected components of validation-to-validation links among clean holdout images.
5. Folds (fixed rule): each cluster's stratum is the RDD2022 partition of most of its images (ties: alphabetical).
   Within each stratum (strata in alphabetical order, stratum k uses seed [2026, k]), clusters are shuffled and
   each is given to the fold furthest below its target image count (target share x stratum images; shares
   development 0.25, S1 0.375, S2 0.375; ties in the order development, S1, S2).

Outputs (in --out): g3_decision.json, clusters.csv, fold_dev.txt, fold_S1.txt, fold_S2.txt, leakage_by_partition.csv,
g3_summary.json (counts, largest cluster, SHA-256 of each fold list).
"""
import argparse, hashlib, json
from pathlib import Path

import numpy as np
import pandas as pd

COUNTRIES = ["China_MotorBike", "China_Drone", "United_States", "Czech", "India", "Japan", "Norway"]
PHASH_LINK, BASE_THR = 4, 0.75
CANDIDATES = [("0.80-0.85", 0.80), ("0.85-0.90", 0.85), ("0.90-1.00", 0.90)]
SHARES = [("dev", 0.25), ("S1", 0.375), ("S2", 0.375)]
SEED = 2026


def country_of(name):
    for c in COUNTRIES:
        if name.startswith(c):
            return c
    return "UNKNOWN"


def decide(groups, answers):
    rows = []
    for g in groups:
        ids = set(g["ids"])
        a = [answers[i] for i in answers if i in ids]
        same, diff, cant = a.count("same"), a.count("different"), a.count("cant_tell")
        dec = same + diff
        rows.append(dict(group=g["group"], bins=g["bins"], n_pairs_total=g["n_pairs_total"], answered=len(a),
                         same=same, different=diff, cant_tell=cant, decided=dec,
                         share_different=(diff / dec) if dec else None,
                         passes=(diff / dec <= 0.5) if dec else None))
    by_bin = {b: r for r in rows for b in r["bins"]}
    base = by_bin["0.75-0.80"]
    if base["passes"] is None:
        raise SystemExit("STOP: no decided answers in the group containing 0.75-0.80")
    if base["passes"]:
        thr, why = BASE_THR, "0.75 kept: at most half of the decided pairs in 0.75-0.80 are different locations"
    else:
        thr, why = None, "no higher bin passes: descriptor criterion dropped, pHash only"
        for b, t in CANDIDATES:
            r = by_bin[b]
            if r is not base and r["passes"]:
                thr, why = t, f"raised to {t}: lowest higher bin with at most half different-location pairs"
                break
    return thr, why, rows


def find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x


def main(a):
    c0, pairs_dir, out = Path(a.c0), Path(a.pairs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(c0 / "c0_meta.json"))
    val = [Path(p).name for p in Path(a.val_list).read_text().split()]
    has_sscd = meta.get("descriptor") == "sscd_disc_mixup"

    if a.force_threshold is not None:
        if not has_sscd:
            raise SystemExit("STOP: a descriptor threshold needs SSCD descriptors")
        thr, why, rows = a.force_threshold, f"prespecified sensitivity: threshold fixed at {a.force_threshold}", []
    elif has_sscd:
        manifest_raw = (pairs_dir / "manifest.json").read_bytes()
        ans = json.load(open(a.answers))
        if ans.get("manifest_sha256") != hashlib.sha256(manifest_raw).hexdigest():
            raise SystemExit("STOP: answers were given on a different manifest")
        groups_meta = json.load(open(pairs_dir / "g3_groups.json"))
        m = json.loads(manifest_raw)
        ids_by_bin = {b["bin"]: [p["id"] for p in b["pairs"]] for b in m["bins"]}
        groups = [dict(gm, ids=[i for b in gm["bins"] for i in ids_by_bin[b]]) for gm in groups_meta]
        thr, why, rows = decide(groups, ans["answers"])
    else:
        thr, why, rows = None, "C0 had no SSCD descriptors: pHash only (protocol Section 3)", []
    json.dump(dict(threshold=thr, reason=why, phash_link_max=PHASH_LINK, groups=rows), open(out / "g3_decision.json", "w"),
              indent=1)
    print(f"decision: {why}")

    def linked(df):
        k = df.phash_dist <= PHASH_LINK
        if thr is not None:
            k |= df.cosine >= thr
        return df[k]

    vt = linked(pd.read_parquet(c0 / "pairs_val_train.parquet"))
    vv = linked(pd.read_parquet(c0 / "pairs_val_val.parquet"))
    leaked = set(vt.val_file)
    clean = [f for f in val if f not in leaked]
    cset = set(clean)

    pos = {f: i for i, f in enumerate(clean)}
    parent = list(range(len(clean)))
    for x, y in zip(vv.file_a, vv.file_b):
        if x in cset and y in cset:
            rx, ry = find(parent, pos[x]), find(parent, pos[y])
            if rx != ry:
                parent[max(rx, ry)] = min(rx, ry)
    roots = [find(parent, i) for i in range(len(clean))]
    cl = pd.DataFrame(dict(file=clean, root=roots))
    cl["country"] = cl.file.map(country_of)
    first = cl.groupby("root").file.min()
    cl["cluster"] = cl.root.map({r: f"c_{f}" for r, f in first.items()})
    size = cl.groupby("cluster").size()

    major = (cl.groupby(["cluster", "country"]).size().reset_index(name="n")
               .sort_values(["cluster", "n", "country"], ascending=[True, False, True])
               .drop_duplicates("cluster").set_index("cluster").country)
    fold_of = {}
    for k, stratum in enumerate(sorted(major.unique())):
        cids = sorted(major[major == stratum].index)
        order = np.random.default_rng([SEED, k]).permutation(len(cids))
        total = int(size[cids].sum())
        got = {f: 0 for f, _ in SHARES}
        for j in order:
            c = cids[j]
            deficit = [(s * total - got[f], -i, f) for i, (f, s) in enumerate(SHARES)]
            f = max(deficit)[2]
            fold_of[c] = f; got[f] += int(size[c])
    cl["fold"] = cl.cluster.map(fold_of)
    cl[["file", "country", "cluster", "fold"]].sort_values("file").to_csv(out / "clusters.csv", index=False)

    hashes = {}
    for f, _ in SHARES:
        txt = "\n".join(sorted(cl.file[cl.fold == f])) + "\n"
        (out / f"fold_{f}.txt").write_text(txt)
        hashes[f] = hashlib.sha256(txt.encode()).hexdigest()

    lk = pd.DataFrame(dict(file=val)); lk["country"] = lk.file.map(country_of); lk["linked_to_train"] = lk.file.isin(leaked)
    lk.groupby("country").linked_to_train.agg(images="size", linked="sum", share="mean").to_csv(out / "leakage_by_partition.csv")

    summ = dict(threshold=thr, n_val=len(val), n_linked_to_train=len(leaked), n_clean=len(clean),
                n_clusters=int(size.size), largest_cluster=int(size.max()) if len(size) else 0,
                largest_cluster_share=float(size.max() / len(clean)) if len(size) else 0.0,
                fold_images={f: int((cl.fold == f).sum()) for f, _ in SHARES},
                fold_images_by_partition=cl.groupby(["country", "fold"]).size().unstack(fill_value=0).to_dict("index"),
                fold_list_sha256=hashes)
    json.dump(summ, open(out / "g3_summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "fold_images_by_partition"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--c0", required=True)
    ap.add_argument("--pairs", default="", help="extracted g3_pairs folder")
    ap.add_argument("--answers", default="", help="g3_answers.json")
    ap.add_argument("--val_list", required=True, help="session1/lists/val_all.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--force_threshold", type=float, default=None,
                    help="sensitivity analysis only: fixed SSCD threshold instead of the G3 decision")
    main(ap.parse_args())

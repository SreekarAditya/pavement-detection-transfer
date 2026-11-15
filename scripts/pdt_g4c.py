#!/usr/bin/env python3
"""Gate G4c, part 1 (laptop, label-free): the setup B scoring set. Protocol 3.3.9, Sections 2, 3 and 10.

DetectionBench's checkpoints were trained on its train split and selected on its validation split (args.yaml:
split val; early stopping). Its published split lists (Hugging Face dronefreak/RDD2022, parquet conversion,
hashes in external_inventory.json) are used as follows.

1. File names: the three lists must be disjoint and every name must be an image of the Kaggle RDD2022 release
   used in setup A (training or validation part, C0 index).
2. Content: the Hugging Face Git LFS SHA-256 of every DetectionBench image is listed here (db_lfs_sha256.csv);
   pod session 2 compares it with the SHA-256 of the Kaggle file of the same name (part 2, before inference).
3. Near-duplicates: a DetectionBench test image is removed if it is linked to any DetectionBench train or
   validation image by the protocol rule (pHash distance 4 or fewer, or SSCD cosine 0.75 or more; C0
   descriptors). The remaining images are the scoring set.
4. Clusters and folds: links among scoring-set images by the same rule; connected components are clusters;
   folds development 25%, S1 37.5%, S2 37.5% by the setup A rule (stratum = majority partition, seed [2026, k]).
5. Strict leakage sensitivity: steps 3 and 4 repeated with SSCD cosine 0.65.
"""
import argparse, hashlib, json, sys, time, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pdt_c0 as c0  # noqa: E402
from pdt_g3_apply import country_of, find, SHARES, SEED  # noqa: E402

PH = 4
HF_TREE = "https://huggingface.co/api/datasets/dronefreak/RDD2022/tree/main/data/images/{split}?recursive=true&limit=1000"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def lfs_listing():
    rows = []
    for split in ("train", "valid", "test"):
        url = HF_TREE.format(split=split)
        while url:
            with urllib.request.urlopen(url, timeout=60) as r:
                items = json.load(r); link = r.headers.get("Link", "")
            for it in items:
                if it["type"] == "file":
                    rows.append(dict(db_split=split, file=Path(it["path"]).name, path=it["path"], size=it["size"],
                                     lfs_sha256=(it.get("lfs") or {}).get("oid")))
            url = None
            for part in link.split(","):
                if 'rel="next"' in part:
                    url = part[part.index("<") + 1:part.index(">")]
        log(f"  {split}: {sum(r['db_split'] == split for r in rows)} files")
    return pd.DataFrame(rows)


def folds(clean, pairs):
    pos = {f: i for i, f in enumerate(clean)}
    parent = list(range(len(clean)))
    for x, y in pairs:
        rx, ry = find(parent, pos[x]), find(parent, pos[y])
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)
    cl = pd.DataFrame(dict(file=clean, root=[find(parent, i) for i in range(len(clean))]))
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
        total = int(size[cids].sum()); got = {f: 0 for f, _ in SHARES}
        for j in order:
            c = cids[j]
            f = max((s * total - got[f], -i, f) for i, (f, s) in enumerate(SHARES))[2]
            fold_of[c] = f; got[f] += int(size[c])
    cl["fold"] = cl.cluster.map(fold_of)
    return cl[["file", "country", "cluster", "fold"]].sort_values("file"), size


def main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sp = {s: pd.read_parquet(Path(a.splits) / s / "0000.parquet").file_name.tolist()
          for s in ("train", "validation", "test")}
    rep = dict(split_list_sha256={s: hashlib.sha256((Path(a.splits) / s / "0000.parquet").read_bytes()).hexdigest()
                                  for s in sp},
               counts={s: len(v) for s, v in sp.items()})
    S = {s: set(v) for s, v in sp.items()}
    rep["name_intersections"] = {"train/validation": len(S["train"] & S["validation"]),
                                 "train/test": len(S["train"] & S["test"]),
                                 "validation/test": len(S["validation"] & S["test"])}
    idx = pd.read_csv(Path(a.c0) / "index.csv")
    kag = dict(zip(idx.file, idx.split))
    allnames = S["train"] | S["validation"] | S["test"]
    rep["names_not_in_kaggle_train_or_val"] = sorted(allnames - set(kag))[:50]
    rep["kaggle_images_not_in_detectionbench"] = len(set(kag) - allnames)
    rep["test_by_kaggle_split"] = pd.Series([kag.get(f, "missing") for f in sp["test"]]).value_counts().to_dict()
    if any(rep["name_intersections"].values()) or rep["names_not_in_kaggle_train_or_val"]:
        json.dump(rep, open(out / "g4c_summary.json", "w"), indent=1)
        sys.exit("STOP: DetectionBench split lists overlap or contain names outside the Kaggle release")

    log("Hugging Face LFS hashes")
    lfs = lfs_listing(); lfs.to_csv(out / "db_lfs_sha256.csv", index=False)
    rep["lfs_listed"] = lfs.db_split.value_counts().to_dict()
    rep["lfs_missing_hash"] = int(lfs.lfs_sha256.isna().sum())
    split_map = {"train": "train", "valid": "validation", "test": "test"}
    mism = [f for s, f in zip(lfs.db_split, lfs.file) if f not in S[split_map[s]]]
    rep["lfs_files_not_in_their_split_list"] = len(mism)

    desc = np.concatenate([np.load(Path(a.c0) / "desc_train.npy"), np.load(Path(a.c0) / "desc_val.npy")]).astype(np.float32)
    row = {f: i for i, f in enumerate(pd.concat([idx[idx.split == "train"].file, idx[idx.split == "val"].file]))}
    ph = dict(zip(idx.file, c0.hex_to_u64(idx.phash)))
    test = sp["test"]; ref = sp["train"] + sp["validation"]
    qd, rd = desc[[row[f] for f in test]], desc[[row[f] for f in ref]]
    qp, rp = np.array([ph[f] for f in test], np.uint64), np.array([ph[f] for f in ref], np.uint64)
    S_tr = qd @ rd.T
    hp = np.zeros(S_tr.shape, bool)
    for i in range(0, len(qp), 512):
        hp[i:i + 512] = c0.popcount64(qp[i:i + 512, None] ^ rp[None, :]) <= PH
    S_tt = qd @ qd.T
    hp_tt = np.zeros(S_tt.shape, bool)
    for i in range(0, len(qp), 512):
        hp_tt[i:i + 512] = c0.popcount64(qp[i:i + 512, None] ^ qp[None, :]) <= PH
    for tag, thr in (("main", 0.75), ("sens065", 0.65)):
        linked = ((S_tr >= thr) | hp).any(1)
        clean = [f for f, l in zip(test, linked) if not l]
        ci = np.array([i for i, l in enumerate(linked) if not l])
        sub = (S_tt[np.ix_(ci, ci)] >= thr) | hp_tt[np.ix_(ci, ci)]
        iu, ju = np.nonzero(np.triu(sub, 1))
        cl, size = folds(clean, [(clean[i], clean[j]) for i, j in zip(iu, ju)])
        d = out / tag; d.mkdir(exist_ok=True)
        cl.to_csv(d / "clusters.csv", index=False)
        h = {}
        for f, _ in SHARES:
            txt = "\n".join(sorted(cl.file[cl.fold == f])) + "\n"
            (d / f"fold_{f}.txt").write_text(txt); h[f] = hashlib.sha256(txt.encode()).hexdigest()
        lk = pd.DataFrame(dict(file=test, linked=linked)); lk["country"] = lk.file.map(country_of)
        rep[tag] = dict(threshold=thr, test_images=len(test), linked_to_train_or_val=int(linked.sum()),
                        clean=len(clean), clusters=int(size.size), largest_cluster=int(size.max()),
                        fold_images={f: int((cl.fold == f).sum()) for f, _ in SHARES}, fold_list_sha256=h,
                        linked_share_by_partition=lk.groupby("country").linked.mean().round(4).to_dict())
        log(f"{tag}: {int(linked.sum())} of {len(test)} test images linked; clean {len(clean)}; "
            f"folds {rep[tag]['fold_images']}")
    json.dump(rep, open(out / "g4c_summary.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True); ap.add_argument("--c0", required=True); ap.add_argument("--out", required=True)
    main(ap.parse_args())

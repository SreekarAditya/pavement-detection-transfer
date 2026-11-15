#!/usr/bin/env python3
"""Gate G3, pod part: draw the image pairs for the near-duplicate boundary check (protocol Section 3).

Label-free: reads only C0 outputs, file lists and images; no labels, predictions or outcomes.

Candidate pairs are all validation-to-training and validation-to-validation pairs with SSCD cosine similarity
of 0.65 or more (from C0). They are grouped into the protocol's bins (0.65 to 0.70, 0.70 to 0.75, 0.75 to 0.80,
0.80 to 0.85, 0.85 to 0.90, 0.90 to 1.00). Within each bin the pairs are put in a random order (seed 2026, one
generator per bin in bin order) and the first --per_bin pairs are exported: the first 20 are the protocol's
sample, the rest are the reserve drawn, in this order, when "can't tell" answers leave a bin under 10 decided
pairs. The full number of pairs per bin is recorded, for the merging rule.

Outputs (in --out), packed as g3_pairs.tar.gz next to it
  manifest.json      bins with full pair counts and the exported pairs in their random order
                     (pair id, image file names and splits only; no similarity values)
  pair_scores.csv    similarity values and hash distances per pair id (not loaded by the viewing page)
  img/<pair_id>_a.jpg, img/<pair_id>_b.jpg   copies resized to at most 1600 pixels on the long side
"""
import argparse, json, tarfile, time
from pathlib import Path

import numpy as np
import pandas as pd

BINS = [(0.65, 0.70), (0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 1.0000001)]
LABELS = ["0.65-0.70", "0.70-0.75", "0.75-0.80", "0.80-0.85", "0.85-0.90", "0.90-1.00"]
SEED, LONG_SIDE = 2026, 1600


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main(a):
    import yaml
    from PIL import Image, ImageOps
    c0, s1, out = Path(a.c0), Path(a.session1), Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    meta = json.load(open(c0 / "c0_meta.json"))
    if meta.get("descriptor") != "sscd_disc_mixup":
        raise SystemExit("STOP: C0 has no SSCD descriptors; the descriptor criterion is dropped and G3 needs no check")
    def remap(p):  # pod paths -> this machine (e.g. --remap /workspace/run/data=/Users/me/rdd2022)
        if a.remap:
            old, new = a.remap.split("=", 1)
            if str(p).startswith(old):
                return new + str(p)[len(old):]
        return str(p)
    train_dir = Path(remap(yaml.safe_load(open(s1 / "data_rdd.yaml"))["train"]))
    val_paths = {Path(p).name: remap(p) for p in (s1 / "lists" / "val_all.txt").read_text().split()}
    missing = [p for p in list(val_paths.values())[:50] if not Path(p).exists()]
    if missing or not train_dir.is_dir():
        raise SystemExit(f"STOP: images not found (first missing: {missing[:1] or train_dir}); check --remap")

    vt = pd.read_parquet(c0 / "pairs_val_train.parquet").rename(columns={"val_file": "file_a", "train_file": "file_b"})
    vt["split_a"], vt["split_b"] = "val", "train"
    vv = pd.read_parquet(c0 / "pairs_val_val.parquet")
    vv["split_a"], vv["split_b"] = "val", "val"
    P = pd.concat([vt, vv], ignore_index=True)
    P = P[P.cosine.notna() & (P.cosine >= BINS[0][0])].copy()
    # deterministic base order before shuffling: sorted by the pair's file names and splits
    P = P.sort_values(["split_b", "file_a", "file_b"]).reset_index(drop=True)

    bins, scores, k = [], [], 0
    for (lo, hi), lab in zip(BINS, LABELS):
        sub = P[(P.cosine >= lo) & (P.cosine < hi)]
        order = np.random.default_rng([SEED, len(bins)]).permutation(len(sub))
        take = sub.iloc[order[:a.per_bin]]
        pairs = []
        for _, r in take.iterrows():
            pid = f"p{k:04d}"; k += 1
            for side, f, sp in [("a", r.file_a, r.split_a), ("b", r.file_b, r.split_b)]:
                src = val_paths[f] if sp == "val" else str(train_dir / f)
                with Image.open(src) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    im.thumbnail((LONG_SIDE, LONG_SIDE))
                    im.save(out / "img" / f"{pid}_{side}.jpg", quality=92)
            pairs.append(dict(id=pid, file_a=r.file_a, split_a=r.split_a, file_b=r.file_b, split_b=r.split_b))
            scores.append(dict(id=pid, bin=lab, cosine=float(r.cosine), phash_dist=int(r.phash_dist),
                               dhash_dist=int(r.dhash_dist)))
        bins.append(dict(bin=lab, n_pairs_total=int(len(sub)), n_exported=len(pairs), pairs=pairs))
        log(f"bin {lab}: {len(sub)} pairs in total, {len(pairs)} exported")

    json.dump(dict(seed=SEED, per_bin_exported=a.per_bin, sample_size=20, min_decided=10, bins=bins,
                   note="Pairs are listed in their random order within each bin. Similarity values are not here."),
              open(out / "manifest.json", "w"), indent=1)
    pd.DataFrame(scores).to_csv(out / "pair_scores.csv", index=False)
    tar = out.parent / "g3_pairs.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        t.add(out, arcname="g3_pairs")
    log(f"packed {tar} ({tar.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--c0", default="/workspace/run/outputs/c0")
    ap.add_argument("--session1", default="/workspace/run/outputs/session1")
    ap.add_argument("--out", default="/workspace/run/outputs/g3_pairs")
    ap.add_argument("--per_bin", type=int, default=80)
    ap.add_argument("--remap", default="", help="OLD=NEW prefix for image paths written on the pod")
    main(ap.parse_args())

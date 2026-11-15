#!/usr/bin/env python3
"""Pod session 2, step 5 (part 2): gates G4 to G6 for the external collections. Protocol 3.3.9.

Label-free except for reference-box counts by class (allowed at step 5). No detector is run.

1. Reference set: every RDD2022 image of the Kaggle release (train and validation descriptors and hashes from
   pod session 1, C0; test images computed here with the same SSCD file and preprocessing).
2. For every image in each collection's pool: pHash, dHash and the SSCD descriptor (C0 code, unchanged).
3. Overlap with RDD2022 (G4): an external image is removed if it is linked to any RDD2022 image by the protocol
   rule (pHash distance 4 bits or fewer, or SSCD cosine 0.75 or more). Candidate pairs (cosine 0.60 or more, or
   pHash 10 bits or fewer) are stored for transparency.
4. Blocks (G5): within each collection, after removal, images are linked if their pHash distance is 4 bits or
   fewer or their SSCD cosine is 0.65 or more; connected components are the independent blocks. The 0.65 value
   is the lowest similarity bin checked by a person at G3, where most decided pairs showed the same location, so
   it groups near views of one place more aggressively than the 0.75 leakage rule. No release provides verified
   road-segment or video identifiers for all its images; this rule is applied to every collection alike.
5. G4 counts: eligible reference boxes by class after removal. Four-class primary: at least 150 boxes in each of
   D00, D10, D20 and D40 and at least 20 blocks. Pothole analysis: at least 150 pothole boxes and 20 blocks.
6. G6: at least 2 eligible collections from different releases and collection teams, for each analysis.

Outputs (in --out): g4_index.csv (all external and RDD2022 test images: hashes), desc_<slug>.npy, desc_test.npy,
pairs_ext_rdd.parquet, pairs_within.parquet, removed.csv, blocks.csv, pool_final_<slug>.txt,
labels_final_<slug>.parquet, g4_g6_decision.json.
"""
import argparse, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pdt_c0 as c0  # noqa: E402
import torch.multiprocessing  # noqa: E402
# Images of many different sizes (EcuPDD: about 330) are held in per-shape batches; the default descriptor-based
# tensor sharing keeps one open file per held image. File-system sharing avoids that; outputs are unchanged.
torch.multiprocessing.set_sharing_strategy("file_system")

LINK_COS, LINK_PH = 0.75, 4
_NCPU = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 8)
NW = min(16, _NCPU)       # SSCD data-loading workers, as in session 1 (more exhausted file handles); outputs do not depend on it
NH = min(32, _NCPU)       # hashing processes, as in session 1
BLOCK_COS = 0.65
MIN_BOXES, MIN_BLOCKS = 150, 20
CLASSES = ["D00", "D10", "D20", "D40"]
# release and collection team (inventory); all differ, recorded for G6
TEAMS = {"rome": "Rome v2 (Zenodo 18528034)", "svrdd": "SVRDD (Zenodo 10100129)", "ecupdd": "EcuPDD (Zenodo)",
         "ird": "IRD (Zenodo)", "hrp4k": "HRP4K (Zenodo)", "kent": "Kent potholes (Mendeley)",
         "pdd2026": "PDD2026 (release)"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def hash_table(paths, names):
    with Pool(NH) as pool:
        h = pool.map(c0.hashes, paths, chunksize=16)
    df = pd.DataFrame(h); df["file"] = names
    return df


def links(qd, rd, qph, rph, cos_t, ph_t, same=False):
    """Index pairs (qi, ri) linked by the rule; descriptors may be None (hash only)."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = set()
    if qd is not None and rd is not None and len(qd) and len(rd):
        qt = torch.from_numpy(qd.astype(np.float32)).to(dev); rt = torch.from_numpy(rd.astype(np.float32)).to(dev)
        for i in range(0, len(qt), 1024):
            s = qt[i:i + 1024] @ rt.T
            if same:
                col = torch.arange(len(rt), device=dev)[None, :]
                row = torch.arange(i, min(i + 1024, len(qt)), device=dev)[:, None]
                s = torch.where(col > row, s, torch.full_like(s, -2.0))
            qi, ri = torch.nonzero(s >= cos_t, as_tuple=True)
            out |= set(zip((qi + i).tolist(), ri.tolist()))
    for i in range(0, len(qph), 512):
        d = c0.popcount64(qph[i:i + 512, None] ^ rph[None, :])
        if same:
            d = np.where(np.arange(len(rph))[None, :] > np.arange(i, i + len(d))[:, None], d, 99)
        qi, ri = np.nonzero(d <= ph_t)
        out |= set(zip((qi + i).tolist(), ri.tolist()))
    return out


def components(n, pairs):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(i) for i in range(n)])


def main(a):
    import torch
    t0 = time.time()
    ext, out, c0d = Path(a.ext), Path(a.out), Path(a.c0)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(c0d / "c0_meta.json"))
    dev = f"cuda:{a.device}" if a.device != "cpu" and torch.cuda.is_available() else "cpu"
    model, mfile = c0.load_sscd(dev)
    if model is not None:
        sha = __import__("hashlib").sha256(open(mfile, "rb").read()).hexdigest()
        if sha != meta["descriptor_file_sha256"]:
            sys.exit("STOP: SSCD file differs from the one used in pod session 1")
    if (model is None) != (meta["descriptor"] is None):
        sys.exit("STOP: descriptor availability differs from pod session 1")

    # reference set: RDD2022 train + val (C0) and test (computed now)
    idx = pd.read_csv(c0d / "index.csv")
    rd = None
    if model is not None:
        rd = np.concatenate([np.load(c0d / "desc_train.npy"), np.load(c0d / "desc_val.npy")]).astype(np.float32)
    tdir = Path(a.kaggle) / "test" / "images"
    tpaths = sorted(str(p) for p in tdir.iterdir() if p.suffix.lower() in c0.EXT)
    log(f"RDD2022 reference: {len(idx)} train and validation images (C0), {len(tpaths)} test images")
    th = hash_table(tpaths, [Path(p).name for p in tpaths]); th.insert(0, "split", "test")
    if model is not None:
        td = c0.embed(tpaths, model, dev, workers=NW); np.save(out / "desc_test.npy", td.astype(np.float16))
        rd = np.concatenate([rd, td])
    ref = pd.concat([idx[["split", "file", "phash"]], th[["split", "file", "phash"]]], ignore_index=True)
    rph = c0.hex_to_u64(ref.phash)
    assert rd is None or len(rd) == len(ref)

    rep = dict(reference_images=len(ref), descriptor=meta["descriptor"], collections={})
    all_rm, all_blocks, pairs_x, pairs_w, hash_rows = [], [], [], [], [th.assign(collection="rdd2022_test")]
    for slug in a.only:
        pool = ext / slug / "pool"
        I = pd.read_csv(pool / "images.csv"); L = pd.read_parquet(pool / "labels.parquet")
        paths = [str(pool / "images" / f) for f in I.file]
        log(f"{slug}: {len(paths)} images")
        h = hash_table(paths, I.file.tolist()); h["collection"] = slug; hash_rows.append(h)
        if h.phash.isna().any():
            sys.exit(f"STOP: {int(h.phash.isna().sum())} images of {slug} could not be read")
        qph = c0.hex_to_u64(h.phash)
        qd = None
        if model is not None:
            qd = c0.embed(paths, model, dev, workers=NW); np.save(out / f"desc_{slug}.npy", qd.astype(np.float16))
        # candidate pairs to RDD2022 for transparency (C0 thresholds), then the rule
        cand, _ = c0.build(I.file.tolist(), ref.file.tolist(), qd, rd, qph, rph,
                           c0.hex_to_u64(h.dhash), c0.hex_to_u64(pd.concat([idx.dhash, th.dhash])), same=False) \
            if model is not None else (pd.DataFrame(), 0)
        if len(cand):
            cand = cand.rename(columns={"a": "ext_file", "b": "rdd_file"}); cand["collection"] = slug
            cand["rdd_split"] = cand.rdd_file.map(dict(zip(ref.file, ref.split))); pairs_x.append(cand)
        lk = links(qd, rd, qph, rph, LINK_COS, LINK_PH)
        rm = sorted({I.file.iloc[q] for q, _ in lk})
        all_rm += [dict(collection=slug, file=f) for f in rm]
        keep = ~I.file.isin(rm)
        I2 = I[keep].reset_index(drop=True); L2 = L[L.file.isin(I2.file)]
        qd2 = qd[keep.to_numpy()] if qd is not None else None; qph2 = qph[keep.to_numpy()]
        wl = links(qd2, qd2, qph2, qph2, BLOCK_COS, LINK_PH, same=True)
        comp = components(len(I2), wl)
        blocks = pd.DataFrame(dict(collection=slug, file=I2.file, block=[f"{slug}_{c}" for c in comp]))
        all_blocks.append(blocks)
        if wl:
            pi, pj = np.array(sorted(wl)).T
            pw = pd.DataFrame(dict(collection=slug, file_a=I2.file.to_numpy()[pi], file_b=I2.file.to_numpy()[pj],
                                   cosine=(np.einsum("ij,ij->i", qd2[pi].astype(np.float32),
                                                     qd2[pj].astype(np.float32)) if qd2 is not None else np.nan),
                                   phash_dist=c0.popcount64(qph2[pi] ^ qph2[pj])))
            pairs_w.append(pw)
        (out / f"pool_final_{slug}.txt").write_text("\n".join(I2.file) + "\n")
        L2.to_parquet(out / f"labels_final_{slug}.parquet")
        cnt = {k: int((L2.mapped == k).sum()) for k in CLASSES}
        nb = int(blocks.block.nunique()); big = int(blocks.block.value_counts().max()) if len(blocks) else 0
        four = all(cnt[k] >= MIN_BOXES for k in CLASSES) and nb >= MIN_BLOCKS
        pot = cnt["D40"] >= MIN_BOXES and nb >= MIN_BLOCKS
        rep["collections"][slug] = dict(
            release_team=TEAMS[slug], images_before=len(I), removed_linked_to_rdd2022=len(rm),
            removed_by_rdd_split=(cand[cand.ext_file.isin(rm)].groupby("rdd_split").ext_file.nunique().to_dict()
                                  if len(cand) else {}),
            images_after=len(I2), boxes_after=cnt, blocks=nb, largest_block_images=big,
            largest_block_share=round(big / max(len(I2), 1), 4),
            eligible_four_class=bool(four), eligible_pothole=bool(pot))
        log(f"  removed {len(rm)}; after: {len(I2)} images, boxes {cnt}, blocks {nb} (largest {big}); "
            f"four-class {'yes' if four else 'no'}, pothole {'yes' if pot else 'no'}")
    pd.DataFrame(all_rm, columns=["collection", "file"]).to_csv(out / "removed.csv", index=False)
    pd.concat(all_blocks).to_csv(out / "blocks.csv", index=False)
    pd.concat(hash_rows, ignore_index=True).to_csv(out / "g4_index.csv", index=False)
    if pairs_x:
        pd.concat(pairs_x).to_parquet(out / "pairs_ext_rdd.parquet")
    if pairs_w:
        pd.concat(pairs_w).to_parquet(out / "pairs_within.parquet")
    c = rep["collections"]
    f4 = sorted(s for s in c if c[s]["eligible_four_class"]); fp = sorted(s for s in c if c[s]["eligible_pothole"])
    rep["G6"] = dict(four_class=dict(eligible=f4, passes=len({c[s]["release_team"] for s in f4}) >= 2),
                     pothole=dict(eligible=fp, passes=len({c[s]["release_team"] for s in fp}) >= 2))
    rep["primary_in_force"] = ("four_class" if rep["G6"]["four_class"]["passes"] else
                               "pothole (fallback 1)" if rep["G6"]["pothole"]["passes"] else
                               "internal study (fallback 2)")
    rep["rules"] = dict(link=f"pHash <= {LINK_PH} or SSCD cosine >= {LINK_COS}",
                        blocks=f"pHash <= {LINK_PH} or SSCD cosine >= {BLOCK_COS}, connected components",
                        min_boxes=MIN_BOXES, min_blocks=MIN_BLOCKS)
    rep["seconds"] = round(time.time() - t0)
    json.dump(rep, open(out / "g4_g6_decision.json", "w"), indent=1)
    log(f"G6: four-class {f4} -> {rep['G6']['four_class']['passes']}; pothole {fp} -> {rep['G6']['pothole']['passes']}; "
        f"primary in force: {rep['primary_in_force']}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", default="/workspace/run/ext")
    ap.add_argument("--c0", default="/workspace/run/inputs/c0")
    ap.add_argument("--kaggle", default="/workspace/run/data")
    ap.add_argument("--out", default="/workspace/run/outputs/g4")
    ap.add_argument("--device", default="0")
    ap.add_argument("--only", nargs="*", default=["rome", "svrdd", "ecupdd", "ird", "hrp4k", "kent", "pdd2026"])
    raise SystemExit(main(ap.parse_args()))

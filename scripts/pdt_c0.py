#!/usr/bin/env python3
"""Experiment C0, pod part: inputs for the near-duplicate rule (protocol Section 3, gate G3).

The protocol links two images if their 64-bit perceptual hashes (pHash) differ by 4 bits or fewer, or if their
copy-detection descriptors (SSCD disc_mixup) have cosine similarity of 0.75 or more. This script computes, for
every training and validation image:
  1. pHash and difference hash (dHash) of the decoded, orientation-corrected image;
  2. the SSCD disc_mixup descriptor with the authors' evaluation preprocessing (short edge resized to 288 pixels,
     aspect ratio preserved, ImageNet normalization). If SSCD cannot be loaded, no other descriptor is used:
     the descriptor criterion is dropped and only hashes are written (protocol Section 3);
  3. every candidate pair, validation to training and validation to validation: all pairs with SSCD cosine
     similarity of 0.60 or more (below the lowest G3 bin, 0.65) and all pairs with pHash distance of 10 bits or
     fewer, each with its cosine similarity, pHash and dHash distances. No threshold of the rule is applied here.

Label-free: no labels, predictions or outcomes are read.

Outputs (in --out)
  index.csv                 split, file, country, width, height, phash, dhash
  desc_train.npy, desc_val.npy   L2-normalized SSCD descriptors (float16), row order as in index.csv
  pairs_val_train.parquet   val_file, train_file, cosine, phash_dist, dhash_dist
  pairs_val_val.parquet     file_a, file_b (a < b in index order), cosine, phash_dist, dhash_dist
  c0_meta.json              descriptor used (or none), preprocessing, counts, per-query cap hits, timing
"""
import argparse, json, os, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

COUNTRIES = ["China_MotorBike", "China_Drone", "United_States", "Czech", "India", "Japan", "Norway"]
SSCD_URL = "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"
SSCD_SHORT_EDGE = 288
COS_MIN, PHASH_MAX, CAP = 0.60, 10, 2000  # stored candidate pairs; CAP = most pairs kept per query image
EXT = {".jpg", ".jpeg", ".png"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def country_of(name):
    for c in COUNTRIES:
        if name.startswith(c):
            return c
    return "UNKNOWN"


def hashes(path):
    from PIL import Image, ImageOps
    import imagehash
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            return dict(file=Path(path).name, width=im.width, height=im.height,
                        phash=str(imagehash.phash(im)), dhash=str(imagehash.dhash(im)))
    except Exception as e:
        return dict(file=Path(path).name, width=None, height=None, phash=None, dhash=None, error=str(e))


def hex_to_u64(h):
    return np.array([int(x, 16) if isinstance(x, str) else 0 for x in h], dtype=np.uint64)


def popcount64(x):
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.int16)
    b = x.view(np.uint8).reshape(*x.shape, 8)
    return np.unpackbits(b, axis=-1).sum(-1).astype(np.int16)


def load_sscd(device):
    import torch
    cache = Path(os.environ.get("MODEL_CACHE", "/workspace/run/models")); cache.mkdir(parents=True, exist_ok=True)
    f = cache / "sscd_disc_mixup.torchscript.pt"
    try:
        if not f.exists():
            torch.hub.download_url_to_file(SSCD_URL, str(f))
        return torch.jit.load(str(f), map_location=device).eval(), f
    except Exception as e:
        log(f"SSCD unavailable ({e!r}): descriptor criterion dropped, hashes only (protocol Section 3)")
        return None, None


class _Imgs:
    def __init__(self, paths):
        import torchvision.transforms as T
        self.paths = paths
        self.tf = T.Compose([T.Resize(SSCD_SHORT_EDGE), T.ToTensor(),
                             T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image, ImageOps
        with Image.open(self.paths[i]) as im:
            return i, self.tf(ImageOps.exif_transpose(im).convert("RGB"))


def embed(paths, model, device, batch=64, workers=16):
    """Batches are formed from images of identical resized shape (aspect ratio is preserved)."""
    import torch
    from torch.utils.data import DataLoader
    dl = DataLoader(_Imgs(paths), batch_size=None, shuffle=False, num_workers=min(workers, os.cpu_count() or 4))
    out = np.zeros((len(paths), 512), np.float32); buckets = {}; done = 0

    def flush(shape):
        nonlocal done
        idx, xs = zip(*buckets.pop(shape))
        with torch.inference_mode():
            f = model(torch.stack(xs).to(device))
        out[list(idx)] = torch.nn.functional.normalize(f.float(), dim=1).cpu().numpy()
        done += len(idx)

    for i, x in dl:
        buckets.setdefault(tuple(x.shape), []).append((i, x))
        if len(buckets[tuple(x.shape)]) >= batch:
            flush(tuple(x.shape))
        if i % 5000 == 0:
            log(f"  embedded {i}/{len(paths)}")
    for s in list(buckets):
        flush(s)
    assert done == len(paths)
    return out


def cos_pairs(q, ref, exclude_upper=False):
    """All (query, ref) pairs with cosine >= COS_MIN, at most CAP per query (highest kept). Returns array of
    (qi, ri, cos) and the number of queries that hit the cap. exclude_upper keeps ri > qi only (val-val)."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    qt, rt = torch.from_numpy(q).to(dev), torch.from_numpy(ref).to(dev)
    rows, capped = [], 0
    for i in range(0, len(qt), 1024):
        s = qt[i:i + 1024] @ rt.T
        if exclude_upper:
            col = torch.arange(len(rt), device=dev)[None, :]
            row = torch.arange(i, min(i + 1024, len(qt)), device=dev)[:, None]
            s = torch.where(col > row, s, torch.full_like(s, -2.0))
        k = min(CAP, s.shape[1])
        v, j = s.topk(k, dim=1)
        v, j = v.cpu().numpy(), j.cpu().numpy()
        capped += int((v[:, -1] >= COS_MIN).sum()) if k == CAP else 0
        qi, r = np.nonzero(v >= COS_MIN)
        rows.append(np.stack([qi + i, j[qi, r], v[qi, r]], 1))
    return (np.concatenate(rows) if rows else np.zeros((0, 3))), capped


def hash_pairs(qh, rh, exclude_upper=False):
    """All (query, ref) pairs with pHash distance <= PHASH_MAX."""
    out = []
    for i in range(0, len(qh), 512):
        d = popcount64(qh[i:i + 512, None] ^ rh[None, :])
        if exclude_upper:
            d = np.where(np.arange(len(rh))[None, :] > np.arange(i, i + len(d))[:, None], d, 99)
        qi, ri = np.nonzero(d <= PHASH_MAX)
        out.append(np.stack([qi + i, ri], 1))
    return np.concatenate(out) if out else np.zeros((0, 2), int)


def build(qnames, rnames, qd, rd, qph, rph, qdh, rdh, same):
    """Union of descriptor and hash candidates, with all three measures for every pair."""
    pairs = set()
    capped = 0
    if qd is not None:
        cp, capped = cos_pairs(qd, rd, exclude_upper=same)
        pairs |= {(int(a), int(b)) for a, b, _ in cp}
    pairs |= {(int(a), int(b)) for a, b in hash_pairs(qph, rph, exclude_upper=same)}
    if not pairs:
        return pd.DataFrame(columns=["a", "b", "cosine", "phash_dist", "dhash_dist"]), capped
    a, b = np.array(sorted(pairs)).T
    cos = (np.einsum("ij,ij->i", qd[a], rd[b]) if qd is not None else np.full(len(a), np.nan)).astype(np.float32)
    df = pd.DataFrame(dict(a=[qnames[x] for x in a], b=[rnames[x] for x in b], cosine=cos,
                           phash_dist=popcount64(qph[a] ^ rph[b]), dhash_dist=popcount64(qdh[a] ^ rdh[b])))
    return df, capped


def main(a):
    import yaml
    t0 = time.time()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(Path(a.session1) / "data_rdd.yaml"))
    tr = sorted(str(p) for p in Path(cfg["train"]).iterdir() if p.suffix.lower() in EXT)
    va = Path(a.session1, "lists", "val_all.txt").read_text().split()
    log(f"{len(tr)} training and {len(va)} validation images")

    log("hashes")
    with Pool(min(32, os.cpu_count() or 8)) as pool:
        h = pool.map(hashes, tr + va, chunksize=128)
    idx = pd.DataFrame(h)
    idx.insert(0, "split", ["train"] * len(tr) + ["val"] * len(va))
    idx["country"] = idx.file.map(country_of)
    idx.to_csv(out / "index.csv", index=False)
    n_hash_fail = int(idx.phash.isna().sum())
    if n_hash_fail:
        log(f"WARNING: {n_hash_fail} images could not be read for hashing")
    ti, vi = idx[idx.split == "train"], idx[idx.split == "val"]
    trp, vap, trd, vad = hex_to_u64(ti.phash), hex_to_u64(vi.phash), hex_to_u64(ti.dhash), hex_to_u64(vi.dhash)
    trn, van = ti.file.tolist(), vi.file.tolist()

    import torch
    dev = f"cuda:{a.device}" if a.device != "cpu" and torch.cuda.is_available() else "cpu"
    model, mfile = load_sscd(dev)
    dtr = dva = None
    if model is not None:
        log("SSCD descriptors (short edge 288, aspect ratio preserved)")
        dtr, dva = embed(tr, model, dev), embed(va, model, dev)
        np.save(out / "desc_train.npy", dtr.astype(np.float16)); np.save(out / "desc_val.npy", dva.astype(np.float16))

    log("candidate pairs, validation to training")
    pvt, cap_vt = build(van, trn, dva, dtr, vap, trp, vad, trd, same=False)
    pvt.rename(columns={"a": "val_file", "b": "train_file"}).to_parquet(out / "pairs_val_train.parquet")
    log(f"  {len(pvt)} pairs; queries at the cap: {cap_vt}")
    log("candidate pairs, validation to validation")
    pvv, cap_vv = build(van, van, dva, dva, vap, vap, vad, vad, same=True)
    pvv.rename(columns={"a": "file_a", "b": "file_b"}).to_parquet(out / "pairs_val_val.parquet")
    log(f"  {len(pvv)} pairs; queries at the cap: {cap_vv}")

    meta = dict(descriptor="sscd_disc_mixup" if model is not None else None,
                descriptor_file_sha256=(__import__("hashlib").sha256(open(mfile, "rb").read()).hexdigest()
                                        if mfile else None),
                preprocessing=f"short edge {SSCD_SHORT_EDGE}, aspect ratio preserved, ImageNet normalization",
                n_train=len(tr), n_val=len(va), hash_read_failures=n_hash_fail,
                stored_pairs=dict(cosine_min=COS_MIN, phash_max=PHASH_MAX, cap_per_query=CAP),
                queries_at_cap=dict(val_train=cap_vt, val_val=cap_vv),
                n_pairs=dict(val_train=len(pvt), val_val=len(pvv)), seconds=round(time.time() - t0))
    json.dump(meta, open(out / "c0_meta.json", "w"), indent=2)
    log("C0 done" + ("" if model is not None else " (hashes only)"))
    return 0 if model is not None and n_hash_fail == 0 else 5


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--session1", default="/workspace/run/outputs/session1")
    ap.add_argument("--out", default="/workspace/run/outputs/c0")
    ap.add_argument("--device", default="0")
    raise SystemExit(main(ap.parse_args()))

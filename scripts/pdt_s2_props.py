#!/usr/bin/env python3
"""Pod session 2, addendum: label-free properties of the external images, for the forecast of each
collection's recall (protocol 3.3.9, Section 8). Run after run_session2.sh, before the pod is terminated.

No label, prediction or outcome is read. For every image in each collection's final pool (after removal of
images linked to RDD2022), the same quantities as E2 for the RDD2022 validation images:
  image properties: source width and height, CIELAB lightness mean and standard deviation, log-ready Laplacian
    variance (sharpness), Canny edge density, grey-level entropy (E2 image_stats, unchanged);
  feature distance: YOLOv12s seed 0 backbone features (layer 8, 640 px letterbox; E2 embed_images, unchanged),
    mean cosine distance to the 10 nearest of the same 10,000 RDD2022 training images used in E2, and the
    distance to the nearest one.
Output: outputs/ext_props/<collection>.parquet and ext_props_manifest.json (SHA-256).
"""
import argparse, hashlib, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pdt_e2 import image_stats, embed_images, knn_cosine  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ft = np.load(Path(a.e2) / "features_train.npy").astype(np.float32)
    dec = json.load(open(Path(a.g4) / "g4_g6_decision.json"))
    man = {}
    for slug in sorted(dec["collections"]):
        files = Path(a.g4, f"pool_final_{slug}.txt").read_text().split()
        if not files:
            continue
        paths = [str(Path(a.ext) / slug / "pool" / "images" / f) for f in files]
        log(f"{slug}: {len(paths)} images")
        with Pool(min(32, os.cpu_count() or 8)) as pool:
            img = pd.DataFrame(pool.map(image_stats, paths, chunksize=32))
        img["file"] = files
        fv = embed_images(a.weights, paths, a.device)
        d_mean, d_1 = knn_cosine(fv, ft)
        img["knn10_cosine_dist"], img["nn1_cosine_dist"] = d_mean, d_1
        img.insert(0, "collection", slug)
        p = out / f"{slug}.parquet"; img.to_parquet(p)
        man[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        log(f"  unreadable: {int((~img.read_ok).sum())}")
    json.dump(man, open(out / "ext_props_manifest.json", "w"), indent=1)
    log("external properties done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--g4", default="/workspace/run/outputs/g4")
    ap.add_argument("--ext", default="/workspace/run/ext")
    ap.add_argument("--e2", default="/workspace/run/inputs/e2")
    ap.add_argument("--weights", default="/workspace/run/weights/yolo12s_seed0_best.pt")
    ap.add_argument("--out", default="/workspace/run/outputs/ext_props")
    ap.add_argument("--device", default="0")
    main(ap.parse_args())

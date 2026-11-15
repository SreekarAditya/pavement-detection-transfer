#!/usr/bin/env python3
"""Experiment E2: measured properties of every validation image and labeled object.

Every variable is computed from the image and its labels alone, never from model errors, so the
attribution in E3 is not circular. The feature-space distance (backbone features of YOLOv12s
seed 0) is computed here for E7 and the descriptive analyses; it is excluded from E3.

Outputs (in --out)
  image_props.parquet   one row per validation image
  object_props.parquet  one row per labeled object (same object ids as E1 labels.parquet)
  exposure.csv          training images and objects per country
  features_val.npy, features_train.npy, features_index.csv   pooled backbone features (layer 8)
  feature_distance.parquet  mean cosine distance of each validation image to its 10 nearest
                            neighbours among 10,000 training images sampled in proportion to country
"""
import argparse, os, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

COUNTRIES = ["China_MotorBike", "China_Drone", "United_States", "Czech", "India", "Japan", "Norway"]
SEED, INPUT, K_NN, N_TRAIN = 2026, 640, 10, 10_000
EMBED_LAYER = 8  # last backbone stage of YOLOv12 (A2C2f, P5/32), before the neck


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def country_of(name):
    for c in COUNTRIES:
        if name.startswith(c):
            return c
    return "UNKNOWN"


# ------------------------------------------------------------------ image properties
def image_stats(path):
    import cv2
    im = cv2.imread(str(path))
    if im is None:
        return dict(file=Path(path).name, read_ok=False)
    H, W = im.shape[:2]
    s = INPUT / max(H, W)
    sm = cv2.resize(im, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)  # 640 px, no padding
    lab = cv2.cvtColor(sm, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0] * 100.0 / 255.0  # OpenCV stores L* scaled to 0..255
    g = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)
    hist = np.bincount(g.ravel(), minlength=256).astype(np.float64)
    p = hist[hist > 0] / hist.sum()
    edges = cv2.Canny(g, 100, 200)
    return dict(file=Path(path).name, read_ok=True, src_w=W, src_h=H, scale_to_input=s,
                brightness_L=float(L.mean()), contrast_L=float(L.std()),
                sharpness_lapvar=float(cv2.Laplacian(g, cv2.CV_64F).var()),
                edge_density=float((edges > 0).mean()), grey_entropy=float(-(p * np.log2(p)).sum()))


def object_props(labels, img):
    """labels: E1 labels.parquet rows (pixel xyxy). img: image_props with src_w, src_h."""
    d = labels.merge(img[["file", "src_w", "src_h"]], on="file", how="left")
    s = INPUT / np.maximum(d.src_w, d.src_h)
    w, h = (d.x2 - d.x1), (d.y2 - d.y1)
    d["w_input"], d["h_input"] = w * s, h * s
    d["long_side_input"] = np.maximum(d.w_input, d.h_input)
    d["area_input"] = d.w_input * d.h_input
    d["aspect"] = np.maximum(w, h) / np.maximum(np.minimum(w, h), 1e-6)
    cx, cy = (d.x1 + d.x2) / 2 / d.src_w, (d.y1 + d.y2) / 2 / d.src_h
    d["cx_frac"], d["cy_frac"] = cx, cy
    d["border_dist_frac"] = np.minimum.reduce([cx, 1 - cx, cy, 1 - cy])
    # crowding: other boxes in the image and largest overlap with another box
    d["n_other"] = d.groupby("file").file.transform("size") - 1
    maxov = np.zeros(len(d))
    for f, g in d.groupby("file"):
        if len(g) < 2:
            continue
        b = g[["x1", "y1", "x2", "y2"]].to_numpy()
        lt = np.maximum(b[:, None, :2], b[None, :, :2]); rb = np.minimum(b[:, None, 2:], b[None, :, 2:])
        inter = np.clip(rb - lt, 0, None).prod(2)
        ar = (b[:, 2:] - b[:, :2]).prod(1)
        iou = inter / (ar[:, None] + ar[None, :] - inter + 1e-7)
        np.fill_diagonal(iou, 0)
        maxov[g.index.to_numpy()] = iou.max(1)
    d["max_iou_other"] = maxov
    return d.drop(columns=["src_w", "src_h"])


# ------------------------------------------------------------------ features
def embed_images(weights, paths, device="0", batch=64, layer=EMBED_LAYER):
    """Pooled features of one backbone layer, on the same 640 px letterboxed input the detector sees."""
    import cv2, torch
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    gpu = device != "cpu" and torch.cuda.is_available()
    net = YOLO(str(weights)).model.to(f"cuda:{device}" if gpu else "cpu").eval()
    net = net.half() if gpu else net.float()
    dev = next(net.parameters()).device
    lb = LetterBox((INPUT, INPUT), auto=False, stride=32)
    out = []
    with torch.inference_mode():
        for i in range(0, len(paths), batch):
            ims = [lb(image=cv2.imread(str(p)))[..., ::-1] for p in paths[i:i + batch]]
            x = torch.from_numpy(np.ascontiguousarray(np.stack(ims).transpose(0, 3, 1, 2))).to(dev)
            x = (x.half() if gpu else x.float()) / 255
            f = net.predict(x, embed=[layer])
            out.append(torch.stack(list(f)).float().cpu().numpy())
            if i % (batch * 20) == 0:
                log(f"  features {i}/{len(paths)}")
    return np.concatenate(out)


def knn_cosine(q, ref, k=K_NN):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    qn = torch.nn.functional.normalize(torch.from_numpy(q).float().to(dev), dim=1)
    rn = torch.nn.functional.normalize(torch.from_numpy(ref).float().to(dev), dim=1)
    sims = []
    for i in range(0, len(qn), 2048):
        s = qn[i:i + 2048] @ rn.T
        sims.append(s.topk(min(k, rn.shape[0]), dim=1).values.cpu())
    sims = torch.cat(sims).numpy()
    return 1 - sims.mean(1), 1 - sims[:, 0]


def train_sample(train_dir, n=N_TRAIN, seed=SEED):
    files = sorted(p for p in Path(train_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    df = pd.DataFrame(dict(path=[str(p) for p in files], file=[p.name for p in files]))
    df["country"] = df.file.map(country_of)
    frac = min(1.0, n / len(df))
    return (df.groupby("country", group_keys=False)
              .apply(lambda g: g.sample(n=min(len(g), max(1, round(len(g) * frac))), random_state=seed))
              .reset_index(drop=True)
              .assign(country=lambda d: d.file.map(country_of)))


def main(a):
    import yaml
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(Path(a.session1) / "data_rdd.yaml"))
    val_paths = Path(a.session1, "lists", "val_all.txt").read_text().split()

    log("image properties")
    with Pool(min(32, os.cpu_count() or 8)) as pool:
        img = pd.DataFrame(pool.map(image_stats, val_paths, chunksize=64))
    img["country"] = img.file.map(country_of)
    L = pd.read_parquet(Path(a.e1) / "labels.parquet")
    cnt = L.groupby("file").size().rename("n_boxes")
    mix = pd.crosstab(L.file, L.cls, normalize="index").add_prefix("share_cls")
    img = img.merge(cnt, left_on="file", right_index=True, how="left").merge(mix, left_on="file", right_index=True, how="left")
    img["n_boxes"] = img.n_boxes.fillna(0).astype(int)
    img.to_parquet(out / "image_props.parquet")
    log(f"  {len(img)} images; unreadable: {(~img.read_ok).sum()}")

    log("object properties")
    object_props(L, img).to_parquet(out / "object_props.parquet")

    log("training exposure")
    parts = list(Path(cfg["train"]).parts)
    parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
    tr_lbl = Path(*parts)
    rows = []
    for f in tr_lbl.glob("*.txt"):
        rows.append((country_of(f.name), sum(1 for l in open(f) if l.strip())))
    ex = pd.DataFrame(rows, columns=["country", "n_obj"]).groupby("country").agg(train_label_files=("n_obj", "size"), train_objects=("n_obj", "sum"))
    ex.to_csv(out / "exposure.csv")

    log("backbone features (YOLOv12s seed 0, layer 8)")
    tr = train_sample(cfg["train"])
    tr.to_csv(out / "features_train_index.csv", index=False)
    fv = embed_images(a.weights, val_paths, a.device)
    ft = embed_images(a.weights, tr.path.tolist(), a.device)
    np.save(out / "features_val.npy", fv.astype(np.float16)); np.save(out / "features_train.npy", ft.astype(np.float16))
    pd.DataFrame(dict(file=[Path(p).name for p in val_paths])).to_csv(out / "features_val_index.csv", index=False)
    d_mean, d_1 = knn_cosine(fv, ft)
    pd.DataFrame(dict(file=[Path(p).name for p in val_paths], country=[country_of(Path(p).name) for p in val_paths],
                      knn10_cosine_dist=d_mean, nn1_cosine_dist=d_1)).to_parquet(out / "feature_distance.parquet")
    log("E2 done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--session1", default="/workspace/run/outputs/session1")
    ap.add_argument("--e1", default="/workspace/run/outputs/e1")
    ap.add_argument("--weights", default="/workspace/run/weights/yolo12s_seed0_best.pt")
    ap.add_argument("--out", default="/workspace/run/outputs/e2")
    ap.add_argument("--device", default="0")
    main(ap.parse_args())

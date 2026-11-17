#!/usr/bin/env python3
"""Inference on the A40 (tasks 06, 07, 09). Predictions only; no label is joined here.

  val        the registered pipeline (scripts/pdt_s2_infer_a.py: Ultralytics validation on an images-only folder,
             settings of scripts/pdt_e1.BASE: 640 pixels, batch 128, half precision, confidence 0.001, suppression
             0.7, at most 300 boxes) for each weight file and each pool; boxes captured in original pixels
  variants   experiment 15: the prediction pipeline for every variant (untiled 640, lower half in two tiles, 2 x 2 tiles,
             1280 pixels, and the downscaled reverse test), all with the same confidence floor, suppression and
             maximum, boxes mapped back to original pixels
Pools are given as name=listfile, where listfile holds one image path per line; the image's base name is the
file key used by every scoring table.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr_common as PC  # noqa: E402


def images_only(name, paths, work):
    d = Path(work) / name / "images"
    d.mkdir(parents=True, exist_ok=True)
    seen = set()
    for p in paths:
        p = Path(p); t = d / p.name
        if p.name in seen:
            sys.exit(f"STOP: duplicate image name {p.name} in pool {name}")
        seen.add(p.name)
        if not t.exists():
            t.symlink_to(p.resolve())
    assert not (Path(work) / name / "labels").exists(), "a labels folder would join labels"
    import yaml
    y = Path(work) / name / "data.yaml"
    yaml.safe_dump(dict(path=str(Path(work) / name), train="images", val="images",
                        names={0: "longitudinal_crack", 1: "transverse_crack", 2: "alligator_crack", 3: "pothole"}),
                   open(y, "w"))
    return y


def parse_pools(items):
    pools = {}
    for it in items:
        name, lst = it.split("=", 1)
        pools[name] = [l.strip() for l in open(lst) if l.strip()]
    return pools


def parse_weights(items):
    w = {}
    for it in items:
        name, p = it.split("=", 1)
        w[name] = p
    return w


def cmd_val(a):
    PC.locked_scripts(a.root)
    import pdt_s2_infer_a as IA
    from pdt_e1 import BASE
    from ultralytics import YOLO
    IA.patch()
    pools, weights = parse_pools(a.pool), parse_weights(a.weights)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = []
    for pn, paths in pools.items():
        y = images_only(pn, paths, a.work)
        for wn, wp in weights.items():
            dst = out / wn / f"{pn}.parquet"
            if dst.exists():
                PC.log(f"skip {wn}/{pn} (done)"); continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            IA.CAPTURE.clear(); t0 = time.time()
            model = YOLO(wp)
            model.val(data=str(y), split="val", device=a.device, plots=False, save_json=True, verbose=False,
                      project=str(Path(a.work).resolve() / "ultra"), name=f"{pn}_{wn}", exist_ok=True, **BASE)
            P = pd.DataFrame(IA.CAPTURE, columns=PC.PRED_COLS)
            P.insert(0, "run", wn)
            tmp = dst.with_suffix(".tmp"); P.to_parquet(tmp); tmp.rename(dst)
            log.append(dict(weights=wn, weights_sha256=PC.sha(wp), pool=pn, images=len(paths),
                            images_with_predictions=int(P.file.nunique()), predictions=len(P),
                            seconds=round(time.time() - t0, 1)))
            PC.log(f"{wn}/{pn}: {len(P)} predictions on {P.file.nunique()}/{len(paths)} images")
            pd.DataFrame(log).to_csv(out / f"val_log_{int(t0)}.csv", index=False)


# ------------------------------------------------------------------ experiment 15
def tiles_2x2(W, H, overlap):
    tw, th = int(np.ceil(W / 2 * (1 + overlap))), int(np.ceil(H / 2 * (1 + overlap)))
    xs, ys = [0, W - tw], [0, H - th]
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


def nms_merge(P, iou):
    import torch, torchvision
    if not len(P):
        return P
    b = torch.tensor(P[PC.BOX].to_numpy(), dtype=torch.float32)
    s = torch.tensor(P.conf.to_numpy(), dtype=torch.float32)
    c = torch.tensor(P.cls.to_numpy(), dtype=torch.int64)
    keep = torchvision.ops.batched_nms(b, s, c, iou).numpy()
    return P.iloc[np.sort(keep)].reset_index(drop=True)


def run_predict(model, img, imgsz, device, base):
    r = model.predict(img, imgsz=imgsz, conf=base["conf"], iou=base["iou"], max_det=base["max_det"],
                      half=base["half"], device=device, verbose=False)[0]
    b = r.boxes
    if b is None or len(b) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    return b.xyxy.cpu().numpy().astype(float), b.conf.cpu().numpy().astype(float), b.cls.cpu().numpy().astype(int)


def variant_preds(model, path, variant, a, base, scale=1.0):
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    arr = np.asarray(im)[:, :, ::-1]               # BGR, as Ultralytics reads files
    H, W = arr.shape[:2]
    rows = []

    def add(xyxy, conf, cls, dx=0.0, dy=0.0, s=1.0):
        for (x1, y1, x2, y2), cf, c in zip(xyxy, conf, cls):
            rows.append((Path(path).name, int(c), float(cf), x1 / s + dx, y1 / s + dy, x2 / s + dx, y2 / s + dy))
    if variant == "untiled_640":
        add(*run_predict(model, arr, 640, a.device, base))
    elif variant == "input_1280":
        add(*run_predict(model, arr, 1280, a.device, base))
    elif variant == "crop":
        # The lower half of the frame (road surface) as two side-by-side tiles with the same overlap as the 2 x 2
        # tiling: a crop of the whole lower half would keep the long side, and with it the scale at model input,
        # unchanged on wide frames such as Norway's
        y0 = H // 2
        tw = int(np.ceil(W / 2 * (1 + a.tile_overlap)))
        for x1 in (0, W - tw):
            add(*run_predict(model, np.ascontiguousarray(arr[y0:, x1:x1 + tw]), 640, a.device, base), dx=x1, dy=y0)
    elif variant == "tile_2x2":
        for (x1, y1, x2, y2) in tiles_2x2(W, H, a.tile_overlap):
            add(*run_predict(model, np.ascontiguousarray(arr[y1:y2, x1:x2]), 640, a.device, base), dx=x1, dy=y1)
    elif variant == "downscaled":
        # Reverse test: shrink the image by `scale` and place it on a grey canvas of the original size, so the
        # letterbox to 640 pixels shows its defects at `scale` times their usual size at model input (a plain
        # downscale would be enlarged back to 640 pixels and keep the defect size at input unchanged).
        import cv2
        sw, sh = max(1, int(round(W * scale))), max(1, int(round(H * scale)))
        canvas = np.full_like(arr, 114)
        canvas[:sh, :sw] = cv2.resize(arr, (sw, sh), interpolation=cv2.INTER_AREA)
        xyxy, cf, cl = run_predict(model, canvas, 640, a.device, base)
        cx, cy = (xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2
        keep = (cx < sw) & (cy < sh)
        xyxy = np.minimum(xyxy[keep], [sw, sh, sw, sh])
        add(xyxy, cf[keep], cl[keep], s=scale)
    else:
        raise ValueError(variant)
    P = pd.DataFrame(rows, columns=PC.PRED_COLS)
    if variant in ("tile_2x2", "crop"):  # duplicates in the tile overlaps merged; never more than max_det per tile
        nt = 4 if variant == "tile_2x2" else 2
        P = nms_merge(P, a.tile_nms_iou).sort_values("conf", ascending=False, kind="stable") \
            .head(base["max_det"] * nt).reset_index(drop=True)
    return P


def cmd_variants(a):
    PC.locked_scripts(a.root)
    from pdt_e1 import BASE
    from ultralytics import YOLO
    pools, weights = parse_pools(a.pool), parse_weights(a.weights)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    scales = json.load(open(a.scales)) if a.scales else {}
    for wn, wp in weights.items():
        model = YOLO(wp)
        for pn, paths in pools.items():
            for v in a.variants:
                dst = out / wn / f"{pn}__{v}.parquet"
                if dst.exists():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                s = scales.get(pn, 1.0) if v == "downscaled" else 1.0
                P = pd.concat([variant_preds(model, p, v, a, BASE, s) for p in paths], ignore_index=True)
                P.insert(0, "run", wn)
                tmp = dst.with_suffix(".tmp"); P.to_parquet(tmp); tmp.rename(dst)
                PC.log(f"{wn}/{pn}/{v}: {len(P)} predictions, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["val", "variants"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--pool", nargs="+", required=True, help="name=listfile")
    ap.add_argument("--weights", nargs="+", required=True, help="name=path")
    ap.add_argument("--out", required=True); ap.add_argument("--work", default="/workspace/pr_work")
    ap.add_argument("--device", default="0")
    ap.add_argument("--variants", nargs="*", default=["untiled_640", "crop", "tile_2x2", "input_1280"])
    ap.add_argument("--tile_overlap", type=float, default=0.2)
    ap.add_argument("--tile_nms_iou", type=float, default=0.7)
    ap.add_argument("--scales", default="", help="JSON {pool: downscale factor} for the reverse test")
    a = ap.parse_args()
    cmd_val(a) if a.cmd == "val" else cmd_variants(a)

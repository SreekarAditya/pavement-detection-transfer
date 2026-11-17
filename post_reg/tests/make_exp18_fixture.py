#!/usr/bin/env python3
"""Synthetic fixture for experiment 18 (tests only): two candidate releases (YOLO text and Pascal VOC), a small
RDD2022 image folder, a root copy with a g4_index.csv, relaxed track minimums, and, after the overlap step,
predictions near the labels."""
import argparse, shutil, sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pr_common as PC  # noqa: E402

NAMES = ["Longitudinal_Crack", "Transverse crack", "Alligator-crack", "Pothole", "crack"]


def img(rng, w=320, h=240):
    a = rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8)
    return Image.fromarray(a).resize((w, h), Image.NEAREST)


def boxes(rng, w, h, n):
    out = []
    for _ in range(n):
        bw, bh = rng.uniform(20, 80), rng.uniform(20, 80)
        x, y = rng.uniform(0, w - bw), rng.uniform(0, h - bh)
        out.append((int(rng.integers(0, 5)), x, y, x + bw, y + bh))
    return out


def build(a):
    rng = np.random.default_rng(1); o = Path(a.out)
    shutil.rmtree(o, ignore_errors=True)
    kag = o / "kaggle"
    for split, n in (("train", 10), ("test", 5)):
        d = kag / split / "images"; d.mkdir(parents=True)
        for i in range(n):
            img(rng).save(d / f"Japan_{split}_{i:06d}.jpg", quality=95)
    rows = []
    for slug, fmt, n in (("alpha", "yolo", 60), ("beta", "voc", 50)):
        di, dl = o / "raw" / slug / "images", o / "raw" / slug / "labels"; di.mkdir(parents=True); dl.mkdir(parents=True)
        for i in range(n):
            if slug == "alpha" and i == 0:
                im = Image.open(kag / "train" / "images" / "Japan_train_000003.jpg")  # overlaps RDD2022
            elif slug == "beta" and i % 5 == 1:
                im = ImageEnhance.Brightness(Image.open(di / f"b{i - 1:04d}.jpg")).enhance(1.05)  # near view
            else:
                im = img(rng)
            name = f"{slug[0]}{i:04d}.jpg"; im.save(di / name, quality=95)
            bx = boxes(rng, im.width, im.height, int(rng.integers(1, 5)))
            if fmt == "yolo":
                (dl / f"{slug[0]}{i:04d}.txt").write_text("".join(
                    f"{c} {(x1 + x2) / 2 / im.width:.6f} {(y1 + y2) / 2 / im.height:.6f} {(x2 - x1) / im.width:.6f} "
                    f"{(y2 - y1) / im.height:.6f}\n" for c, x1, y1, x2, y2 in bx))
            else:
                objs = "".join(f"<object><name>{NAMES[c]}</name><bndbox><xmin>{x1:.1f}</xmin><ymin>{y1:.1f}</ymin>"
                               f"<xmax>{x2:.1f}</xmax><ymax>{y2:.1f}</ymax></bndbox></object>" for c, x1, y1, x2, y2 in bx)
                (dl / f"{slug[0]}{i:04d}.xml").write_text(f"<annotation>{objs}</annotation>")
        rows.append(dict(slug=slug, release=f"{slug} release", team=f"{slug} team", format=fmt, images_dir=str(di),
                         labels_dir=str(dl), class_names=";".join(NAMES), license="CC BY 4.0", url="https://example.org",
                         download_urls="file:///test", sha256="test"))
    (o / "rules").mkdir()
    pd.DataFrame(rows).to_csv(o / "rules" / "exp18_candidates.csv", index=False)
    # root copy: symlinks to the synthetic root, plus g4_index.csv for the seven used collections
    r = o / "root"; r.mkdir()
    for p in Path(a.root).iterdir():
        if p.name != "s2":
            (r / p.name).symlink_to(p.resolve())
    (r / "s2").mkdir()
    for p in (Path(a.root) / "s2").iterdir():
        if p.name != "g4":
            (r / "s2" / p.name).symlink_to(p.resolve())
    (r / "s2" / "g4").mkdir()
    for p in (Path(a.root) / "s2" / "g4").iterdir():
        (r / "s2" / "g4" / p.name).symlink_to(p.resolve())
    gi = [dict(split="", file=f"{c}_{i}.jpg", phash=f"{rng.integers(0, 2**63):016x}", dhash="0" * 16, collection=c)
          for c in PC.EXT_ALL for i in range(5)]
    PC.locked_scripts(a.root); c0 = sys.modules.get("pdt_c0") or __import__("pdt_c0")
    tests = sorted((kag / "test" / "images").iterdir()); trains = sorted((kag / "train" / "images").iterdir())
    th = pd.DataFrame([c0.hashes(str(p)) for p in tests]); th["split"] = "test"; th["collection"] = "rdd2022_test"
    pd.concat([pd.DataFrame(gi), th[["split", "file", "phash", "dhash", "collection"]]], ignore_index=True) \
        .to_csv(r / "s2" / "g4" / "g4_index.csv", index=False)
    for c in PC.EXT_ALL:
        v = rng.normal(size=(5, 512)); v /= np.linalg.norm(v, axis=1, keepdims=True)
        np.save(r / "s2" / "g4" / f"desc_{c}.npy", v.astype(np.float16))
    # C0 stand-in: RDD2022 training images (no validation images in the fixture)
    cd = o / "c0"; cd.mkdir()
    idx = pd.DataFrame([c0.hashes(str(p)) for p in trains]); idx.insert(0, "split", "train")
    idx.to_csv(cd / "index.csv", index=False)
    model, mfile = c0.load_sscd("cpu")
    if model is not None:
        np.save(cd / "desc_train.npy", c0.embed([str(p) for p in trains], model, "cpu", workers=2).astype(np.float16))
        np.save(cd / "desc_val.npy", np.zeros((0, 512), np.float16))
        np.save(r / "s2" / "g4" / "desc_test.npy", c0.embed([str(p) for p in tests], model, "cpu", workers=2).astype(np.float16))
    import json
    dp = r / "s2" / "g4" / "g4_g6_decision.json"; dd = json.load(open(dp)); dp.unlink()
    dd.update(reference_images=len(trains) + len(tests), descriptor="sscd_disc_mixup"); json.dump(dd, open(dp, "w"))
    json.dump(dict(descriptor="sscd_disc_mixup", descriptor_file_sha256=PC.sha(mfile) if mfile else None),
              open(cd / "c0_meta.json", "w"))
    # rules copy with relaxed minimums
    rr = o / "rules"
    for p in Path(a.rules).iterdir():
        if p.name != "params.yaml":
            (rr / p.name).symlink_to(p.resolve())
    t = (Path(a.rules) / "params.yaml").read_text()
    t = t.replace("min_per_class: 150", "min_per_class: 5").replace("min_total: 450", "min_total: 15") \
         .replace("pothole: {min: 150}", "pothole: {min: 5}").replace("min_blocks: 20", "min_blocks: 3")
    (rr / "params.yaml").write_text(t)


def preds(a):
    rng = np.random.default_rng(2); o = Path(a.out)
    for slug in ("alpha", "beta"):
        L = pd.concat([pd.read_parquet(o / "ext18" / slug / "open" / "labels_calibration.parquet"),
                       pd.read_parquet(o / "ext18" / "private" / f"heldout_labels_{slug}.parquet")])
        L = L[L.cls >= 0]
        P = L[["file", "cls", "x1", "y1", "x2", "y2"]].copy()
        P[["x1", "y1", "x2", "y2"]] += rng.normal(0, 3, (len(P), 4))
        P["conf"] = rng.uniform(0.05, 0.95, len(P))
        F = P.sample(frac=0.6, random_state=0).copy(); F["conf"] = rng.uniform(0.001, 0.6, len(F))
        F[["x1", "x2"]] += 120
        P = pd.concat([P, F])[PC.PRED_COLS]
        for det in ("yolo12m_seed0", "yolo12s_seed0", "yolov8s"):
            d = o / "pred18" / det; d.mkdir(parents=True, exist_ok=True); P.to_parquet(d / f"{slug}.parquet")
    PC.write_manifest(o / "pred18")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "preds"])
    ap.add_argument("--root", default=""); ap.add_argument("--rules", default=""); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a) if a.cmd == "build" else preds(a)

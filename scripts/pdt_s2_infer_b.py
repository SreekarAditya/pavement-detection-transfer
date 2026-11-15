#!/usr/bin/env python3
"""Pod session 2: gate G4c part 2 and setup B inference (predictions only). Protocol 3.3.9.

Runs in a separate environment: Ultralytics 8.4.153 (the version that saved DetectionBench's YOLO checkpoints)
and the rfdetr package; installed versions are recorded.

G4c part 2 (before any inference)
  a. Content: SHA-256 of every Kaggle image named in DetectionBench's lists equals the Hugging Face Git LFS SHA-256
     of the DetectionBench file (db_lfs_sha256.csv from part 1). Any mismatch on a scoring-set image stops the run.
  b. Checkpoints: downloaded at the recorded revisions; YOLO26 best.pt SHA-256 must equal the G0 freeze.
  c. Class order: every checkpoint's class list must be longitudinal_crack, transverse_crack, alligator_crack,
     pothole. RF-DETR label ids: DetectionBench trains RF-DETR on its canonical COCO conversion of RDD2022, whose
     category ids are 0 to 3 in that order (datasets/rdd2022.py and utils/convert_yolo_to_coco.py, commit
     18dc0ab945cfc9d8995aa838bb607ba08988b2f6), and its evaluation compares predicted ids with those categories
     directly; so the offset is 0. The RF-DETR head has one output more than the number of classes; predictions
     with id 4 belong to no class, are kept in the prediction files as released, and are not eligible alerts.
     The ids seen on the development fold are recorded; an id outside 0 to 4 stops the run.

Inference: each model's own pipeline at its released resolution (YOLO 640, RF-DETR Nano 384, Small 512),
confidence floor 0.001, at most 300 detections per image, suppression overlap 0.7 where the model uses suppression
(YOLOv8); YOLO26 end-to-end. Pools: the G4c scoring set (development, S1, S2) and every eligible external
collection. Predictions in original pixels go to sealed_b/<pool>/<model>.parquet and are hashed; no label is read.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

NAMES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]
MODELS = {  # repo, file, kind, resolution, expected SHA-256 (G0) or None
    "yolov8n": ("dronefreak/rdd2022-yolov8n", "best.pt", "yolo", 640, None, None),
    "yolov8s": ("dronefreak/rdd2022-yolov8s", "best.pt", "yolo", 640, None, None),
    "yolo26s": ("dronefreak/rdd2022-yolo26s", "best.pt", "yolo", 640, "2f1b3d34e1f56bdcbdd1933b46e13b33d7bdc064",
                "b4e45aaacc0a1e14368ddadabeec3edb8694b77a91df163ea46fdb904f29e8c7"),
    "yolo26m": ("dronefreak/rdd2022-yolo26m", "best.pt", "yolo", 640, "a57f517eb5614a8954737cd0c4fa300826f79892",
                "eaa5869b6f4bc4abcc49fe7efceb04c2ef172127a51b6234dbadbd46deb65aa6"),
    "rfdetr_nano": ("dronefreak/rdd2022-rfdetr-nano", "checkpoint_best_total.pth", "rfdetr", 384, None, None),
    "rfdetr_small": ("dronefreak/rdd2022-rfdetr-small", "checkpoint_best_total.pth", "rfdetr", 512, None, None),
}
CONF, IOU, MAXDET = 0.001, 0.7, 300


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def kaggle_path(name, kdir, kag_split):
    return Path(kdir) / {"train": "train", "val": "val"}[kag_split[name]] / "images" / name


def g4c_content(a, rep):
    lfs = pd.read_csv(Path(a.g4c) / "db_lfs_sha256.csv").dropna(subset=["lfs_sha256"])
    idx = pd.read_csv(Path(a.c0) / "index.csv"); ks = dict(zip(idx.file, idx.split))
    lfs = lfs[lfs.file.isin(ks)]
    got = [sha(kaggle_path(f, a.kaggle, ks)) for f in lfs.file]
    lfs["kaggle_sha256"] = got; lfs["match"] = lfs.kaggle_sha256 == lfs.lfs_sha256
    lfs.to_csv(Path(a.out) / "g4c_content_check.csv", index=False)
    scoring = set()
    for f in ("dev", "S1", "S2"):
        scoring |= set((Path(a.g4c) / "main" / f"fold_{f}.txt").read_text().split())
    bad = lfs[~lfs.match]
    rep["content"] = dict(checked=len(lfs), mismatches=len(bad),
                          scoring_set_images=len(scoring),
                          scoring_set_checked=int(lfs.file.isin(scoring).sum()),
                          scoring_set_mismatches=int(bad.file.isin(scoring).sum()))
    log(f"G4c content: {len(lfs)} files checked, {len(bad)} mismatches "
        f"({rep['content']['scoring_set_mismatches']} in the scoring set)")
    if rep["content"]["scoring_set_mismatches"] or rep["content"]["scoring_set_checked"] != len(scoring):
        json.dump(rep, open(Path(a.out) / "g4c_part2.json", "w"), indent=1)
        sys.exit("STOP: scoring-set content check failed")
    return sorted(scoring), ks


def get_models(a, rep, sel):
    from huggingface_hub import hf_hub_download, model_info
    rep["checkpoints"] = {}
    paths = {}
    for k, (repo, fn, kind, res, rev, want) in MODELS.items():
        if k not in sel:
            continue
        rev = rev or model_info(repo).sha
        p = hf_hub_download(repo, fn, revision=rev, cache_dir=a.cache)
        if kind == "rfdetr":
            cfg = json.load(open(hf_hub_download(repo, "config.json", revision=rev, cache_dir=a.cache)))
            if cfg.get("class_names") != NAMES:
                sys.exit(f"STOP: {k} class list {cfg.get('class_names')}")
        s = sha(p)
        if want and s != want:
            sys.exit(f"STOP: {k} best.pt differs from the G0 freeze")
        rep["checkpoints"][k] = dict(repo=repo, revision=rev, file=fn, sha256=s, resolution=res)
        paths[k] = p
    return paths


def run_yolo(path, imgs, device, batch=64):
    from ultralytics import YOLO
    m = YOLO(path)
    if [m.names[i] for i in range(len(m.names))] != NAMES:
        sys.exit(f"STOP: class list {m.names}")
    rows = []
    for i in range(0, len(imgs), batch):
        for r in m.predict(source=[str(p) for p in imgs[i:i + batch]], imgsz=640, conf=CONF, iou=IOU, max_det=MAXDET,
                           half=True, device=device, verbose=False, stream=True, batch=32):
            b = r.boxes
            f = Path(r.path).name
            for (x1, y1, x2, y2), c, k in zip(b.xyxy.tolist(), b.conf.tolist(), b.cls.tolist()):
                rows.append((f, int(k), c, x1, y1, x2, y2))
    return pd.DataFrame(rows, columns=["file", "cls", "conf", "x1", "y1", "x2", "y2"])


def rfdetr_model(name, path):
    import rfdetr
    return (rfdetr.RFDETRNano if name == "rfdetr_nano" else rfdetr.RFDETRSmall)(pretrain_weights=path)


def run_rfdetr(model, imgs, offset, batch=16):
    from PIL import Image, ImageOps
    rows = []
    for i in range(0, len(imgs), batch):
        ims = []
        for p in imgs[i:i + batch]:
            with Image.open(p) as im:
                ims.append(ImageOps.exif_transpose(im).convert("RGB"))
        dets = model.predict(ims, threshold=CONF)
        dets = dets if isinstance(dets, list) else [dets]
        for p, d in zip(imgs[i:i + batch], dets):
            k = np.argsort(-d.confidence)[:MAXDET]
            for (x1, y1, x2, y2), c, cl in zip(d.xyxy[k].tolist(), d.confidence[k].tolist(), d.class_id[k].tolist()):
                rows.append((Path(p).name, int(cl) - offset, c, x1, y1, x2, y2))
    return pd.DataFrame(rows, columns=["file", "cls", "conf", "x1", "y1", "x2", "y2"])


def main(a):
    import importlib.metadata as md
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rep = dict(versions={p: md.version(p) for p in ("ultralytics", "rfdetr", "torch", "supervision")})
    if rep["versions"]["ultralytics"] != "8.4.153":
        sys.exit(f"STOP: setup B needs Ultralytics 8.4.153, found {rep['versions']['ultralytics']}")
    sel = a.models or list(MODELS)
    tag = "_".join(sel) if a.models else "all"
    scoring, ks = g4c_content(a, rep)
    paths = get_models(a, rep, sel)
    dev = (Path(a.g4c) / "main" / "fold_dev.txt").read_text().split()
    pools = {"rdd2022_scoring": [kaggle_path(f, a.kaggle, ks) for f in scoring]}
    dec = json.load(open(Path(a.g4) / "g4_g6_decision.json"))
    for s, v in sorted(dec["collections"].items()):
        if v["eligible_four_class"] or v["eligible_pothole"]:
            files = (Path(a.g4) / f"pool_final_{s}.txt").read_text().split()
            pools[s] = [(Path(a.ext) / s / "pool" / "images" / f) for f in files]
    sealed = out / "sealed_b"
    rep["rfdetr_offset"] = {}
    runs = []
    for k, (repo, fn, kind, res, *_ ) in MODELS.items():
        if k not in sel:
            continue
        model = None
        if kind == "rfdetr":
            model = rfdetr_model(k, paths[k])
            probe = run_rfdetr(model, [kaggle_path(f, a.kaggle, ks) for f in dev], offset=0)
            ids = set(probe.cls.unique().tolist())
            if not ids <= {0, 1, 2, 3, 4}:
                sys.exit(f"STOP: {k} label ids {sorted(ids)} on the development fold")
            off = 0  # DetectionBench category ids 0 to 3 (see docstring)
            rep["rfdetr_offset"][k] = dict(offset=off, ids_seen=sorted(ids),
                                           dev_predictions_by_id=probe.cls.value_counts().sort_index().to_dict())
            log(f"{k}: label-id offset {off} (ids {sorted(ids)})")
        for pool, imgs in pools.items():
            p = sealed / pool / f"{k}.parquet"
            if p.exists():
                continue
            p.parent.mkdir(parents=True, exist_ok=True); t0 = time.time()
            P = run_yolo(paths[k], imgs, a.device) if kind == "yolo" else run_rfdetr(model, imgs, rep["rfdetr_offset"][k]["offset"])
            P.insert(0, "model", k); P.insert(1, "pool", pool)
            tmp = p.with_suffix(".parquet.tmp"); P.to_parquet(tmp); tmp.replace(p)  # atomic
            runs.append(dict(model=k, pool=pool, images=len(imgs), images_with_predictions=P.file.nunique(),
                             predictions=len(P), seconds=round(time.time() - t0, 1)))
            log(f"  {k} on {pool}: {len(P)} predictions on {P.file.nunique()}/{len(imgs)} images")
            pd.DataFrame(runs).to_csv(out / f"b_runs_{tag}.csv", index=False)
    json.dump(rep, open(out / f"g4c_part2_{tag}.json", "w"), indent=1)
    if not a.models:
        finalize(out)
    log("setup B done")


def finalize(out):
    """Merge per-process records and write the sealed manifest (after all model groups finish)."""
    out = Path(out); sealed = out / "sealed_b"
    reps = [json.load(open(p)) for p in sorted(out.glob("g4c_part2_*.json"))]
    rep = dict(reps[0]); rep["checkpoints"] = {}; rep["rfdetr_offset"] = {}
    for r in reps:
        assert r["content"] == reps[0]["content"] and r["versions"] == reps[0]["versions"]
        rep["checkpoints"].update(r["checkpoints"]); rep["rfdetr_offset"].update(r.get("rfdetr_offset", {}))
    json.dump(rep, open(out / "g4c_part2.json", "w"), indent=1)
    runs = [pd.read_csv(p) for p in sorted(out.glob("b_runs_*.csv"))]
    if runs:
        pd.concat(runs).to_csv(out / "b_runs.csv", index=False)
    man = {str(p.relative_to(sealed)): sha(p) for p in sorted(sealed.rglob("*.parquet"))}
    json.dump(man, open(out / "sealed_b_manifest.json", "w"), indent=1)
    log(f"setup B records merged; sealed manifest: {len(man)} files")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--g4c", default="/workspace/run/inputs/g4c")
    ap.add_argument("--g4", default="/workspace/run/outputs/g4")
    ap.add_argument("--c0", default="/workspace/run/inputs/c0")
    ap.add_argument("--kaggle", default="/workspace/run/data")
    ap.add_argument("--ext", default="/workspace/run/ext")
    ap.add_argument("--cache", default="/workspace/run/hf_b")
    ap.add_argument("--out", default="/workspace/run/outputs/s2b")
    ap.add_argument("--device", default="0")
    ap.add_argument("--models", nargs="*", choices=list(MODELS), help="run only these (parallel groups)")
    ap.add_argument("--finalize", action="store_true", help="only merge records and write the manifest")
    a = ap.parse_args()
    finalize(a.out) if a.finalize else main(a)

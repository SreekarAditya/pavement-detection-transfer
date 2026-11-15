#!/usr/bin/env python3
"""Pod session 2, step 6: setup A external inference (predictions only) and the July per-country rerun.
Protocol 3.3.9. Runs in the Ultralytics 8.4.24 environment (the version that saved the setup A checkpoints).

External inference. For every collection that is eligible for the four-class primary or the pothole analysis
(g4_g6_decision.json), the 20 setup A checkpoints are run through the same Ultralytics validation pipeline and
settings as E1 (input 640, batch 128, half precision, confidence floor 0.001, suppression overlap 0.7, at most 300
detections per image; YOLOv10s end-to-end). The pool folder holds images only, with no label files, so no label
is joined and no score is computed. Predictions are captured at full precision in original image pixels and
written to sealed_ext/<collection>/<run>.parquet, hashed in sealed_ext_manifest.json. These files stay unopened
until the lock (protocol Section 9, step 7).

July per-country rerun. YOLOv12s seed 0 per RDD2022 partition at confidence 0.25, the July manuscript's setting
for per-country mAP50 (July Section 3.6); compared with the published July values (viewable: it reproduces
published figures).
"""
import argparse, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pdt_e1 import BASE, run_specs  # noqa: E402

CAPTURE = []
# July manuscript, Table 4 (YOLOv12s, tau = 0.25, IoU 0.50): mAP50, precision, recall (02_plan/July_manuscript_text.md)
JULY_TABLE4 = {"Japan": (0.4767, 0.6232, 0.5676), "India": (0.2808, 0.4723, 0.3357), "Czech": (0.2738, 0.6379, 0.3200),
               "Norway": (0.2197, 0.6485, 0.2597), "United_States": (0.5514, 0.6486, 0.6326),
               "China_MotorBike": (0.9265, 0.9109, 0.9209), "China_Drone": (0.6053, 0.7633, 0.6721)}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def patch():
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.utils import ops

    def pred_to_json_full(self, predn, pbatch):
        box = ops.xyxy2xywh(predn["bboxes"]); box[:, :2] -= box[:, 2:] / 2
        f = Path(pbatch["im_file"]).name
        for b, s_, c in zip(box.tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
            CAPTURE.append((f, int(c), s_, b[0], b[1], b[0] + b[2], b[1] + b[3]))
    DetectionValidator.pred_to_json = pred_to_json_full


def build_pool(slug, g4, work, ext):
    """Images-only folder of the final pool (after removal of images linked to RDD2022)."""
    files = Path(g4, f"pool_final_{slug}.txt").read_text().split()
    d = Path(work) / slug / "images"
    d.mkdir(parents=True, exist_ok=True)
    src = Path(ext) / slug / "pool" / "images"
    for f in files:
        t = d / f
        if not t.exists():
            t.symlink_to((src / f).resolve())
    assert not (Path(work) / slug / "labels").exists(), "a labels folder would join labels"
    names = {0: "longitudinal_crack", 1: "transverse_crack", 2: "alligator_crack", 3: "pothole"}
    y = Path(work) / slug / "data.yaml"
    import yaml
    yaml.safe_dump(dict(path=str(Path(work) / slug), train="images", val="images", names=names), open(y, "w"))
    return y, files


def external(a):
    from ultralytics import YOLO
    patch()
    dec = json.load(open(Path(a.g4) / "g4_g6_decision.json"))
    slugs = sorted(s for s, v in dec["collections"].items() if v["eligible_four_class"] or v["eligible_pothole"])
    log(f"eligible collections: {slugs}")
    specs = [s for s in run_specs(Path(a.weights)) if s["group"] == "main"]
    assert len(specs) == 20
    out = Path(a.out) / "sealed_ext"; out.mkdir(parents=True, exist_ok=True)
    runs = []
    for slug in slugs:
        y, files = build_pool(slug, a.g4, a.work, a.ext)
        (out / slug).mkdir(exist_ok=True)
        for sp in specs:
            p = out / slug / f"{sp['run']}.parquet"
            if p.exists():
                log(f"skip {slug}/{sp['run']} (done)"); continue
            CAPTURE.clear(); t0 = time.time()
            model = YOLO(str(sp["weights"]))
            model.val(data=str(y), split="val", device=a.device, plots=False, save_json=True, verbose=False,
                      project=str(Path(a.work) / "ultra"), name=f"{slug}_{sp['run']}", exist_ok=True, **BASE)
            P = pd.DataFrame(CAPTURE, columns=["file", "cls", "conf", "x1", "y1", "x2", "y2"])
            P.insert(0, "run", sp["run"]); P.insert(1, "collection", slug)
            P.to_parquet(p)
            seen = P.file.nunique()
            runs.append(dict(collection=slug, run=sp["run"], images=len(files), images_with_predictions=seen,
                             predictions=len(P), seconds=round(time.time() - t0, 1)))
            log(f"  {slug}/{sp['run']}: {len(P)} predictions on {seen}/{len(files)} images")
            pd.DataFrame(runs).to_csv(Path(a.out) / "ext_runs.csv", index=False)
    man = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*.parquet"))}
    json.dump(man, open(Path(a.out) / "sealed_ext_manifest.json", "w"), indent=1)
    log(f"sealed external manifest: {len(man)} files")


def country025(a):
    from ultralytics import YOLO
    import yaml
    from pdt_gate import run_val
    s1 = Path(a.session1)
    base = yaml.safe_load(open(s1 / "data_rdd.yaml"))
    model = YOLO(str(Path(a.weights) / "yolo12s_seed0_best.pt"))
    settings = dict(BASE, conf=0.25)
    out = Path(a.out) / "country_conf025"; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for lst in sorted((s1 / "lists").glob("val_*.txt")):
        c = lst.stem[4:]
        if c in ("all", "UNKNOWN"):
            continue
        y = out / f"data_{c}.yaml"
        yaml.safe_dump(dict(base, val=str(lst)), open(y, "w"), sort_keys=False)
        r = run_val(model, y, f"country025_{c}", settings, out, a.device)
        jm, jp, jr = JULY_TABLE4[c]
        r.update(country=c, n_images=sum(1 for _ in open(lst)), july_map50=jm, july_precision=jp, july_recall=jr)
        rows.append(r)
        log(f"  {c}: mAP50 {r['map50']:.4f} (July {jm}), precision {r['precision']:.4f} (July {jp}), "
            f"recall {r['recall']:.4f} (July {jr})")
    json.dump(rows, open(out / "per_country_conf025.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["external", "country025"])
    ap.add_argument("--weights", default="/workspace/run/weights")
    ap.add_argument("--g4", default="/workspace/run/outputs/g4")
    ap.add_argument("--work", default="/workspace/run/infer")
    ap.add_argument("--ext", default="/workspace/run/ext")
    ap.add_argument("--session1", default="/workspace/run/outputs/session1")
    ap.add_argument("--out", default="/workspace/run/outputs/s2a")
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    external(a) if a.cmd == "external" else country025(a)

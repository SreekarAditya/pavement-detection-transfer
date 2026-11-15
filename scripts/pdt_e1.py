#!/usr/bin/env python3
"""Experiment E1: per-image inference for every checkpoint, scored by one matcher.

For each run, Ultralytics validation is executed with save_json=True, so the saved predictions
are exactly those behind Ultralytics' own score. The predictions are then re-scored here with a
re-implementation of Ultralytics' matching and average-precision code; the recomputed mean
average precision at 0.5 overlap must agree with Ultralytics' figure within 0.002 for every run
(the check that the matcher is right). The same matcher then produces the per-object and
per-image outcomes used by every later analysis.

Runs
  main      20 checkpoints at 640 px: YOLOv5s, v8s, v9s, v10s, v11s, v12s at seeds 0, 42, 123;
            YOLOv12n and v12m at seed 0
  v10nms    YOLOv10s at the three seeds with standard non-maximum suppression (one-to-many head)
  cap1000   YOLOv12s seed 0 with up to 1,000 detections per image (recall-ceiling definition)
  repeat    YOLOv12s seed 0 run a second time (repeatability)
  res       the 480 px and 800 px YOLOv12s models at their own input size (feeds E4)

Matcher check (gate G2). During each validation run the script records, per image, the reference boxes
and predictions exactly as Ultralytics holds them (its resized input space) and Ultralytics' own
true-positive flags. The re-implemented matcher is run in that same space; the check compares flags
prediction by prediction and requires the recomputed mean average precision at 0.5 overlap to agree
within 0.002. Outcomes are then computed in original image pixels, the geometrically correct space that
applies to every model family and input size; Ultralytics' resizing rounds the short side of non-square
images, which shifts some overlaps near 0.5. The size of that effect is recorded as space_difference.

Outputs (in --out). Only runs.csv, repeatability.json and labels.parquet may be opened before the
registration lock (protocol Section 9); everything under sealed/ is hashed in sealed_manifest.json
and stays unopened until then.
  runs.csv                 one row per run: settings, matcher-check differences and flag mismatches,
                           check, prediction and label counts, seconds (no scores)
  sealed/predictions/<run>.parquet  every saved prediction with its match outcome
  sealed/summary__<run>.json        per-run outcome summaries
  sealed/objects.parquet   one row per labeled object per run (detected at 0.25, best confidence,
                           reachable at any confidence, matched-box geometry)
  sealed/images.parquet    one row per image per run (labels, true, false and missed at 0.25)
  labels.parquet           the reference labels as scored (after Ultralytics' duplicate removal)
"""
import argparse, json, re, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

IOUV = np.linspace(0.5, 0.95, 10)
OP_CONF = 0.25          # operating threshold for detection outcomes
CHECK_TOL = 0.002
COUNTRIES = ["China_MotorBike", "China_Drone", "United_States", "Czech", "India", "Japan", "Norway"]
BASE = dict(imgsz=640, batch=128, half=True, conf=0.001, iou=0.7, max_det=300)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def country_of(name):
    for c in COUNTRIES:
        if name.startswith(c):
            return c
    return "UNKNOWN"


def run_specs(wdir):
    specs = []
    for arch in ["yolov5s", "yolov8s", "yolov9s", "yolov10s", "yolo11s", "yolo12s"]:
        for seed in [0, 42, 123]:
            specs.append(dict(run=f"{arch}_seed{seed}", weights=wdir / f"{arch}_seed{seed}_best.pt", group="main"))
    for arch in ["yolo12n", "yolo12m"]:
        specs.append(dict(run=f"{arch}_seed0", weights=wdir / f"{arch}_seed0_best.pt", group="main"))
    for seed in [0, 42, 123]:
        specs.append(dict(run=f"yolov10s_nms_seed{seed}", weights=wdir / f"yolov10s_seed{seed}_best.pt",
                          group="v10nms", settings=dict(end2end=False)))
    specs.append(dict(run="yolo12s_seed0_cap1000", weights=wdir / "yolo12s_seed0_best.pt", group="cap1000",
                      settings=dict(max_det=1000)))
    specs.append(dict(run="yolo12s_seed0_repeat", weights=wdir / "yolo12s_seed0_best.pt", group="repeat"))
    for px in [480, 800]:
        specs.append(dict(run=f"yolo12s_{px}px_seed0", weights=wdir / f"yolo12s_{px}px_seed0_best.pt", group="res",
                          settings=dict(imgsz=px, batch=64 if px == 800 else 128)))
    return specs


# ------------------------------------------------------------------ capture of Ultralytics' own matching
CAPTURE = []  # per image: label classes and boxes, predictions and Ultralytics' flags, in its input space


def patch_validator():
    """Ultralytics rounds saved boxes to 3 decimals and scores to 5. Replace its JSON writer with one that keeps
    full precision, so the recomputed score is not affected by rounding or by ties it creates."""
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.utils import ops

    def pred_to_json_full(self, predn, pbatch):
        path = Path(pbatch["im_file"])
        box = ops.xyxy2xywh(predn["bboxes"])
        box[:, :2] -= box[:, 2:] / 2
        for b, s_, c in zip(box.tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
            self.jdict.append({"image_id": path.stem, "file_name": path.name,
                               "category_id": self.class_map[int(c)], "bbox": b, "score": s_})
    DetectionValidator.pred_to_json = pred_to_json_full

    orig_process = DetectionValidator._process_batch

    def process_batch_capture(self, preds, batch):
        res = orig_process(self, preds, batch)
        CAPTURE.append(dict(
            file=Path(batch["im_file"]).name,
            lcls=batch["cls"].detach().cpu().numpy().astype(np.int64).reshape(-1),
            lbox=batch["bboxes"].detach().float().cpu().numpy().reshape(-1, 4),
            pcls=preds["cls"].detach().cpu().numpy().astype(np.int64).reshape(-1),
            pbox=preds["bboxes"].detach().float().cpu().numpy().reshape(-1, 4),
            conf=preds["conf"].detach().cpu().numpy().reshape(-1),
            tp=np.asarray(res["tp"], dtype=bool)))
        return res
    DetectionValidator._process_batch = process_batch_capture


def matcher_check(cap):
    """Re-match every image in Ultralytics' own input space with this script's matcher (float32 overlap, as
    Ultralytics computes it) and compare. Returns flag mismatches, prediction and label counts and the
    recomputed mAP50 (Ultralytics' sort order and AP routine reproduced)."""
    tps, confs, pcls, tcls, mism = [], [], [], [], 0
    for c in cap:
        nd = len(c["pcls"])
        tp = np.zeros((nd, len(IOUV)), bool)
        if len(c["lcls"]) and nd:
            a, b = c["lbox"].astype(np.float32), c["pbox"].astype(np.float32)
            lt = np.maximum(a[:, None, :2], b[None, :, :2]); rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
            inter = np.clip(rb - lt, 0, None).prod(2)
            iou = inter / ((a[:, None, 2:] - a[:, None, :2]).prod(2) + (b[None, :, 2:] - b[None, :, :2]).prod(2)
                           - inter + np.float32(1e-7))
            iou = iou * (c["lcls"][:, None] == c["pcls"][None, :])
            tp, _ = ultra_match(iou, IOUV)
        mism += int((tp != c["tp"]).sum())
        tps.append(tp); confs.append(c["conf"]); pcls.append(c["pcls"]); tcls.append(c["lcls"])
    tp, conf = np.concatenate(tps), np.concatenate(confs)
    pc, tc = np.concatenate(pcls), np.concatenate(tcls)
    i = np.argsort(-conf)  # Ultralytics' order (default sort on the same array)
    classes, nt = np.unique(tc, return_counts=True)
    ap = np.zeros((len(classes), tp.shape[1]))
    for ci, cl in enumerate(classes):
        k = pc[i] == cl
        if k.sum() == 0 or nt[ci] == 0:
            continue
        t = tp[i][k]
        fpc, tpc = (1 - t).cumsum(0), t.cumsum(0)
        rec, prec = tpc / (nt[ci] + 1e-16), tpc / (tpc + fpc)
        for j in range(tp.shape[1]):
            ap[ci, j] = compute_ap(rec[:, j], prec[:, j])
    return dict(flag_mismatch=mism, n_pred=int(len(conf)), n_labels=int(len(tc)), map50=float(ap[:, 0].mean()))


# ------------------------------------------------------------------ labels
def load_labels(val_list, names_n):
    """Reference boxes in original pixels, cleaned as Ultralytics cleans them (duplicates removed)."""
    from PIL import Image
    from ultralytics.data.utils import exif_size, img2label_paths
    imgs = [Path(p) for p in Path(val_list).read_text().split()]
    lbl_files = img2label_paths([str(p) for p in imgs])
    rows, shapes, dup_removed = [], {}, 0
    for im, lf in zip(imgs, lbl_files):
        with Image.open(im) as I:
            w, h = exif_size(I)
        shapes[im.name] = (w, h)
        lf = Path(lf)
        if not lf.exists():
            continue
        lb = [l.split() for l in lf.read_text().strip().splitlines() if l.strip()]
        if not lb:
            continue
        out = []
        for r in lb:
            c, v = int(float(r[0])), np.array(r[1:], dtype=np.float64)
            if len(v) > 4:  # polygon -> box, as Ultralytics does
                xs, ys = v[0::2], v[1::2]
                v = np.array([(xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2, xs.max() - xs.min(), ys.max() - ys.min()])
            out.append([c, *v[:4]])
        lb = np.array(out, dtype=np.float32)
        _, idx = np.unique(lb, axis=0, return_index=True)
        if len(idx) < len(lb):
            dup_removed += len(lb) - len(idx)
            lb = lb[np.sort(idx)]
        for k, (c, x, y, bw, bh) in enumerate(lb):
            rows.append((im.name, k, int(c), (x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h))
    L = pd.DataFrame(rows, columns=["file", "obj", "cls", "x1", "y1", "x2", "y2"])
    L["country"] = L.file.map(country_of)
    L["img_w"] = L.file.map(lambda f: shapes[f][0])
    L["img_h"] = L.file.map(lambda f: shapes[f][1])
    return L, shapes, dup_removed


# ------------------------------------------------------------------ matcher
def box_iou(a, b):
    """IoU between (n,4) and (m,4) xyxy arrays, same formula as Ultralytics (eps 1e-7)."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area_a = (a[:, 2:] - a[:, :2]).prod(1)
    area_b = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-7)


def ultra_match(iou, thresholds):
    """Exact copy of Ultralytics' greedy matching. iou is labels x detections with wrong classes zeroed.
    Returns correct (D x T) and, per threshold, the matched (label, detection) pairs."""
    D = iou.shape[1]
    correct = np.zeros((D, len(thresholds)), bool)
    pairs = []
    for i, t in enumerate(thresholds):
        m = np.array(np.nonzero(iou >= t)).T
        if m.shape[0]:
            if m.shape[0] > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int), i] = True
        pairs.append(m)
    return correct, pairs


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    f = getattr(np, "trapezoid", None) or np.trapz
    return f(np.interp(x, mrec, mpre), x)


def ap_per_class(tp, conf, pred_cls, target_cls, eps=1e-16):
    i = np.argsort(-conf, kind="stable")
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    classes, nt = np.unique(target_cls, return_counts=True)
    ap = np.zeros((len(classes), tp.shape[1]))
    for ci, c in enumerate(classes):
        k = pred_cls == c
        if k.sum() == 0 or nt[ci] == 0:
            continue
        fpc, tpc = (1 - tp[k]).cumsum(0), tp[k].cumsum(0)
        rec, prec = tpc / (nt[ci] + eps), tpc / (tpc + fpc)
        for j in range(tp.shape[1]):
            ap[ci, j] = compute_ap(rec[:, j], prec[:, j])
    return classes, ap


def score_run(P, L, shapes, run, names):
    """P: predictions (file, cls, conf, x1..y2). Returns predictions with flags, objects, images, scores."""
    P = P.sort_values(["file", "conf"], ascending=[True, False]).reset_index(drop=True)
    P["tp50"] = False; P["tp50_op"] = False
    tp_all = np.zeros((len(P), len(IOUV)), bool)
    obj_rows, img_rows = [], []
    Pg = dict(tuple(P.groupby("file", sort=False)))
    Lg = dict(tuple(L.groupby("file", sort=False)))
    for f in shapes:
        lab = Lg.get(f); prd = Pg.get(f)
        nl = 0 if lab is None else len(lab)
        np_ = 0 if prd is None else len(prd)
        if nl and np_:
            lb, pb = lab[["x1", "y1", "x2", "y2"]].to_numpy(), prd[["x1", "y1", "x2", "y2"]].to_numpy()
            iou = box_iou(lb, pb) * (lab.cls.to_numpy()[:, None] == prd.cls.to_numpy()[None, :])
            corr, _ = ultra_match(iou, IOUV)
            tp_all[prd.index.to_numpy()] = corr
            # operating point: same matcher, predictions at or above 0.25 only
            op = prd.conf.to_numpy() >= OP_CONF
            iou_op = iou * op[None, :]
            corr_op, pairs_op = ultra_match(iou_op, [0.5])
            P.loc[prd.index[corr_op[:, 0]], "tp50_op"] = True
            matched = {int(l): int(d) for l, d in pairs_op[0]} if len(pairs_op[0]) else {}
            reach = iou >= 0.5
            best_conf = np.where(reach, prd.conf.to_numpy()[None, :], 0).max(1)
        else:
            matched, best_conf = {}, np.zeros(nl)
        for k in range(nl):
            r = lab.iloc[k]
            d = matched.get(k)
            g = dict(run=run, file=f, obj=int(r.obj), country=r.country, cls=int(r.cls),
                     detected=d is not None, best_conf=float(best_conf[k]), reachable=bool(best_conf[k] > 0))
            if d is not None:
                q = prd.iloc[d]
                g.update(iou=float(iou[k, d]), conf=float(q.conf), px1=q.x1, py1=q.y1, px2=q.x2, py2=q.y2)
            obj_rows.append(g)
        n_op = 0 if prd is None else int((prd.conf >= OP_CONF).sum())
        tp_op = len(matched)
        img_rows.append(dict(run=run, file=f, country=country_of(f), n_labels=nl, n_pred_op=n_op,
                             tp=tp_op, fp=n_op - tp_op, fn=nl - tp_op))
    P["tp50"] = tp_all[:, 0]
    classes, ap = ap_per_class(tp_all.astype(float), P.conf.to_numpy(), P.cls.to_numpy(), L.cls.to_numpy())
    scores = dict(map50=float(ap[:, 0].mean()), map50_95=float(ap.mean()),
                  ap50_per_class={names[int(c)]: float(a) for c, a in zip(classes, ap[:, 0])})
    return P, pd.DataFrame(obj_rows), pd.DataFrame(img_rows), scores


# ------------------------------------------------------------------ main
def main(a):
    from ultralytics import YOLO
    import yaml
    patch_validator()
    out = Path(a.out); (out / "sealed").mkdir(parents=True, exist_ok=True)
    s1 = Path(a.session1)
    data = s1 / "data_rdd.yaml"; val_list = s1 / "lists" / "val_all.txt"
    names = yaml.safe_load(open(data))["names"]
    log("loading reference labels")
    L, shapes, dup = load_labels(val_list, len(names))
    L.to_parquet(out / "labels.parquet")
    log(f"{len(shapes)} images, {len(L)} labeled objects, {dup} duplicate label rows removed")

    specs = run_specs(Path(a.weights))
    missing = [str(sp["weights"]) for sp in specs if not Path(sp["weights"]).exists()]
    if missing:
        sys.exit("STOP: missing checkpoints:\n" + "\n".join(missing))
    if a.only:
        specs = [s for s in specs if re.search(a.only, s["run"])]
    runs = pd.read_csv(out / "runs.csv").to_dict("records") if (out / "runs.csv").exists() else []
    done = {r["run"] for r in runs if r.get("check") == "PASS"}
    for sp in specs:
        if sp["run"] in done:
            log(f"skip {sp['run']} (done)"); continue
        st = dict(BASE, **sp.get("settings", {}))
        log(f"run {sp['run']} {st}")
        CAPTURE.clear()
        model = YOLO(str(sp["weights"]))
        t0 = time.time()
        m = model.val(data=str(data), split="val", device=a.device, plots=False, save_json=True, verbose=False,
                      project=str(out / "ultra"), name=sp["run"], exist_ok=True, **st)
        js = Path(m.save_dir) / "predictions.json"
        J = pd.DataFrame(json.load(open(js)))
        P = pd.DataFrame(dict(file=J.file_name, cls=J.category_id - 1, conf=J.score,
                              x1=J.bbox.str[0], y1=J.bbox.str[1]))
        P["x2"] = P.x1 + J.bbox.str[2]; P["y2"] = P.y1 + J.bbox.str[3]
        P, O, I, sc = score_run(P, L, shapes, sp["run"], names)
        P.insert(0, "run", sp["run"]); P["country"] = P.file.map(country_of)
        (out / "sealed" / "predictions").mkdir(parents=True, exist_ok=True)
        P.to_parquet(out / "sealed" / "predictions" / f"{sp['run']}.parquet")
        for df, nm in [(O, "objects"), (I, "images")]:
            df.to_parquet(out / "sealed" / f"{nm}__{sp['run']}.parquet")
        g2 = matcher_check(CAPTURE)
        diff = g2["map50"] - float(m.box.map50)
        # Viewable before the lock (protocol Section 9, step 2): matcher-check differences and counts only.
        row = dict(run=sp["run"], group=sp["group"], **{k: v for k, v in st.items()},
                   difference=round(diff, 6), flag_mismatch=g2["flag_mismatch"],
                   check="PASS" if abs(diff) <= CHECK_TOL else "FAIL",
                   space_difference=round(sc["map50"] - float(m.box.map50), 6),
                   n_pred=len(P), n_pred_ultra=g2["n_pred"], labels_ours=len(L), labels_ultra=g2["n_labels"],
                   seconds=round(time.time() - t0, 1))
        runs = [r for r in runs if r["run"] != sp["run"]] + [row]
        pd.DataFrame(runs).to_csv(out / "runs.csv", index=False)
        # Sealed until the lock: per-run outcome summaries.
        sealed = out / "sealed"; sealed.mkdir(exist_ok=True)
        json.dump(dict(run=sp["run"], ultra_map50=float(m.box.map50), ultra_space_recomputed_map50=g2["map50"],
                       original_space_map50=sc["map50"], ultra_map50_95=float(m.box.map),
                       original_space_map50_95=sc["map50_95"],
                       recall_op=float(O.detected.mean()), reachable=float(O.reachable.mean()),
                       ap50_per_class=sc["ap50_per_class"]), open(sealed / f"summary__{sp['run']}.json", "w"))
        log(f"  matcher check {row['check']} (difference {row['difference']:+.6f}, flag mismatches "
            f"{row['flag_mismatch']}, predictions {row['n_pred']}/{row['n_pred_ultra']}, labels "
            f"{row['labels_ours']}/{row['labels_ultra']}); outcome files sealed")
    # merge per-run tables
    for nm in ["objects", "images"]:
        parts = sorted((out / "sealed").glob(f"{nm}__*.parquet"))
        if parts:
            pd.concat([pd.read_parquet(p) for p in parts]).to_parquet(out / "sealed" / f"{nm}.parquet")
    # repeatability: identical predictions?
    a0, a1 = out / "sealed" / "predictions" / "yolo12s_seed0.parquet", out / "sealed" / "predictions" / "yolo12s_seed0_repeat.parquet"
    if a0.exists() and a1.exists():
        x, y = pd.read_parquet(a0), pd.read_parquet(a1)
        same = len(x) == len(y) and np.allclose(x[["x1", "y1", "x2", "y2", "conf"]].to_numpy(),
                                                y[["x1", "y1", "x2", "y2", "conf"]].to_numpy())
        json.dump(dict(n_first=len(x), n_repeat=len(y), identical=bool(same)), open(out / "repeatability.json", "w"))
        log(f"repeat run identical: {same}")
    import hashlib
    man = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted((out / "sealed").rglob("*")) if p.is_file()}
    json.dump(man, open(out / "sealed_manifest.json", "w"), indent=1)
    log(f"sealed manifest written: {len(man)} files hashed")
    bad = [r["run"] for r in runs if r["check"] == "FAIL"]
    log("E1 done" + (f"; matcher check FAILED for {bad}" if bad else "; matcher check passed for all runs"))
    sys.exit(3 if bad else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--session1", default="/workspace/run/outputs/session1")
    ap.add_argument("--weights", default="/workspace/run/weights")
    ap.add_argument("--out", default="/workspace/run/outputs/e1")
    ap.add_argument("--device", default="0")
    ap.add_argument("--only", default="", help="regex to run a subset, e.g. 'yolo12s_seed0$'")
    main(ap.parse_args())

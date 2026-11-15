#!/usr/bin/env python3
"""Pod session 1, part 1: dataset check, image-size survey and reproduction gate (G).

Stages
  prepare  find the Kaggle dataset, build data_rdd.yaml, check split sizes and class order
           against the checkpoint, write per-country validation lists, survey image sizes
  gate     validate YOLOv12s seed 0 on the 5,757 validation images; PASS if mean average
           precision at 0.5 overlap is within 0.002 of 0.6192; then score each country
           separately (recorded, not gated)

Pre-stated settings (fixed before running): imgsz 640, batch 128, half precision, conf 0.001,
iou 0.7, max_det 300. These match how Ultralytics validated during training (batch doubled,
exponential-moving-average weights in half precision). If the primary run fails, two
diagnostic runs (batch 64 half; batch 128 full precision) are recorded but do not change
the verdict.
"""
import argparse, csv, hashlib, json, os, platform, re, subprocess, sys, time, zipfile
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
COUNTRIES = ["China_MotorBike", "China_Drone", "United_States", "Czech", "India", "Japan", "Norway"]
EXPECTED_VAL_IMAGES = 5757
GATE_TARGET, GATE_TOL = 0.6192, 0.002
JULY_COUNTRY_MAP50 = {"Norway": 0.2197, "China_MotorBike": 0.9265}  # from July manuscript; check only
DCODE = {"D00": "longitudinal_crack", "D10": "transverse_crack", "D20": "alligator_crack", "D40": "pothole"}
PRIMARY = dict(imgsz=640, batch=128, half=True, conf=0.001, iou=0.7, max_det=300)
DIAGNOSTICS = {"batch64_half": dict(PRIMARY, batch=64), "batch128_fp32": dict(PRIMARY, half=False)}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def country_of(name):
    for c in COUNTRIES:  # longest-first order matters for China_*
        if name.startswith(c + "_") or name.startswith(c):
            return c
    return "UNKNOWN"


def images_in(d):
    return sorted(p for p in Path(d).iterdir() if p.suffix.lower() in IMG_EXT) if Path(d).is_dir() else []


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def ckpt_meta(weights):
    """Read version and class names from the checkpoint pickle without importing Ultralytics."""
    z = zipfile.ZipFile(weights)
    data = z.read(next(n for n in z.namelist() if n.endswith("data.pkl")))
    ver = re.findall(rb"version.{1,12}?(\d+\.\d+\.\d+)", data, re.S)
    names = re.findall(rb"(longitudinal_crack|transverse_crack|alligator_crack|pothole)", data)
    seen = []
    for n in names:
        n = n.decode()
        if n not in seen:
            seen.append(n)
    return (ver[0].decode() if ver else None), seen


# ---------------------------------------------------------------- prepare
def find_split_dirs(root):
    """Return {split: images_dir} using either */images/<split> or <split>/images layouts."""
    alias = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}
    found = defaultdict(list)
    for d in Path(root).rglob("*"):
        if not d.is_dir():
            continue
        a, b = d.parent.name.lower(), d.name.lower()
        if a == "images" and b in alias:
            found[alias[b]].append(d)
        elif b == "images" and a in alias:
            found[alias[a]].append(d)
    out = {}
    for split, dirs in found.items():
        dirs = [d for d in dirs if images_in(d)]
        if not dirs:
            continue
        if split == "val":
            exact = [d for d in dirs if len(images_in(d)) == EXPECTED_VAL_IMAGES]
            dirs = exact or dirs
        out[split] = max(dirs, key=lambda d: len(images_in(d)))
    return out


def label_dir_for(img_dir):
    parts = list(img_dir.parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts)


def yaml_names(root):
    import yaml
    for y in sorted(Path(root).rglob("*.yaml")) + sorted(Path(root).rglob("*.yml")):
        try:
            d = yaml.safe_load(open(y))
        except Exception:
            continue
        if isinstance(d, dict) and "names" in d:
            n = d["names"]
            n = [n[k] for k in sorted(n)] if isinstance(n, dict) else list(n)
            return y, n
    return None, None


def size_of(p):
    from PIL import Image
    try:
        with Image.open(p) as im:
            return im.size
    except Exception:
        return (None, None)


def prepare(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rep = {"dataset_root": str(a.data)}
    splits = find_split_dirs(a.data)
    if "val" not in splits or "train" not in splits:
        sys.exit(f"STOP: could not find train/val image folders under {a.data}: {splits}")
    rep["split_dirs"] = {k: str(v) for k, v in splits.items()}

    counts = {}
    for k, d in splits.items():
        imgs = images_in(d); ld = label_dir_for(d)
        nlab = len(list(ld.glob("*.txt"))) if ld.is_dir() else 0
        counts[k] = {"images": len(imgs), "label_files": nlab, "label_dir": str(ld)}
    rep["counts"] = counts
    log(f"split counts: {counts}")

    yml, names = yaml_names(a.data)
    ver, ck_names = ckpt_meta(a.weights)
    rep.update(kaggle_yaml=str(yml), kaggle_names=names, ckpt_version=ver, ckpt_names=ck_names)
    def key(n):  # 'Longitudinal', 'longitudinal_crack', 'D00' -> 'longitudinal'
        n = DCODE.get(str(n).strip(), str(n)).strip().lower()
        return re.split(r"[_\s]", n)[0]
    if names is None:
        log("WARNING: no yaml with class names in the dataset; using checkpoint names")
        names = ck_names
    if [key(n) for n in names] != [key(n) for n in ck_names]:
        sys.exit(f"STOP: class order differs. dataset {names} vs checkpoint {ck_names}")
    log(f"class order matches: dataset {names} = checkpoint {ck_names}")

    vimgs = images_in(splits["val"])
    if len(vimgs) != EXPECTED_VAL_IMAGES:
        rep["val_count_problem"] = f"{len(vimgs)} images, expected {EXPECTED_VAL_IMAGES}"
        log("WARNING: " + rep["val_count_problem"])
    missing_lab = [p.name for p in vimgs if not (label_dir_for(splits["val"]) / (p.stem + ".txt")).exists()]
    rep["val_images_without_label_file"] = len(missing_lab)

    by_c = defaultdict(list)
    for p in vimgs:
        by_c[country_of(p.name)].append(p)
    rep["val_per_country"] = {c: len(v) for c, v in sorted(by_c.items())}
    lists = out / "lists"; lists.mkdir(exist_ok=True)
    for c, v in by_c.items():
        (lists / f"val_{c}.txt").write_text("\n".join(str(p) for p in v) + "\n")
    (lists / "val_all.txt").write_text("\n".join(str(p) for p in vimgs) + "\n")
    rep["val_list_sha256"] = hashlib.sha256("\n".join(p.name for p in vimgs).encode()).hexdigest()

    import yaml
    cfg = {"path": "/", "train": str(splits["train"]), "val": str(splits["val"]),
           "names": {i: n for i, n in enumerate(ck_names)}}
    if "test" in splits:
        cfg["test"] = str(splits["test"])
    yaml.safe_dump(cfg, open(out / "data_rdd.yaml", "w"), sort_keys=False)

    log("surveying image sizes (header read only)")
    rows = []
    with Pool(min(16, os.cpu_count() or 4)) as pool:
        for k, d in splits.items():
            imgs = images_in(d)
            for p, (w, h) in zip(imgs, pool.map(size_of, imgs, chunksize=256)):
                rows.append((k, p.name, country_of(p.name), w, h))
    with open(out / "image_sizes.csv", "w", newline="") as f:
        csv.writer(f).writerows([("split", "file", "country", "width", "height")] + rows)
    summ = defaultdict(Counter)
    for k, _, c, w, h in rows:
        summ[(k, c)][f"{w}x{h}"] += 1
    rep["image_sizes"] = {f"{k}/{c}": dict(v.most_common(5)) | {"_n": sum(v.values())}
                          for (k, c), v in sorted(summ.items())}
    json.dump(rep, open(out / "prepare_report.json", "w"), indent=2)
    log(f"val per country: {rep['val_per_country']}")
    log("prepare done -> prepare_report.json, image_sizes.csv, data_rdd.yaml")


# ---------------------------------------------------------------- gate
def run_val(model, data, name, settings, out, device):
    m = model.val(data=str(data), split="val", device=device, plots=False, save_json=False,
                  project=str(out / "ultra"), name=name, exist_ok=True, verbose=False, **settings)
    names = model.names
    per_class = {names[int(c)]: float(ap) for c, ap in zip(getattr(m.box, 'ap_class_index', None) if getattr(m.box, 'ap_class_index', None) is not None else m.ap_class_index, m.box.ap50)}
    return {"map50": float(m.box.map50), "map50_95": float(m.box.map), "precision": float(m.box.mp),
            "recall": float(m.box.mr), "ap50_per_class": per_class, "settings": settings,
            "speed_ms_per_image": m.speed}


def env_record(a):
    import torch, ultralytics
    r = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
         "ultralytics": ultralytics.__version__, "weights": str(a.weights),
         "weights_sha256": sha256(a.weights), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        r["pip_freeze"] = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                                         text=True).stdout.splitlines()
    except Exception:
        pass
    return r


def gate(a):
    from ultralytics import YOLO
    import yaml
    out = Path(a.out)
    data = out / "data_rdd.yaml"
    if not data.exists():
        sys.exit("STOP: run the prepare stage first")
    res = {"env": env_record(a), "target": GATE_TARGET, "tolerance": GATE_TOL}
    ver, _ = ckpt_meta(a.weights)
    if ver and res["env"]["ultralytics"] != ver:
        log(f"WARNING: Ultralytics {res['env']['ultralytics']} installed, checkpoint saved with {ver}")
    model = YOLO(a.weights)

    log("gate: primary run")
    p = run_val(model, data, "gate_primary", PRIMARY, out, a.device)
    res["primary"] = p
    diff = p["map50"] - GATE_TARGET
    res["difference"] = round(diff, 5)
    res["verdict"] = "PASS" if abs(diff) <= GATE_TOL else "FAIL"
    log(f"gate: mAP50 {p['map50']:.4f} vs {GATE_TARGET} (difference {diff:+.4f}) -> {res['verdict']}")

    if res["verdict"] == "FAIL":
        res["diagnostics"] = {}
        for k, s in DIAGNOSTICS.items():
            log(f"diagnostic run {k}")
            res["diagnostics"][k] = run_val(model, data, f"diag_{k}", s, out, a.device)
            log(f"  {k}: mAP50 {res['diagnostics'][k]['map50']:.4f}")
    json.dump(res, open(out / "gate_results.json", "w"), indent=2, default=str)

    if res["verdict"] == "PASS" or a.country_even_if_fail:
        base = yaml.safe_load(open(data))
        rows = []
        for lst in sorted((out / "lists").glob("val_*.txt")):
            c = lst.stem[4:]
            if c == "all" or c == "UNKNOWN":
                continue
            y = out / f"data_{c}.yaml"
            yaml.safe_dump(dict(base, val=str(lst)), open(y, "w"), sort_keys=False)
            r = run_val(model, y, f"country_{c}", PRIMARY, out, a.device)
            r["country"], r["n_images"] = c, sum(1 for _ in open(lst))
            r["july_value"] = JULY_COUNTRY_MAP50.get(c)
            rows.append(r)
            log(f"  {c}: mAP50 {r['map50']:.4f}" + (f" (July {r['july_value']})" if r["july_value"] else ""))
        json.dump(rows, open(out / "per_country_yolo12s_seed0.json", "w"), indent=2, default=str)
        with open(out / "per_country_yolo12s_seed0.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["country", "n_images", "map50", "map50_95", "precision", "recall"])
            for r in rows:
                w.writerow([r["country"], r["n_images"], r["map50"], r["map50_95"], r["precision"], r["recall"]])
    log("gate stage done")
    if res["verdict"] == "FAIL":
        sys.exit(2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["version", "prepare", "gate"])
    ap.add_argument("--data", default="/workspace/run/data")
    ap.add_argument("--weights", default="/workspace/run/weights/yolo12s_seed0_best.pt")
    ap.add_argument("--out", default="/workspace/run/outputs/session1")
    ap.add_argument("--device", default="0")
    ap.add_argument("--country-even-if-fail", action="store_true")
    a = ap.parse_args()
    if a.stage == "version":
        print(ckpt_meta(a.weights)[0] or "")
    elif a.stage == "prepare":
        prepare(a)
    else:
        gate(a)

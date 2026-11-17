"""Synthetic copy of the locked root, same paths and columns, with known truth, for testing the post-registration
code without opening any real outcome. Usage: python pr_synth.py --out DIR [--scale 1.0] [--seed 7]"""
import argparse, json, shutil, sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import pr_common as PC  # noqa: E402

EXT = {"ecupdd": (300, 3.2, True), "ird": (500, 1.6, True), "svrdd": (700, 1.3, True), "rome": (250, 0.6, False),
       "hrp4k": (300, 1.2, False), "kent": (150, 1.6, False), "pdd2026": (400, 0.3, False)}
EXT_DIFF = {"ecupdd": -1.6, "ird": -0.5, "svrdd": -0.2, "rome": -1.0, "hrp4k": -1.5, "kent": -1.2, "pdd2026": 0.0}
COUNTRY_DIFF = {"China_MotorBike": 1.5, "China_Drone": 0.6, "United_States": 1.0, "Czech": -0.4, "India": -0.4,
                "Japan": 0.2, "Norway": -1.0}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_image_boxes(rng, n_boxes, W, H, pothole_only=False):
    rows = []
    for _ in range(n_boxes):
        w, h = rng.uniform(0.05, 0.4) * W, rng.uniform(0.05, 0.3) * H
        x1, y1 = rng.uniform(0, W - w), rng.uniform(0, H - h)
        c = 3 if pothole_only else int(rng.choice(4, p=[0.35, 0.2, 0.25, 0.2]))
        rows.append((c, x1, y1, x1 + w, y1 + h))
    return rows


def predict(rng, boxes, W, H, skill, base, fp_rate=25):
    """Predictions for one image: each box detected with probability sigmoid(skill + base + box effect);
    detections jitter the box slightly; plus low-confidence false alarms."""
    out = []
    for (c, x1, y1, x2, y2, eff) in boxes:
        p = sigmoid(skill + base + eff)
        u = rng.random()
        if u < p:
            conf = float(np.clip(0.25 + 0.7 * rng.beta(4, 2), 0.001, 0.999))
        else:
            conf = float(rng.uniform(0.001, 0.2)) if rng.random() < 0.6 else None
        if conf is None:
            continue
        j = 0.03 * (x2 - x1)
        cc = c if rng.random() < 0.9 else int(rng.integers(0, 4))
        out.append((cc, conf, x1 + rng.normal(0, j), y1 + rng.normal(0, j), x2 + rng.normal(0, j), y2 + rng.normal(0, j)))
    for _ in range(rng.poisson(fp_rate)):
        w, h = rng.uniform(0.03, 0.3) * W, rng.uniform(0.03, 0.3) * H
        x1, y1 = rng.uniform(0, W - w), rng.uniform(0, H - h)
        conf = float(rng.beta(1, 8) * 0.9 + 0.001)
        out.append((int(rng.integers(0, 4)), conf, x1, y1, x1 + w, y1 + h))
    return out


def main(a):
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    for d in ["e1/sealed/predictions", "g3/main", "g3/sens065", "s2/g4", "s2/s2a/sealed_ext", "s2/s2b/sealed_b",
              "g4c/main", "g4c/sens065", "e2", "s2/ext_props", "forecast", "docs", "scripts"]:
        (out / d).mkdir(parents=True, exist_ok=True)
    lock = Path(a.lock_scripts)
    for f in lock.glob("pdt_*.py"):
        shutil.copy(f, out / "scripts" / f.name)
    skill = {r: rng.normal(0.4, 0.25) for r in PC.RUNS_A}
    skill_b = {m: rng.normal(0.3, 0.25) for m in PC.ARCH_B}
    sc = a.scale
    # ---------------- internal validation set
    labs, preds, imgs, objs, clus = [], {r: [] for r in PC.RUNS_A}, [], [], []
    predb = {m: [] for m in PC.ARCH_B}
    W, H = 600, 600
    nper = int(110 * sc)
    for country in PC.COUNTRIES:
        Wc = 3650 if country == "Norway" else W
        Hc = 2044 if country == "Norway" else H
        cl = 0
        for i in range(nper):
            f = f"{country}_{i:06d}.jpg"
            nb = rng.poisson(1.4)
            raw = make_image_boxes(rng, nb, Wc, Hc)
            boxes = [(c, x1, y1, x2, y2, COUNTRY_DIFF[country] + rng.normal(0, 1)) for c, x1, y1, x2, y2 in raw]
            for j, (c, x1, y1, x2, y2, e) in enumerate(boxes):
                labs.append(dict(file=f, obj=j, cls=c, x1=x1, y1=y1, x2=x2, y2=y2, country=country, img_w=Wc, img_h=Hc))
                s = 640 / max(Wc, Hc)
                objs.append(dict(file=f, obj=j, cls=c, x1=x1, y1=y1, x2=x2, y2=y2, country=country, img_w=Wc, img_h=Hc,
                                 w_input=(x2 - x1) * s, h_input=(y2 - y1) * s, long_side_input=max(x2 - x1, y2 - y1) * s,
                                 area_input=(x2 - x1) * (y2 - y1) * s * s, aspect=(x2 - x1) / (y2 - y1), cx_frac=0.5,
                                 cy_frac=0.5, border_dist_frac=0.1, n_other=nb - 1, max_iou_other=0.0))
            for r in PC.RUNS_A:
                for (c, conf, x1, y1, x2, y2) in predict(rng, boxes, Wc, Hc, skill[r], 0.0):
                    preds[r].append((r, f, c, conf, x1, y1, x2, y2, country))
            for m in PC.ARCH_B:
                for (c, conf, x1, y1, x2, y2) in predict(rng, boxes, Wc, Hc, skill_b[m], 0.0):
                    predb[m].append((m, "rdd2022_scoring", f, c, conf, x1, y1, x2, y2))
            imgs.append(dict(file=f, read_ok=True, src_w=Wc, src_h=Hc, scale_to_input=640 / max(Wc, Hc),
                             brightness_L=50.0, contrast_L=20.0, sharpness_lapvar=100.0, edge_density=0.1,
                             grey_entropy=7.0, country=country, n_boxes=nb, share_cls0=0, share_cls1=0, share_cls2=0,
                             share_cls3=0))
            if i % 3 == 0:
                cl += 1
            u = rng.random()
            linked = u < 0.25
            if not linked:
                fold = "dev" if u < 0.25 + 0.75 * 0.25 else ("S1" if u < 0.25 + 0.75 * 0.625 else "S2")
                clus.append(dict(file=f, country=country, cluster=f"c_{country}_{cl}", fold=fold))
    LA = pd.DataFrame(labs)
    LA.to_parquet(out / "e1" / "labels.parquet")
    pd.DataFrame(objs).to_parquet(out / "e2" / "object_props.parquet")
    pd.DataFrame(imgs).to_parquet(out / "e2" / "image_props.parquet")
    pd.DataFrame(dict(file=[i["file"] for i in imgs], country=[i["country"] for i in imgs],
                      knn10_cosine_dist=rng.uniform(0.02, 0.2, len(imgs)), nn1_cosine_dist=0.01)
                 ).to_parquet(out / "e2" / "feature_distance.parquet")
    for r in PC.RUNS_A:
        P = pd.DataFrame(preds[r], columns=["run", "file", "cls", "conf", "x1", "y1", "x2", "y2", "country"])
        P["tp50"] = False; P["tp50_op"] = False
        P.to_parquet(out / "e1" / "sealed" / "predictions" / f"{r}.parquet")
    CL = pd.DataFrame(clus)
    for kind in ("main", "sens065"):
        CL.to_csv(out / "g3" / kind / "clusters.csv", index=False)
        for fold in ("dev", "S1", "S2"):
            (out / "g3" / kind / f"fold_{fold}.txt").write_text("\n".join(sorted(CL[CL.fold == fold].file)) + "\n")
        lk = pd.DataFrame(dict(country=PC.COUNTRIES))
        lk["images"] = [nper] * len(PC.COUNTRIES)
        lk["linked"] = [nper - int((CL.country == c).sum()) for c in PC.COUNTRIES]
        lk["share"] = lk.linked / lk.images
        lk.to_csv(out / "g3" / kind / "leakage_by_partition.csv", index=False)
    # setup B folds (same files) and DetectionBench-style labels
    for kind in ("main", "sens065"):
        CL.to_csv(out / "g4c" / kind / "clusters.csv", index=False)
        for fold in ("dev", "S1", "S2"):
            (out / "g4c" / kind / f"fold_{fold}.txt").write_text("\n".join(sorted(CL[CL.fold == fold].file)) + "\n")
    rows = []
    for f, g in LA.groupby("file"):
        rows.append(dict(file_name=f, objects=dict(categories=g.cls.astype(int).tolist(),
                                                     bbox=[[float(r.x1), float(r.y1), float(r.x2 - r.x1), float(r.y2 - r.y1)]
                                                           for r in g.itertuples()])))
    pd.DataFrame(rows).to_parquet(out / "g4c" / "db_test.parquet")
    (out / "s2" / "s2b" / "sealed_b" / "rdd2022_scoring").mkdir(parents=True, exist_ok=True)
    # ---------------- external collections
    blocks, dec = [], {}
    for c, (n0, dens, four) in EXT.items():
        n = int(n0 * sc)
        files, Lr = [], []
        pr = {r: [] for r in PC.RUNS_A}; pb = {m: [] for m in PC.ARCH_B}
        props = []
        b = 0
        Wc, Hc = (1920, 1080) if c != "pdd2026" else (1024, 1024)
        for i in range(n):
            f = f"{c}__images__img_{i:05d}.jpg"
            files.append(f)
            if rng.random() < 0.7:
                b += 1
            blocks.append(dict(collection=c, file=f, block=f"{c}_{b}"))
            raw = make_image_boxes(rng, rng.poisson(dens), Wc, Hc, pothole_only=not four)
            bx = [(cc, x1, y1, x2, y2, EXT_DIFF[c] + rng.normal(0, 1)) for cc, x1, y1, x2, y2 in raw]
            for (cc, x1, y1, x2, y2, e) in bx:
                Lr.append(dict(file=f, raw_class=str(cc), mapped=f"D{[0, 10, 20, 40][cc]:02d}", cls=cc, x1=x1, y1=y1,
                               x2=x2, y2=y2))
            for r in PC.RUNS_A:
                for (cc, conf, x1, y1, x2, y2) in predict(rng, bx, Wc, Hc, skill[r], 0.0):
                    pr[r].append((r, c, f, cc, conf, x1, y1, x2, y2))
            for m in PC.ARCH_B:
                for (cc, conf, x1, y1, x2, y2) in predict(rng, bx, Wc, Hc, skill_b[m], 0.0):
                    pb[m].append((m, c, f, cc, conf, x1, y1, x2, y2))
            props.append(dict(collection=c, file=f, read_ok=True, src_w=Wc, src_h=Hc, scale_to_input=640 / max(Wc, Hc),
                              brightness_L=50.0, contrast_L=20.0, sharpness_lapvar=100.0, edge_density=0.1,
                              grey_entropy=7.0, knn10_cosine_dist=float(rng.uniform(0.05, 0.3)), nn1_cosine_dist=0.02))
        (out / "s2" / "g4" / f"pool_final_{c}.txt").write_text("\n".join(files) + "\n")
        L = pd.DataFrame(Lr, columns=["file", "raw_class", "mapped", "cls", "x1", "y1", "x2", "y2"])
        L.to_parquet(out / "s2" / "g4" / f"labels_final_{c}.parquet")
        (out / "s2" / "s2a" / "sealed_ext" / c).mkdir(parents=True, exist_ok=True)
        (out / "s2" / "s2b" / "sealed_b" / c).mkdir(parents=True, exist_ok=True)
        for r in PC.RUNS_A:
            pd.DataFrame(pr[r], columns=["run", "collection", "file", "cls", "conf", "x1", "y1", "x2", "y2"]).to_parquet(
                out / "s2" / "s2a" / "sealed_ext" / c / f"{r}.parquet")
        for m in PC.ARCH_B:
            pd.DataFrame(pb[m], columns=["model", "pool", "file", "cls", "conf", "x1", "y1", "x2", "y2"]).to_parquet(
                out / "s2" / "s2b" / "sealed_b" / c / f"{m}.parquet")
        pd.DataFrame(props).to_parquet(out / "s2" / "ext_props" / f"{c}.parquet")
        cnt = L.groupby("mapped").size().to_dict()
        dec[c] = dict(images_after=n, boxes_after={k: int(cnt.get(k, 0)) for k in ["D00", "D10", "D20", "D40"]},
                      blocks=b, eligible_four_class=four, eligible_pothole=True)
    for m in PC.ARCH_B:
        pd.DataFrame(predb[m], columns=["model", "pool", "file", "cls", "conf", "x1", "y1", "x2", "y2"]).to_parquet(
            out / "s2" / "s2b" / "sealed_b" / "rdd2022_scoring" / f"{m}.parquet")
    pd.DataFrame(blocks).to_csv(out / "s2" / "g4" / "blocks.csv", index=False)
    json.dump(dict(collections=dec, primary_in_force="four_class"), open(out / "s2" / "g4" / "g4_g6_decision.json", "w"))
    fc = {t: dict(F0_dev_recall=0.55, forecasts={c: dict(F0=0.55, F1=float(rng.uniform(0.1, 0.7)),
                                                           F2=float(rng.uniform(0.4, 0.7)), images=EXT[c][0])
                                                 for c in EXT if (t == "pothole" or EXT[c][2])})
          for t in ("four_class", "pothole")}
    json.dump(dict(targets=fc), open(out / "forecast" / "forecasts.json", "w"))
    json.dump(dict(skill=skill, skill_b=skill_b), open(out / "truth.json", "w"))
    print(f"synthetic root at {out}: {len(LA)} internal boxes, {len(blocks)} external images")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--lock_scripts", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    main(ap.parse_args())

#!/usr/bin/env python3
"""Experiment 5, local adaptation (tasks 02 and 07).

  plan      write the run table (size curve, ranking check, ceiling) from params.exp5
  samples   for every (collection, draw): pilot, training and audit image lists from the calibration part only
            (whole blocks; pilot first, then further blocks; the post-adaptation audit is disjoint from both)
  train     run the rows of the run table on the card: fine-tune, then predict with the registered validation
            pipeline on the held-out part, the calibration part, S1 and S2 (and the unadapted checkpoint on the
            held-out part, once per checkpoint and collection, in the same session)
  timing    task 02: time one fine-tune per given checkpoint on n calibration images, no scoring
Held-out labels are never read here.
"""
import argparse, json, shutil, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr_common as PC  # noqa: E402


def plan(a, params):
    p = params["exp5"]; sel = params["selected"]
    rows = []
    for c in PC.EXT_FOUR:
        for n in p["sizes"]:
            for d in range(p["draws_size_curve"]):
                rows.append(dict(kind="size_curve", checkpoint=sel, collection=c, size=n, draw=d))
        for ck in PC.RUNS_A:
            rows.append(dict(kind="ranking", checkpoint=ck, collection=c, size=p["ranking_size"], draw=0))
        rows.append(dict(kind="ceiling", checkpoint=sel, collection=c, size="all", draw=0))
    T = pd.DataFrame(rows).drop_duplicates(["checkpoint", "collection", "size", "draw"]).reset_index(drop=True)
    T.insert(0, "run_id", [f"r{i:03d}" for i in range(len(T))])
    T.to_csv(Path(a.out) / "exp5_run_table.csv", index=False)
    PC.log(f"run table: {len(T)} runs")


def block_order(files, blocks, seed, cidx, draw):
    ub = sorted({blocks[f] for f in files})
    rng = np.random.default_rng(np.random.SeedSequence([seed, 5, cidx, draw]))
    return [ub[i] for i in rng.permutation(len(ub))]


def samples(a, params):
    PC.locked_scripts(a.root); R = PC.Root(a.root)
    p = params["exp5"]; out = Path(a.out) / "samples"; out.mkdir(parents=True, exist_ok=True)
    T = pd.read_csv(Path(a.run_table))
    for c in PC.EXT_FOUR:
        cal, _ = PC.read_split(a.rules, c)
        B = R.blocks(c); L = R.labels_external_subset(c, cal)
        nbox = L.groupby("file").size().to_dict()
        members = {}
        for f in cal:
            members.setdefault(B[f], []).append(f)
        for d in sorted(T[T.collection == c].draw.unique()):
            order = block_order(cal, B, params["seed"], PC.EXT_ALL.index(c), int(d))
            pilot, audit, rest = [], [], []
            cnt = 0; i = 0
            while i < len(order) and cnt < p["pilot_boxes"]:
                pilot.append(order[i]); cnt += sum(nbox.get(f, 0) for f in members[order[i]]); i += 1
            cnt = 0
            while i < len(order) and cnt < p["audit_boxes"]:
                audit.append(order[i]); cnt += sum(nbox.get(f, 0) for f in members[order[i]]); i += 1
            rest = order[i:]
            train_order = pilot + rest
            rec = dict(collection=c, draw=int(d), pilot_blocks=pilot, audit_blocks=audit, train_order=train_order,
                       pilot_images=[f for b in pilot for f in members[b]],
                       audit_images=[f for b in audit for f in members[b]])
            for _, r in T[(T.collection == c) & (T.draw == d)].iterrows():
                n = r["size"]
                imgs = []
                for b in train_order:
                    if n != "all" and len(imgs) >= int(n):
                        break
                    imgs += members[b]
                rec[f"train_{n}"] = imgs
            assert not set(rec["audit_images"]) & {f for k_, v in rec.items() if k_.startswith("train_") for f in v}
            rec["shortfall"] = dict(audit_boxes=int(sum(nbox.get(f, 0) for f in rec["audit_images"])) < p["audit_boxes"],
                                    **{f"train_{n}": (n != "all" and len(rec[f"train_{n}"]) < int(n))
                                       for n in T[T.collection == c]["size"].astype(str).unique()})
            if any(rec["shortfall"].values()):
                PC.log(f"WARNING {c} draw {d}: shortfall {rec['shortfall']}")
            PC.jdump(rec, out / f"{c}_draw{int(d)}.json")
    PC.log("samples written")


def yolo_dataset(R, coll, files, root_imgs, work, name, L=None, S=None):
    """Images (links) and four-class labels of calibration images, normalized by the size after EXIF orientation.
    L (labels) and S (file -> src_w, src_h) may be given; otherwise they come from the locked root."""
    d = Path(work) / name
    if d.exists():
        shutil.rmtree(d)
    (d / "images").mkdir(parents=True); (d / "labels").mkdir()
    L = R.labels_external_subset(coll, files) if L is None else L[L.file.isin(set(files))]
    S = R.ext_props(coll).set_index("file")[["src_w", "src_h"]] if S is None else S
    g = L.groupby("file")
    for f in files:
        (d / "images" / f).symlink_to((Path(root_imgs) / coll / "pool" / "images" / f).resolve())
        w, h = S.loc[f]
        lines = []
        if f in g.groups:
            for r in g.get_group(f).itertuples():
                cx, cy = (r.x1 + r.x2) / 2 / w, (r.y1 + r.y2) / 2 / h
                bw, bh = (r.x2 - r.x1) / w, (r.y2 - r.y1) / h
                lines.append(f"{int(r.cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (d / "labels" / (Path(f).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
    import yaml
    y = d / "data.yaml"
    yaml.safe_dump(dict(path=str(d), train="images", val="images",
                        names={0: "longitudinal_crack", 1: "transverse_crack", 2: "alligator_crack", 3: "pothole"}),
                   open(y, "w"))
    return y


def finetune(weights, data, recipe, work, name, device):
    from ultralytics import YOLO
    m = YOLO(weights)
    t0 = time.time()
    extra = {k: recipe[k] for k in ("warmup_epochs", "close_mosaic", "weight_decay", "cos_lr") if k in recipe}
    m.train(data=str(data), epochs=recipe["epochs"], lr0=recipe["lr0"], imgsz=recipe["imgsz"], batch=recipe["batch"],
            optimizer=recipe["optimizer"], seed=recipe["seed"], deterministic=True, val=False, plots=False,
            freeze=recipe.get("freeze", 0) or None, project=str(Path(work).resolve() / "train"), name=name, exist_ok=True,
            device=device, verbose=False, workers=8, **extra)
    w = Path(work).resolve() / "train" / name / "weights" / "last.pt"
    if not w.exists():
        sys.exit(f"STOP: no last.pt after training {name}")
    return w, time.time() - t0


def predict(weights_map, pools, a, out):
    import types
    import pr_infer as I
    lists = {}
    for pn, paths in pools.items():
        lf = Path(a.work) / "lists" / f"{pn}.txt"; lf.parent.mkdir(parents=True, exist_ok=True)
        lf.write_text("\n".join(str(p) for p in paths) + "\n"); lists[pn] = lf
    ns = types.SimpleNamespace(root=a.root, pool=[f"{k}={v}" for k, v in lists.items()],
                               weights=[f"{k}={v}" for k, v in weights_map.items()], out=str(out), work=a.work,
                               device=a.device)
    I.cmd_val(ns)


def train(a, params):
    PC.locked_scripts(a.root); R = PC.Root(a.root)
    T = pd.read_csv(a.run_table); recipe = params["exp5"]["recipe"]
    out = Path(a.out); (out / "weights").mkdir(parents=True, exist_ok=True)
    rdd = {l.strip().split("/")[-1]: l.strip() for l in open(a.rdd_list) if l.strip()}
    s1 = [rdd[f] for f in R.fold("S1")]; s2 = [rdd[f] for f in R.fold("S2")]
    imgp = lambda c, fs: [str(Path(a.ext_root) / c / "pool" / "images" / f) for f in fs]  # noqa: E731
    log_p = out / "e5_runs.csv"
    done = set(pd.read_csv(log_p).run_id) if log_p.exists() else set()
    rows = pd.read_csv(log_p).to_dict("records") if log_p.exists() else []
    t_start = time.time()
    for r in T.itertuples():
        if a.only and r.run_id not in a.only:
            continue
        if r.run_id in done:
            continue
        if a.max_hours and (time.time() - t_start) / 3600 > a.max_hours:
            PC.log("time budget reached; stopping at a run boundary"); break
        cal, held = PC.read_split(a.rules, r.collection)
        smp = json.load(open(Path(a.samples) / f"{r.collection}_draw{r.draw}.json"))
        files = smp[f"train_{r.size}"]
        wsrc = Path(a.weights_dir) / f"{r.checkpoint}_best.pt"
        y = yolo_dataset(R, r.collection, files, a.ext_root, a.work, f"ds_{r.run_id}")
        w, secs = finetune(wsrc, y, recipe, a.work, r.run_id, a.device)
        wdst = out / "weights" / f"{r.run_id}_{r.checkpoint}_{r.collection}.pt"
        shutil.copy(w, wdst)
        pools = {f"{r.collection}_heldout": imgp(r.collection, held), f"{r.collection}_calibration": imgp(r.collection, cal),
                 "S1": s1, "S2": s2}
        predict({f"adapted_{r.run_id}": str(wdst)}, pools, a, out / "pred")
        base = f"unadapted_{r.checkpoint}"
        if not (out / "pred" / base / f"{r.collection}_heldout.parquet").exists():
            predict({base: str(wsrc)}, {f"{r.collection}_heldout": imgp(r.collection, held), "S1": s1, "S2": s2}, a,
                    out / "pred")
        rows.append(dict(run_id=r.run_id, kind=r.kind, checkpoint=r.checkpoint, collection=r.collection, size=r.size,
                         draw=r.draw, train_images=len(files), train_seconds=round(secs, 1),
                         weights_sha256=PC.sha(wdst), base_sha256=PC.sha(wsrc)))
        pd.DataFrame(rows).to_csv(log_p, index=False)
        shutil.rmtree(Path(a.work) / f"ds_{r.run_id}", ignore_errors=True)


def timing(a, params):
    """Task 02: fine-tune timing on n calibration images of one collection; no scoring, weights deleted."""
    PC.locked_scripts(a.root); R = PC.Root(a.root)
    cal = sorted(Path(a.cal_list).read_text().split())
    rng = np.random.default_rng(params["seed"] if params else 2026)
    files = sorted(rng.choice(cal, a.n, replace=False))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    recipe = dict(epochs=a.epochs, lr0=0.001, imgsz=640, batch=16, optimizer="AdamW", seed=0, freeze=0)
    res = []
    for ck in a.checkpoints:
        y = yolo_dataset(R, a.collection, files, a.ext_root, a.work, f"timing_{ck}")
        try:
            import torch
            cuda = torch.cuda.is_available()
            if cuda:
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            cuda = False
        w, secs = finetune(Path(a.weights_dir) / f"{ck}_best.pt", y, recipe, a.work, f"timing_{ck}", a.device)
        res.append(dict(checkpoint=ck, images=len(files), epochs=a.epochs, seconds=round(secs, 1),
                        seconds_per_epoch=round(secs / a.epochs, 2), weights_bytes=w.stat().st_size,
                        dataloader_workers=8,
                        peak_card_memory_gb=round(torch.cuda.max_memory_reserved() / 1e9, 2) if cuda else None))
        shutil.rmtree(Path(a.work) / "train" / f"timing_{ck}", ignore_errors=True)
    PC.jdump(res, out / "timing.json")
    PC.log(json.dumps(res))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "samples", "train", "timing"])
    ap.add_argument("--root", required=True); ap.add_argument("--rules", default="")
    ap.add_argument("--out", required=True); ap.add_argument("--work", default="/workspace/pr_work")
    ap.add_argument("--run_table", default=""); ap.add_argument("--samples", default="")
    ap.add_argument("--ext_root", default="/workspace/run/ext"); ap.add_argument("--rdd_list", default="")
    ap.add_argument("--weights_dir", default="/workspace/run/weights"); ap.add_argument("--device", default="0")
    ap.add_argument("--only", nargs="*", default=None); ap.add_argument("--max_hours", type=float, default=0)
    ap.add_argument("--cal_list", default=""); ap.add_argument("--collection", default="ecupdd")
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--checkpoints", nargs="*", default=["yolo12m_seed0", "yolov5s_seed0"])
    ap.add_argument("--manifest_sha", default="ef49673bdc6301009223fb7eecd0aa71be7f53eb6220395d30262b64b19228c7")
    ap.add_argument("--rules_sha", default=""); ap.add_argument("--params_sha", default="")
    ap.add_argument("--run_table_sha", default="")
    ap.add_argument("--skip_gate", action="store_true", help="synthetic tests only")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if not a.skip_gate:
        PC.verify_lock(a.root, a.manifest_sha)
        if a.cmd in ("samples", "train"):
            PC.rule_gate(a.rules, a.rules_sha, a.params_sha)
            if PC.sha(a.run_table) != a.run_table_sha:
                sys.exit("STOP: exp5_run_table.csv differs from the approved hash")
    params = PC.load_params(a.rules) if a.rules else None
    {"plan": plan, "samples": samples, "train": train, "timing": timing}[a.cmd](a, params)

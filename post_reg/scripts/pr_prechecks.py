#!/usr/bin/env python3
"""task 01, pre-checks before the rule document. Label-free and outcome-free: counts reference boxes (as gate
G4 did) but never joins a label to a prediction and computes no recall, precision or match.

  sizes      source image sizes by RDD2022 partition and external collection (from the locked E2 and external
             property tables, which were computed from the image files)
  preds      predictions available at confidence >= 0.001 against the quota at each alert rate
  tokens     file-name tokens that could carry order (session, sequence, time)
  split      draft calibration and held-out block split, box counts by class and class set
  weights    SHA-256 of downloaded best.pt files against weights_hashes.csv
"""
import argparse, re, sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr_common as PC  # noqa: E402

K_GRID = [25, 50, 75, 100, 152.49, 200, 250, 300]
K_FOUR, K_POT, K_CRACK = 152.49, 15.80, 136.69
M_GRID = [50, 100, 200, 400, 800]


def sizes(R, out):
    I = R.image_props()
    rows = []
    for c, g in I.groupby("country"):
        for (w, h), n in g.groupby(["src_w", "src_h"]).size().items():
            rows.append(dict(source="RDD2022", group=c, width=int(w), height=int(h), images=int(n)))
    for c in PC.EXT_ALL:
        E = R.ext_props(c)
        for (w, h), n in E.groupby(["src_w", "src_h"]).size().items():
            rows.append(dict(source="external", group=c, width=int(w), height=int(h), images=int(n)))
    D = pd.DataFrame(rows)
    D.to_csv(out / "image_sizes.csv", index=False)
    multi = D.groupby(["source", "group"]).size()
    flags = dict(more_than_one_size=sorted(multi[multi > 1].index.get_level_values(1).tolist()),
                 norway=D[D.group == "Norway"].to_dict("records"))
    PC.jdump(flags, out / "image_sizes_flags.json")
    PC.log(f"sizes: {len(D)} rows; groups with more than one size: {flags['more_than_one_size']}")


def preds(R, out):
    rows = []
    pools = {f: R.fold(f) for f in ("dev", "S1", "S2")}
    for run in PC.RUNS_A:
        P = R.pred_internal(run)
        for pn, files in pools.items():
            _count(rows, run, pn, P[P.file.isin(set(files))], len(files))
        for c in PC.EXT_ALL:
            files = R.pool(c)
            _count(rows, run, c, R.pred_external(run, c), len(files))
    # setup B, crack classes, at the crack rate
    gb = {f: sorted((R.r / "g4c" / "main" / f"fold_{f}.txt").read_text().split()) for f in ("dev", "S1", "S2")}
    for m in PC.ARCH_B:
        P = R.pred_b(m, "rdd2022_scoring")
        for pn, files in gb.items():
            p = P[P.file.isin(set(files)) & P.cls.isin(PC.CRACK)]
            rows.append(dict(checkpoint=m, pool=f"B_{pn}", class_set="merged_crack", k=K_CRACK,
                             available=len(p), quota=int(np.floor(K_CRACK * len(files) / 100)),
                             lowest_conf=float(p.conf.min()) if len(p) else None))
        for c in PC.EXT_FOUR:
            p = R.pred_b(m, c); p = p[p.cls.isin(PC.CRACK)]; n = len(R.pool(c))
            rows.append(dict(checkpoint=m, pool=c, class_set="merged_crack", k=K_CRACK, available=len(p),
                             quota=int(np.floor(K_CRACK * n / 100)), lowest_conf=float(p.conf.min()) if len(p) else None))
    D = pd.DataFrame(rows)
    D["enough"] = D.available >= D.quota
    D.to_csv(out / "prediction_counts.csv", index=False)
    short = D[~D.enough]
    PC.log(f"preds: {len(D)} rows; {len(short)} without enough predictions")
    PC.jdump(dict(rows_without_enough=short.to_dict("records")), out / "prediction_counts_short.json")


def _count(rows, run, pool, P, n):
    p4 = P[P.cls.isin(PC.FOUR)]
    pp = P[P.cls.isin(PC.POTHOLE)]
    pc = P[P.cls.isin(PC.CRACK)]
    lo = float(P.conf.min()) if len(P) else None
    for k in K_GRID:
        rows.append(dict(checkpoint=run, pool=pool, class_set="four_class", k=k, available=len(p4),
                         quota=int(np.floor(k * n / 100)), lowest_conf=lo))
        kp = K_POT * k / K_FOUR
        rows.append(dict(checkpoint=run, pool=pool, class_set="pothole", k=round(kp, 4), available=len(pp),
                         quota=int(np.floor(kp * n / 100)), lowest_conf=lo))
        kc = K_CRACK * k / K_FOUR
        rows.append(dict(checkpoint=run, pool=pool, class_set="merged_crack", k=round(kc, 4), available=len(pc),
                         quota=int(np.floor(kc * n / 100)), lowest_conf=lo))


TOKEN_PATTERNS = {
    "prefix_running_number": r"^(?P<prefix>.*?[A-Za-z_\-])(?P<num>\d{3,})(?:\.[A-Za-z]+)$",
    "date_time": r"(?P<date>(19|20)\d{2}[-_]?\d{2}[-_]?\d{2})[-_ T]?(?P<time>\d{2}[-_:]?\d{2}[-_:]?\d{2})?",
    "gps": r"(?P<lat>-?\d{1,2}\.\d{4,})[_,](?P<lon>-?\d{1,3}\.\d{4,})",
    "video_frame": r"(?i)(?P<video>(vid|video|clip|seq|session)[\w\-]*?)[_\-]?(f|frame)?(?P<frame>\d{2,})",
}


def tokens(R, out):
    lines = ["# File-name tokens (label-free; order is never treated as verified)", ""]
    groups = {c: R.pool(c) for c in PC.EXT_ALL}
    I = R.image_props()
    for cc, g in I.groupby("country"):
        groups[f"RDD2022 {cc}"] = sorted(g.file)
    summary = {}
    for name, files in groups.items():
        base = [f.split("__")[-1] for f in files]
        res = {}
        for t, pat in TOKEN_PATTERNS.items():
            rx = re.compile(pat)
            hits = [rx.search(b) for b in base]
            n = sum(h is not None for h in hits)
            ex = [b for b, h in zip(base, hits) if h][:3]
            extra = {}
            if t == "prefix_running_number" and n:
                pre = Counter(h.group("prefix") for h in hits if h)
                extra = dict(prefixes=len(pre), top_prefixes=pre.most_common(3))
            res[t] = dict(images=n, share=round(n / max(len(base), 1), 3), examples=ex, **extra)
        summary[name] = res
        found = [f"{t} ({v['images']} of {len(base)} images; e.g. {', '.join(v['examples'])})"
                 for t, v in res.items() if v["images"]]
        lines.append(f"- **{name}**: " + ("; ".join(found) if found else "no order-like token found"))
    lines += ["", "Regular expressions:", ""] + [f"- `{t}`: `{p}`" for t, p in TOKEN_PATTERNS.items()]
    (out / "filename_tokens.md").write_text("\n".join(lines) + "\n")
    PC.jdump(summary, out / "filename_tokens.json")
    PC.log("tokens written")


def split(R, out, share, seed):
    d = out / "split_draft"; d.mkdir(exist_ok=True)
    rows = []
    for c in PC.EXT_ALL:
        files = R.pool(c); B = R.blocks(c)
        cal, held = PC.make_split(files, B, share, seed)
        (d / f"{c}_calibration.txt").write_text("\n".join(cal) + "\n")
        (d / f"{c}_heldout.txt").write_text("\n".join(held) + "\n")
        L = R.labels_external(c)
        for part, fs in (("calibration", cal), ("heldout", held)):
            l = L[L.file.isin(set(fs))]
            cnt = l.cls.value_counts().to_dict()
            row = dict(collection=c, part=part, images=len(fs), blocks=len({B[f] for f in fs}),
                       D00=int(cnt.get(0, 0)), D10=int(cnt.get(1, 0)), D20=int(cnt.get(2, 0)), D40=int(cnt.get(3, 0)))
            row["four_class"] = row["D00"] + row["D10"] + row["D20"] + row["D40"]
            row["merged_crack"] = row["D00"] + row["D10"] + row["D20"]
            row["pothole"] = row["D40"]
            for cs in ("four_class", "merged_crack", "pothole"):
                for m in M_GRID:
                    row[f"{cs}_M{m}_feasible"] = row[cs] >= m
            rows.append(row)
    D = pd.DataFrame(rows)
    D.to_csv(out / "split_draft_counts.csv", index=False)
    PC.jdump(dict(share=share, seed=seed, unit="block", balance="images"), d / "split_rule.json")
    PC.log("draft split written")


def weights(out, wdir, table):
    T = pd.read_csv(table)
    rows = []
    for r in T.itertuples():
        p = Path(wdir) / f"{r.checkpoint}_best.pt"
        got = PC.sha(p) if p.exists() else None
        rows.append(dict(checkpoint=r.checkpoint, file=p.name, expected=r.best_sha256, got=got, ok=got == r.best_sha256))
    D = pd.DataFrame(rows)
    D.to_csv(out / "weights_check.csv", index=False)
    if not D.ok.all():
        sys.exit(f"STOP: weight hashes differ: {D[~D.ok].checkpoint.tolist()}")
    PC.log("weights: all 20 best.pt match")


def main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if not a.skip_verify:
        PC.verify_lock(a.root, a.manifest_sha)
    PC.locked_scripts(a.root)
    R = PC.Root(a.root)
    PC.env_record(out)
    steps = a.steps or ["sizes", "preds", "tokens", "split"]
    for s in steps:
        if s == "sizes":
            sizes(R, out)
        elif s == "preds":
            preds(R, out)
        elif s == "tokens":
            tokens(R, out)
        elif s == "split":
            split(R, out, a.share, a.seed)
        elif s == "weights":
            weights(out, a.weights, a.weights_table)
    PC.write_manifest(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--manifest_sha", default="ef49673bdc6301009223fb7eecd0aa71be7f53eb6220395d30262b64b19228c7")
    ap.add_argument("--skip_verify", action="store_true", help="synthetic tests only")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", nargs="*", choices=["sizes", "preds", "tokens", "split", "weights"])
    ap.add_argument("--share", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--weights", default="")
    ap.add_argument("--weights_table", default="")
    main(ap.parse_args())

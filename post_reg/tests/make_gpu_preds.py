#!/usr/bin/env python3
"""Synthetic stand-ins for the A40 prediction directories (tests only): sealed predictions with jittered
confidences, written in the layouts that tasks 06 and 07 produce, so the scoring code can be exercised
without a graphics card."""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pr_common as PC  # noqa: E402


def jitter(P, rng, s=0.05):
    P = P.copy()
    P["conf"] = np.clip(P.conf.to_numpy() + rng.normal(0, s, len(P)), 0.0011, 0.999)
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--rules", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--runs", default="")
    a = ap.parse_args()
    R = PC.Root(a.root); out = Path(a.out); rng = np.random.default_rng(0)
    folds = {f: set(R.fold(f)) for f in ("S1", "S2", "dev")}
    # task 06: best_<run>/<pool>.parquet and last_<run>/<pool>.parquet
    for run in PC.RUNS_A:
        PI = R.pred_internal(run)
        for tag in ("best", "last"):
            d = out / "e6" / f"{tag}_{run}"; d.mkdir(parents=True, exist_ok=True)
            for f in ("S1", "S2"):
                jitter(PI[PI.file.isin(folds[f])], rng).to_parquet(d / f"{f}.parquet")
            for c in PC.EXT_FOUR:
                jitter(R.pred_external(run, c), rng).to_parquet(d / f"{c}.parquet")
    # task 06: experiment 15 variants, <det>/<pool>__<variant>.parquet
    clean = folds["S1"] | folds["S2"] | folds["dev"]
    for det in ("yolo12m_seed0", "yolo12s_seed0"):
        PI = R.pred_internal(det); d = out / "e15" / det; d.mkdir(parents=True, exist_ok=True)
        for c in ("Norway", "United_States", "Japan"):
            P = PI[PI.file.isin(clean) & PI.file.str.startswith(c)]
            vs = ["untiled_640", "crop", "tile_2x2", "input_1280"] if c == "Norway" else ["untiled_640"]
            for v in vs:
                jitter(P, rng).to_parquet(d / f"{c}__{v}.parquet")
            if c != "Norway":
                jitter(P, rng).to_parquet(d / f"{c}_down__downscaled.parquet")
    # task 07: adapted_<run_id>/<coll>_{heldout,calibration}.parquet, S1, S2; unadapted_<ckpt>/same
    if a.runs:
        T = pd.read_csv(a.runs); d5 = out / "e5"
        for ck in T.checkpoint.unique():
            for c in T[T.checkpoint == ck].collection.unique():
                cal, held = PC.read_split(a.rules, c)
                PE = R.pred_external(ck, c); dd = d5 / f"unadapted_{ck}"; dd.mkdir(parents=True, exist_ok=True)
                PE[PE.file.isin(set(held))].to_parquet(dd / f"{c}_heldout.parquet")
                PE[PE.file.isin(set(cal))].to_parquet(dd / f"{c}_calibration.parquet")
            PI = R.pred_internal(ck)
            for f in ("S1", "S2"):
                PI[PI.file.isin(folds[f])].to_parquet(d5 / f"unadapted_{ck}" / f"{f}.parquet")
        for r in T.itertuples():
            cal, held = PC.read_split(a.rules, r.collection)
            PE = R.pred_external(r.checkpoint, r.collection); PI = R.pred_internal(r.checkpoint)
            dd = d5 / f"adapted_{r.run_id}"; dd.mkdir(parents=True, exist_ok=True)
            boost = PE.copy(); boost["conf"] = np.clip(boost.conf * 1.1, 0.0011, 0.999)
            boost[boost.file.isin(set(held))].to_parquet(dd / f"{r.collection}_heldout.parquet")
            boost[boost.file.isin(set(cal))].to_parquet(dd / f"{r.collection}_calibration.parquet")
            for f in ("S1", "S2"):
                jitter(PI[PI.file.isin(folds[f])], rng).to_parquet(dd / f"{f}.parquet")
    print(json.dumps({"written": str(out)}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Scoring of the experiments whose predictions were made on the A40 (processor-only; tasks 06 and 07).

  scales  experiment 15 reverse test: downscale factor per partition = median longest box side at model input in
          Norway's clean holdout divided by the same median in the partition's clean holdout (E2 object table)
  e6      registered primary analysis on the new best.pt predictions and on the last.pt predictions (same folds,
          quota, bootstrap and Fieller settings), and the paired difference of external mean recall per architecture
  e15     recall at the four-class quota per variant (Norway; reverse test on United States and Japan), paired
          cluster-bootstrap differences against the untiled run, and recall at confidence 0.25 per variant
  e5      local adaptation: adapted against unadapted on the held-out part (quota and recalibrated threshold),
          optimism, forgetting, size curve, ranking check, ceilings, label cost
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr_common as PC  # noqa: E402


def clean_files(R):
    return sorted(set(R.fold("dev")) | set(R.fold("S1")) | set(R.fold("S2")))


def scales(a, params):
    R = PC.Root(a.root)
    O = pd.read_parquet(Path(a.root) / "e2" / "object_props.parquet")
    O = O[O.file.isin(set(clean_files(R)))]
    med = O.groupby("country").long_side_input.median()
    s = {c: float(min(1.0, med["Norway"] / med[c])) for c in ("United_States", "Japan")}
    PC.jdump(dict(scales={f"{c}_down": v for c, v in s.items()}, medians=med.to_dict()), Path(a.out) / "e15_scales.json")
    json.dump({f"{c}_down": v for c, v in s.items()}, open(Path(a.out) / "e15_scales_flat.json", "w"))
    PC.log(f"reverse-test scales {s}")


def e6(a, params):
    R = PC.Root(a.root); PC.locked_scripts(a.root)
    pdir = Path(a.pred)
    pools = {"S1": PC.internal_pool(R, "S1", R.fold("S1")), "S2": PC.internal_pool(R, "S2", R.fold("S2"))}
    for c in PC.EXT_FOUR:
        pools[c] = PC.external_pool(R, c)
    res, vals, recs, boots = {}, {}, [], []
    for tag in ("best", "last"):
        pred = lambda r, p, tag=tag: pd.read_parquet(pdir / f"{tag}_{r}" / f"{p}.parquet", columns=PC.PRED_COLS)  # noqa
        pt, bt, det = PC.panel(PC.RUNS_A, pred, pools, PC.FOUR, params["k"]["four_class"], params["bootstrap"]["draws"],
                               params["seed"])
        r = PC.ratio_from_panel(pt, bt, PC.GROUPS_A, "S1", "S2", PC.EXT_FOUR)
        r["each_collection"] = {c: PC.ratio_from_panel(pt, bt, PC.GROUPS_A, "S1", "S2", [c]) for c in PC.EXT_FOUR}
        res[tag] = r
        vals[tag] = (pt, bt)
        recs.append(pd.DataFrame(det).assign(weights=tag)); boots.append(PC.boot_long(bt, weights=tag))
        PC.jdump(r, Path(a.out) / f"e6_{tag}.json")
    rows = []
    for arch, runs in PC.GROUPS_A.items():
        d_pt = np.mean([np.mean([vals["last"][0][(r, c)] - vals["best"][0][(r, c)] for r in runs]) for c in PC.EXT_FOUR])
        d_bt = np.mean([np.mean([vals["last"][1][(r, c)] - vals["best"][1][(r, c)] for r in runs], 0)
                        for c in PC.EXT_FOUR], 0)
        rows.append(dict(architecture=arch, last_minus_best_external_mean=float(d_pt), ci95=PC.pct(d_bt)))
    pd.DataFrame(rows).to_csv(Path(a.out) / "e6_paired.csv", index=False)
    pd.concat(recs, ignore_index=True).to_csv(Path(a.out) / "e6_recalls.csv", index=False)
    pd.concat(boots, ignore_index=True).to_parquet(Path(a.out) / "e6_bootstrap.parquet")
    PC.log(f"e6: best {res['best']['outcome']}, last {res['last']['outcome']}")


def e15(a, params):
    R = PC.Root(a.root); PC.locked_scripts(a.root)
    pdir = Path(a.pred); k = params["k"]["four_class"]; B = params["bootstrap"]["draws"]
    clean = clean_files(R)
    by_c = {c: [f for f in clean if f.startswith(c)] for c in ("Norway", "United_States", "Japan")}
    sets = [("Norway", "Norway", list(params["exp15"]["variants"])),
            ("United_States", "United_States", ["untiled_640"]), ("United_States", "United_States_down", ["downscaled"]),
            ("Japan", "Japan", ["untiled_640"]), ("Japan", "Japan_down", ["downscaled"])]
    rows, diffs, boots = [], [], []
    for det in a.detectors:
        for country in ("Norway", "United_States", "Japan"):
            runs = [(pn, v) for c, pn, vs in sets if c == country for v in vs]
            pool = {country: PC.internal_pool(R, country, by_c[country])}
            pred = lambda rv, p, det=det: pd.read_parquet(pdir / det / f"{rv.split('|')[0]}__{rv.split('|')[1]}.parquet",  # noqa
                                                          columns=PC.PRED_COLS)
            names = [f"{pn}|{v}" for pn, v in runs]
            pt, bt, dd = PC.panel(names, pred, pool, PC.FOUR, k, B, params["seed"])
            boots.append(PC.boot_long(bt, detector=det, country=country))
            base = f"{country}|untiled_640"
            for n_ in names:
                P = pred(n_, country); P = P[P.cls.isin(PC.FOUR) & (P.conf >= 0.25)]
                lm, _, _ = PC.match_flags(P, pool[country].L[pool[country].L.cls.isin(PC.FOUR)])
                rows.append(dict(detector=det, country=country, variant=n_.split("|")[1], pool=n_.split("|")[0],
                                 recall_at_quota=pt[(n_, country)], ci95=PC.pct(bt[(n_, country)]),
                                 recall_at_conf_025=float(lm.mean()) if len(lm) else None,
                                 note_conf_025="not matched in alert volume"))
                if n_ != base:
                    d = bt[(n_, country)] - bt[(base, country)]
                    diffs.append(dict(detector=det, country=country, variant=n_.split("|")[1],
                                      minus_untiled=pt[(n_, country)] - pt[(base, country)], ci95=PC.pct(d)))
    V = pd.DataFrame(rows)
    V[V.country == "Norway"].to_csv(Path(a.out) / "e15_variants.csv", index=False)
    V[V.country != "Norway"].to_csv(Path(a.out) / "e15_reverse.csv", index=False)
    pd.concat(boots, ignore_index=True).to_parquet(Path(a.out) / "e15_bootstrap.parquet")
    pd.DataFrame(diffs).to_csv(Path(a.out) / "e15_paired.csv", index=False)
    PC.log("e15 done")


def e5(a, params):
    R = PC.Root(a.root); PC.locked_scripts(a.root)
    pdir = Path(a.pred); k = params["k"]["four_class"]; B = params["bootstrap"]["draws"]
    runs = pd.read_csv(a.runs)
    rows, boots = [], []
    for r in runs.itertuples():
        cal, held = PC.read_split(a.rules, r.collection)
        smp = json.load(open(Path(a.samples) / f"{r.collection}_draw{r.draw}.json"))
        train = set(smp[f"train_{r.size}"])
        ad, un = f"adapted_{r.run_id}", f"unadapted_{r.checkpoint}"
        rd = lambda name, p: pd.read_parquet(pdir / name / f"{p}.parquet", columns=PC.PRED_COLS)  # noqa: E731
        hp = {"held": PC.external_pool(R, r.collection, held)}
        pred = lambda name, p: rd(name, f"{r.collection}_heldout")  # noqa: E731
        pt, bt, _ = PC.panel([ad, un], pred, hp, PC.FOUR, k, B, params["seed"])
        gain = bt[(ad, "held")] - bt[(un, "held")]
        boots.append(pd.DataFrame(dict(run_id=r.run_id, draw=np.arange(len(gain)), recall_adapted=bt[(ad, "held")],
                                       recall_unadapted=bt[(un, "held")], gain=gain)))
        # recalibrated threshold: adapted predictions on calibration images not used for training
        rest = [f for f in cal if f not in train]
        pc = rd(ad, f"{r.collection}_calibration")
        pc = pc[pc.file.isin(set(rest)) & pc.cls.isin(PC.FOUR)].conf.sort_values(ascending=False).to_numpy()
        t = PC.threshold_for_rate(pc, len(rest), k) if rest else np.nan
        recal_ok = bool(np.isfinite(t))  # false when no unused calibration images or their quota is zero
        TR = PC.ThresholdRun(rd(ad, f"{r.collection}_heldout"), R.labels_external(r.collection), held, PC.FOUR)
        al, m, x = TR.per_image_at(t if recal_ok else np.inf)
        # optimism: adapted recall on its own training images at the same threshold
        TT = PC.ThresholdRun(rd(ad, f"{r.collection}_calibration"), R.labels_external(r.collection), sorted(train), PC.FOUR)
        _, mt, xt = TT.per_image_at(t if recal_ok else np.inf)
        row = dict(run_id=r.run_id, kind=r.kind, checkpoint=r.checkpoint, collection=r.collection, size=r.size,
                   draw=r.draw, recall_adapted=pt[(ad, "held")], recall_unadapted=pt[(un, "held")],
                   gain=pt[(ad, "held")] - pt[(un, "held")], gain_ci95=PC.pct(gain),
                   recalibrated_threshold=t if recal_ok else None, recalibration_possible=recal_ok,
                   unused_calibration_images=len(rest),
                   recall_at_recalibrated=(m.sum() / x.sum() if x.sum() else None) if recal_ok else None,
                   alerts_per_100_at_recalibrated=100 * al.sum() / len(held) if recal_ok else None,
                   recall_on_training_images=(mt.sum() / xt.sum() if xt.sum() else None) if recal_ok else None,
                   pilot_images=len(smp["pilot_images"]), topup_images=len(train - set(smp["pilot_images"])),
                   audit_images=len(smp["audit_images"]))
        for f in ("S1", "S2"):
            ip = {f: PC.internal_pool(R, f, R.fold(f))}
            p2, _, _ = PC.panel([ad, un], lambda name, p: rd(name, p), ip, PC.FOUR, k, 1, params["seed"])
            row[f"forgetting_{f}"] = p2[(ad, f)] - p2[(un, f)]
        rows.append(row)
    D = pd.DataFrame(rows); D.to_csv(Path(a.out) / "e5_scores.csv", index=False); D.to_parquet(Path(a.out) / "e5_scores.parquet")
    pd.concat(boots, ignore_index=True).to_parquet(Path(a.out) / "e5_bootstrap.parquet")
    sc = D[D.kind == "size_curve"].copy()
    for c_ in ("gain", "recall_adapted", "recall_unadapted", "recall_on_training_images", "forgetting_S1", "forgetting_S2"):
        sc[c_] = pd.to_numeric(sc[c_], errors="coerce")
    if len(sc):
        sc.groupby(["collection", "size"]).agg(runs=("run_id", "size"), mean_gain=("gain", "mean"),
                                              min_gain=("gain", "min"), max_gain=("gain", "max"),
                                              mean_recall_adapted=("recall_adapted", "mean"),
                                              recall_unadapted=("recall_unadapted", "first"),
                                              mean_recall_on_training_images=("recall_on_training_images", "mean"),
                                              mean_forgetting_S1=("forgetting_S1", "mean"),
                                              mean_forgetting_S2=("forgetting_S2", "mean")) \
            .reset_index().to_csv(Path(a.out) / "e5_summary.csv", index=False)
    # ranking check and ceilings per collection
    rk = []
    from scipy.stats import spearmanr
    for c in PC.EXT_FOUR:
        g = D[(D.collection == c) & (D.kind.isin(["ranking"]) | ((D.kind == "size_curve") & (D["size"].astype(str) == str(params["exp5"]["ranking_size"])) & (D.draw == 0)))]
        g = g.drop_duplicates("checkpoint").set_index("checkpoint")
        if set(PC.RUNS_A) <= set(g.index):
            un = [np.mean([g.loc[r, "recall_unadapted"] for r in PC.GROUPS_A[x]]) for x in PC.ARCH_A]
            ad = [np.mean([g.loc[r, "recall_adapted"] for r in PC.GROUPS_A[x]]) for x in PC.ARCH_A]
            rk.append(dict(collection=c, spearman=float(spearmanr(un, ad).correlation),
                           top_unadapted=PC.ARCH_A[int(np.argmax(un))], top_adapted=PC.ARCH_A[int(np.argmax(ad))],
                           best_unadapted_choice_recall=float(g.recall_unadapted.max()),
                           best_unadapted_choice=str(g.recall_unadapted.idxmax()),
                           ceiling_full_calibration=float(D[(D.collection == c) & (D.kind == "ceiling")].recall_adapted.max())
                           if ((D.collection == c) & (D.kind == "ceiling")).any() else None))
    pd.DataFrame(rk).to_csv(Path(a.out) / "e5_ranking_ceilings.csv", index=False)
    PC.log("e5 done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scales", "e6", "e15", "e5"])
    ap.add_argument("--root", required=True); ap.add_argument("--rules", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--pred", default="")
    ap.add_argument("--runs", default=""); ap.add_argument("--samples", default="")
    ap.add_argument("--detectors", nargs="*", default=["yolo12m_seed0", "yolo12s_seed0"])
    ap.add_argument("--manifest_sha", default="ef49673bdc6301009223fb7eecd0aa71be7f53eb6220395d30262b64b19228c7")
    ap.add_argument("--rules_sha", default=""); ap.add_argument("--params_sha", default="")
    ap.add_argument("--skip_gate", action="store_true", help="synthetic tests only")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if not a.skip_gate:
        PC.verify_lock(a.root, a.manifest_sha); PC.rule_gate(a.rules, a.rules_sha, a.params_sha)
    params = PC.load_params(a.rules)
    PC.locked_scripts(a.root)
    {"scales": scales, "e6": e6, "e15": e15, "e5": e5}[a.cmd](a, params)
    PC.write_manifest(a.out)

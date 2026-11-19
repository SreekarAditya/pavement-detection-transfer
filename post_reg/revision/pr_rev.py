#!/usr/bin/env python3
"""Run the frozen-chain revision checks on verified inputs."""
import argparse
from contextlib import contextmanager
import json
import math
import os
import shutil
from pathlib import Path
import sys
import time
from types import FunctionType, SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import norm
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import pr_common as PC
import pr_exp1 as E1
import pr_exp18 as E18
from pr_exp2 import block_samples

KEYS = ["detector", "collection", "class_set", "arm"]


def need(condition, message):
    if not condition:
        raise SystemExit(f"STOP: {message}")


def checked(path, expected):
    need(isinstance(expected, str) and len(expected) == 64, f"Missing SHA-256 for {path.name}")
    need(path.is_file() and PC.sha(path) == expected, f"{path.name} SHA-256 differs")


def frozen(a):
    checked(a.revision.with_name("RULES_revision.md"), os.environ.get("REVISION_SHA"))
    checked(a.revision, os.environ.get("REVISION_PARAMS_SHA"))
    revision = yaml.safe_load(a.revision.read_text())
    items = revision["frozen"]
    need(items["evaluator_tag"] == "qualification-evaluator-v1", "Evaluator tag differs")
    paths = {
        "rules_sha": a.rules / "RULES.md", "params_sha": a.rules / "params.yaml",
        "constants_sha": a.constants or a.rules / "exp1_constants.yaml",
        "registered_summary_sha": a.registered,
        "exp18_commitments_sha": HERE / "exp18_commitments.json",
    }
    for key, path in paths.items():
        need(path is not None, f"Missing path for {key}")
        checked(path, items[key])
    for key, label in (("rules_sha", "RULES_SHA"), ("params_sha", "PARAMS_SHA"), ("constants_sha", "CONSTANTS_SHA")):
        need(os.environ.get(label, items[key]) == items[key], f"{label} differs from the frozen revision value")
    return revision, PC.load_params(a.rules), yaml.safe_load(paths["constants_sha"].read_text())


def frozen_seed(params, seed):
    return {**params, "seed": seed}


def make_lists(root, names, seed, directory):
    split = directory / "split"
    split.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for coll in names:
        pool, blocks = root.pool(coll), root.blocks(coll)
        cal, held = PC.make_split(pool, blocks, 0.5, seed, balance="images")
        need(cal and held and set(cal).isdisjoint(held) and set(cal) | set(held) == set(pool),
             f"Invalid split for {coll}, seed {seed}")
        need({blocks[f] for f in cal}.isdisjoint(blocks[f] for f in held), f"Block crosses split for {coll}")
        for part, files in (("calibration", cal), ("heldout", held)):
            path = split / f"{coll}_{part}.txt"
            path.write_text("\n".join(files) + "\n")
            hashes[path.name] = PC.sha(path)
    (split / "hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")


def networks(params):
    return [(coll, cs) for cs in params["exp1"]["class_sets"]
            for coll in (PC.EXT_ALL if params["class_sets"][cs]["networks"] == "all_seven"
                         else params["class_sets"][cs]["networks"])]


def r2_candidates(conf, delta, pred_blocks, block_refs, floor, z):
    """Evaluate every distinct threshold using changes in the locked per-image prefix counts."""
    if not len(conf) or len(block_refs) < 2 or not block_refs.sum():
        return None
    vals, loc = np.unique(conf, return_inverse=True)
    increase = np.zeros((len(vals), len(block_refs)), dtype=np.int64)
    np.add.at(increase, (len(vals) - 1 - loc, pred_blocks), delta)
    matched = np.cumsum(increase, axis=0)
    receiver = np.asarray(block_refs, float)
    total = receiver.sum()
    recall = matched.sum(axis=1) / total
    residuals = matched - recall[:, None] * receiver
    se = np.sqrt(len(receiver) / (len(receiver) - 1) * np.square(residuals).sum(axis=1)) / total
    passing = np.flatnonzero(recall - z * se >= floor)
    return None if not len(passing) else float(vals[-1 - passing[0]])


def coverage_for(matched, refs, cal_blocks, census, heldout, levels):
    out = {}
    factor = 1 - len(refs) / cal_blocks
    for level in levels:
        z = norm.ppf(0.5 + level / 2)
        estimate, lower, upper, se = PC.ratio_ci(matched, refs, z)
        width = z * se * math.sqrt(factor)
        key = str(round(level * 100))
        for kind, left, right in (("ordinary", lower, upper), ("fpc", estimate - width, estimate + width)):
            for label, truth in (("census", census), ("heldout", heldout)):
                out[f"cover{key}_{kind}_{label}"] = float(left <= truth <= right) if np.isfinite([left, right, truth]).all() else np.nan
    out["fpc_variance_factor"] = factor
    return out


def endpoint_rows(root, coll, cs, rules, params, revision, rows, cidx):
    first = rows.iloc[0]
    n, m, floor, detector = int(first.n), int(first.M), float(first.recall_floor), first.detector
    view = E1.Net(root, coll, cs, detector, rules, params)
    blocks = np.array(sorted(set(view.cal_block)))
    code = {block: i for i, block in enumerate(blocks)}
    img_block = np.array([code[block] for block in view.cal_block])
    cal_ref = view.Tc.R.nref
    by_ref = np.bincount(img_block, weights=cal_ref, minlength=len(blocks))
    block_images = np.bincount(img_block, minlength=len(blocks))
    rng = np.random.default_rng(np.random.SeedSequence([params["seed"], 1, cidx, list(PC.CLASS_SET_CLASSES).index(cs), m]))
    decision_z = norm.ppf(0.5 + params["exp3"]["decision_level"] / 2)
    prefixes = view.Tc.R
    pos = prefixes.off[prefixes.img] + prefixes.P.groupby("img", sort=False).cumcount().to_numpy() + 1
    delta = prefixes.M[pos] - prefixes.M[pos - 1]
    pred_block = img_block[prefixes.img]
    conf = view.Tc.conf
    census_threshold = PC.threshold_for_rate(np.sort(conf)[::-1], len(view.cal), view.k)
    census_rate, _ = view.heldout(census_threshold)
    grid_n = max(params["exp2"]["n_grid"])
    draws_n = block_samples(view.cal, view.B, grid_n, len(rows), params["seed"], cidx)
    thresholds = {}
    evaluation = {}
    alternatives = {}
    floor_grid = sorted(set([floor, *params["exp3"]["floor_grid"]]))
    details = []
    for i, r in enumerate(rows.sort_values("draw").itertuples()):
        if r.threshold not in evaluation:
            _, matches, refs = view.Tc.per_image_at(r.threshold)
            evaluation[r.threshold] = (np.bincount(img_block, weights=matches, minlength=len(blocks)),
                                       matches.sum() / refs.sum() if refs.sum() else np.nan)
        match_by_block, census_recall = evaluation[r.threshold]
        perm = rng.permutation(len(blocks))
        count = min(int(np.searchsorted(np.cumsum(by_ref[perm]), m)) + 1, len(blocks))
        pilot = perm[:count]
        need(count == r.pilot_blocks and int(by_ref[pilot].sum()) == r.pilot_boxes,
             f"Pilot stream differs from frozen output for {coll}/{cs}")
        pilot_key = tuple(sorted(pilot))
        if pilot_key not in thresholds:
            block_codes = np.full(len(blocks), -1, dtype=int)
            block_codes[pilot] = np.arange(len(pilot))
            prediction_codes = block_codes[pred_block]
            selected = prediction_codes >= 0
            thresholds[pilot_key] = r2_candidates(conf[selected], delta[selected], prediction_codes[selected],
                                                   by_ref[pilot], floor, decision_z)
        t = thresholds[pilot_key]
        if t is not None and t not in alternatives:
            alternatives[t] = view.heldout(t)
        alternative_rate, alternative_recall = alternatives.get(t, (np.nan, np.nan))
        sample_n = draws_n[i]
        sample_conf = np.concatenate([view.cal_conf[j] for j in sample_n])
        t_grid = PC.threshold_for_rate(np.sort(sample_conf)[::-1], len(sample_n), view.k)
        if t_grid not in alternatives:
            alternatives[t_grid] = view.heldout(t_grid)
        grid_rate, _ = alternatives[t_grid]
        part = dict(calibration_images=len(view.cal), calibration_blocks=len(blocks), calibration_boxes=int(by_ref.sum()),
                    pilot_images=int(block_images[pilot].sum()), pilot_whole_part=count == len(blocks),
                    threshold_sample_whole_part=r.threshold_images == len(view.cal),
                    pilot_image_fraction=float(block_images[pilot].sum() / len(view.cal)),
                    pilot_box_fraction=float(by_ref[pilot].sum() / by_ref.sum()) if by_ref.sum() else np.nan,
                    threshold_fraction=r.threshold_images / len(view.cal), census_recall=census_recall,
                    r2_threshold=np.nan if t is None else t, r2_no_threshold=t is None,
                    r2_true_recall=alternative_recall, r2_rate_rel_error=alternative_rate / view.k - 1,
                    r2_floor_met=float(alternative_recall >= floor) if np.isfinite(alternative_recall) else np.nan,
                    procedure_floor_met=float(r.true_recall >= floor) if np.isfinite(r.true_recall) else np.nan,
                    r3_threshold=census_threshold, r3_rate_rel_error=census_rate / view.k - 1,
                    r3_grid_n=grid_n, r3_grid_shortfall=len(sample_n) < grid_n, r3_grid_rate_rel_error=grid_rate / view.k - 1)
        part.update(coverage_for(match_by_block[pilot], by_ref[pilot], len(blocks), census_recall,
                                 r.true_recall, revision["r4"]["intervals"]))
        for f in floor_grid:
            k = f"{f:.2f}"
            known = np.isfinite(r.true_recall)
            actual = r.true_recall >= f
            call = np.isfinite(r.stated_lo) and r.stated_lo >= f
            near = known and abs(r.true_recall - f) <= params["exp3"]["margin"]
            part[f"accuracy_{k}_near"] = float(call == actual) if near else np.nan
            part[f"accuracy_{k}_far"] = float(call == actual) if known and not near else np.nan
            part[f"false_accept_{k}"] = float(call) if known and not actual else np.nan
        details.append(part)
    details = pd.DataFrame(details)
    return pd.concat([rows.sort_values("draw").reset_index(drop=True), details], axis=1)


def chain(a, params, out, use_adapted):
    out.mkdir(parents=True, exist_ok=True)
    E1.chain(SimpleNamespace(root=a.root, rules=out, constants=a.constants, out=out, sensitivity=False,
                e5=a.e5 if use_adapted else None, samples=a.samples, pred5=a.pred5), params)
    return pd.read_parquet(out / "e1_draws.parquet")


def verify_e18(a):
    need(a.ext18 and a.pred18, "Phase A and B Experiment 18 inputs are required")
    commitments = json.loads((HERE / "exp18_commitments.json").read_text())
    inputs = commitments["inputs"]
    checked(a.constants, inputs["constants_sha256"])
    checked(a.ext18 / "exp18_eligibility.json", inputs["eligibility_sha256"])
    checked(a.pred18 / "MANIFEST.sha256", inputs["prediction_manifest_sha256"])
    for name, digest in inputs["split_lists_sha256"].items():
        checked(a.ext18 / "rules_split/split" / name, digest)
    eligible = json.loads((a.ext18 / "exp18_eligibility.json").read_text())["networks"]
    for slug, item in eligible.items():
        checked(E18.heldout_path(a.ext18, slug), item["heldout_labels_sha256"])
    for line in (a.pred18 / "MANIFEST.sha256").read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        if relative.startswith("./"):
            relative = relative[2:]
        checked(a.pred18 / relative, digest)
    return eligible, commitments


def split_command(a, revision, params, constants):
    seeds = revision["r1"]["split_seeds"]
    r1 = revision["r1"]
    original = a.seed == r1["registered_seed"]
    need(original or seeds["first"] <= a.seed <= seeds["last"], "Split seed is not listed in the revision")
    draws = a.draws if a.draws is not None else r1["registered_draws"] if original else r1["draws"]
    need(draws == (r1["registered_draws"] if original else r1["draws"]), "Draw count differs from the approved revision")
    need(params["selected"] == r1["detector"], "The selected detector differs from the frozen rules")
    seed_params = {**frozen_seed(params, a.seed), "sample_draws": draws}
    PC.log(f"Using draw seed {a.seed} in memory for {draws} draws")
    PC.locked_scripts(a.root)
    registered = PC.Root(a.root)
    pairs = networks(params)
    make_lists(registered, dict.fromkeys(c for c, _ in pairs), a.seed, a.out)
    completed = chain(a, seed_params, a.out, original)
    procedure = completed[completed.arm == "procedure"]
    details = [endpoint_rows(registered, c, cs, a.out, seed_params, revision,
                             procedure[(procedure.collection == c) & (procedure.class_set == cs)],
                             PC.EXT_ALL.index(c)).assign(post_hoc_network=False) for c, cs in pairs]
    eligible, committed = verify_e18(a)
    post = E18.Root18(a.root, a.ext18, a.pred18)
    all_names = sorted(eligible)
    post_names = [s for s in all_names if any(s.endswith("_" + tag) for tag in r1["include_post_registration_networks"])]
    need(len(post_names) == len(r1["include_post_registration_networks"]), "Both post hoc networks are required")
    post_split = a.out / "post_hoc"
    make_lists(post, post_names, a.seed, post_split)
    track = "merged_crack"
    cs = E18.TRACK_CLASS_SET[track]
    unseen = constants["values"][cs]["unseen"]
    for slug in post_names:
        need(eligible[slug]["eligible"][track], f"Committed merged-crack endpoint missing for {slug}")
        row, = [r for r in committed["procedure"] if r["collection"] == slug and r["detector"] == r1["detector"]
                and r["track"] == track and r["class_set"] == cs]
        need((int(row["n"]), int(row["M"])) == (unseen["n"], unseen["M"]), f"Committed sample counts differ for {slug}")
        cidx = 100 + all_names.index(slug)
        nd = E1.run_chain(post, slug, cs, r1["detector"], post_split, seed_params, unseen["n"], unseen["M"],
                          float(row["recall_floor"]), cidx, draws)
        nd.attrs = {}
        details.append(endpoint_rows(post, slug, cs, post_split, seed_params, revision, nd[nd.arm == "procedure"],
                                     cidx).assign(post_hoc_network=True))
    table = pd.concat(details, ignore_index=True)
    table.insert(0, "seed", a.seed)
    table.to_parquet(a.out / "revision_draws.parquet", index=False)
    metrics(table, params).to_csv(a.out / "per_split_summary.csv", index=False)


def replace_match_constant(original, iou):
    need(sum(type(v) is float and v == 0.5 for v in original.__code__.co_consts) == 1,
         "The frozen matching function no longer has exactly one matching constant")
    code = original.__code__.replace(co_consts=tuple(iou if type(v) is float and v == 0.5 else v
                                                    for v in original.__code__.co_consts))
    return FunctionType(code, original.__globals__, original.__name__, original.__defaults__, original.__closure__)


@contextmanager
def match_iou(value):
    core = sys.modules["pdt_core"]
    initializer, matcher = core.QuotaRun.__init__, core.n_matched
    if value != 0.5:
        core.QuotaRun.__init__ = replace_match_constant(initializer, value)
        core.n_matched = replace_match_constant(matcher, value)
    PC.log(f"In-memory same-class IoU gate {value}")
    try:
        yield
    finally:
        core.QuotaRun.__init__, core.n_matched = initializer, matcher


def r7_command(a, revision, params):
    need(a.iou in revision["r7"]["match_iou_sweep"], "IoU is outside the frozen sweep")
    PC.locked_scripts(a.root)
    a.out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(a.rules / "split", a.out / "split")
    with match_iou(a.iou):
        table = chain(a, {**params, "exp1": {**params["exp1"], "class_sets": ["four_class"]}}, a.out, False)
    table = table[KEYS + ["draw", "threshold", "true_recall", "stated_recall", "recall_error", "heldout_rate"]].copy()
    table.insert(0, "iou", a.iou)
    table.to_parquet(a.out / "r7_draws.parquet", index=False)
    summary = table.groupby(["iou", *KEYS], as_index=False)[["true_recall", "stated_recall", "recall_error", "heldout_rate"]].median()
    summary.to_csv(a.out / "r7_summary.csv", index=False)


def comparable(left, right):
    if pd.isna(left) and pd.isna(right):
        return True
    return bool(left == right)


def differing_cells(registered, actual, keys, check):
    registered = registered.set_index(keys)
    actual = actual.set_index(keys)
    need(registered.index.is_unique and actual.index.is_unique, f"Duplicate gate row keys for {check}")
    result = []
    for name in registered.index.union(actual.index):
        common = dict(zip(keys, name))
        if name not in registered.index or name not in actual.index:
            result.append(dict(check=check, **common, column="row",
                               registered="present" if name in registered.index else "missing",
                               recomputed="present" if name in actual.index else "missing"))
            continue
        for col in registered.columns:
            first = registered.at[name, col]
            second = actual.at[name, col] if col in actual.columns else None
            if col not in actual.columns or not comparable(first, second):
                result.append(dict(check=check, **common, column=col, registered=first, recomputed=second))
    return result


def gate_command(a):
    ref = pd.read_csv(a.registered, float_precision="round_trip")
    got = pd.read_csv(a.split / "e1_summary.csv", float_precision="round_trip")
    old = pd.read_parquet(a.registered.with_name("e1_draws.parquet"), columns=KEYS+["draw","true_recall"])
    old = old[old.class_set == "four_class"]
    now = pd.read_parquet(a.iou_dir / "r7_draws.parquet", columns=KEYS+["draw","true_recall"])
    differences = differing_cells(ref, got, KEYS, "seed_2026")
    differences += differing_cells(old, now, KEYS+["draw"], "iou_0.5")
    a.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(differences, columns=["check",*KEYS,"draw","column","registered","recomputed"]).to_csv(
        a.out / "gate_differences.csv", index=False)
    PC.jdump(dict(registered_summary_rows=len(ref), heldout_recall_rows=len(old), differences=len(differences),
                  passed=not differences), a.out / "gate_checks.json")
    need(not differences, f"Exact reproduction gate differs in {len(differences)} cells, listed in gate_differences.csv")


def metric_rows(group, params, nominal_risk):
    crit, tol = params["exp2"]["criterion_rel"], params["exp3"]["margin"]
    raw = {
        "recall_within": group.recall_error.abs().le(tol).astype(float),
        "alert_within": group.rate_rel_error.abs().le(crit).astype(float),
        "pilot_whole_part": group.pilot_whole_part.astype(float),
        "threshold_sample_whole_part": group.threshold_sample_whole_part.astype(float),
        "pilot_shortfall": group.pilot_shortfall.astype(float),
        "threshold_shortfall": group.threshold_shortfall.astype(float),
        "recall_within_pilot_not_whole": group.loc[~group.pilot_whole_part, "recall_error"].abs().le(tol).astype(float),
        "r2_no_qualifying_threshold": group.r2_no_threshold.astype(float),
        "r2_floor_met_qualifying": group.r2_floor_met,
        "r2_floor_met_all": group.r2_floor_met.where(~group.r2_no_threshold, 0.0),
        "r2_rate_rel_error": group.r2_rate_rel_error,
        "procedure_floor_met": group.procedure_floor_met,
        "procedure_rate_rel_error": group.rate_rel_error,
        "r3_rate_rel_error": group.r3_rate_rel_error.iloc[:1],
        "r3_alert_within": group.r3_rate_rel_error.iloc[:1].abs().le(crit).astype(float),
        "r3_grid_rate_rel_error": group.r3_grid_rate_rel_error,
        "r3_grid_alert_within": group.r3_grid_rate_rel_error.abs().le(crit).astype(float),
    }
    raw.update({c: group[c] for c in group if c.startswith(("cover", "false_accept_", "accuracy_"))})
    for name, values in raw.items():
        clean = values[np.isfinite(values)].astype(float)
        n = len(clean)
        target = (params["exp2"]["criterion_share"] if name in ("recall_within", "alert_within") else
                  float(name[5:7]) / 100 if name.startswith("cover") else
                  nominal_risk if name.startswith("false_accept") else
                  1 - nominal_risk if name.startswith("r2_floor_met") else np.nan)
        yield dict(metric=name, numerator=float(clean.sum()), denominator=n, value=float(clean.mean()) if n else np.nan,
                   within_split_mcse=float(clean.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan, target=target)


def metrics(table, params):
    result = []
    for key, group in table.groupby(["seed","collection","class_set","M","post_hoc_network"], sort=False):
        identity = dict(zip(["seed","collection","class_set","M","post_hoc_network"], key))
        nominal_risk = (1 - params["exp3"]["decision_level"]) / 2
        for row in metric_rows(group, params, nominal_risk):
            result.append(identity | row)
    return pd.DataFrame(result)


def interval_class(low, high, criterion):
    if not np.isfinite([low, high]).all():
        return "not resolved"
    if low >= criterion:
        return "met"
    return "not met" if high < criterion else "not resolved"


def summarize_metrics(split_metrics, revision):
    group_keys = ["collection","class_set","M","post_hoc_network","metric"]
    z = norm.ppf(0.5 + revision["monte_carlo"]["interval"] / 2)
    expected = revision["r1"]["split_seeds"]["last"] - revision["r1"]["split_seeds"]["first"] + 1
    output = []
    for key, rows in split_metrics.groupby(group_keys):
        values = rows.value.dropna()
        denominator = int(rows.denominator.sum())
        point = float(rows.numerator.sum() / denominator) if denominator else np.nan
        se = float(values.std(ddof=1) / math.sqrt(expected)) if len(values) == expected else np.nan
        lo, hi = point-z*se, point+z*se
        target = float(rows.target.iloc[0])
        row = dict(zip(group_keys, key), pooled_value=point, mcse=se, low95=lo, high95=hi,
                   denominators=denominator, splits_with_estimate=len(values), nominal_target=target,
                   per_split_median=values.median(), per_split_p10=values.quantile(.10), per_split_p90=values.quantile(.90))
        metric = row["metric"]
        if metric in ("recall_within", "alert_within"):
            row.update(status=interval_class(lo, hi, target), splits_meeting=int(values.ge(target).sum()))
        if metric.startswith("cover"):
            row["coverage_flag"] = bool(hi < target or lo > target)
        if metric.startswith("false_accept"):
            row["risk_flag"] = bool(lo > target)
        output.append(row)
    return pd.DataFrame(output)


def aggregate(a, revision, params):
    a.out.mkdir(parents=True, exist_ok=True)
    config = revision["r1"]["split_seeds"]
    seeds = range(config["first"], config["last"] + 1)
    folders = [a.splits / f"seed_{seed}" for seed in seeds]
    table = pd.concat([pd.read_csv(p / "per_split_summary.csv") for p in folders], ignore_index=True)
    need(set(table.seed) == set(seeds), "The split metrics do not cover the approved seeds")
    stats = summarize_metrics(table, revision)
    stats.to_csv(a.out / "pooled_metrics.csv", index=False)
    table.to_csv(a.out / "split_metrics.csv", index=False)
    measurements = pd.concat([pd.read_parquet(p / "revision_draws.parquet", columns=[
        "seed", "collection", "class_set", "post_hoc_network", "pilot_images", "pilot_image_fraction",
        "pilot_box_fraction", "threshold_fraction", "pilot_shortfall", "threshold_shortfall", "calibration_images",
        "calibration_boxes", "pilot_whole_part", "threshold_sample_whole_part", "n", "M"]) for p in folders])
    keys = ["collection","class_set","post_hoc_network"]
    sizes = measurements.groupby(keys)[["pilot_images","pilot_image_fraction","pilot_box_fraction","threshold_fraction"]].median().reset_index()
    limited = measurements.assign(calibration_part_shortfall=(measurements.n > measurements.calibration_images) |
                                    (measurements.M > measurements.calibration_boxes))
    flags = limited.groupby([*keys,"seed"]).calibration_part_shortfall.max().groupby(level=keys).sum()
    sizes = sizes.merge(flags.rename("splits_with_calibration_part_shortfall").reset_index(), on=keys)
    sizes.to_csv(a.out / "sampling_summary.csv", index=False)
    registered = pd.read_csv(a.gate_dir / "seed_2026/per_split_summary.csv")
    registered.to_csv(a.out / "registered_split_metrics.csv", index=False)
    r7_parts = [pd.read_csv(a.gate_dir / "iou_050/r7_summary.csv")]
    for x in revision["r7"]["match_iou_sweep"]:
        if x != revision["r7"]["registered_match_iou"]:
            r7_parts.append(pd.read_csv(a.r7_dir / f"iou_{x}/r7_summary.csv"))
    pd.concat(r7_parts).sort_values(["iou",*KEYS]).to_csv(a.out / "r7_sweep.csv", index=False)
    started = float(os.environ.get("REVISION_CPU_STARTED", time.time()))
    PC.jdump(dict(pod_minutes=(time.time()-started)/60), a.out / "cost_stamp.json")
    final = pd.read_csv(a.out / "pooled_metrics.csv")
    selected = final[final.metric.isin(["recall_within","alert_within"])]
    parts = ["# Revision results", "",
             "These are post hoc sensitivities of the registered analysis. The seed-2026 gate is reported separately.", "",
             "| Collection | Class set | Quantity | Pooled share | MCSE | 95% interval | Status |",
             "| --- | --- | --- | ---: | ---: | --- | --- |"]
    for r in selected.itertuples():
        parts.append(f"| {r.collection} | {r.class_set} | {r.metric} | {r.pooled_value:.6f} | {r.mcse:.6f} | "
                     f"{r.low95:.6f} to {r.high95:.6f} | {r.status} |")
    parts += ["", "The pooled metrics file reports the recall-controlled comparison, whole-part threshold, coverage with "
              "and without the finite population correction, false accepts and decision accuracy. The sampling summary "
              "reports whole-part and shortfall support alongside the split metrics. The full IoU sweep is in the accompanying CSV."]
    (a.out / "REVISION_RESULTS.md").write_text("\n".join(parts)+"\n")


def verify_answers(path):
    recorded = HERE / "clauses/ANSWERS_SHA256.txt"
    need(recorded.is_file(), "Answers tag is required")
    expected = recorded.read_text().split()[0]
    need(os.environ.get("ANSWERS_SHA") == expected, "Answer hash differs from the answers tag")
    checked(path, expected)


def clause_command(a):
    verify_answers(a.answers)
    freeze = yaml.safe_load((HERE.parent / "rules/params_revision.yaml").read_text())
    need(a.evaluator_tag == freeze["frozen"]["evaluator_tag"], "Evaluator tag differs")
    sys.path.insert(0, str(HERE.parent / "supplements/qualification"))
    from transfer import cases, extended
    from pyshacl import validate
    from rdflib import Graph, Literal, Namespace
    from rdflib.namespace import RDF

    fixtures = pd.read_csv(a.fixtures, keep_default_na=False)
    inventory = pd.read_csv(a.inventory, keep_default_na=False)
    answers = pd.read_csv(a.answers, keep_default_na=False)
    for column in ["expected_answer", "answer_basis", "interpretation_needed"]:
        need(fixtures[column].eq("").all(), "Public fixtures contain answer fields")
    need(answers.fixture_id.is_unique and fixtures.fixture_id.is_unique and set(answers.fixture_id)==set(fixtures.fixture_id),
         "Answers and fixtures differ in coverage")
    need(answers.expected_answer.ne("").all() and answers.answer_basis.ne("").all(), "An answer or its basis is blank")
    by_id = answers.set_index("fixture_id")
    reports = []
    capabilities = []
    ns = Namespace("urn:qualification:revision:")
    shapes = Graph().parse(a.shapes)
    for row in fixtures.itertuples():
        given = by_id.loc[row.fixture_id]
        rec = json.loads(row.input_record_json)
        if str(given.interpretation_needed).casefold() == "yes":
            reports.append(dict(fixture_id=row.fixture_id,clause_id=row.clause_id,group=row.group,
                                interpretation_needed=True,scored=False))
            continue
        frozen_result = extended(rec)
        found = frozen_result["action"]
        expected = str(given.expected_answer)
        category = ("correctly_refused" if found in {"insufficient_measurement","unit_error"} and found == expected else
                    "expressible_unchanged" if found == expected else
                    "requires_new_operator" if found == "unrepresentable" and rec["requirement"].get("extension_operator") else
                    "not_representable")
        graph = Graph()
        node = ns[row.fixture_id]
        graph.add((node, RDF.type, ns[row.clause_id]))
        for k, value in rec["evidence"].items():
            if isinstance(value, (str, bool, int, float)):
                graph.add((node, ns[k], Literal(value)))
        conforms, reports_graph, _ = validate(graph, shacl_graph=shapes)
        reports.append(dict(fixture_id=row.fixture_id,clause_id=row.clause_id,group=row.group,
                            interpretation_needed=False,scored=True,category=category,answer=found,agreement=found==expected,
                            shacl_conforms=conforms))
        capabilities.append(dict(fixture_id=row.fixture_id,clause_id=row.clause_id,
                                 record_field_count=len(rec["requirement"])+len(rec["evidence"]),
                                 record_uncertainty=rec["requirement"]["operator"]=="recall_lower_bound",
                                 validation_triples=len(reports_graph),shacl_conforms=conforms))
    for record, _, _ in cases():
        graph = Graph()
        node = ns[record["id"]]
        graph.add((node,RDF.type,ns[record["requirement"]["operator"]]))
        for k,value in record["evidence"].items():
            if isinstance(value,(str,bool,int,float)):
                graph.add((node,ns[k],Literal(value)))
        ok, results, _ = validate(graph, shacl_graph=shapes)
        capabilities.append(dict(fixture_id=record["id"],clause_id="registered_case",
                                 record_field_count=len(record["requirement"])+len(record["evidence"]),
                                 validation_triples=len(results),shacl_conforms=ok))
    a.out.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(reports)
    results.to_csv(a.out / "fixture_outcomes.csv", index=False)
    scored = results[results.scored].copy()
    for key, name in (("clause_id","clause_outcomes.csv"),("group","group_agreement.csv")):
        scored.groupby([key,"category"], dropna=False).agreement.agg(["size","sum","mean"]).reset_index().to_csv(a.out / name,index=False)
    pd.DataFrame(capabilities).to_csv(a.out / "representation_capabilities.csv",index=False)
    summary = pd.read_csv(a.out / "group_agreement.csv")
    report = ["# Clause revision results", "",
              "Counts and agreements are from the scored fixture output. Interpretation cases are listed in the detail table.", ""]
    for item in summary.itertuples():
        report.append(f"- Group {item.group}, {item.category}. Agreement {item.sum} of {item.size}.")
    destination = a.out / "report"
    destination.mkdir(exist_ok=True)
    (destination / "REVISION_RESULTS.md").write_text("\n".join(report)+"\n")


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["split","r7","gate","aggregate","clauses","sensitivity"])
    for name in ("root","e5","samples","pred5","ext18","pred18","registered","constants","split","splits","out",
                 "fixtures","answers","inventory","shapes"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--rules", type=Path, default=HERE.parent / "rules")
    parser.add_argument("--revision", type=Path, default=HERE.parent / "rules/params_revision.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--draws", type=int)
    parser.add_argument("--iou")
    parser.add_argument("--r7", type=Path)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--evaluator_tag", default="qualification-evaluator-v1")
    a = parser.parse_args()
    a.constants = a.constants or a.rules / "exp1_constants.yaml"
    if a.command=="clauses":
        verify_answers(a.answers)
    revision, params, constants = frozen(a)
    if a.command=="split":
        split_command(a, revision, params, constants)
    elif a.command=="r7":
        a.iou = float(a.iou)
        r7_command(a, revision, params)
    elif a.command=="gate":
        a.iou_dir = Path(a.iou)
        gate_command(a)
    elif a.command=="aggregate":
        a.gate_dir, a.r7_dir = a.gate, a.r7
        aggregate(a, revision, params)
    elif a.command=="clauses":
        clause_command(a)
    else:
        r1=revision["r1"]
        need(r1["split_seeds"]["first"] <= a.seed <= r1["split_seeds"]["last"], "Sensitivity seed is outside the revision")
        PC.locked_scripts(a.root)
        base=PC.Root(a.root)
        make_lists(base,dict.fromkeys(c for c,_ in networks(params)),a.seed,a.out)
        E1.chain(SimpleNamespace(root=a.root,rules=a.out,constants=a.constants,out=a.out,
                                 e5=None,samples=None,pred5=None,sensitivity=True),
                 {**frozen_seed(params,a.seed),"sample_draws":r1["draws"]})


if __name__ == "__main__":
    cli()

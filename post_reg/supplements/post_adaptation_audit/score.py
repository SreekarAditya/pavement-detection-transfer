"""Retrospective nine-fit requalification using the study's frozen matching and interval code."""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm


FIT_IDS = {
    "ecupdd": ["r003", "r004", "r005"],
    "ird": ["r032", "r033", "r034"],
    "svrdd": ["r061", "r062", "r063"],
}
CHECKPOINT = "yolo12m_seed0"
BLOCK_COLUMNS = ["run_id", "collection", "draw", "block", "images", "matched", "refs"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def match_number(actual, expected, label):
    # CSV round-tripping can perturb the last bit. Decisions use the original
    # unrounded computed values and have no tolerance at the recall floor.
    require(np.isclose(actual, expected, rtol=0, atol=1e-12, equal_nan=True),
            f"{label}: computed {actual}, source {expected}")


def frozen_module(inputs):
    sys.path.insert(0, str(inputs / "post_reg" / "scripts"))
    import pr_common as common
    core = common.locked_scripts(inputs)
    for module, path in [(core, inputs / "scripts/pdt_core.py"),
                         (common, inputs / "post_reg/scripts/pr_common.py")]:
        require(Path(module.__file__).resolve() == path.resolve(),
                f"Frozen module was loaded from a different path: {module.__file__}")
    return common


def decisions(point, lower, heldout, floor):
    point_accept = bool(np.isfinite(point) and point >= floor)
    lower_accept = bool(np.isfinite(lower) and lower >= floor)
    require(not lower_accept or point_accept, "Lower-bound acceptance requires point acceptance")

    def result(accepted):
        if not np.isfinite(heldout):
            return "unscorable"
        if heldout >= floor:
            return "correct_accept" if accepted else "unnecessary_adaptation"
        return "false_accept" if accepted else "correct_adapt"

    return dict(point_action="accept" if point_accept else "adapt",
                interval_action="accept" if lower_accept else "adapt",
                action_changed=point_accept != lower_accept,
                heldout_floor=("unscorable" if not np.isfinite(heldout) else
                               "met" if heldout >= floor else "below"),
                point_outcome=result(point_accept), interval_outcome=result(lower_accept))


def selected_runs(table):
    selected = table[(table.kind == "size_curve") & (table.checkpoint == CHECKPOINT)
                     & (table["size"].astype(str) == "100")].sort_values(["collection", "draw"])
    actual = list(selected[["collection", "draw", "run_id"]].itertuples(index=False, name=None))
    expected = [(collection, draw, run) for collection, runs in FIT_IDS.items()
                for draw, run in enumerate(runs)]
    require(actual == expected, f"Nine-run identity mismatch: {actual}")
    return selected


def e1_trace(inputs, expected_draws, floor):
    columns = ["collection", "class_set", "detector", "arm", "draw", "decision", "recall_floor",
               "adapted_run", "adapted_accept", "adapted_threshold", "adapted_stated_recall",
               "adapted_recall_at_threshold", "adapted_rate"]
    rows = pd.read_parquet(inputs / "e1_draws.parquet", columns=columns)
    rows = rows[(rows.class_set == "four_class") & (rows.arm == "procedure")
                & (rows.detector == CHECKPOINT)]
    require(set(rows.collection) == set(FIT_IDS), "Task 08 collection mismatch")
    for collection, group in rows.groupby("collection"):
        require(sorted(group.draw) == list(range(expected_draws)), f"{collection}: Task 08 draw IDs differ")
        require(group.recall_floor.eq(floor).all(), f"{collection}: Task 08 floor differs")
        require(group.decision.isin(["accept", "adapt"]).all(), f"{collection}: unknown initial decision")
        mapped = group[group.decision == "adapt"]
        expected_ids = [FIT_IDS[collection][d % 3] for d in mapped.draw]
        require(mapped.adapted_run.tolist() == expected_ids, f"{collection}: wrong adaptation mapping")
        require(group[group.decision == "accept"].adapted_run.isna().all(),
                f"{collection}: accepted initial run mapped to adaptation")
    return rows


def check_historical_fit(row, trace):
    selected = trace[trace.adapted_run == row["run_id"]]
    fields = {"adapted_threshold": "threshold", "adapted_stated_recall": "audit_recall",
              "adapted_recall_at_threshold": "heldout_recall", "adapted_rate": "heldout_alerts_per_100"}
    for column, field in fields.items():
        for value in selected[column].unique():
            match_number(row[field], value, f"{row['run_id']}: Task 08 {column}")
    expected = row["point_action"] == "accept"
    require(selected.adapted_accept.notna().all() and selected.adapted_accept.eq(expected).all(),
            f"{row['run_id']}: Task 08 original point decision differs")
    return len(selected)


def validate_sample(sample, fit, cal, held, blocks):
    require((sample["collection"], sample["draw"]) == (fit.collection, fit.draw), "Sample identity mismatch")
    aud, train, pilot = sample["audit_images"], sample["train_100"], sample["pilot_images"]
    for files in [aud, train, pilot]:
        require(len(set(files)) == len(files) and set(files) <= set(cal), "Invalid calibration sample")
    require(len(cal) == len(set(cal)) and len(held) == len(set(held)) and bool(held),
            "Duplicate split entries or empty held-out split")
    require(not set(cal) & set(held), "Split images overlap")
    aud_blocks = {blocks[f] for f in aud}
    require(set(sample["audit_blocks"]) == aud_blocks and len(aud_blocks) == len(sample["audit_blocks"]),
            "Audit block membership differs")
    require({f for f in cal if blocks[f] in aud_blocks} == set(aud), "Audit contains partial block")
    require(not aud_blocks & {blocks[f] for f in train + pilot}, "Audit overlaps training or pilot blocks")
    require(not {blocks[f] for f in cal} & {blocks[f] for f in held}, "Held-out block overlap")


def per_image(common, predictions, labels, files, threshold):
    # Discarding candidates below the fixed cutoff leaves the same prefixes in
    # the locked QuotaRun order, and avoids computing their unused match tables.
    pred = predictions[predictions.conf >= threshold]
    run = common.ThresholdRun(pred, labels, files, common.FOUR)
    alerts, matched, refs = run.per_image_at(threshold)
    return pd.DataFrame(dict(file=run.files, alerts=alerts, matched=matched, refs=refs))


def score_fit(inputs, common, fit, rules, labels, blocks, params, floor, z):
    c, run_id = fit.collection, fit.run_id
    sample = json.loads((inputs / "samples" / f"{c}_draw{fit.draw}.json").read_text())
    cal, held = common.read_split(rules, c)
    validate_sample(sample, fit, cal, held, blocks)
    audit, train = sample["audit_images"], set(sample["train_100"])
    rest = set(cal) - train
    require(len(rest) == fit.unused_calibration_images and len(audit) == fit.audit_images,
            f"{run_id}: archived sample sizes differ")
    pred_dir = inputs / "pred" / f"adapted_{run_id}"
    cal_pred = pd.read_parquet(pred_dir / f"{c}_calibration.parquet", columns=common.PRED_COLS)
    held_pred = pd.read_parquet(pred_dir / f"{c}_heldout.parquet", columns=common.PRED_COLS)
    for pred, files in [(cal_pred, cal), (held_pred, held)]:
        require(pred.file.isin(files).all(), f"{run_id}: prediction on wrong split")
    eligible = cal_pred.file.isin(rest) & cal_pred.cls.isin(common.FOUR)
    scores = cal_pred.loc[eligible, "conf"].sort_values(ascending=False).to_numpy()
    t = common.threshold_for_rate(scores, len(rest), params["k"]["four_class"]) if rest else np.nan
    possible = bool(np.isfinite(t))
    require(fit.recalibration_possible == possible, f"{run_id}: threshold availability differs")
    t = float(t) if possible else np.nan
    match_number(t, fit.recalibrated_threshold, f"{run_id}: archived threshold")

    # Zero-reference audit blocks remain cluster units. Do not round or clip bounds.
    block_table = pd.DataFrame(columns=["block", "images", "matched", "refs"])
    point = lo = hi = se = hrec = rate = np.nan
    ha = hm = hx = am = ax = np.nan
    status = "no_recalibrated_threshold"
    if possible:
        if audit:
            counts = per_image(common, cal_pred, labels, audit, t)
            counts["block"] = counts.file.map(blocks)
            block_table = counts.groupby("block", as_index=False).agg(
                images=("file", "size"), matched=("matched", "sum"), refs=("refs", "sum"))
            am, ax = int(block_table.matched.sum()), int(block_table.refs.sum())
            point, lo, hi, se = common.ratio_ci(block_table.matched, block_table.refs, z)
            status = "ok" if np.isfinite(lo) else ("zero_audit_references" if not ax else "too_few_blocks")
        else:
            am = ax = 0
            status = "empty_audit"

        truth = per_image(common, held_pred, labels, held, t)
        ha, hm, hx = (int(truth[col].sum()) for col in ["alerts", "matched", "refs"])
        hrec = hm / hx if hx else np.nan
        rate = 100 * ha / len(held)

    match_number(hrec, fit.recall_at_recalibrated, f"{run_id}: Task 07 recall at threshold")
    match_number(rate, fit.alerts_per_100_at_recalibrated, f"{run_id}: Task 07 alert rate")
    row = dict(collection=c, run_id=run_id, draw=fit.draw, checkpoint=fit.checkpoint,
               train_images=len(train), threshold=t, recall_floor=floor,
               audit_status=status, audit_images=len(audit), audit_blocks=len(sample["audit_blocks"]),
               audit_box_shortfall=sample["shortfall"]["audit_boxes"], audit_matched=am, audit_refs=ax,
               audit_recall=point, audit_se=se, audit_lo90=lo, audit_hi90=hi,
               heldout_images=len(held), heldout_alerts=ha, heldout_matched=hm, heldout_refs=hx,
               heldout_recall=hrec, heldout_alerts_per_100=rate, matched_quota_recall_gain_task07=fit.gain,
               **decisions(point, lo, hrec, floor))
    block_table = block_table.assign(run_id=run_id, collection=c, draw=fit.draw)
    return row, block_table[BLOCK_COLUMNS]


def summarize(fits):
    groups = {}
    for name, group in fits.groupby("collection", sort=True):
        groups[name] = dict(fits=len(group), fits_mapped_in_task08=int((group.task08_mapped_draws > 0).sum()),
                            changes=int(group.action_changed.sum()),
                            point_outcomes=group.point_outcome.value_counts().to_dict(),
                            interval_outcomes=group.interval_outcome.value_counts().to_dict(),
                            incomplete_audits=int((group.audit_status != "ok").sum()))
    return dict(analysis="retrospective", unit="one distinct 100-image-target fit",
                fits=len(fits), networks=groups,
                interpretation="Three fits per network share one held-out split. Task 08 mapping counts "
                               "indicate reuse by historical draws. Gain is a separate matched-quota metric.")


def score(inputs, output):
    common = frozen_module(inputs)
    rules = inputs / "post_reg/rules"
    params = common.load_params(rules)
    constants = yaml.safe_load((rules / "exp1_constants.yaml").read_text())
    floor = common.recall_floor(constants["recall_floor"], "four_class")
    level = params["exp3"]["decision_level"]
    require((params["selected"], params["exp1"]["adapt_size"], level, floor) == (CHECKPOINT, 100, .90, .30),
            "Frozen rule settings differ from the protocol")
    require(floor == common.recall_floor(params["exp1"]["recall_floor"], "four_class"), "Floor mismatch")
    z = norm.ppf(.5 + level / 2)
    selected_runs(pd.read_csv(rules / "exp5_run_table.csv", dtype={"size": str}))
    runs = selected_runs(pd.read_csv(inputs / "e5_scores.csv", dtype={"size": str}, float_precision="round_trip"))
    trace = e1_trace(inputs, params["sample_draws"], floor)
    data = common.Root(inputs)
    rows, blocks_by_fit = [], []
    for collection, group in runs.groupby("collection"):
        labels, blocks = data.labels_external(collection), data.blocks(collection)
        for fit in group.itertuples(index=False):
            row, table = score_fit(inputs, common, fit, rules, labels, blocks, params, floor, z)
            row["task08_mapped_draws"] = check_historical_fit(row, trace)
            rows.append(row)
            blocks_by_fit.append(table)
            print(f"Verified fit {fit.run_id} ({collection}, draw {fit.draw})", flush=True)
    fits = pd.DataFrame(rows)
    fits.to_csv(output / "fits.csv", index=False)
    pd.concat(blocks_by_fit, ignore_index=True).to_csv(output / "audit_blocks.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summarize(fits), indent=2, allow_nan=False) + "\n")
    common.write_manifest(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", "--bundle", required=True, type=Path)
    parser.add_argument("--output", "--out", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
            f"Output directory must be new or empty: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".audit-") as tmp:
        stage = Path(tmp) / "outputs"
        stage.mkdir()
        score(args.inputs.resolve(), stage)
        os.rename(stage, output)


if __name__ == "__main__":
    main()

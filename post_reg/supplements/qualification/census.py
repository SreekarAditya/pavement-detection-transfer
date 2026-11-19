"""Retrospective recall census and alert workload at archived thresholds."""
import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import audit_policy as A
import paired
import pr_common as PC
from pr_exp1 import Net
from pr_exp2 import block_samples


TABLES = ("paired_draws.parquet", "endpoint_summary.csv", "original_e1_reconciliation.csv",
          "paired_uncertainty.csv", "coverage_diagnostics.csv")


def need(test, reason):
    if not test:
        raise ValueError(reason)


def ratio(n, d):
    return n / d if d else np.nan


def side(value, floor):
    if not np.isfinite(value):
        return "unscorable"
    return "pass" if value >= floor else "below"


def transition(audit, census, held):
    if "unscorable" in (audit, census, held):
        return "unscorable"
    if census == held:
        return "agreement_unchanged" if audit == held else "corrected"
    return "introduced" if audit == held else "disagreement_persisted"


def band(value, target, relative_band):
    if value < target * (1 - relative_band):
        return "low"
    if value > target * (1 + relative_band):
        return "high"
    return "within"


def decomposition(first, middle, last):
    signed = first - middle, middle - last
    if all(np.isfinite(v) for v in (first, middle, last)):
        paired.check(sum(signed), first - last, "recall decomposition")
    return signed


def workload_decomposition(sample, census, held, target):
    signed = sample - target, census - sample, held - census
    paired.check(sum(signed), held - target, "workload decomposition")
    return signed


def replay(args):
    args.out.mkdir(parents=True)
    replayed = args.out / "task2_replay"
    paired.main(argparse.Namespace(lock=args.lock, rules=args.rules, draws=args.draws, out=replayed))
    checks = []
    for name in TABLES:
        load = pd.read_parquet if name.endswith(".parquet") else pd.read_csv
        old, new = [load(p / name) for p in (args.reference, replayed)]
        pd.testing.assert_frame_equal(old, new, check_exact=False, atol=1e-12, rtol=0)
        checks.append(dict(table=name, rows=len(new), columns=len(new.columns), reproduced=True))
    pd.DataFrame(checks).to_csv(args.out / "task2_reproduction.csv", index=False)
    draws = pd.read_parquet(replayed / "paired_draws.parquet")
    return draws[draws.arm == "full_95"], pd.read_csv(replayed / "coverage_diagnostics.csv")


def census_endpoint(R, rules, params, const, original, reproduced, coll, cs):
    started = time.monotonic()
    N = Net(R, coll, cs, params["selected"], rules, params)
    nterms = const["values"][cs][coll]["n"]
    floor = PC.recall_floor(const["recall_floor"], cs)
    relative = params["exp2"]["criterion_rel"]
    cidx, csi = PC.EXT_ALL.index(coll), list(PC.CLASS_SET_CLASSES).index(cs)
    draws = params["sample_draws"]
    samples = block_samples(N.cal, N.B, nterms, draws, params["seed"], cidx)
    old = original[(original.collection == coll) & (original.class_set == cs)].sort_values("draw")
    fixed = reproduced[(reproduced.collection == coll) & (reproduced.class_set == cs)].sort_values("draw")
    need(old.draw.tolist() == fixed.draw.tolist() == list(range(draws)), "Draw index differs")
    blocks = sorted(set(N.cal_block))
    blockids = {b: i for i, b in enumerate(blocks)}
    ci = np.array([blockids[b] for b in N.cal_block])
    images = np.bincount(ci, minlength=len(blocks))
    cr = N.Tc.R.nref
    refs = np.bincount(ci, weights=cr, minlength=len(blocks)).astype(int)
    rows, cache = [], {}
    for d, (idx, e1, a2) in enumerate(zip(samples, old.itertuples(), fixed.itertuples())):
        t = e1.threshold
        conf = np.concatenate([N.cal_conf[i] for i in idx]) if len(idx) else np.array([])
        paired.check(PC.threshold_for_rate(np.sort(conf)[::-1], len(idx), N.k), t, "Threshold differs")
        if t not in cache:
            ca, cm, _ = N.Tc.per_image_at(t)
            ha, hm, hr = N.Th.per_image_at(t)
            matched = np.bincount(ci, weights=cm, minlength=len(blocks)).astype(int)
            cache[t] = ca, matched, int(hm.sum()), int(hr.sum()), int(ha.sum())
        ca, m, held_match, held_ref, held_alert = cache[t]
        cal_match, cal_ref = int(m.sum()), int(refs.sum())
        cal, held = ratio(cal_match, cal_ref), ratio(held_match, held_ref)
        order = A.ordered_blocks(len(blocks), [2026, 302, cidx, csi, d])
        audit = A.look(order, images, m, refs, A.BUDGETS[1])
        p = audit["recall"]
        audit_match = int(m[order[:audit["blocks"]]].sum())
        paired.check(p, ratio(audit_match, audit["boxes"]), "Sampled audit matches differ")
        action, reason = A.three_way(audit, 95, floor)
        for field in ("recall", "lo95", "hi95", "se", "boxes", "images", "blocks", "target"):
            paired.check(audit[field], getattr(a2, field), f"Replayed audit field differs: {field}")
        need((audit["valid"], action, reason) == (a2.valid, a2.action, a2.action_reason), "Replayed action differs")
        sampling_gap, sets_gap = decomposition(p, cal, held)
        audit_side, cal_side, held_side = [side(x, floor) for x in (p, cal, held)]
        lo, hi = audit["lo95"], audit["hi95"]
        cal_in = bool(lo <= cal <= hi) if audit["valid"] and np.isfinite(cal) else None
        held_in = bool(lo <= held <= hi) if audit["valid"] and np.isfinite(held) else None
        interval = "unscorable" if None in (cal_in, held_in) else (
            "both" if cal_in and held_in else "calibration_only" if cal_in else "heldout_only" if held_in else "neither")
        sample_alert, cal_alert = int(ca[idx].sum()), int(ca.sum())
        sample_rate, cal_rate, held_rate = (100 * ratio(n, den) for n, den in
            ((sample_alert, len(idx)), (cal_alert, len(N.cal)), (held_alert, len(N.held))))
        paired.check([held, held_rate], [e1.true_recall, e1.heldout_rate], "E1 held-out values differ")
        workload = workload_decomposition(sample_rate, cal_rate, held_rate, N.k)
        rows.append(dict(collection=coll, class_set=cs, detector=params["selected"], draw=d, threshold=t,
            recall_floor=floor, audit_recall=p, calibration_recall=cal, heldout_recall=held,
            audit_images=audit["images"], audit_blocks=audit["blocks"], audit_references=audit["boxes"],
            audit_matches=audit_match, audit_valid=audit["valid"], audit_invalid_reason=audit["evidence_reason"],
            audit_shortfall=audit["images"] < audit["target"], audit_target_images=audit["target"],
            audit_lo95=lo, audit_hi95=hi, audit_se=audit["se"], full_95_action=action, full_95_action_reason=reason,
            calibration_images=len(N.cal), calibration_blocks=len(blocks), calibration_references=cal_ref,
            calibration_matches=cal_match, heldout_images=len(N.held), heldout_blocks=len({N.B[f] for f in N.held}),
            heldout_references=held_ref, heldout_matches=held_match,
            audit_side=audit_side, calibration_side=cal_side, heldout_side=held_side,
            audit_to_census=transition(audit_side, cal_side, held_side), audit_minus_calibration=sampling_gap,
            calibration_minus_heldout=sets_gap, audit_minus_heldout=p-held, calibration_in_interval95=cal_in,
            heldout_in_interval95=held_in, interval_joint=interval, threshold_sample_target_images=nterms,
            threshold_sample_images=len(idx), threshold_sample_blocks=len(set(ci[idx])),
            threshold_sample_shortfall=len(idx) < nterms, threshold_sample_alerts=sample_alert,
            calibration_alerts=cal_alert, heldout_alerts=held_alert,
            alert_target_per_100=N.k, alert_relative_band=relative, threshold_sample_rate=sample_rate,
            calibration_rate=cal_rate, heldout_rate=held_rate, sample_rate_minus_target=workload[0],
            calibration_rate_minus_sample=workload[1], heldout_rate_minus_calibration=workload[2],
            heldout_rate_minus_target=held_rate-N.k, threshold_sample_band=band(sample_rate,N.k,relative),
            calibration_band=band(cal_rate,N.k,relative), heldout_band=band(held_rate,N.k,relative)))
    elapsed = time.monotonic() - started
    print(f"{coll}/{cs}: {elapsed:.2f} seconds", flush=True)
    return pd.DataFrame(rows), elapsed


def reduce_endpoint(df, elapsed, coverage):
    x = df.iloc[0]
    row = dict(collection=x.collection, class_set=x.class_set, draws=len(df), elapsed_seconds=elapsed,
               recall_floor=x.recall_floor, alert_target_per_100=x.alert_target_per_100)
    counter = Counter(df.audit_to_census)
    for group in ("corrected", "introduced", "disagreement_persisted", "agreement_unchanged", "unscorable"):
        row[group] = counter[group]
    row["three_way_estimable"] = len(df) - counter["unscorable"]
    for name in ("audit", "calibration", "heldout"):
        row[f"{name}_unscorable"] = int((df[name+"_side"] == "unscorable").sum())
    pair = Counter(zip(df.calibration_side, df.heldout_side))
    for a in ("pass","below"):
        for b in ("pass","below"):
            row[f"census_{a}_heldout_{b}"] = pair[a,b]
    row["census_heldout_estimable"] = sum(row[f"census_{a}_heldout_{b}"] for a in ("pass","below") for b in ("pass","below"))
    row["census_heldout_unscorable"] = len(df) - row["census_heldout_estimable"]
    for name in ("audit_minus_calibration", "calibration_minus_heldout", "audit_minus_heldout",
                 "sample_rate_minus_target", "calibration_rate_minus_sample", "heldout_rate_minus_calibration"):
        row[name+"_estimable"] = int(df[name].notna().sum())
        row[name+"_median_signed"] = df[name].median()
        row[name+"_median_absolute"] = df[name].abs().median()
    for action in ("accept", "adapt", "unresolved"):
        row["full_95_"+action] = int((df.full_95_action == action).sum())
    row["audit_invalid"] = int((~df.audit_valid).sum())
    for flag in ("audit_shortfall","threshold_sample_shortfall"):
        row[flag] = int(df[flag].sum())
    interval = Counter(df.interval_joint)
    for it in ("both", "calibration_only","heldout_only","neither","unscorable"):
        row["interval_"+it] = interval[it]
    row["interval_joint_estimable"] = len(df) - interval["unscorable"]
    marginal = df.heldout_in_interval95.dropna()
    n, inc = len(marginal), int(marginal.sum())
    need(n == coverage.full_n, "Task 2 full_n differs")
    paired.check(ratio(inc,n), coverage.full_inclusion, "Task 2 held-out inclusion differs")
    if (x.collection, x.class_set) == ("kent","pothole"):
        need((inc,n) == (1470,2000), "Kent inclusion differs")
    row.update(heldout_inclusion_n=n, heldout_inclusion_count=inc, heldout_inclusion_rate=ratio(inc,n),
               task2_full_n=int(coverage.full_n),task2_full_inclusion=coverage.full_inclusion)
    for a in ("within","low","high"):
        for b in ("within","low","high"):
            row[f"workload_calibration_{a}_heldout_{b}"] = int(((df.calibration_band==a) & (df.heldout_band==b)).sum())
    cgood, hgood = df.calibration_band=="within", df.heldout_band=="within"
    row.update(workload_estimable=len(df), calibration_within_band=int(cgood.sum()), heldout_within_band=int(hgood.sum()),
               workload_miss_emerges_heldout=int((cgood & ~hgood).sum()),
               workload_miss_only_calibration=int((~cgood & hgood).sum()),
               workload_miss_both=int((~cgood & ~hgood).sum()),
               workload_within_both=int((cgood & hgood).sum()),
               threshold_sample_within_band=int((df.threshold_sample_band=="within").sum()))
    return row


def report(table, out):
    def section(title, columns):
        yield "## " + title
        yield ""
        yield "| " + " | ".join(label for _,label in columns) + " |"
        yield "| " + " | ".join("---" for _ in columns) + " |"
        for row in table[[name for name,_ in columns]].itertuples(index=False,name=None):
            yield "| " + " | ".join(f"{v:.4f}" if isinstance(v,float) else str(v) for v in row) + " |"
        yield ""
    key = [("collection","Collection"),("class_set","Classes")]
    lines = ["# Task 2A · Full-calibration census at the archived threshold","",
        "The first table compares the sampled point decision with the full-calibration point decision against the held-out side of the registered recall floor. Each endpoint uses 2,000 archived draws. Equality clears the floor.",""]
    lines += list(section("Did the census correct the audit?", key + [
        ("three_way_estimable","Estimable"),("corrected","Corrected"),("introduced","Introduced"),
        ("disagreement_persisted","Still wrong"),("agreement_unchanged","Still right"),("unscorable","Unscorable")]))
    lines += list(section("Full calibration versus held-out recall", key + [
        ("census_heldout_estimable","Estimable"),("census_pass_heldout_pass","Both clear"),
        ("census_pass_heldout_below","Calibration clears only"),("census_below_heldout_pass","Held-out clears only"),
        ("census_below_heldout_below","Both below")]))
    lines += list(section("Alert workload within the registered target band", key + [
        ("workload_estimable","Estimable"),("calibration_within_band","Calibration within"),
        ("heldout_within_band","Held-out within"),("workload_within_both","Both within"),
        ("workload_miss_emerges_heldout","Miss appears held-out"),("workload_miss_only_calibration","Miss only calibration"),
        ("workload_miss_both","Both miss")]))
    lines += list(section("Audit validity and interval reconciliation", key + [
        ("full_95_accept","Accept"),("full_95_adapt","Adapt"),("full_95_unresolved","Unresolved"),
        ("audit_shortfall","Shortfall"),("audit_invalid","Invalid"),("heldout_inclusion_count","Held-out included"),
        ("heldout_inclusion_n","Valid for held-out inclusion")]))
    lines += ["All five Task 2 tables reproduced before the census summaries. Kent potholes reproduced 1,470 of 2,000 held-out interval inclusions.",
        "", "The endpoint table gives the full interval and workload cross-tabs, point recall denominators, and median signed and absolute differences. The draw table records the actual whole-block sampling budgets, counts, decisions and signed decompositions.",
        "", "These retrospective comparisons describe repeated draws on fixed splits. The sampled audit is part of the census. The two signed differences sum to the audit-to-held-out error for estimable draws. Their magnitudes need not add. Differences across sets do not identify a cause and the census point comparison is a diagnostic."]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n")


def main(args):
    full, coverage = replay(args)
    Path("/workspace/census.stage").write_text("Running Task 2A endpoint census\n")
    PC.locked_scripts(args.lock)
    params = PC.load_params(args.rules)
    const = yaml.safe_load((args.rules/"exp1_constants.yaml").read_text())
    original = pd.read_parquet(args.draws)
    original = original[(original.arm=="procedure") & (original.detector==params["selected"])]
    need(params["selected"]=="yolo12m_seed0" and params["sample_draws"]==2000, "Locked detector/draws differ")
    cov = coverage[coverage.level==95].set_index(["collection","class_set"])
    need(len(cov)==10 and len(original)==20000 and len(full)==20000, "Unexpected Task 2 endpoint/draw count")
    evidence, table = [], []
    for (coll, cs), expected in cov.iterrows():
        df, elapsed = census_endpoint(PC.Root(args.lock),args.rules,params,const,original,full,coll,cs)
        evidence.append(df)
        table.append(reduce_endpoint(df,elapsed,expected))
    endpoint = pd.DataFrame(table)
    endpoint.to_csv(args.out/"endpoint_summary.csv",index=False)
    pd.concat(evidence,ignore_index=True).to_parquet(args.out/"draw_evidence.parquet",index=False)
    endpoint[["collection","class_set","heldout_inclusion_count","heldout_inclusion_n","task2_full_n",
              "task2_full_inclusion"]].to_csv(args.out/"interval_reconciliation.csv",index=False)
    report(endpoint,args.out)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    for arg in ("lock","rules","draws","reference","out"):
        parser.add_argument("--"+arg,type=Path,required=True)
    main(parser.parse_args())

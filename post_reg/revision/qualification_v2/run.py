"""Run the frozen v2 fixture and source checks with private source passages."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

from evaluator import evaluate
from fixtures import build, encode, make_requirement
from text_checks import m3, m3b, measurable, original_directions, physical, threshold_values

HERE = Path(__file__).resolve().parent
MEASURES = "10_clauses/measures/"


def read_csv(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def indexed(rows, key):
    output = {row[key]: row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"Duplicate {key}")
    return output


def write_csv(path, rows):
    rows = [{key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (list, dict)) else value
             for key, value in row.items()} for row in rows]
    keys = list(dict.fromkeys(key for row in rows for key in row)) or ["clause_id"]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def archive_csv(archive, name):
    with io.TextIOWrapper(archive.extractfile(MEASURES + name)) as source:
        return list(csv.DictReader(source))


def review(row, failed, source, before, after, correction, rule):
    observed = []
    for item in before["exceptions"]:
        if item["missing_value"]:
            observed.append(f"bound {item['bound']} absent from cited passage")
        if item["missing_unit"]:
            observed.append(f"unit {item['unit']} absent from cited passage")
    if correction:
        outcome, reason = "encoding corrected", correction["reason"]
    elif rule is None:
        outcome, reason = "no encoding correction", "No supported comparison established for the cited rule."
    elif before["status"] == "passed":
        outcome, reason = "no encoding correction", "The original bound and unit match under v2 source normalization."
    else:
        outcome, reason = ("unresolved source consistency",
                           "The cited passage did not independently substantiate every bound or unit. "
                           "The encoding is retained and the omission is reported, with no inferred correction.")
    return dict(clause_id=row["clause_id"], source_id=row["source_id"], locator=row["locator"],
                disposition=outcome, reason=reason, original_threshold=row["threshold_or_limit"],
                corrected_threshold=(correction or {}).get("changes", {}).get("threshold_or_limit", ""),
                original_missing_values=failed["missing_values"], original_missing_units=failed["missing_units"],
                v2_observations="; ".join(observed), revised_status=after["status"],
                passage_sha256=hashlib.sha256(source.encode()).hexdigest())


def routing(row):
    # These are limits of the development encodings, recorded before checking the evaluator.
    key = row["clause_id"]
    if key.startswith("hif18022-table8-"):
        return row | {"missing_capability": "measurement rounding and surface or speed applicability"}
    if key == "hif18022-unpaved":
        return row | {"missing_capability": "conditional eligibility based on agency determination"}
    return row


def evaluate_rows(inventory, passages, corrections, failures, original):
    counts, fixtures, reviews, encoded = [], [], [], {}
    for row in inventory:
        key = row["clause_id"]
        item = routing(row) if original else row
        source = passages[key]["text"]
        correction = corrections.get(key)
        updated = item | (correction or {}).get("changes", {})
        previous_rule, previous_missing = encode(item)
        rule, missing = encode(updated)
        if rule is not None:
            category = "executes"
        elif missing.startswith(("comparison to", "measurement rounding", "conditional eligibility")):
            category = "needs comparison type"
        else:
            category = "not representable"
        req = make_requirement(updated, rule, missing)
        consistency = m3(req, source)
        direction = m3b(threshold_values(rule), source)
        old_direction = m3b(original_directions(original[key]), source) if original else None
        rows = list(build(updated, rule, missing))
        results = {fixture["kind"]: evaluate(fixture["record"])["action"]
                   for fixture in rows if fixture["kind"] in {"box-only", "units"}}
        is_measurable = measurable(source)
        is_physical = physical(updated["quantity"], updated["unit"])
        counts.append(dict(clause_id=key, source_id=row["source_id"], locator=updated["locator"],
                           coverage=category, missing_capability=missing, measurable=is_measurable,
                           physical_quantity=is_physical, m2=results["box-only"] if is_measurable and is_physical else "",
                           m3=consistency["status"], m3_exceptions=consistency["exceptions"],
                           m3b=direction["status"], m3b_comparisons=direction["comparisons"],
                           original_m3b=old_direction["status"] if old_direction else "",
                           original_m3b_comparisons=old_direction["comparisons"] if old_direction else [],
                           m4=results["units"], comparison=rule))
        for fixture in rows:
            fixtures.append(fixture | {"decision": evaluate(fixture["record"])})
        encoded[key] = req
        if key in failures:
            before = m3(make_requirement(item, previous_rule, previous_missing), source)
            reviews.append(review(row, failures[key], source, before, consistency, correction, rule))
    if {row["clause_id"] for row in reviews} != set(failures):
        raise ValueError("Review does not cover the original M3 failures")
    return counts, fixtures, reviews, encoded


def worked_cases(published, legacy_fixtures, encoded):
    output = []
    for row in published:
        fixture = legacy_fixtures[row["fixture_id"]]
        req = encoded[fixture["clause_id"]]
        evidence = json.loads(fixture["input_record_json"])["evidence"]
        result = evaluate(dict(requirement=req, evidence=evidence))
        output.append(row | {"actual": result["action"], "agreement": result["action"] == row["published_action"]})
    return output


def summary(rows, published):
    report = []
    slices = {"pooled": rows}
    for name in sorted({row["source_id"] for row in rows}):
        slices[name] = [row for row in rows if row["source_id"] == name]
    for name, entries in slices.items():
        values = {
            "M0": ["measurable" if row["measurable"] else "not measurable" for row in entries],
            "M1": [row["coverage"] for row in entries if row["measurable"]],
            "M2": [row["m2"] for row in entries if row["m2"]],
            "M3": [row["m3"] for row in entries if row["measurable"] and row["m3"] != "not_applicable"],
            "M3b": [row["m3b"] for row in entries],
            "M4": [row["m4"] for row in entries],
        }
        if any(row["original_m3b"] for row in entries):
            values["Original M3b"] = [row["original_m3b"] for row in entries]
        examples = [row for row in published if name == "pooled" or row["source_id"] == name]
        if examples:
            values["M5"] = ["agreement" if row["agreement"] else "difference" for row in examples]
        for measure, labels in values.items():
            frequencies = Counter(labels) or {"no applicable cases": 0}
            for category, count in sorted(frequencies.items()):
                report.append(dict(scope=name, measure=measure, category=category, count=count, denominator=len(labels)))
    return report


def develop(args):
    if sha(args.baseline) != args.baseline_sha or sha(args.passages) != args.passages_sha:
        raise ValueError("Development evidence checksum differs")
    inv = read_csv(args.inventory)
    if len(inv) != 564:
        raise ValueError("Development inventory does not contain 564 clauses")
    indexed(inv, "clause_id")
    passages = json.loads(args.passages.read_text())
    if set(passages) != {row["clause_id"] for row in inv}:
        raise ValueError("Development passage coverage differs")
    with tarfile.open(args.baseline) as baseline:
        archived_m3 = archive_csv(baseline, "m3_source_consistency_private.csv")
        baseline_summary = archive_csv(baseline, "measures_summary.csv")
        original_published = archive_csv(baseline, "published_outcomes.csv")
        original = {row["clause_id"]: json.loads(row["encoding"])
                    for row in archive_csv(baseline, "m1_coverage.csv")}
    failures = {row["clause_id"]: row for row in archived_m3 if row["encoded"] == "True" and row["passed"] == "False"}
    if len(failures) != 114:
        raise ValueError("Expected 114 original M3 failures")
    corrections = json.loads((HERE / "development_corrections.json").read_text())
    if not set(corrections) <= set(failures):
        raise ValueError("Correction is outside the original failure list")
    rows, fixtures, review_rows, encoded = evaluate_rows(inv, passages, corrections, failures, original)
    legacy_fixtures = indexed(read_csv(args.fixtures), "fixture_id")
    published = read_csv(args.published)
    if len(published) != 23 or any("--route-aaa-" not in row["fixture_id"] for row in published):
        raise ValueError("Expected 23 Route AAA published cases")
    worked = worked_cases(published, legacy_fixtures, encoded)
    inherited = [row | {"agreement": row["agreement"] == "True"} for row in original_published
                 if row["source_id"] == "registered_vendor_guideline"]
    # These original vendor cases are reported alongside the newly evaluated Route AAA cases.
    inherited = [row | {"phase": "frozen original comparator"} for row in inherited]
    summaries = summary(rows, worked)
    args.out.mkdir(parents=True, exist_ok=False)
    write_csv(args.out / "development_summary.csv", summaries)
    write_csv(args.out / "amendment_3_summary.csv", baseline_summary)
    write_csv(args.out / "development_clauses.csv", rows)
    write_csv(args.out / "development_fixtures.csv", fixtures)
    write_csv(args.out / "m3_original_failure_review.csv", review_rows)
    write_csv(args.out / "encoding_corrections.csv", [row for row in review_rows if row["disposition"] == "encoding corrected"])
    write_csv(args.out / "published_worked_cases.csv", worked + inherited)
    before = Counter(row["disposition"] for row in review_rows)
    (args.out / "REPORT.md").write_text(
        "# Amendment 4 development results\n\n"
        "These results use the 564 development clauses. The held-out test is a separate stage.\n\n"
        "The original Amendment 3 measures are copied from the verified archive alongside the v2 measures. "
        "The original 114 M3 failures each have a review disposition. Unresolved source support remains visible.\n\n"
        + "".join(f"- {key.capitalize()} — {value}\n" for key, value in sorted(before.items()))
        + "\nThe 23 Route AAA outcomes were evaluated as development cases. "
          "The published outcome table lists every match and difference.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("development",))
    for name in ["inventory", "fixtures", "published", "passages", "baseline", "out"]:
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--baseline-sha", required=True)
    parser.add_argument("--passages-sha", required=True)
    develop(parser.parse_args())


if __name__ == "__main__":
    main()

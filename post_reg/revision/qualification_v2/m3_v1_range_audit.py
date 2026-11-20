"""Recheck the sealed v1 M3 inputs with one range-tokenization correction."""

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import re
import tarfile

from run import archive_csv, indexed, sha, write_csv

ORIGINAL = Path(__file__).resolve().parents[1] / "measures/text_checks.py"
original_spec = importlib.util.spec_from_file_location("amendment3_text_checks", ORIGINAL)
original = importlib.util.module_from_spec(original_spec)
original_spec.loader.exec_module(original)

NUMBER = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
RANGE = re.compile(rf"([\[(])\s*({NUMBER})\s*,\s*({NUMBER})\s*([\])])")


def corrected_check(requirement, text):
    check = dict(requirement)
    if "criterion" in check:
        check["criterion"] = RANGE.sub(
            lambda match: f"{match[1]}{match[2]} to {match[3]}{match[4]}",
            str(check["criterion"]))
    return original.consistency(check, text)


def run(args):
    if sha(args.baseline) != args.baseline_sha or sha(args.passages) != args.passages_sha:
        raise ValueError("Amendment 3 input checksum differs")
    with tarfile.open(args.baseline) as archive:
        encodings = indexed(archive_csv(archive, "m1_coverage.csv"), "clause_id")
        recorded = indexed(archive_csv(archive, "m3_source_consistency_private.csv"), "clause_id")
    passages = json.loads(args.passages.read_text())
    if len(recorded) != 564 or set(recorded) != set(encodings) or set(recorded) != set(passages):
        raise ValueError("Recorded M3 inputs do not cover the same 564 clauses")
    diagnoses = json.loads(Path(__file__).with_name("m3_review_notes.json").read_text())
    rows = []
    for key, row in recorded.items():
        if row["encoded"] != "True":
            continue
        req = json.loads(encodings[key]["encoding"])
        source = passages[key]["text"]
        before = original.consistency(req, source)
        if before["passed"] != (row["passed"] == "True"):
            raise ValueError(f"Original M3 result differs for {key}")
        after = corrected_check(req, source)
        if before["passed"] and not after["passed"]:
            raise ValueError(f"Range correction regressed original M3 at {key}")
        entries = dict(clause_id=key, source_id=encodings[key]["source_id"], locator=row["locator"],
                       original_passed=before["passed"], corrected_passed=after["passed"],
                       original_missing_values=before["missing_values"], original_missing_units=before["missing_units"],
                       corrected_missing_values=after["missing_values"], corrected_missing_units=after["missing_units"],
                       assessment="", reason="")
        if not before["passed"]:
            if after["passed"]:
                entries.update(assessment="range parsing fixed",
                               reason="The previous checker concatenated the two interval endpoints at the comma.")
            elif key in diagnoses:
                entries.update(diagnoses[key])
            elif before["missing_values"] != after["missing_values"]:
                entries.update(assessment="range parsing fixed; other mismatch unresolved",
                               reason="The range is now read as two endpoints. The remaining evidence gap needs source review.")
            else:
                entries.update(assessment="unresolved",
                               reason="The cited passage and the v1 encoding do not establish which side accounts for the mismatch.")
        rows.append(entries)
    if len(rows) != 552 or sum(row["original_passed"] for row in rows) != 438:
        raise ValueError("Original M3 denominator or numerator differs")
    old_failures = [row for row in rows if not row["original_passed"]]
    unresolved = [row for row in old_failures if not row["corrected_passed"]]
    if len(old_failures) != 114 or not set(diagnoses) <= {row["clause_id"] for row in unresolved}:
        raise ValueError("Failure review coverage differs")
    args.out.mkdir(parents=True, exist_ok=False)
    write_csv(args.out / "m3_v1_original_and_corrected.csv", rows)
    write_csv(args.out / "m3_v1_original_failures.csv", old_failures)
    write_csv(args.out / "m3_v1_remaining_failures.csv", unresolved)
    grouped = Counter(row["assessment"] for row in unresolved)
    passes = sum(row["corrected_passed"] for row in rows)
    (args.out / "REPORT.md").write_text(
        "# Amendment 3 M3 range-parser correction\n\n"
        "| Checker | Matches | Encodings |\n| --- | ---: | ---: |\n"
        f"| Original, as recorded | 438 | 552 |\n| Range parsing corrected | {passes} | 552 |\n\n"
        "The original checker was reproduced on every encoding before applying the correction. "
        "Only M3 was evaluated. Both passes used the identical v1 encodings and cited source passages.\n\n"
        + "".join(f"- {name.capitalize()} — {count}\n" for name, count in sorted(grouped.items()))
        + "\nThe failure tables retain the clause locators and missing values or units. "
          "A failure without sufficient source evidence remains unresolved.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "passages", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("baseline-sha", "passages-sha"):
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

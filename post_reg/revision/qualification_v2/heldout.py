"""Apply the frozen evaluator to a separately sealed held-out inventory."""

import argparse
import json
from pathlib import Path

from run import evaluate_rows, indexed, read_csv, sha, summary, write_csv

REQUIRED_FILES = {"inventory.csv", "clause_duplicates.json", "sources.csv"}


def evaluate(args):
    if sha(args.seals) != args.seals_sha:
        raise ValueError("Held-out inventory seal differs")
    seals = json.loads(args.seals.read_text())
    if set(seals["files"]) != REQUIRED_FILES:
        raise ValueError("Required held-out files are missing from the seal")
    for name, expected in seals["files"].items():
        if sha(args.seals.parent / name) != expected:
            raise ValueError(f"Sealed held-out input differs: {name}")
    if sha(args.passages) != seals["private_passages_sha256"]:
        raise ValueError("Held-out source passages differ")

    original = read_csv(args.seals.parent / "inventory.csv")
    if any(key in row for row in original for key in ("comparison_json", "threshold_or_limit", "record")):
        raise ValueError("The tagged inventory must precede encoding")
    inventory = indexed(original, "clause_id")
    encodings = indexed(read_csv(args.encodings), "clause_id")
    if set(inventory) != set(encodings):
        raise ValueError("Encoding coverage differs from held-out inventory")
    sources = indexed(read_csv(args.seals.parent / "sources.csv"), "source_id")
    if {row["source_id"] for row in inventory.values()} - {key for key, value in sources.items()
                                                          if value["disposition"] == "included"}:
        raise ValueError("Inventory includes a source that did not pass intake")
    groups = json.loads((args.seals.parent / "clause_duplicates.json").read_text())
    represented, eligible = [], {}
    for key, members in groups.items():
        subjects = [entry["clause_id"] for entry in members if entry["phase"] == "heldout"]
        represented.extend(subjects)
        if subjects and not any(entry["phase"] == "development" for entry in members):
            eligible[key] = sorted(subjects)
    if len(represented) != len(inventory) or set(represented) != set(inventory):
        raise ValueError("Duplicate map does not fully partition the inventory")
    passages = json.loads(args.passages.read_text())
    if set(passages) != set(inventory):
        raise ValueError("Held-out passages do not fully cover the inventory")
    inputs = [
        {**value, **{field: encodings[key][field] for field in
                     ("quantity", "unit", "threshold_or_limit", "action", "group_guess",
                      "missing_capability", "comparison_json")}}
        for key, value in inventory.items()
    ]
    details, fixtures, _, encoded = evaluate_rows(inputs, passages, {}, {}, {})
    actual = indexed(details, "clause_id")
    tests = []
    if args.published:
        from evaluator import evaluate as evaluate_record
        known = read_csv(args.published)
        indexed(known, "fixture_id")
        for case in known:
            key = case["clause_id"]
            result = evaluate_record(dict(requirement=encoded[key], evidence=json.loads(case["evidence_json"])))
            tests.append(case | {"actual": result["action"], "agreement": case["published_action"] == result["action"],
                                 "source_id": actual[key]["source_id"]})

    output = []
    unique_counts = []
    for label, include_template in [("all programs", True), ("without template programs", False)]:
        pool = []
        per_source = {source: [] for source, metadata in sources.items()
                      if metadata["disposition"] == "included"
                      and (include_template or metadata["federal_template_flag"] == "false")}
        for members in eligible.values():
            filtered = [key for key in members if actual[key]["source_id"] in per_source]
            if not filtered:
                continue
            pool.append(actual[filtered[0]])
            for source in sorted({actual[key]["source_id"] for key in filtered}):
                represented_key = next(key for key in filtered if actual[key]["source_id"] == source)
                per_source[source].append(actual[represented_key])
        # The same exact text can recur across sources. Pooled counts use one
        # representative, while each source receives its own unique-clause count.
        selections = [("pooled", pool), *per_source.items()]
        for scope, rows in selections:
            ids = {row["clause_id"] for row in rows}
            cases = [case for case in tests if case["clause_id"] in ids]
            output.extend(row | {"scope": scope, "program_set": label}
                          for row in summary(rows, cases) if row["scope"] == "pooled")
            raw = sum(1 for row in inventory.values() if
                      (scope == "pooled" and row["source_id"] in per_source) or row["source_id"] == scope)
            unique_counts.append(dict(program_set=label, scope=scope, raw_clauses=raw, unique_test_clauses=len(rows)))

    args.out.mkdir(parents=True, exist_ok=False)
    write_csv(args.out / "heldout_summary.csv", output)
    write_csv(args.out / "unique_and_raw_counts.csv", unique_counts)
    write_csv(args.out / "heldout_clauses.csv", details)
    write_csv(args.out / "heldout_fixtures.csv", fixtures)
    write_csv(args.out / "heldout_published_outcomes.csv", tests)
    write_csv(args.out / "sources.csv", list(sources.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("seals", "passages", "encodings", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seals-sha", required=True)
    parser.add_argument("--published", type=Path)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()

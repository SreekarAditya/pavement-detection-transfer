"""List missing revision inputs and dangling links before any analysis runs."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

COLLECTIONS = ("ecupdd", "hrp4k", "ird", "kent", "pdd2026", "rome", "svrdd")


def inspect_inputs(root, rules, t07, t08, ext18, pred18):
    problems = []

    def require(path):
        if not path.is_file():
            problems.append(f"Missing readable file: {path}")
            return False
        if not os.access(path, os.R_OK) or path.stat().st_size == 0:
            problems.append(f"Unreadable or empty file: {path}")
            return False
        return True

    def structured(path, loader):
        if not require(path):
            return {}
        try:
            with path.open() as source:
                return loader(source)
        except (OSError, ValueError, yaml.YAMLError) as error:
            problems.append(f"Cannot read {path}: {error}")
            return {}

    for directory in (t07, t08):
        if not directory.is_dir():
            problems.append(f"Missing input directory: {directory}")
            continue
        result = subprocess.run(["find", "-L", str(directory), "-type", "l"],
                                capture_output=True, text=True)
        problems.extend(f"Dangling symlink: {line}" for line in result.stdout.splitlines())
        if result.returncode:
            problems.append(f"Symlink scan failed for {directory}: {result.stderr.strip()}")

    params = structured(rules / "params.yaml", yaml.safe_load)
    selected = params.get("selected")
    if not selected:
        problems.append("Selected detector is absent from params.yaml")
    for path in (root / "s2/g4/blocks.csv", root / "e1/labels.parquet",
                 root / "g3/main/fold_dev.txt", root / "g3/main/fold_S1.txt",
                 root / "g3/main/fold_S2.txt", t07 / "scores/e5_scores.csv",
                 t08 / "selected/e1_summary.csv"):
        require(path)
    if selected:
        require(root / f"e1/sealed/predictions/{selected}.parquet")
    for collection in COLLECTIONS:
        require(root / f"s2/g4/pool_final_{collection}.txt")
        require(root / f"s2/g4/labels_final_{collection}.parquet")
        if selected:
            require(root / f"s2/s2a/sealed_ext/{collection}/{selected}.parquet")
        for part in ("calibration", "heldout"):
            require(rules / f"split/{collection}_{part}.txt")

    table = rules / "exp5_run_table.csv"
    if require(table):
        try:
            with table.open(newline="") as source:
                reader = csv.DictReader(source)
                if not {"run_id", "collection", "draw"} <= set(reader.fieldnames or ()):
                    problems.append(f"Required run table columns missing: {table}")
                else:
                    rows = list(reader)
                    if not rows:
                        problems.append(f"Run table has no rows: {table}")
                    for row in rows:
                        require(t07 / f"samples/{row['collection']}_draw{row['draw']}.json")
                        require(t07 / f"pred/adapted_{row['run_id']}/{row['collection']}_calibration.parquet")
        except (OSError, csv.Error) as error:
            problems.append(f"Cannot read {table}: {error}")

    constants = structured(rules / "exp1_constants.yaml", yaml.safe_load)
    commitments = structured(Path(__file__).with_name("exp18_commitments.json"), json.load)
    eligibility = structured(ext18 / "exp18_eligibility.json", json.load)
    require(ext18 / "blocks.csv")
    require(pred18 / "MANIFEST.sha256")
    networks = eligibility.get("networks", {})
    unseen = constants.get("values", {}).get("merged_crack", {}).get("unseen", {})
    for country in ("georgia", "turkiye"):
        matching = [name for name in networks if name.endswith("_" + country)]
        if len(matching) != 1:
            problems.append(f"Expected one Experiment 18 network for {country}, found {len(matching)}")
            continue
        collection = matching[0]
        for path in (ext18 / collection / "pool_final.txt",
                     ext18 / collection / "open/labels_calibration.parquet",
                     ext18 / f"private/heldout_labels_{collection}.parquet"):
            require(path)
        if selected:
            require(pred18 / selected / f"{collection}.parquet")
        for part in ("calibration", "heldout"):
            require(ext18 / f"rules_split/split/{collection}_{part}.txt")
        matching = [row for row in commitments.get("procedure", [])
                    if row.get("collection") == collection and row.get("detector") == selected
                    and row.get("track") == "merged_crack" and row.get("class_set") == "merged_crack"]
        if len(matching) != 1:
            problems.append(f"Missing unique Experiment 18 constants for {collection}")
            continue
        row = matching[0]
        if any(row.get(key) is None for key in ("n", "M", "recall_floor")):
            problems.append(f"Incomplete Experiment 18 constants for {collection}")
        if any(unseen.get(key) is None or row.get(key) != unseen.get(key) for key in ("n", "M")):
            problems.append(f"Experiment 18 sample constants differ for {collection}")
    return list(dict.fromkeys(problems))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "rules", "t07", "t08", "ext18", "pred18"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    problems = inspect_inputs(**vars(args))
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(f"Input preflight failed with {len(problems)} problems.", file=sys.stderr)
        return 1
    print("Input preflight passed. Required files exist and input symlinks resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

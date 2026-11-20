"""Call registered Experiment 1 functions on the sealed external pothole pools."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

REVISION = Path(__file__).resolve().parents[1]
PR = REVISION.parent
sys.path[:0] = [str(PR / "scripts"), str(REVISION)]
import pr_common as pc
import pr_exp1 as e1
import pr_rev

NAMES = ("highrpd", "dashcam", "uneti", "m2s", "attain", "wasit", "capstone")
LABEL = "eligibility decided after screening"
DETECTOR = "yolo12m_seed0"
CLASS_SET = "pothole"


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


class Root(pc.Root):
    def __init__(self, args, name, scored):
        super().__init__(args.lock)
        self.folder = args.splits / name
        self.seal = json.loads((self.folder / "split_seal.json").read_text())["sha256"]
        self.prediction_folder = args.predictions
        self.scored = scored
        for item in ("calibration.txt", "heldout.txt", "blocks.csv", "calibration_labels.parquet"):
            need(pc.sha(self.folder / item) == self.seal[item], f"Split byte seal differs: {name}/{item}")

    def blocks(self, _):
        return dict(pd.read_csv(self.folder / "blocks.csv", dtype=str).itertuples(index=False, name=None))

    def pool(self, _):
        return sorted(self.blocks(None))

    def labels_external(self, _):
        calibration = pd.read_parquet(self.folder / "calibration_labels.parquet")
        if not self.scored:
            return calibration
        private = "private/heldout_labels.parquet"
        need(pc.sha(self.folder / private) == self.seal[private], "Held-out label seal differs")
        return pd.concat([calibration, pd.read_parquet(self.folder / private)], ignore_index=True)

    def pred_external(self, detector, name):
        need(detector == DETECTOR, "Wrong detector")
        return pd.concat([pd.read_parquet(self.prediction_folder / f"{name}_{part}.parquet", columns=pc.PRED_COLS)
                          for part in ("calibration", "heldout")], ignore_index=True)


def json_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_split(root, name, directory, seed):
    target = directory / "split"
    target.mkdir(parents=True, exist_ok=True)
    if seed == 2026:
        parts = [(root.folder / f"{part}.txt").read_text().splitlines() for part in ("calibration", "heldout")]
    else:
        parts = pc.make_split(root.pool(name), root.blocks(name), 0.5, seed, balance="images")
    need(all(parts) and not set(parts[0]).intersection(parts[1]), f"Invalid split for {name} seed {seed}")
    need(set(parts[0] + parts[1]) == set(root.pool(name)), f"Split pool differs for {name}")
    for part, values in zip(("calibration", "heldout"), parts):
        (target / f"{name}_{part}.txt").write_text("\n".join(values) + "\n")


def trial(args, name, seed, draws, scored):
    root = Root(args, name, scored)
    base = args.out / "split_lists" / str(seed) / name
    write_split(root, name, base, seed)
    params = deepcopy(args.params)
    params["seed"] = seed
    rows = e1.run_chain(root, name, CLASS_SET, DETECTOR, base, params, 500, 800, 0.25,
                        100 + NAMES.index(name), draws, procedure_only=not scored)
    rows.attrs = {}
    rows["seed"] = seed
    rows["analysis"] = LABEL
    rows["block_type"] = args.block_types[name]
    return rows


def procedure(rows):
    return [{key: json_value(row[key]) for key in [*e1.CHECK_COLS, "collection", "threshold_images",
                                                 "threshold_shortfall", "pilot_shortfall", "pilot_blocks",
                                                 "block_type", "analysis"]}
            for row in rows[rows.arm == "procedure"].to_dict(orient="records")]


def one_split(payload):
    args, name, seed = payload
    response = trial(args, name, seed, 500, True)
    draws = response[response.arm == "procedure"].copy()
    destination = args.out / "sensitivity_draws" / name
    destination.mkdir(parents=True, exist_ok=True)
    draws.to_parquet(destination / f"{seed}.parquet", index=False)
    measures = dict(recall_within=draws.recall_error.abs().le(.05),
                    alert_within=draws.rate_rel_error.abs().le(.10))
    return [dict(collection=name, class_set=CLASS_SET, M=800, post_hoc_network=True,
                 metric=metric, numerator=int(flags.sum()), denominator=len(flags), value=float(flags.mean()),
                 target=.90, seed=seed, analysis=LABEL, block_type=args.block_types[name])
            for metric, flags in measures.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["commit", "score"])
    for attr in ("lock", "splits", "predictions", "constants", "block-types", "out"):
        parser.add_argument("--"+attr, type=Path, required=True)
    parser.add_argument("--commitments", type=Path)
    parser.add_argument("--commitments-sha")
    args = parser.parse_args()
    args.params = yaml.safe_load((PR / "rules/params.yaml").read_text())
    revision = yaml.safe_load((PR / "rules/params_revision.yaml").read_text())
    const = yaml.safe_load(args.constants.read_text())
    need(pc.sha(args.constants) == "b0e41e96804816b8e0894e371a6c8dae628086b6a40fd76352719d8e456311e5", "Constants SHA differs")
    need(args.params["selected"] == DETECTOR and args.params["class_sets"][CLASS_SET]["k"] == 15.80
         and args.params["exp3"]["decision_level"] == .90 and
         const["values"][CLASS_SET]["unseen"]["n"] == 500 and
         const["values"][CLASS_SET]["unseen"]["M"] == 800 and const["recall_floor"][CLASS_SET] == .25,
         "Frozen procedure parameters differ")
    args.block_types = json.loads(args.block_types.read_text())
    pc.locked_scripts(args.lock)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.phase == "commit":
        rows = [entry for name in NAMES for entry in procedure(trial(args, name, 2026, 1, False))]
        need(len(rows) == len(NAMES), "Commitment count differs")
        sealed = dict(procedure=rows, label=LABEL, prediction_manifest_sha256=pc.sha(args.predictions/"MANIFEST.sha256"),
                      split_seals={name: pc.sha(args.splits/name/"split_seal.json") for name in NAMES})
        (args.out/"post-reg-nine-commitments.json").write_text(json.dumps(sealed, indent=2, allow_nan=False)+"\n")
    else:
        need(args.commitments is not None and pc.sha(args.commitments) == args.commitments_sha,
             "Commitment tag and SHA-256 are required before opening any held-out label")
        sealed = json.loads(args.commitments.read_text())
        need(sealed["prediction_manifest_sha256"] == pc.sha(args.predictions/"MANIFEST.sha256") and
             all(sealed["split_seals"][name] == pc.sha(args.splits/name/"split_seal.json") for name in NAMES),
             "Commitment input seal differs")
        pure = [entry for name in NAMES for entry in procedure(trial(args, name, 2026, 1, False))]
        need(pure == sealed["procedure"], "Calibration-only procedure differs from committed values")
        # This launch has no approved adaptation run. An adaptation decision is retained
        # in the commitments and stops scoring before held-out labels are read.
        need(all(item["decision"] == "accept" for item in pure),
             "At least one committed decision requires Experiment 18 adaptation before primary scoring")
        primary = pd.concat([trial(args, name, 2026, 1, True) for name in NAMES], ignore_index=True)
        need(procedure(primary) == pure, "Full primary scoring procedure differs from commitments")
        primary["recall_within"] = primary.recall_error.abs().le(.05)
        primary["alert_within"] = primary.rate_rel_error.abs().le(.10)
        primary.to_csv(args.out/"primary.csv", index=False)
        fixed = [(args, name, seed) for seed in range(3001,3051) for name in NAMES]
        with ProcessPoolExecutor(6) as pool:
            stats = pd.DataFrame([row for rows in pool.map(one_split, fixed, chunksize=1) for row in rows])
        stats.to_csv(args.out/"split_summary.csv", index=False)
        aggregate = pr_rev.summarize_metrics(stats, revision)
        aggregate["analysis"] = LABEL
        aggregate["block_type"] = aggregate.collection.map(args.block_types)
        aggregate.to_csv(args.out/"sensitivity_summary.csv", index=False)


if __name__ == "__main__":
    main()

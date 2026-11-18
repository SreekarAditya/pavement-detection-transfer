"""Synthetic CPU tests. These use the frozen code and invented images and boxes."""
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

from score import FIT_IDS, CHECKPOINT, decisions, selected_runs, score, validate_sample


CODE_ROOT = Path(os.environ.get("AUDIT_INPUTS", Path(__file__).resolve().parents[2] / "08_lock/github_repo"))
Z = norm.ppf(.95)
LABEL_COLUMNS = ["file", "cls", "x1", "y1", "x2", "y2"]
PRED_COLUMNS = ["file", "cls", "conf", "x1", "y1", "x2", "y2"]


def build_inputs(root):
    for path in ("scripts/pdt_core.py", "post_reg/scripts/pr_common.py"):
        source = CODE_ROOT / path
        if not source.is_file():
            raise RuntimeError(f"Set AUDIT_INPUTS to the extracted input tree (missing {source})")
        dest = root / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(source.resolve())
    rules = root / "post_reg/rules"
    (rules / "split").mkdir(parents=True)
    (root / "samples").mkdir()
    g4 = root / "s2/g4"
    g4.mkdir(parents=True)
    params = dict(selected=CHECKPOINT, sample_draws=3, exp1=dict(adapt_size=100, recall_floor={"four_class": .3}),
                  exp3=dict(decision_level=.90), k={"four_class": 152.49})
    (rules / "params.yaml").write_text(yaml.safe_dump(params))
    (rules / "exp1_constants.yaml").write_text(yaml.safe_dump({"recall_floor": {"four_class": .3}}))

    cal, held = ["train", "A1", "A2", "B", "Z"], ["H1", "H2"]
    file_block = {"train": "train", "A1": "A", "A2": "A", "B": "B", "Z": "Z", "H1": "H1", "H2": "H2"}
    first = (0., 0., .25, .25)
    second = (.5, .5, .75, .75)
    lab = [(f, 0, *first) for f in ["train", "A1", "A2", "B", "H1", "H2"]] + [("B", 0, *second)]
    labels = pd.DataFrame(lab, columns=LABEL_COLUMNS)
    cal_pred = pd.DataFrame([
        ("A1", 0, .9, *first), ("A2", 0, .8, *first), ("B", 1, .7, *first),
        ("Z", 1, .6, *first), ("Z", 1, .5, *second), ("Z", 2, .4, *first),
        ("B", 0, .1, *second), ("train", 0, .99, *first)], columns=PRED_COLUMNS)
    held_pred = pd.DataFrame([("H1", 0, .9, *first), ("H2", 1, .4, *first), ("H1", 1, .4, *second),
                              ("H2", 0, .1, *first)], columns=PRED_COLUMNS)
    catalog, scored, history, blocks = [], [], [], []
    for name, runs in FIT_IDS.items():
        labels.to_parquet(g4 / f"labels_final_{name}.parquet")
        blocks.extend((name, f, b) for f, b in file_block.items())
        for label, files in [("calibration", cal), ("heldout", held)]:
            (rules / "split" / f"{name}_{label}.txt").write_text("\n".join(files))
        for draw, rid in enumerate(runs):
            ident = dict(run_id=rid, kind="size_curve", checkpoint=CHECKPOINT, collection=name, size=100, draw=draw)
            catalog.append(ident)
            scored.append(dict(**ident, unused_calibration_images=4, audit_images=4,
                               recalibration_possible=True, recalibrated_threshold=.4,
                               recall_at_recalibrated=.5, alerts_per_100_at_recalibrated=150., gain=.1))
            smp = dict(collection=name, draw=draw, audit_images=cal[1:], audit_blocks=["A", "B", "Z"],
                       pilot_images=["train"], train_100=["train"], shortfall={"audit_boxes": True})
            (root / "samples" / f"{name}_draw{draw}.json").write_text(json.dumps(smp))
            pred_dir = root / "pred" / f"adapted_{rid}"
            pred_dir.mkdir(parents=True)
            cal_pred.to_parquet(pred_dir / f"{name}_calibration.parquet")
            held_pred.to_parquet(pred_dir / f"{name}_heldout.parquet")
            used = draw != 2
            history.append(dict(collection=name, class_set="four_class", detector=CHECKPOINT, arm="procedure",
                                draw=draw, decision="adapt" if used else "accept", recall_floor=.3,
                                adapted_run=rid if used else None, adapted_accept=True if used else None,
                                adapted_threshold=.4 if used else None, adapted_stated_recall=.5 if used else None,
                                adapted_recall_at_threshold=.5 if used else None, adapted_rate=150. if used else None))
    pd.DataFrame(catalog).to_csv(rules / "exp5_run_table.csv", index=False)
    pd.DataFrame(scored).to_csv(root / "e5_scores.csv", index=False)
    pd.DataFrame(history).to_parquet(root / "e1_draws.parquet")
    pd.DataFrame(blocks, columns=["collection", "file", "block"]).to_csv(g4 / "blocks.csv", index=False)


class AuditTests(unittest.TestCase):
    def test_synthetic_nine_fit_run(self):
        with tempfile.TemporaryDirectory() as work:
            inputs, output = Path(work) / "inputs", Path(work) / "out"
            output.mkdir()
            build_inputs(inputs)
            score(inputs, output)
            fits = pd.read_csv(output / "fits.csv")
            self.assertEqual(len(fits), 9)
            self.assertTrue((fits.audit_refs == 4).all())
            self.assertTrue((fits.audit_matched == 2).all())
            self.assertTrue((fits.audit_blocks == 3).all())
            self.assertTrue((fits.heldout_floor == "met").all())
            self.assertTrue((fits.point_action == "accept").all())
            self.assertTrue((fits.interval_outcome == "unnecessary_adaptation").all())
            np.testing.assert_allclose(fits.audit_se, np.sqrt(3) / 4)
            np.testing.assert_allclose(fits.audit_lo90, .5 - Z * np.sqrt(3) / 4)
            self.assertEqual(sum(fits.task08_mapped_draws == 0), 3)
            blocks = pd.read_csv(output / "audit_blocks.csv")
            self.assertEqual(sum((blocks.block == "Z") & (blocks.refs == 0)), 9)

            historical = pd.read_parquet(inputs / "e1_draws.parquet")
            historical.loc[0, "adapted_stated_recall"] = .7
            historical.to_parquet(inputs / "e1_draws.parquet")
            with self.assertRaisesRegex(ValueError, "Task 08 adapted_stated_recall"):
                score(inputs, output)

    def test_floor_equality_and_nonfinite_audits(self):
        self.assertEqual(decisions(.30, .30, .30, .30)["interval_outcome"], "correct_accept")
        self.assertEqual(decisions(.31, .30, .29, .30)["interval_outcome"], "false_accept")
        result = decisions(.35, np.nan, np.nan, .30)
        self.assertEqual(result["interval_action"], "adapt")
        self.assertEqual(result["interval_outcome"], "unscorable")
        self.assertEqual(decisions(np.nan, np.nan, .29, .3)["point_outcome"], "correct_adapt")
        with self.assertRaisesRegex(ValueError, "requires point acceptance"):
            decisions(.29, .31, .3, .30)

    def test_catches_identity_and_block_overlap(self):
        ident = dict(run_id="r003", kind="size_curve", checkpoint=CHECKPOINT, collection="ecupdd", size=100, draw=0)
        with self.assertRaisesRegex(ValueError, "Nine-run identity mismatch"):
            selected_runs(pd.DataFrame([ident] * 9))
        sample = dict(collection="ecupdd", draw=0, audit_images=["a"], train_100=["a"], pilot_images=[],
                      audit_blocks=["A"])
        with self.assertRaisesRegex(ValueError, "overlaps training"):
            validate_sample(sample, pd.Series(ident), ["a"], ["h"], {"a": "A", "h": "H"})


if __name__ == "__main__":
    unittest.main()

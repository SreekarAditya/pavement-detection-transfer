#!/usr/bin/env python3
"""Experiment 5: merge the run logs of the fine-tuning workers and stop if any row of the run table is not done."""
import argparse, sys
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--run_table", required=True)
    a = ap.parse_args()
    out, t = Path(a.out), pd.read_csv(a.run_table)
    logs = [pd.read_csv(p) for p in sorted(out.glob("worker_*/e5_runs.csv"))]
    done = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame(columns=["run_id"])
    if done.run_id.duplicated().any():
        sys.exit(f"STOP: rows run twice: {sorted(done[done.run_id.duplicated()].run_id)}")
    done.to_csv(out / "e5_runs.csv", index=False)
    left = sorted(set(t.run_id) - set(done.run_id))
    (out / "rows_not_done.txt").write_text("\n".join(left) + ("\n" if left else ""))
    print(f"{len(done)} of {len(t)} rows done")
    if left:
        sys.exit("STOP: rows still to run (rerun the task script on a new pod); scoring waits for the full table")


if __name__ == "__main__":
    main()

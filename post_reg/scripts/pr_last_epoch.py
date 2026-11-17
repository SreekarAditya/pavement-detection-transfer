#!/usr/bin/env python3
"""Task 06, step 5: each last.pt must be the weights after the final (100th) epoch. Ultralytics strips the
checkpoints when training ends (epoch set to -1); a stripped file with 100 planned epochs and a patience of at
least 100 (so early stopping could not end training sooner) is accepted, as is an unstripped file saved at epoch
index 99. Anything else stops the task. Writes one row per file."""
import sys
from pathlib import Path

import pandas as pd


def main(d, out):
    import torch
    rows = []
    for p in sorted(Path(d).glob("*_last.pt")):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        ta = ck.get("train_args") or {}
        ta = ta if isinstance(ta, dict) else vars(ta)
        ep, n, pat = ck.get("epoch"), ta.get("epochs"), ta.get("patience")
        ok = (ep == 99 and n == 100) or (ep == -1 and n == 100 and pat is not None and pat >= 100)
        rows.append(dict(file=p.name, epoch_field=ep, planned_epochs=n, patience=pat, date=ck.get("date"), ok=ok))
    D = pd.DataFrame(rows)
    D.to_csv(out, index=False)
    if len(D) != 20:
        sys.exit(f"STOP: {len(D)} last.pt files found, 20 expected")
    if not D.ok.all():
        sys.exit(f"STOP: last.pt epoch check failed for {D[~D.ok].file.tolist()}")
    print("last.pt: all 20 are final-epoch weights of 100-epoch runs")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

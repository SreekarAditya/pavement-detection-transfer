#!/usr/bin/env python3
"""Image lists for inference (one image path per line; the base name is the file key of every scoring table).

  S1, S2, dev            clean-holdout folds (validation images, paths from pod session 1's val_all.txt)
  Norway, United_States, Japan
                         every clean-holdout image (development fold and both scoring folds) of that partition
  <collection>           the final pool of each external collection (prepared folder of pdt_s2_external.py)
Stops if any listed image is missing on disk.
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr_common as PC  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--val_list", default="", help="pod session 1 lists/val_all.txt (paths of validation images)")
    ap.add_argument("--ext_root", default="", help="prepared external collections (<c>/pool/images)")
    ap.add_argument("--collections", nargs="*", default=PC.EXT_ALL)
    a = ap.parse_args()
    R = PC.Root(a.root); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lists = {}
    if a.val_list:
        paths = [l.strip() for l in open(a.val_list) if l.strip()]
        by = {Path(p).name: p for p in paths}
        if len(by) != len(paths):
            sys.exit("STOP: repeated base names in the validation list")
        folds = {f: R.fold(f) for f in ("S1", "S2", "dev")}
        for f, files in folds.items():
            miss = [x for x in files if x not in by]
            if miss:
                sys.exit(f"STOP: {len(miss)} images of fold {f} are not in the validation list, e.g. {miss[:3]}")
            lists[f] = [by[x] for x in files]
        clean = sorted(set().union(*folds.values()))
        for c in ("Norway", "United_States", "Japan"):
            lists[c] = [by[x] for x in clean if x.startswith(c)]
    if a.ext_root:
        for c in a.collections:
            lists[c] = [str(Path(a.ext_root) / c / "pool" / "images" / f) for f in R.pool(c)]
    missing = [p for v in lists.values() for p in v if not Path(p).exists()]
    if missing:
        sys.exit(f"STOP: {len(missing)} listed images are missing, e.g. {missing[:3]}")
    for k, v in lists.items():
        (out / f"{k}.txt").write_text("\n".join(v) + "\n")
        PC.log(f"{k}: {len(v)} images")


if __name__ == "__main__":
    main()

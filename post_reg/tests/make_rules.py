#!/usr/bin/env python3
"""Small rule folder for the synthetic tests: params.yaml with few draws, and a block split of every collection."""
import argparse, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pr_common as PC  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True); ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); (out / "split").mkdir(parents=True, exist_ok=True)
    t = Path(a.params).read_text()
    t = re.sub(r"bootstrap: \{draws: \d+\}", "bootstrap: {draws: 20}", t)
    t = re.sub(r"(?m)^sample_draws: \d+", "sample_draws: 5", t)
    t = re.sub(r"chance_repeats: \d+", "chance_repeats: 2", t)
    (out / "params.yaml").write_text(t)
    p = PC.load_params(out)
    R = PC.Root(a.root)
    for c in PC.EXT_ALL:
        cal, held = PC.make_split(R.pool(c), R.blocks(c), p["split"]["calibration_share"], p["split"]["seed"],
                                  p["split"].get("balance", "images"))
        (out / "split" / f"{c}_calibration.txt").write_text("\n".join(cal) + "\n")
        (out / "split" / f"{c}_heldout.txt").write_text("\n".join(held) + "\n")
    (out / "RULES.md").write_text("synthetic test rules\n")
    print("rules written", out)


if __name__ == "__main__":
    main()

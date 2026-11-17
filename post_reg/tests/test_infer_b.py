#!/usr/bin/env python3
"""Plumbing test of pr_infer_b.py without a card: the locked setup B runners, the download and the version check
are replaced by stand-ins; checks file layout, revision and SHA-256 checks, and the stop on a changed checkpoint."""
import hashlib, json, sys, tempfile, types
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(HERE))


def run(lock_scripts):
    tmp = Path(tempfile.mkdtemp())
    root = tmp / "root"; (root / "s2" / "s2b").mkdir(parents=True)
    (root / "scripts").symlink_to(Path(lock_scripts).resolve())
    w = tmp / "w.pt"; w.write_bytes(b"weights")
    h = hashlib.sha256(b"weights").hexdigest()
    sys.path.insert(0, str(root / "scripts"))
    import pdt_s2_infer_b as IB
    rec = dict(versions=dict(ultralytics="8.4.153"),
               checkpoints={m: dict(revision="r" + m, sha256=h) for m in IB.MODELS},
               rfdetr_offset={"rfdetr_nano": dict(offset=0), "rfdetr_small": dict(offset=0)})
    json.dump(rec, open(root / "s2" / "s2b" / "g4c_part2.json", "w"))
    lst = tmp / "imgs.txt"; lst.write_text("/x/a__1.jpg\n/x/a__2.jpg\n")
    seen = []
    fake = lambda *a_, **k: pd.DataFrame(dict(file=["a__1.jpg"], cls=[0], conf=[0.5], x1=[0.], y1=[0.], x2=[5.], y2=[5.]))  # noqa
    IB.run_yolo = lambda p, imgs, dev: (seen.append(("yolo", p)), fake())[1]
    IB.run_rfdetr = lambda m, imgs, off: (seen.append(("rfdetr", off)), fake())[1]
    IB.rfdetr_model = lambda name, p: object()
    sys.modules["ultralytics"] = types.SimpleNamespace(__version__="8.4.153")
    import huggingface_hub
    huggingface_hub.hf_hub_download = lambda repo, fn, revision, cache_dir: (seen.append(("rev", revision)), str(w))[1]
    import pr_infer_b
    pr_infer_b.main(["--root", str(root), "--out", str(tmp / "out"), "--pool", f"p={lst}"])
    files = sorted(p.relative_to(tmp / "out").as_posix() for p in (tmp / "out").rglob("*.parquet"))
    assert files == sorted(f"{m}/p.parquet" for m in IB.MODELS), files
    assert ("rev", "ryolov8n") in seen
    # a changed checkpoint stops the run
    rec["checkpoints"]["yolov8n"]["sha256"] = "0" * 64
    json.dump(rec, open(root / "s2" / "s2b" / "g4c_part2.json", "w"))
    try:
        pr_infer_b.main(["--root", str(root), "--out", str(tmp / "out2"), "--pool", f"p={lst}", "--models", "yolov8n"])
        raise AssertionError("changed checkpoint not caught")
    except SystemExit as e:
        assert "differs" in str(e)
    print("setup B inference plumbing test passed")


if __name__ == "__main__":
    run(sys.argv[1])

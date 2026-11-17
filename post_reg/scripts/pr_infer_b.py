#!/usr/bin/env python3
"""Setup B predictions on new pools (task 09, phase B; merged-crack endpoint). The registered pipeline of
scripts/pdt_s2_infer_b.py: each model at its released resolution, confidence floor 0.001, at most 300 boxes,
suppression 0.7 where the model uses it; RF-DETR label offset as recorded in pod session 2. Checkpoints are
downloaded at the revisions recorded in s2/s2b/g4c_part2.json and must match its SHA-256 values. Runs in the
setup B environment (Ultralytics 8.4.153, rfdetr 1.12.1). Predictions only; no label is read.
Output: <out>/<model>/<pool>.parquet (same layout as pr_infer.py)."""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr_common as PC  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--pool", nargs="+", required=True, help="name=listfile")
    ap.add_argument("--models", nargs="*", default=None); ap.add_argument("--cache", default="/workspace/run/hf_b")
    ap.add_argument("--device", default="0")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(Path(a.root) / "scripts"))
    import pdt_s2_infer_b as IB
    from huggingface_hub import hf_hub_download
    rec = json.load(open(Path(a.root) / "s2" / "s2b" / "g4c_part2.json"))
    import ultralytics
    if ultralytics.__version__ != rec["versions"]["ultralytics"]:
        sys.exit(f"STOP: setup B needs Ultralytics {rec['versions']['ultralytics']}, found {ultralytics.__version__}")
    pools = {}
    for s in a.pool:
        n, f = s.split("=", 1)
        pools[n] = [Path(l.strip()) for l in open(f) if l.strip()]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = []
    for m in a.models or list(IB.MODELS):
        repo, fn, kind = IB.MODELS[m][:3]
        ck = rec["checkpoints"][m]
        p = hf_hub_download(repo, fn, revision=ck["revision"], cache_dir=a.cache)
        if PC.sha(p) != ck["sha256"]:
            sys.exit(f"STOP: {m} differs from the checkpoint used in pod session 2")
        model = IB.rfdetr_model(m, p) if kind == "rfdetr" else None
        for pn, imgs in pools.items():
            dst = out / m / f"{pn}.parquet"
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            if kind == "yolo":
                P = IB.run_yolo(p, imgs, a.device)
            else:
                P = IB.run_rfdetr(model, imgs, rec["rfdetr_offset"][m]["offset"])
            tmp = dst.with_suffix(".tmp"); P.to_parquet(tmp); tmp.rename(dst)
            log.append(dict(model=m, pool=pn, images=len(imgs), predictions=len(P), seconds=round(time.time() - t0, 1),
                            revision=ck["revision"], sha256=ck["sha256"]))
            PC.log(f"{m}/{pn}: {len(P)} predictions on {P.file.nunique()}/{len(imgs)} images")
    import pandas as pd
    pd.DataFrame(log).to_csv(out / f"setup_b_log_{int(time.time())}.csv", index=False)


if __name__ == "__main__":
    main()

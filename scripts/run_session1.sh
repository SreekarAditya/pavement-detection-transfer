#!/usr/bin/env bash
# Pod session 1, part 1: data, weights, reproduction gate.
# Usage on the pod (from the folder holding this file and pdt_gate.py):
#   export KAGGLE_USERNAME=sreekaraditya KAGGLE_KEY=...   # or put kaggle.json in ~/.kaggle/
#   bash run_session1.sh 2>&1 | tee session1.log
set -euo pipefail
ROOT=/workspace/run
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ROOT"/{data,weights,outputs}

echo "== system packages and Python packages"
(apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 >/dev/null) || true
pip install -q kaggle huggingface_hub pyyaml pillow pandas pyarrow imagehash

echo "== dataset from Kaggle (skipped if already present)"
if [ -z "$(find "$ROOT/data" -name '*.jpg' -print -quit)" ]; then
  if [ ! -f ~/.kaggle/kaggle.json ] && [ -z "${KAGGLE_KEY:-}" ]; then
    echo "STOP: no Kaggle token. Set KAGGLE_USERNAME and KAGGLE_KEY, or copy kaggle.json to ~/.kaggle/"; exit 1
  fi
  kaggle datasets download -d sreekaraditya/rdd2022-yolo-crackscan-v2 -p "$ROOT/data" --unzip
fi
df -h "$ROOT" | tail -1

echo "== all 31 checkpoints from Hugging Face (skipped if already present)"
python - <<EOF
from huggingface_hub import snapshot_download
from pathlib import Path
d = snapshot_download("SreekarAditya/yolo-rdd2022-benchmark", allow_patterns=["yolo-rdd2022-benchmark/*.pt"],
                      local_dir="$ROOT/hf")
src = Path(d) / "yolo-rdd2022-benchmark"
for p in src.glob("*.pt"):
    t = Path("$ROOT/weights") / p.name
    if not t.exists():
        t.symlink_to(p)
print(len(list(Path("$ROOT/weights").glob("*.pt"))), "checkpoints")
EOF

echo "== Ultralytics at the version that saved the checkpoint"
V=$(python "$HERE/pdt_gate.py" version --weights "$ROOT/weights/yolo12s_seed0_best.pt")
echo "checkpoint saved with Ultralytics $V"
pip install -q "ultralytics==$V"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

OUT="$ROOT/outputs/session1"
python "$HERE/pdt_gate.py" prepare --out "$OUT"
set +e
python "$HERE/pdt_gate.py" gate --out "$OUT"
STATUS=$?
set -e

cp "$HERE"/*.log "$OUT"/ 2>/dev/null || true
tar -czf "$ROOT/outputs/session1_outputs.tar.gz" -C "$ROOT/outputs" session1
echo "== packed: $ROOT/outputs/session1_outputs.tar.gz"
if [ $STATUS -ne 0 ]; then echo "GATE FAILED: stop here and send me the archive."; exit $STATUS; fi
echo "GATE PASSED. Continuing with E1 (per-image inference), E2 (image properties), C0 (near-duplicate inputs) and inference speed."

set +e
python "$HERE/pdt_e1.py" --out "$ROOT/outputs/e1"; E1=$?
python "$HERE/pdt_e2.py" --out "$ROOT/outputs/e2"; E2=$?
python "$HERE/pdt_c0.py" --out "$ROOT/outputs/c0"; C0=$?
python "$HERE/pdt_latency.py" --out "$ROOT/outputs/latency"; LAT=$?
set -e
echo "E1=$E1 E2=$E2 C0=$C0 LAT=$LAT" | tee "$ROOT/outputs/session1_status.txt"
tar --exclude="e1/ultra" --exclude="e1/objects__*" --exclude="e1/images__*" \
    -czf "$ROOT/outputs/session1_all_outputs.tar.gz" -C "$ROOT/outputs" session1_status.txt session1 e1 e2 c0 latency
echo "== packed: $ROOT/outputs/session1_all_outputs.tar.gz (download it, then STOP the pod)"
if [ $E1 -ne 0 ] || [ $E2 -ne 0 ] || [ $C0 -ne 0 ] || [ $LAT -ne 0 ]; then
  echo "WARNING: a step failed (E1=$E1 E2=$E2 C0=$C0 LAT=$LAT; E1=3 means the matcher check failed, C0=5 means SSCD could not be loaded or an image could not be hashed, LAT=4 means some models were not timed). Send me the archive."
fi

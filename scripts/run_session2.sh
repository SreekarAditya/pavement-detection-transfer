#!/usr/bin/env bash
# Pod session 2 (protocol 3.3.9, steps 5 and 6): external collections, gates G4 to G6 and G4c part 2,
# setup A and setup B inference (predictions only), July per-country rerun at confidence 0.25,
# label-free properties of the external images.
# Usage on the pod, from the folder holding this file, the rdd_*.py scripts and inputs/:
#   export KAGGLE_USERNAME=... KAGGLE_KEY=...      # or ~/.kaggle/kaggle.json; removed at the end
#   bash run_session2.sh 2>&1 | tee session2.log
# Sealed outputs (never open): outputs/s2a/sealed_ext, outputs/s2b/sealed_b.
set -euo pipefail
ROOT=/workspace/run
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ROOT"/{data,weights,outputs,inputs}
cp -r "$HERE"/inputs/* "$ROOT/inputs/"
OUT="$ROOT/outputs"

echo "== packages"
(apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 >/dev/null) || true
pip install -q kaggle huggingface_hub pyyaml pillow pandas pyarrow imagehash

echo "== RDD2022 (Kaggle) and setup A checkpoints, as in session 1"
if [ -z "$(find "$ROOT/data" -name '*.jpg' -print -quit)" ]; then
  kaggle datasets download -d sreekaraditya/rdd2022-yolo-crackscan-v2 -p "$ROOT/data" --unzip
fi
python - <<EOF
from huggingface_hub import snapshot_download
from pathlib import Path
d = snapshot_download("SreekarAditya/yolo-rdd2022-benchmark", allow_patterns=["yolo-rdd2022-benchmark/*.pt"],
                      local_dir="$ROOT/hf")
for p in (Path(d) / "yolo-rdd2022-benchmark").glob("*.pt"):
    t = Path("$ROOT/weights") / p.name
    if not t.exists():
        t.symlink_to(p)
EOF
V=$(python "$HERE/pdt_gate.py" version --weights "$ROOT/weights/yolo12s_seed0_best.pt")
pip install -q "ultralytics==$V"
python -c "import ultralytics, torch; print('setup A: ultralytics', ultralytics.__version__, 'torch', torch.__version__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h "$ROOT" | tail -1

echo "== session 1 lists rebuilt; the validation list must match session 1"
python "$HERE/pdt_gate.py" prepare --out "$OUT/session1"
python - <<EOF
import json, hashlib
r = json.load(open("$OUT/session1/prepare_report.json"))
want = "f84063d2f222ddc9823c827656a049b14f9d3dfc1e36a9c317e3af59e1160a62"
assert r["val_list_sha256"] == want, ("validation list differs from session 1", r["val_list_sha256"])
print("validation list matches session 1")
EOF
python "$HERE/pdt_gate.py" gate --out "$OUT/session1"
python - <<EOF
import json
g = json.load(open("$OUT/session1/gate_results.json"))["primary"]["map50"]
assert abs(g - 0.6182) <= 0.001, ("environment differs from session 1: gate mAP50", g)
print("gate mAP50", round(g, 4), "matches session 1 (0.6182)")
EOF

echo "== external collections: download, checksums, preparation (labels may be read)"
python "$HERE/pdt_s2_external.py" fetch --inventory "$ROOT/inputs/external_inventory.json" --root "$ROOT/ext"
python "$HERE/pdt_s2_external.py" prepare --root "$ROOT/ext"
df -h "$ROOT" | tail -1

echo "== gates G4 to G6"
MODEL_CACHE="$ROOT/models" python "$HERE/pdt_s2_g4.py" --ext "$ROOT/ext" --c0 "$ROOT/inputs/c0" \
  --kaggle "$ROOT/data" --out "$OUT/g4"

echo "== setup A: external inference (predictions only) and the July per-country rerun"
python "$HERE/pdt_s2_infer_a.py" external --weights "$ROOT/weights" --g4 "$OUT/g4" --ext "$ROOT/ext" \
  --work "$ROOT/infer" --out "$OUT/s2a"
python "$HERE/pdt_s2_infer_a.py" country025 --weights "$ROOT/weights" --session1 "$OUT/session1" --out "$OUT/s2a"

echo "== setup B environment (Ultralytics 8.4.153, rfdetr 1.12.1) and inference"
python -m venv --system-site-packages "$ROOT/envB"
"$ROOT/envB/bin/pip" install -q "ultralytics==8.4.153" "rfdetr==1.12.1" "supervision==0.31.7" huggingface_hub \
  pandas pyarrow pillow pyyaml
"$ROOT/envB/bin/python" -c "import torch; assert torch.cuda.is_available(), 'setup B: no CUDA'; print('setup B torch', torch.__version__)"
"$ROOT/envB/bin/python" "$HERE/pdt_s2_infer_b.py" --g4c "$ROOT/inputs/g4c" --g4 "$OUT/g4" --c0 "$ROOT/inputs/c0" \
  --kaggle "$ROOT/data" --ext "$ROOT/ext" --cache "$ROOT/hf_b" --out "$OUT/s2b"

echo "== label-free properties of the external images (forecast inputs; reads no labels or predictions)"
python "$HERE/pdt_s2_props.py" --g4 "$OUT/g4" --ext "$ROOT/ext" --e2 "$ROOT/inputs/e2" \
  --weights "$ROOT/weights/yolo12s_seed0_best.pt" --out "$OUT/ext_props"

echo "== packing"
mkdir -p "$OUT/ext_labels"
for d in "$ROOT"/ext/*/pool; do s=$(basename "$(dirname "$d")"); mkdir -p "$OUT/ext_labels/$s"; cp "$d"/labels.parquet "$d"/images.csv "$OUT/ext_labels/$s/"; done
cp "$ROOT/ext/prepare_report.json" "$ROOT/ext/fetch_report.json" "$OUT/"
cp -r "$ROOT/ext/overlays" "$OUT/"
cp "$HERE"/session2.log "$OUT/" 2>/dev/null || true
tar -czf "$ROOT/session2_outputs.tar.gz" -C "$OUT" --exclude="session1/ultra" --exclude="*/ultra" \
  session1 g4 s2a s2b ext_props ext_labels overlays prepare_report.json fetch_report.json $( [ -f "$OUT/session2.log" ] && echo session2.log )
sha256sum "$ROOT/session2_outputs.tar.gz"
rm -f ~/.kaggle/kaggle.json
echo "== done: download $ROOT/session2_outputs.tar.gz, then terminate the pod"

#!/usr/bin/env bash
# Pod session 2, resume after the G4 step stopped (data-loading workers capped at 16 and file-system tensor sharing in pdt_s2_g4.py).
# Everything before G4 (packages, RDD2022, checkpoints, reproduction gate, downloads, checksums, preparation)
# already completed in this pod and is reused. The steps below are identical to run_session2.sh.
set -euo pipefail
ROOT=/workspace/run
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$ROOT/outputs"
ulimit -n 65536 2>/dev/null || true
rm -rf "$OUT/g4"
echo "== gates G4 to G6"
MODEL_CACHE="$ROOT/models" python "$HERE/pdt_s2_g4.py" --ext "$ROOT/ext" --c0 "$ROOT/inputs/c0" \
  --kaggle "$ROOT/data" --out "$OUT/g4"

echo "== setup A: external inference (predictions only) and the July per-country rerun"
python "$HERE/pdt_s2_infer_a.py" external --weights "$ROOT/weights" --g4 "$OUT/g4" --ext "$ROOT/ext" \
  --work "$ROOT/infer" --out "$OUT/s2a"
python "$HERE/pdt_s2_infer_a.py" country025 --weights "$ROOT/weights" --session1 "$OUT/session1" --out "$OUT/s2a"

echo "== setup B environment (Ultralytics 8.4.153, rfdetr 1.12.1) and inference"
python -m venv --system-site-packages "$ROOT/envB"
"$ROOT/envB/bin/pip" install -q "ultralytics==8.4.153" "rfdetr==1.12.1" "supervision==0.31.7" huggingface_hub
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

#!/usr/bin/env bash
# Pod session 2, resume at setup B. Setup A inference and the July rerun are complete in this pod and are kept.
# The setup B environment did not see the setup A packages (they live in /opt/venv, not the system Python),
# so it now installs its own; nothing else changes. The steps below are identical to run_session2.sh.
set -euo pipefail
ROOT=/workspace/run
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$ROOT/outputs"
rm -rf "$ROOT/envB" "$OUT/s2b"
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

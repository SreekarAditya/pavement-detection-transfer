#!/usr/bin/env bash
# Task 02: timing of local fine-tuning on the A40 (no scoring). Graphics-card pod, auto-stop at 2 hours.
# Needs the draft calibration list of task 01: SPLIT_DRAFT=<folder holding ecupdd_calibration.txt>.
source "$(dirname "$0")/common.sh"
need SPLIT_DRAFT
start_task 02
setup_env gpu
fetch_lock
par fetch_best fetch_best
par fetch_ecupdd fetch_ext ecupdd
wait_all
( while true; do nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader; sleep 5; done ) \
  >"$OUT/nvidia_smi.csv" & SMI=$!
python "$S/pr_finetune.py" timing --root "$LOCK" --out "$OUT" --work "$WS/pr_work" --ext_root "$RUN/ext" \
  --weights_dir "$RUN/weights" --cal_list "$SPLIT_DRAFT/ecupdd_calibration.txt" --collection ecupdd --n 100 \
  --epochs 30 --checkpoints yolo12m_seed0 yolov5s_seed0 --manifest_sha "$MANIFEST_SHA"
# step 5: inference timing on the whole EcuPDD pool (registered pipeline); only the row count is kept
python "$S/pr_lists.py" --root "$LOCK" --out "$OUT/lists" --ext_root "$RUN/ext" --collections ecupdd
T0=$(date +%s)
python "$S/pr_infer.py" val --root "$LOCK" --out "$WS/t02_pred" --work "$WS/t02_work" \
  --weights yolo12m_seed0="$RUN/weights/yolo12m_seed0_best.pt" --pool ecupdd="$OUT/lists/ecupdd.txt"
T1=$(date +%s)
python - "$WS/t02_pred/yolo12m_seed0/ecupdd.parquet" "$OUT/lists/ecupdd.txt" "$((T1 - T0))" "$OUT/timing_inference.json" <<'PYEOF'
import json, sys
import pyarrow.parquet as pq
rows = pq.ParquetFile(sys.argv[1]).metadata.num_rows
n = sum(1 for l in open(sys.argv[2]) if l.strip()); secs = int(sys.argv[3])
json.dump(dict(checkpoint="yolo12m_seed0", pool="ecupdd", images=n, prediction_rows=rows, seconds=secs,
               images_per_second=round(n / max(secs, 1), 2)), open(sys.argv[4], "w"), indent=1)
PYEOF
rm -rf "$WS/t02_pred" "$WS/t02_work"   # predictions are not kept (task 02, step 5)
kill "$SMI" || true
finish_task

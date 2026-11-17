#!/usr/bin/env bash
# Task 06: inference runs for experiments 6 and 15 on one A40 (48 GB), then their scoring on the same pod.
# Before starting, send the Mac folder ~/Desktop/yolo-rdd2022-benchmark/epoch100/ to $RUN/weights_last/ with
# runpodctl (never through a public location). Three inference processes share the card at the same time:
# experiment 6 best.pt, experiment 6 last.pt, experiment 15 variants.
source "$(dirname "$0")/common.sh"
need RULES_SHA PARAMS_SHA
start_task 06
setup_env gpu
fetch_lock
fetch_best
par rdd fetch_rdd
par ext fetch_ext ecupdd ird svrdd
wait_all
check_weights last "$RUN/weights_last"
python "$S/pr_last_epoch.py" "$RUN/weights_last" "$OUT/last_epoch_check.csv"
reproduction_gate | tee "$OUT/reproduction_gate.txt"
python "$S/pr_lists.py" --root "$LOCK" --out "$OUT/lists" --val_list "$RUN/session1/lists/val_all.txt" \
  --ext_root "$RUN/ext" --collections ecupdd ird svrdd
python "$S/pr_score_gpu.py" scales --root "$LOCK" --out "$OUT/scores" "${GATE[@]}"
RUNS=$(python -c "import sys; sys.path.insert(0, '$S'); import pr_common as PC; print(' '.join(PC.RUNS_A))")
L="$OUT/lists"
POOLS6=(S1="$L/S1.txt" S2="$L/S2.txt" ecupdd="$L/ecupdd.txt" ird="$L/ird.txt" svrdd="$L/svrdd.txt")
WB=(); WL=()
for r in $RUNS; do WB+=("best_$r=$RUN/weights/${r}_best.pt"); WL+=("last_$r=$RUN/weights_last/${r}_last.pt"); done
read -r TOV TNMS <<<"$(python -c "import yaml; e = yaml.safe_load(open('$RULES/params.yaml'))['exp15']; print(e['tile_overlap'], e['tile_merge']['iou'])")"
e15() {
  local W=(yolo12m_seed0="$RUN/weights/yolo12m_seed0_best.pt" yolo12s_seed0="$RUN/weights/yolo12s_seed0_best.pt")
  python "$S/pr_infer.py" variants --root "$LOCK" --out "$OUT/pred_e15" --work "$WS/w15" --weights "${W[@]}" --tile_overlap "$TOV" --tile_nms_iou "$TNMS" \
    --pool Norway="$L/Norway.txt" --variants untiled_640 crop tile_2x2 input_1280
  python "$S/pr_infer.py" variants --root "$LOCK" --out "$OUT/pred_e15" --work "$WS/w15" --weights "${W[@]}" \
    --pool United_States="$L/United_States.txt" Japan="$L/Japan.txt" --variants untiled_640
  python "$S/pr_infer.py" variants --root "$LOCK" --out "$OUT/pred_e15" --work "$WS/w15" --weights "${W[@]}" \
    --pool United_States_down="$L/United_States.txt" Japan_down="$L/Japan.txt" --variants downscaled \
    --scales "$OUT/scores/e15_scales_flat.json"
}
( while true; do nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader; sleep 30; done ) \
  >"$OUT/nvidia_smi.csv" & SMI=$!
par e6_best python "$S/pr_infer.py" val --root "$LOCK" --out "$OUT/pred_e6" --work "$WS/w6b" --weights "${WB[@]}" --pool "${POOLS6[@]}"
par e6_last python "$S/pr_infer.py" val --root "$LOCK" --out "$OUT/pred_e6" --work "$WS/w6l" --weights "${WL[@]}" --pool "${POOLS6[@]}"
par e15 e15
wait_all
kill "$SMI" || true
# scoring (processor work on this pod); predictions are complete and hashed first
python -c "import sys; sys.path.insert(0, '$S'); import pr_common as PC; PC.write_manifest('$OUT/pred_e6'); PC.write_manifest('$OUT/pred_e15')"
par score_e6 python "$S/pr_score_gpu.py" e6 --root "$LOCK" --out "$OUT/scores_e6" --pred "$OUT/pred_e6" "${GATE[@]}"
par score_e15 python "$S/pr_score_gpu.py" e15 --root "$LOCK" --out "$OUT/scores_e15" --pred "$OUT/pred_e15" "${GATE[@]}"
wait_all
finish_task

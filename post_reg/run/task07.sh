#!/usr/bin/env bash
# Task 07: local fine-tuning for experiment 5 on A40 (48 GB) pods, then its scoring.
# WORKERS fine-tuning processes share the card (default 2; each takes every WORKERS-th row of the run table).
# MAX_HOURS stops each worker at a run boundary; rerunning the script resumes (finished rows are skipped).
source "$(dirname "$0")/common.sh"
need RULES_SHA PARAMS_SHA RUN_TABLE_SHA
WORKERS="${WORKERS:-2}"; MAX_HOURS="${MAX_HOURS:-14}"
start_task 07
setup_env gpu
fetch_lock
par best fetch_best
wait_all
par rdd fetch_rdd
par ext fetch_ext ecupdd ird svrdd
wait_all
reproduction_gate | tee "$OUT/reproduction_gate.txt"
T="$RULES/exp5_run_table.csv"
G=(--rules "$RULES" --rules_sha "$RULES_SHA" --params_sha "$PARAMS_SHA" --run_table "$T" --run_table_sha "$RUN_TABLE_SHA"
   --manifest_sha "$MANIFEST_SHA")
[ -d "$OUT/samples" ] || python "$S/pr_finetune.py" samples --root "$LOCK" --out "$OUT" "${G[@]}"
for w in $(seq 0 $((WORKERS - 1))); do
  ROWS=$(python -c "import pandas as pd; t = pd.read_csv('$T'); print(' '.join(t.run_id[$w::$WORKERS]))")
  par "train_$w" python "$S/pr_finetune.py" train --root "$LOCK" --out "$OUT/worker_$w" --work "$WS/w7_$w" \
    --samples "$OUT/samples" --ext_root "$RUN/ext" --rdd_list "$RUN/session1/lists/val_all.txt" \
    --weights_dir "$RUN/weights" --max_hours "$MAX_HOURS" --only $ROWS "${G[@]}"
done
wait_all
# one prediction folder for scoring (links into the workers' folders) and one run log
mkdir -p "$OUT/pred"
for w in $(seq 0 $((WORKERS - 1))); do
  for d in "$OUT/worker_$w/pred"/*/; do
    [ -d "$d" ] || continue
    n=$(basename "$d"); [ -e "$OUT/pred/$n" ] || ln -s "$d" "$OUT/pred/$n"
  done
done
python "$S/pr_runs_merge.py" --out "$OUT" --run_table "$T"
python -c "import sys; sys.path.insert(0, '$S'); import pr_common as PC; PC.write_manifest('$OUT/pred')"
python "$S/pr_score_gpu.py" e5 --root "$LOCK" --out "$OUT/scores" --pred "$OUT/pred" --runs "$T" \
  --samples "$OUT/samples" "${GATE[@]}"
finish_task

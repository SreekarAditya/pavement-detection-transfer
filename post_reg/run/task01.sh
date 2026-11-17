#!/usr/bin/env bash
# Task 01: pre-checks before the rule document (label-free, outcome-free). Processor-only pod.
# Usage: bash task01.sh            (the four table steps run at the same time; weights check after)
source "$(dirname "$0")/common.sh"
start_task 01
setup_env cpu
fetch_lock
for s in sizes preds tokens split; do
  par "$s" python "$S/pr_prechecks.py" --root "$LOCK" --out "$OUT/$s" --steps "$s"
done
par fetch_best fetch_best
wait_all
python "$S/pr_prechecks.py" --root "$LOCK" --out "$OUT/weights" --steps weights --weights "$RUN/weights" \
  --weights_table "$PR/run/weights_hashes.csv"
finish_task

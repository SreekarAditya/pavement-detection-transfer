#!/usr/bin/env bash
# Task 04: benchmark, robustness and survey tables (experiments 7, 8, 9, 10, 12, 13, 14, 16, 19) and the
# code-path check. Processor-only pod. Parts run at the same time, each in its own folder.
source "$(dirname "$0")/common.sh"
need RULES_SHA PARAMS_SHA
start_task 04
setup_env cpu
fetch_lock
# the code-path check runs first, alone, on every processor; nothing else starts unless it reproduces the
# registered setup A and setup B primary values exactly
export PR_WORKERS="$NCPU"
python "$S/pr_bench.py" --root "$LOCK" --out "$OUT/codepath" --parts codepath --scoring "$(dirname "$SCORE_JSON")" \
  "${GATE[@]}" || die "code-path check failed; see $OUT/codepath/codepath_check.csv"
PARTS=(e7 e8 e9 e10 e12 e13_14_19 e16)
export PR_WORKERS=$(( NCPU / ${#PARTS[@]} > 1 ? NCPU / ${#PARTS[@]} : 1 ))
for p in "${PARTS[@]}"; do
  par "$p" python "$S/pr_bench.py" --root "$LOCK" --out "$OUT/$p" --parts "$p" --scoring "$(dirname "$SCORE_JSON")" "${GATE[@]}"
done
wait_all
finish_task

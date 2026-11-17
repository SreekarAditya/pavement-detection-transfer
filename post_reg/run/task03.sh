#!/usr/bin/env bash
# Task 03: procedure steps on the sealed predictions (experiments 2, 3 and 4). Processor-only pod.
# Experiment 2 and experiments 3 and 4 run at the same time, each on half of the processors.
source "$(dirname "$0")/common.sh"
need RULES_SHA PARAMS_SHA
start_task 03
setup_env cpu
fetch_lock
export PR_WORKERS=$(( NCPU / 2 > 1 ? NCPU / 2 : 1 ))
par e2 python "$S/pr_exp2.py" --root "$LOCK" --out "$OUT/e2" "${GATE[@]}"
par e3_e4 python "$S/pr_exp3.py" --root "$LOCK" --out "$OUT/e3" --part all "${GATE[@]}"
wait_all
finish_task

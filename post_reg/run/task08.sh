#!/usr/bin/env bash
# Task 08: experiment 1 (primary). Processor-only pod.
#   bash task08.sh constants   writes exp1_constants.yaml from task 03's outputs; Aditya approves, commits it to
#                              post_reg/rules/ and tags post-reg-exp1-constants
#   bash task08.sh chain       the chain against floors and ceiling (needs CONSTANTS_SHA)
# Inputs from the Mac, unpacked on the pod: T03=<task 03 output folder>, T07=<task 07 output folder>.
source "$(dirname "$0")/common.sh"
need RULES_SHA PARAMS_SHA T03
STEP="${1:?constants or chain}"
start_task "08_$STEP"
setup_env cpu
fetch_lock
case "$STEP" in
  constants)
    python "$S/pr_exp1.py" constants --root "$LOCK" --out "$OUT" --e2 "$T03/e2" --e3 "$T03/e3" "${GATE[@]}" ;;
  chain)
    need CONSTANTS_SHA T07
    C="$RULES/exp1_constants.yaml"
    COMMON=(--root "$LOCK" --constants "$C" --constants_sha "$CONSTANTS_SHA" --e5 "$T07/scores"
            --samples "$T07/samples" --pred5 "$T07/pred")
    export PR_WORKERS=$(( NCPU / 2 > 1 ? NCPU / 2 : 1 ))
    par selected python "$S/pr_exp1.py" chain --out "$OUT/selected" "${COMMON[@]}" "${GATE[@]}"
    par sensitivity python "$S/pr_exp1.py" chain --out "$OUT/all_checkpoints" --sensitivity "${COMMON[@]}" "${GATE[@]}"
    wait_all ;;
  *) die "step must be constants or chain" ;;
esac
finish_task

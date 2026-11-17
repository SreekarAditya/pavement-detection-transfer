#!/usr/bin/env bash
# Task 05: what the review queue contains (experiment 11). Processor-only pod, container disk 100 GB.
# Usage: bash task05.sh a        classification from labels (Part A)
#        bash task05.sh b        rating packets for the two raters (sent to the Mac; nothing pre-filled)
#        bash task05.sh c ANSWERS_RATER1.json ANSWERS_RATER2.json [ADJUDICATION.json]
source "$(dirname "$0")/common.sh"
need RULES_SHA PARAMS_SHA
PART="${1:?part a, b or c}"
start_task "05${PART^^}"
setup_env cpu
fetch_lock
fetch_ext ecupdd ird svrdd
case "$PART" in
  a) python "$S/pr_exp11.py" a --root "$LOCK" --out "$OUT" --ext_root "$RUN/ext" "${GATE[@]}" ;;
  b) [ -d "$OUTBASE/05A" ] || die "run part a first (its output is the sample frame)"
     cp -r "$OUTBASE/05A/." "$OUT/" ; rm -f "$OUT/MANIFEST.sha256"
     python "$S/pr_exp11.py" b --root "$LOCK" --out "$OUT" --ext_root "$RUN/ext" "${GATE[@]}" ;;
  c) [ -n "${2:-}" ] && [ -n "${3:-}" ] || die "part c needs both raters' answer files"
     cp -r "$OUTBASE/05B/." "$OUT/" ; rm -f "$OUT/MANIFEST.sha256"
     python "$S/pr_exp11.py" c --root "$LOCK" --out "$OUT" --ext_root "$RUN/ext" --answers "$2" "$3" \
       ${4:+--adjudication "$4"} "${GATE[@]}" ;;
  *) die "part must be a, b or c" ;;
esac
finish_task

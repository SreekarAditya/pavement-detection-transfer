#!/usr/bin/env bash
# Task 10: revision analyses (post_reg/rules/RULES_revision.md). One runner, processor-only pod.
#   bash task10.sh cpu       R1-R4 over 51 splits and the R7 sweep, in parallel (needs the revision tag)
#   bash task10.sh clauses   R5-R6: frozen evaluator and SHACL on the clause fixtures (needs ANSWERS_SHA)
#   bash task10.sh all       cpu, then clauses if ANSWERS_SHA is set (otherwise stops after cpu and says so)
# Environment (set from the task header):
#   RULES_SHA PARAMS_SHA CONSTANTS_SHA   registered rules, as in task 08
#   REVISION_SHA REVISION_PARAMS_SHA      sha256 of RULES_revision.md and params_revision.yaml at post-reg-revision-v1
#   T08                                   unpacked post_reg_08_chain_outputs (registered summary for the gate)
#   T07                                   unpacked post_reg_07_outputs (scores, samples, pred), as in task 08
#   ANSWERS_SHA                           sha256 of the completed expected-answer file (clauses step only)
source "$(dirname "$0")/common.sh"
REV="$PR/revision"
STEP="${1:?cpu, clauses or all}"

check_revision_tag() {
  # The pod receives the repository as files, not as a git clone (as in tasks 03-09), so tags are verified on
  # the Mac before sending; here the committed file hashes are checked instead.
  [ "$(sha256sum "$RULES/RULES_revision.md" | cut -d' ' -f1)" = "$REVISION_SHA" ] || die "RULES_revision.md hash differs"
  [ "$(sha256sum "$RULES/params_revision.yaml" | cut -d' ' -f1)" = "$REVISION_PARAMS_SHA" ] || die "params_revision.yaml hash differs"
}

run_splits() {   # seeds 3001-3050 (checked against params_revision.yaml by pr_rev.py), bounded parallelism
  local cmd="$1" first="$2" folder="$3" group="$4"
  local slots=$(( NCPU > 2 ? NCPU - 2 : 1 )) fail=0 s
  for s in $(seq "$first" 3050); do
    while [ "$(jobs -rp | wc -l)" -ge "$slots" ]; do wait -n || fail=1; done
    python "$REV/pr_rev.py" "$cmd" --seed "$s" --out "$OUT/$folder/seed_$s" "${COMMON[@]}" >"$LOGDIR/${group}_$s.log" 2>&1 &
  done
  while [ "$(jobs -rp | wc -l)" -gt 0 ]; do wait -n || fail=1; done
  [ "$fail" = 0 ] || { echo "at least one split failed; see split_*.log"; return 1; }
}

run_cpu() {
  need RULES_SHA PARAMS_SHA CONSTANTS_SHA REVISION_SHA REVISION_PARAMS_SHA T07 T08 E18 PRED18 REVISION_CPU_RATE REVISION_CPU_STARTED
  check_revision_tag
  NCPU="${REVISION_CPUS:-$NCPU}"
  export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
  start_task "10_cpu"
  setup_env cpu
  fetch_lock
  COMMON=(--root "$LOCK" --rules "$RULES" --constants "$RULES/exp1_constants.yaml"
          --revision "$RULES/params_revision.yaml" --e5 "$T07/scores" --samples "$T07/samples" --pred5 "$T07/pred"
          --registered "$T08/selected/e1_summary.csv" --ext18 "$E18" --pred18 "$PRED18")
  export PR_WORKERS=1

  # 1. Gate: registered split, 2,000 draws, must equal the registered summary exactly; R7 gate at 0.5 likewise.
  log "gate: registered split (seed 2026, 2000 draws) and matching at 0.5"
  par gate_split python "$REV/pr_rev.py" split --seed 2026 --draws 2000 --out "$OUT/gate/seed_2026" "${COMMON[@]}"
  par gate_iou   python "$REV/pr_rev.py" r7 --iou 0.5 --out "$OUT/gate/iou_050" "${COMMON[@]}"
  wait_all
  python "$REV/pr_rev.py" gate --registered "$T08/selected/e1_summary.csv" \
         --split "$OUT/gate/seed_2026" --iou "$OUT/gate/iou_050" --out "$OUT/gate" "${COMMON[@]}" \
    || die "reproduction gate failed: nothing else runs"

  # 2. Fifty splits in parallel (one process per split, NCPU-2 at a time), and the R7 sweep beside them.
  log "50 splits and the R7 sweep (0.1-0.4; 0.5 is the gate run) on $NCPU processors"
  par r7 bash -c 'for x in 0.1 0.2 0.3 0.4; do python "$0/pr_rev.py" r7 --iou "$x" --out "$1/r7/iou_$x" "${@:2}" || exit 1; done' \
      "$REV" "$OUT" "${COMMON[@]}"
  first_start=$(date +%s)
  python "$REV/pr_rev.py" split --seed 3001 --out "$OUT/splits/seed_3001" "${COMMON[@]}" >"$LOGDIR/split_3001.log" 2>&1
  first_seconds=$(( $(date +%s) - first_start ))
  python - "$OUT/projection.json" "$first_seconds" "$NCPU" "$RULES/params_revision.yaml" <<'PROJECTION'
from datetime import datetime, timezone
from pathlib import Path
import json,os,sys,time,yaml
dest,first,cpus,rules=sys.argv[1:]
config=yaml.safe_load(Path(rules).read_text())
seconds=int(first)
slots=max(1,int(cpus)-2)
rate=float(os.environ["REVISION_CPU_RATE"])
elapsed=time.time()-float(os.environ["REVISION_CPU_STARTED"])
def usd(t): return rate*t/3600
selected=usd(elapsed+seconds*(49/slots+4))
extra=usd(19*50*seconds/slots)
sensitivity_limit=config["r1"]["sensitivity_checkpoints"]["run_only_if_projected_cost_below_usd"]
note=dict(first_split_seconds=seconds,slots=slots,projected_selected_usd=selected,
          projected_sensitivity_extra_usd=extra,quoted_hourly_usd=rate,elapsed_seconds=elapsed,
          run_sensitivity=extra<sensitivity_limit,user_removed_task_spending_stop=os.environ.get("REVISION_UNCAPPED")=="1")
Path(dest).write_text(json.dumps(note,indent=2)+"\n")
print(json.dumps(note))
PROJECTION
  if [ "${REVISION_UNCAPPED:-0}" != 1 ] && python -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["projected_selected_usd"]>2 else 1)' "$OUT/projection.json"; then
    die "projected selected run exceeds USD 2; see projection.json"
  fi
  par splits run_splits split 3002 splits split
  wait_all
  if python -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["run_sensitivity"] else 1)' "$OUT/projection.json"; then
    run_splits sensitivity 3001 sensitivity sensitivity
  fi

  # 3. Pool and report R1-R4 (and the seed-2026 row beside them); cost stamp for the sensitivity decision.
  python "$REV/pr_rev.py" aggregate --splits "$OUT/splits" --gate "$OUT/gate" --r7 "$OUT/r7" --out "$OUT/report" "${COMMON[@]}"
  finish_task
}

run_clauses() {
  need ANSWERS_SHA REVISION_SHA REVISION_PARAMS_SHA T08
  check_revision_tag
  A="$WS/run/inputs/revision_answers.csv"     # private copy (holds short AASHTO R 85 quotes), sent from the Mac
  C="$REV/clauses/ANSWERS_SHA256.txt"         # committed at tag post-reg-revision-answers
  [ -f "$C" ] || die "committed answer hash missing: answers tag not in the files sent"
  [ "$(cut -d' ' -f1 "$C")" = "$ANSWERS_SHA" ] || die "ANSWERS_SHA differs from the committed hash"
  [ "$(sha256sum "$A" | cut -d' ' -f1)" = "$ANSWERS_SHA" ] || die "answer file hash differs from the committed hash"
  start_task "10_clauses"
  setup_env cpu
  uv pip install -q pyshacl rdflib
  python "$REV/pr_rev.py" clauses --fixtures "$REV/clauses/fixtures.csv" --answers "$A" \
         --inventory "$REV/clauses/clause_inventory.csv" --shapes "$REV/clauses/shapes.ttl" \
         --evaluator_tag qualification-evaluator-v1 --registered "$T08/selected/e1_summary.csv" --out "$OUT"
  finish_task
}

case "$STEP" in
  cpu) run_cpu ;;
  clauses) run_clauses ;;
  all)
    run_cpu
    if [ -n "${ANSWERS_SHA:-}" ]; then run_clauses
    else log "clauses not run: ANSWERS_SHA not set (expected answers not yet written and committed)"; fi ;;
  *) die "step must be cpu, clauses or all" ;;
esac

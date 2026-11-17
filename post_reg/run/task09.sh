#!/usr/bin/env bash
# Task 09: networks never opened before (experiment 18). Run once; each phase ends with a commit and tag of its
# hashes before the next starts, and its archive goes to the Mac.
#   bash task09.sh A        processor pod: download (hash-checked), preparation, overlap removal, blocks,
#                           eligibility, split; held-out labels to private/ (hashed, not read again before D)
#   bash task09.sh B        A40 pod: predictions of the 20 setup A checkpoints on every eligible network, and of
#                           the six setup B checkpoints on networks eligible for the merged-crack endpoint
#   bash task09.sh C        processor pod: threshold, pilot estimate and decision (calibration labels only)
#   bash task09.sh C_adapt  A40 pod, only if phase C listed networks in adapt_networks.txt
#   bash task09.sh C_commit processor pod: exp18_commitments.json (commit and tag post-reg-exp18-committed)
#   bash task09.sh D        processor pod: scoring (held-out labels opened after their hashes are checked)
# Before the rules tag: every field of exp18_candidates.csv filled (the run stops on blank or '?'). Folder and
# label paths come from the archive's member names, read from the zip directory alone (no image downloaded):
#   python scripts/pr_list_archive.py URL --out <slug>_members.csv
# Optional columns: copy_regex (export-tool copies, group 1 = source name), parent_regex (crops of one
# photograph, group 1 = photograph id), max_kept_share (largest share the copy collapse may keep).
# Environment, from the earlier phases (unpacked from the Mac where a new pod is used):
#   CANDIDATES_SHA (A)            E18_TAR: phase-A archive (B to D on a new pod)
#   PRED18, PRED18_MANIFEST_SHA   phase-B prediction folder and the hash of its MANIFEST.sha256 (C to D)
#   CONSTANTS_SHA (C to D)        DECISIONS: phase-C e18_decisions.csv (C_adapt, C_commit)
#   ADAPTED: C_adapt output folder, if it ran (C_commit, D)   COMMITMENTS, COMMITMENTS_SHA (D)
source "$(dirname "$0")/common.sh"
need RULES_SHA PARAMS_SHA
PHASE="${1:?A, B, C, C_adapt, C_commit or D}"
E18="$WS/ext18"; RAW="$WS/raw18"
CAND="$RULES/exp18_candidates.csv"
start_task "09_$PHASE"
X18=(--root "$LOCK" --ext18 "$E18" --candidates "$CAND" --raw "$RAW")
SETUP_B=(yolov8n yolov8s yolo26s yolo26m rfdetr_nano rfdetr_small)
sens() {   # sensitivity checkpoints of phase B (after setup_env: needs the analysis environment)
  local others
  read -r -a others <<<"$(python -c "import sys; sys.path.insert(0, '$S'); import pr_common as PC; print(' '.join(r for r in PC.RUNS_A if r != PC.SELECTED))")"
  SENS=(--extra_detectors "${others[@]}" --setup_b "${SETUP_B[@]}")
}
restore_e18() {
  if [ ! -f "$E18/exp18_eligibility.json" ]; then
    need E18_TAR
    tar -xf "$E18_TAR" -C "$WS"
  fi
  python "$S/pr_exp18.py" download "${X18[@]}" --rules "$RULES" --skip_gate   # images behind pool/images links
}
eligible() {   # eligible ENDPOINT|any : network names
  python3 -c "import json; d = json.load(open('$E18/exp18_eligibility.json'))['networks']; \
print(' '.join(s for s, v in sorted(d.items()) if ('$1' == 'any' and any(v['eligible'].values())) or v['eligible'].get('$1')))"
}
case "$PHASE" in
  A)
    need CANDIDATES_SHA
    setup_env cpu; fetch_lock
    sha_ok "$CAND" "$CANDIDATES_SHA"
    [ -d "$RUN/inputs/c0" ] || die "send pod session 2's inputs/c0 (Mac: 05_scripts/p1_session2_bundle.tar.gz) to $RUN/inputs/c0"
    python "$S/pr_exp18.py" download "${X18[@]}" "${GATE[@]}"
    python "$S/pr_exp18.py" prepare "${X18[@]}" "${GATE[@]}"
    MODEL_CACHE="$RUN/models" python "$S/pr_exp18.py" overlap "${X18[@]}" --c0 "$RUN/inputs/c0" --device cpu "${GATE[@]}"
    cp "$E18/exp18_download.json" "$E18/exp18_prepare.json" "$E18/exp18_eligibility.json" "$E18/blocks.csv" "$OUT/"
    cp -r "$E18/rules_split" "$OUT/"
    tar -cf "$OUT/ext18_phaseA.tar" -C "$WS" ext18
    sha256sum "$OUT/ext18_phaseA.tar" "$OUT/exp18_eligibility.json" | tee "$OUT/PHASE_A_SHA.txt" ;;
  B)
    setup_env gpu; fetch_lock; fetch_best; restore_e18
    reproduction_gate | tee "$OUT/reproduction_gate.txt"
    PA=(); for n in $(eligible any); do PA+=("$n=$E18/lists/$n.txt"); done
    [ "${#PA[@]}" -gt 0 ] || die "no eligible network; experiment 18 reports the screening only"
    W=(); for r in $(python -c "import sys; sys.path.insert(0, '$S'); import pr_common as PC; print(' '.join(PC.RUNS_A))"); do
      W+=("$r=$RUN/weights/${r}_best.pt"); done
    par setup_a python "$S/pr_infer.py" val --root "$LOCK" --out "$OUT/pred18" --work "$WS/w18" --weights "${W[@]}" --pool "${PA[@]}"
    PB=(); for n in $(eligible merged_crack); do PB+=("$n=$E18/lists/$n.txt"); done
    if [ "${#PB[@]}" -gt 0 ]; then
      setup_env_b
      par setup_b "$WS/envB/bin/python" "$S/pr_infer_b.py" --root "$LOCK" --out "$OUT/pred18" --pool "${PB[@]}" \
        --cache "$RUN/hf_b"
    fi
    wait_all
    python -c "import sys; sys.path.insert(0, '$S'); import pr_common as PC; PC.write_manifest('$OUT/pred18')"
    sha256sum "$OUT/pred18/MANIFEST.sha256" | tee "$OUT/PRED18_MANIFEST_SHA.txt" ;;
  C)
    need CONSTANTS_SHA PRED18_MANIFEST_SHA PRED18
    setup_env cpu; fetch_lock; restore_e18; sens
    python "$S/pr_exp18.py" decide "${X18[@]}" --pred "$PRED18" --out "$OUT" "${SENS[@]}" \
      --constants "$RULES/exp1_constants.yaml" --constants_sha "$CONSTANTS_SHA" \
      --pred_manifest_sha "$PRED18_MANIFEST_SHA" "${GATE[@]}" ;;
  C_adapt)
    need PRED18 DECISIONS
    setup_env gpu; fetch_lock; fetch_best; restore_e18
    mapfile -t NETS < "$(dirname "$DECISIONS")/adapt_networks.txt"
    [ "${#NETS[@]}" -gt 0 ] || die "no network needs adaptation"
    python "$S/pr_exp18.py" adapt "${X18[@]}" --pred "$PRED18" --out "$OUT" --slugs "${NETS[@]}" \
      --work "$WS/w18a" --weights_dir "$RUN/weights" "${GATE[@]}" ;;
  C_commit)
    need CONSTANTS_SHA PRED18_MANIFEST_SHA PRED18 DECISIONS
    setup_env cpu; fetch_lock; restore_e18
    python "$S/pr_exp18.py" commit "${X18[@]}" --pred "$PRED18" --out "$OUT" --decisions "$DECISIONS" \
      ${ADAPTED:+--adapted "$ADAPTED"} --constants "$RULES/exp1_constants.yaml" --constants_sha "$CONSTANTS_SHA" \
      --pred_manifest_sha "$PRED18_MANIFEST_SHA" "${GATE[@]}"
    sha256sum "$OUT/exp18_commitments.json" | tee "$OUT/COMMITMENTS_SHA.txt" ;;
  D)
    need CONSTANTS_SHA PRED18_MANIFEST_SHA PRED18 COMMITMENTS COMMITMENTS_SHA
    setup_env cpu; fetch_lock; restore_e18; sens
    python "$S/pr_exp18.py" score "${X18[@]}" --pred "$PRED18" --out "$OUT" "${SENS[@]}" \
      --constants "$RULES/exp1_constants.yaml" --constants_sha "$CONSTANTS_SHA" \
      --pred_manifest_sha "$PRED18_MANIFEST_SHA" --commitments "$COMMITMENTS" --commitments_sha "$COMMITMENTS_SHA" \
      ${ADAPTED:+--adapted "$ADAPTED"} "${GATE[@]}" ;;
  *) die "phase must be A, B, C, C_adapt, C_commit or D" ;;
esac
finish_task

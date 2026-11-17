#!/usr/bin/env bash
# Synthetic end-to-end test of every analysis script (no real data, no labels, a few minutes on a processor).
# Run on each new pod after setup_env, before any real step:  bash run_tests.sh [TEST_DIR]
# Needs the locked scripts (fetch_lock) because the analysis code imports them. GPU=1 also tests inference and
# fine-tuning with two checkpoints on the card.
source "$(dirname "$0")/common.sh"
T="${1:-$WS/selftest}"
rm -rf "$T"; mkdir -p "$T"
export PYTHONPATH="$S" PR_WORKERS="${PR_WORKERS:-4}"
LS="${LOCK_SCRIPTS:-$LOCK/scripts}"
[ -f "$LS/pdt_core.py" ] || die "locked scripts not found in $LS (run fetch_lock first)"
SK=(--skip_gate)
step() { log "test: $1"; }

step "synthetic root and rules"
python "$PR/tests/pr_synth.py" --out "$T/root" --lock_scripts "$LS" --scale "${SCALE:-0.2}" >/dev/null
python "$PR/tests/make_rules.py" --params "$PR/rules_draft/params.yaml" --root "$T/root" --out "$T/rules"
R=(--root "$T/root" --rules "$T/rules")
LOGDIR="$T/logs"; mkdir -p "$LOGDIR"

step "estimators (coverage of the ratio and prediction-powered intervals)"
python "$PR/tests/test_estimators.py"

step "task 01 pre-checks"
python "$S/pr_prechecks.py" --root "$T/root" --out "$T/01" --skip_verify --steps sizes preds tokens split

step "task 03: experiments 2, 3, 4 (in parallel)"
par t_e2 python "$S/pr_exp2.py" "${R[@]}" --out "$T/03/e2" "${SK[@]}"
par t_e3 python "$S/pr_exp3.py" "${R[@]}" --out "$T/03/e3" --part all "${SK[@]}"
wait_all

step "task 04: benchmark parts (code path against the locked scoring on the same synthetic root)"
python "$PR/tests/run_locked_score.py" "$T/root" "$T/score" >/dev/null 2>&1 || log "locked scoring skipped (small root)"
BP=(e7 e8 e9 e10 e12 e13_14_19 e16); [ -f "$T/score/score_results.json" ] && BP+=(codepath)
for p in "${BP[@]}"; do
  par "t_$p" python "$S/pr_bench.py" "${R[@]}" --out "$T/04/$p" --parts "$p" --scoring "$T/score" "${SK[@]}"
done
wait_all

step "task 05: review queue parts a and b"
python "$PR/tests/make_ext_root.py" "$T/root" "$T/ext"
python "$S/pr_exp11.py" a "${R[@]}" --out "$T/05" --ext_root "$T/ext" "${SK[@]}"
python "$S/pr_exp11.py" b "${R[@]}" --out "$T/05" --ext_root "$T/ext" "${SK[@]}"

step "task 07 planning: run table and samples"
python "$S/pr_finetune.py" plan --root "$T/root" --rules "$T/rules" --out "$T/07" --skip_gate
python "$S/pr_finetune.py" samples --root "$T/root" --rules "$T/rules" --out "$T/07" --run_table "$T/07/exp5_run_table.csv" --skip_gate

step "tasks 06 and 07 scoring on stand-in predictions"
python "$PR/tests/make_gpu_preds.py" --root "$T/root" --rules "$T/rules" --out "$T/gpu" --runs "$T/07/exp5_run_table.csv" >/dev/null
python "$S/pr_score_gpu.py" scales "${R[@]}" --out "$T/06" "${SK[@]}"
par t_s6 python "$S/pr_score_gpu.py" e6 "${R[@]}" --out "$T/06/e6" --pred "$T/gpu/e6" "${SK[@]}"
par t_s15 python "$S/pr_score_gpu.py" e15 "${R[@]}" --out "$T/06/e15" --pred "$T/gpu/e15" "${SK[@]}"
par t_s5 python "$S/pr_score_gpu.py" e5 "${R[@]}" --out "$T/07/scores" --pred "$T/gpu/e5" \
  --runs "$T/07/exp5_run_table.csv" --samples "$T/07/samples" "${SK[@]}"
wait_all

step "task 08: constants and chain (with the adaptation arm)"
python "$S/pr_exp1.py" constants "${R[@]}" --out "$T/08" --e2 "$T/03/e2" --e3 "$T/03/e3" "${SK[@]}"
python "$S/pr_exp1.py" chain "${R[@]}" --out "$T/08/chain" --constants "$T/08/exp1_constants.yaml" \
  --e5 "$T/07/scores" --samples "$T/07/samples" --pred5 "$T/gpu/e5" "${SK[@]}"

step "task 09: prepare, overlap (hashes), decide, commit, score, and tampered commitments"
F="$T/x18"
python "$PR/tests/make_exp18_fixture.py" build --root "$T/root" --rules "$T/rules" --out "$F" >/dev/null 2>&1
X=(--root "$F/root" --rules "$F/rules" --ext18 "$F/ext18" --candidates "$F/rules/exp18_candidates.csv")
SENS=(--extra_detectors yolo12s_seed0 --setup_b yolov8s)
C1=(--constants "$T/08/exp1_constants.yaml")
python "$S/pr_exp18.py" prepare "${X[@]}" "${SK[@]}"
python "$S/pr_exp18.py" overlap "${X[@]}" --c0 "$F/c0" --device cpu --hash_only_test "${SK[@]}"
python "$PR/tests/make_exp18_fixture.py" preds --out "$F"
python "$S/pr_exp18.py" decide "${X[@]}" --pred "$F/pred18" --out "$F/dec" "${SENS[@]}" "${C1[@]}" "${SK[@]}"
python "$S/pr_exp18.py" commit "${X[@]}" --pred "$F/pred18" --out "$F/com" --decisions "$F/dec/e18_decisions.csv" \
  "${C1[@]}" "${SK[@]}"
python "$S/pr_exp18.py" score "${X[@]}" --pred "$F/pred18" --out "$F/e18" --commitments "$F/com/exp18_commitments.json" \
  "${SENS[@]}" "${C1[@]}" "${SK[@]}"
python - "$F/com/exp18_commitments.json" "$F/com/tampered.json" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1])); r = c["procedure"][0]
r["decision"] = "adapt" if r["decision"] == "accept" else "accept"
json.dump(c, open(sys.argv[2], "w"))
PYEOF
if python "$S/pr_exp18.py" score "${X[@]}" --pred "$F/pred18" --out "$F/e18_bad" --commitments "$F/com/tampered.json" \
     "${SENS[@]}" "${C1[@]}" "${SK[@]}" >/dev/null 2>&1; then
  die "tampered commitments were not caught"
fi
python "$PR/tests/test_infer_b.py" "$LS"
python "$PR/tests/test_exp18_inputs.py" "$F"

if [ "${GPU:-0}" = 1 ]; then
  step "graphics card: inference (registered pipeline and variants) and one-epoch fine-tuning"
  W=(yolo12n_seed0="$RUN/weights/yolo12n_seed0_best.pt" yolov5s_seed0="$RUN/weights/yolov5s_seed0_best.pt")
  find "$F/raw/alpha/images" -name '*.jpg' | sort | sed -n '1,8p' > "$T/imgs.txt"   # sed reads the whole stream
  python "$S/pr_infer.py" val --root "$LOCK" --out "$T/gpu_val" --work "$T/w" --weights "${W[@]}" --pool p="$T/imgs.txt"
  python "$S/pr_infer.py" variants --root "$LOCK" --out "$T/gpu_var" --work "$T/w" --weights "${W[@]}" \
    --pool p="$T/imgs.txt" --variants untiled_640 crop tile_2x2 input_1280
fi
log "all synthetic tests passed"

#!/usr/bin/env bash
set -euo pipefail
umask 077
export SSL_CERT_FILE=/etc/ssl/cert.pem
export PYTHONDONTWRITEBYTECODE=1
repo="$PWD/02_plan/revision/.release"
code="$repo/post_reg/revision/amendment41"
private="$PWD/02_plan/revision/private/amendment41"
out="$PWD/06_pod_outputs/post_reg/revision/remaining_experiments/attempt_01"
python3 "$code/preflight.py" "$PWD/08_lock/root" > "$out/shared-preflight.log"
git -C "$repo" diff --exit-code qualification-evaluator-v2 -- post_reg/revision/qualification_v2 post_reg/scripts
python3 -m unittest discover -s "$repo/post_reg/revision/qualification_v2" -p 'test_*.py' >> "$out/shared-preflight.log" 2>&1
lane_a() {
  python3 "$code/transcript.py" \
    --sources "$PWD/02_plan/aei_qualification_tasks/sources/revision" \
    --restricted-pdf "$private/AASHTO_R85-18.pdf" \
    --forms "$PWD/02_plan/revision/private/clause_forms.csv" \
    --development "$PWD/06_pod_outputs/post_reg/revision/amendment_4/attempt_01/amendment-4-development.tar.gz" \
    --out "$private/development_transcript.json"
  python3 "$code/freeze_amendment.py" "$repo" "$private/development_transcript.json"
}
lane_a > "$out/lane_a.log" 2>&1 &
lane_a_pid=$!
python3 "$code/lane_b_gate.py" > "$out/lane_b.log" 2>&1 &
lane_b_pid=$!
status_a=0
status_b=0
wait "$lane_a_pid" || status_a=$?
wait "$lane_b_pid" || status_b=$?
printf 'lane_a_exit=%s\nlane_b_exit=%s\n' "$status_a" "$status_b" > "$out/preflight_exit.txt"
if [ "$status_a" -ne 0 ] || [ "$status_b" -ne 0 ]; then exit 1; fi

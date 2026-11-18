#!/usr/bin/env bash
set -euo pipefail
umask 077
cd /workspace
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PR_WORKERS=8
export RULES_SHA=cf1db87134e8f147054df0e9534f41fccf5d46b34c0064d36eab1c172b4cef80
export PARAMS_SHA=2623e57590da09aea111c734a79d17540676ee996b205a1efe6e91343778fb31
tag=post-reg-qualification-supplement-v1
repo=https://github.com/SreekarAditya/pavement-detection-transfer.git
printf '%s\n' 'input verification' > qualification.stage
sha256sum -c INPUTS.sha256
read -r approved < RELEASE_COMMIT
test "$(git ls-remote --refs "$repo" "refs/tags/$tag" | cut -f1)" = "$approved"
tar -xzf qualification-code.tar.gz
tar -xzf post-reg-code.tar.gz
tar -xzf original-rules.tar.gz
tar -xzf reference-c0.tar.gz
mkdir -p qualification-tests locked-test runlogs qualifications post_reg_out
tar -xzf locked-test-scripts.tar.gz -C locked-test
Q=/workspace/post_reg/supplements/qualification
python -m pip install --quiet --upgrade uv
uv python install 3.13.18
source post_reg/run/common.sh
setup_env cpu
sha_ok "$RULES/RULES.md" "$RULES_SHA"
sha_ok "$RULES/params.yaml" "$PARAMS_SHA"
sha_ok "$RULES/exp1_constants.yaml" b0e41e96804816b8e0894e371a6c8dae628086b6a40fd76352719d8e456311e5
printf '%s\n' 'synthetic tests' > qualification.stage
LOCK_SCRIPTS=/workspace/locked-test/scripts bash post_reg/run/run_tests.sh >runlogs/original-tests.log 2>&1
export PYTHONPATH="/workspace/post_reg/scripts:$Q"
python -m unittest discover -s "$Q" -p 'test_qualification.py' >runlogs/supplement-tests.log 2>&1
printf '%s\n' 'original lock verification' > qualification.stage
fetch_lock >runlogs/lock.log 2>&1
tar -xzf analysis-inputs.tar.gz
pack() {
    local task="$1" out="/workspace/qualifications/$1" arc="/workspace/post_reg_out/qualification-$1.tar.gz"
    mkdir -p "$out/logs"
    cp runlogs/{original-tests,supplement-tests,lock}.log "$out/logs/"
    cp "runlogs/task-$task.log" "$out/logs/"
    cp INPUTS.sha256 RELEASE_COMMIT "$out/"
    python - "$out" <<'PY'
from pathlib import Path
import sys
import pr_common as p
out=Path(sys.argv[1])
p.env_record(out)
p.write_manifest(out)
PY
    tar -czf "$arc" -C /workspace/qualifications "$task"
    sha256sum "$arc" | cut -d' ' -f1 > "$arc.sha256"
}
status=0
printf '%s\n' 'Tasks 1 and 1A' > qualification.stage
if (
    set -e
    python "$Q/conformance.py" /workspace/audit /workspace/qualifications/01
    python "$Q/transfer.py" /workspace/qualifications/01/transfer
    cp "$Q"/schema{,-transfer}.json "$Q"/{qualification,transfer,conformance,test_qualification}.py /workspace/qualifications/01/
    cp "$Q/data/source_versions.json" /workspace/qualifications/01/transfer/
) >runlogs/task-01.log 2>&1
then
    pack 01
else
    printf 'Task 1 or 1A stopped\n'
    status=1
fi
printf '%s\n' 'Task 2 retrospective audit' > qualification.stage
if python "$Q/paired.py" --lock "$LOCK" --rules "$RULES" --draws /workspace/e1_draws.parquet --out /workspace/qualifications/02 >runlogs/task-02.log 2>&1
then
    cp "$Q"/{audit_policy,paired,test_qualification}.py /workspace/qualifications/02/
    pack 02
else
    printf 'Task 2 stopped\n'
    status=1
fi
printf '%s\n' 'Task 3 eligibility screen' > qualification.stage
if python "$Q/external.py" screen --root "$LOCK" --rules "$RULES" --c0 /workspace/inputs/c0 --sources /workspace/external_sources --out /workspace/qualifications/03 --workers 8 >runlogs/task-03.log 2>&1
then
    cp "$Q"/{external,audit_policy,test_qualification}.py /workspace/qualifications/03/
    pack 03
else
    printf 'Task 3 screen stopped\n'
    status=1
fi
exit "$status"

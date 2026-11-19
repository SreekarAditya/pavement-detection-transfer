#!/usr/bin/env bash
set -euo pipefail
umask 077
cd /workspace
trap 'echo $? > census.exit' EXIT
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PR_WORKERS=8
export RULES_SHA=cf1db87134e8f147054df0e9534f41fccf5d46b34c0064d36eab1c172b4cef80
export PARAMS_SHA=2623e57590da09aea111c734a79d17540676ee996b205a1efe6e91343778fb31
mkdir -p census-logs post_reg_out
stage() { printf '%s\n' "$1" > census.stage; }
stage "Verifying pinned input archives"
sha256sum -c INPUTS.sha256
test "$(git ls-remote --refs https://github.com/SreekarAditya/pavement-detection-transfer.git refs/tags/post-reg-qualification-census-v1 | cut -f1)" = "$(cat RELEASE_COMMIT)"
tar -xzf qualification-census-code.tar.gz
tar -xzf post-reg-code.tar.gz
tar -xzf original-rules.tar.gz
mkdir -p locked-test
tar -xzf locked-test-scripts.tar.gz -C locked-test
Q=/workspace/post_reg/supplements/qualification
python -m pip install -q uv
uv python install 3.13.18
source post_reg/run/common.sh
setup_env cpu
sha_ok "$RULES/RULES.md" "$RULES_SHA"
sha_ok "$RULES/params.yaml" "$PARAMS_SHA"
sha_ok "$RULES/exp1_constants.yaml" b0e41e96804816b8e0894e371a6c8dae628086b6a40fd76352719d8e456311e5
stage "Running synthetic checks"
LOCK_SCRIPTS=/workspace/locked-test/scripts bash post_reg/run/run_tests.sh >census-logs/original-tests.log 2>&1
export PYTHONPATH="/workspace/post_reg/scripts:$Q"
python -m unittest discover -s "$Q" -p 'test_census.py' >census-logs/census-tests.log 2>&1
stage "Verifying the full locked tree"
fetch_lock >census-logs/lock.log 2>&1
tar -xzf analysis-inputs.tar.gz
mkdir task2
tar -xzf qualification-02.tar.gz -C task2
(cd task2/02 && sha256sum -c MANIFEST.sha256) >census-logs/task2-inputs.log
stage "Reproducing Task 2 on CPU"
python "$Q/census.py" --lock "$LOCK" --rules "$RULES" --draws /workspace/e1_draws.parquet \
    --reference /workspace/task2/02 --out /workspace/02A >census-logs/census.log 2>&1
stage "Sealing Task 2A output archive"
cp -r census-logs 02A/logs
cp INPUTS.sha256 RELEASE_COMMIT "$Q"/{census.py,test_census.py,CENSUS_ADDENDUM.md} 02A/
python - <<'PY'
from pathlib import Path
import pr_common as pc
p=Path('02A')
pc.env_record(p)
pc.write_manifest(p)
PY
tar -czf post_reg_out/qualification-02A.tar.gz 02A
sha256sum post_reg_out/qualification-02A.tar.gz | cut -d ' ' -f1 >census.sha256
stage "Task 2A archive ready"

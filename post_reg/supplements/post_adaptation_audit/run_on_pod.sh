#!/usr/bin/env bash
# Run on a Linux CPU pod. Keep the run directory for diagnosis if a step fails.
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: run_on_pod.sh INPUTS.tar.gz TRUSTED_INPUT_SHA256 NEW_RUN_DIRECTORY" >&2
    exit 2
fi

code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
archive="$1"
expected_sha="$2"
if [[ "$expected_sha" != e13e7f389b52c971bd357deada731c4930b2e4c4eb804cddc1b5648d39a08765 ]]; then
    echo "Input identity differs from the reviewed addendum." >&2
    exit 2
fi
mkdir -p -- "$(dirname -- "$3")"
mkdir -- "$3"
run_dir="$(cd "$3" && pwd)"
inputs="$run_dir/inputs"
receipt="$run_dir/return"
mkdir "$receipt"
trap 'echo "Run interrupted. Keep $run_dir and inspect its logs before retrying." >&2' ERR

export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LC_ALL=C TZ=UTC
unset PYTHONPATH

python3 "$code_dir/verify_return.py" --unpack-inputs "$archive" "$expected_sha" "$inputs"
for file in requirements.txt score.py test_score.py verify_return.py run_on_pod.sh; do
    test -s "$code_dir/$file"
done

# A previous matching environment can be supplied explicitly for recovery.
if [[ -n "${AUDIT_PYTHON:-}" ]]; then
    py="$AUDIT_PYTHON"
else
    if command -v uv >/dev/null 2>&1; then
        uv_bin="$(command -v uv)"
    else
        python3 -m venv "$run_dir/bootstrap"
        "$run_dir/bootstrap/bin/python" -m pip install --disable-pip-version-check --retries 1 uv
        uv_bin="$run_dir/bootstrap/bin/uv"
    fi
    export UV_HTTP_RETRIES=1 UV_HTTP_TIMEOUT=45 UV_NO_PROGRESS=1
    "$uv_bin" venv --python 3.13.18 "$run_dir/venv"
    py="$run_dir/venv/bin/python"
    "$uv_bin" pip install --python "$py" --only-binary :all: -r "$code_dir/requirements.txt"
fi
py="$("$py" -c 'import sys; print(sys.executable)')"
export AUDIT_INPUTS="$inputs"

"$py" - "$code_dir" "$receipt" "$expected_sha" <<'PY'
import hashlib
import importlib.metadata as md
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

code, receipt = map(Path, sys.argv[1:3])
packages = {}
for line in (code / "requirements.txt").read_text().splitlines():
    if line.strip() and not line.startswith("#"):
        package, expected = line.strip().split("==")
        actual = md.version(package)
        if actual != expected:
            raise SystemExit(f"{package}: expected {expected}, found {actual}")
        packages[package] = actual

files = ("run_on_pod.sh", "verify_return.py", "requirements.txt", "score.py", "test_score.py",
         "protocol_gate.sh", "ADDITIONAL_RULE.md")
metadata = {
    "input_archive_sha256": sys.argv[3].lower(),
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "python": platform.python_version(),
    "platform": platform.platform(),
    "packages": packages,
    "code_sha256": {name: hashlib.sha256((code / name).read_bytes()).hexdigest() for name in files},
}
(receipt / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
PY

bash "$code_dir/protocol_gate.sh" "$code_dir" "$inputs" "$run_dir/protocol" "$py" 2>&1 | tee "$receipt/protocol.log"
(
    cd "$code_dir"
    "$py" -m unittest -v test_score
) 2>&1 | tee "$receipt/test.log"
"$py" "$code_dir/score.py" --inputs "$inputs" --output "$receipt/results" 2>&1 | tee "$receipt/score.log"

"$py" "$code_dir/verify_return.py" --pack-return "$receipt" "$run_dir/return.tar.gz" "$expected_sha" "$code_dir"
echo "Download $run_dir/return.tar.gz and $run_dir/return.tar.gz.sha256"

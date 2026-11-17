#!/usr/bin/env bash
# Shared settings and helpers for the analyses after registration. Sourced by every task runner.
# Layout on the pod (the locked scripts use the same defaults):
#   /workspace/run/data      RDD2022 from Kaggle          /workspace/run/ext       external collections
#   /workspace/run/weights   best.pt (Hugging Face)        /workspace/run/weights_last  last.pt (sent from the Mac)
#   /workspace/run/inputs    session-2 inputs (c0, inventory; task 09 only)
#   /workspace/lock          locked tree + private inputs  /workspace/post_reg_out/<task>  outputs
# Hashes the gates need are read from the environment (Aditya sets them from the task header):
#   RULES_SHA PARAMS_SHA [RUN_TABLE_SHA] [CONSTANTS_SHA] [CANDIDATES_SHA] [PRED18_MANIFEST_SHA]
set -euo pipefail

PR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"     # post_reg/ in the repository
S="$PR/scripts"
RULES="${RULES:-$PR/rules}"
WS="${WS:-/workspace}"
RUN="$WS/run"
LOCK="$WS/lock"
OUTBASE="$WS/post_reg_out"
VENV="$WS/venv"
MANIFEST_SHA=ef49673bdc6301009223fb7eecd0aa71be7f53eb6220395d30262b64b19228c7
LOCK_TAR_SHA=ffc1209fe446871a0f31cac86ce5baac4d488e6d6a5783c98e9d0d67afc1cfb1
PRIVATE_SHA=1eda9d8c9f758d0d98026ee59224138392dedee468c4d3f13b039195d5e0088c
SCORING_SHA=87894c62b43a22ce75d272036d950e73925081d270b8fc52f44f7aee885f7d58
VAL_LIST_SHA=f84063d2f222ddc9823c827656a049b14f9d3dfc1e36a9c317e3af59e1160a62
GATE_MAP50=0.6182
NCPU="$(nproc)"
mkdir -p "$RUN" "$OUTBASE"

log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "STOP: $*" >&2; exit 1; }
need() { for v in "$@"; do [ -n "${!v:-}" ] || die "environment variable $v is not set (task header)"; done; }

# ---------------------------------------------------------------- environment
# One Python 3.13.18 environment for everything: the analysis versions of the scoring run, plus the
# inference and training versions on graphics-card pods (common rules, "Versions").
setup_env() {   # setup_env cpu|gpu
  local kind="$1"
  (apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 curl >/dev/null) || true
  command -v uv >/dev/null || pip install -q uv
  [ -x "$VENV/bin/python" ] || uv venv -q -p 3.13.18 "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  uv pip install -q numpy==2.5.4 pandas==3.1.1 pyarrow==26.0.1 scipy==1.18.1 statsmodels==0.15.0 \
    pyyaml pillow imagehash huggingface_hub kaggle
  if [ "$kind" = gpu ]; then
    uv pip install -q torch==2.15.0 torchvision --index-url https://download.pytorch.org/whl/cu132
    uv pip install -q ultralytics==8.4.24
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  else
    uv pip install -q torch==2.15.0 torchvision --index-url https://download.pytorch.org/whl/cpu
  fi
  python - <<'EOF'
import sys, numpy, pandas, pyarrow, scipy, statsmodels
v = dict(python=sys.version.split()[0], numpy=numpy.__version__, pandas=pandas.__version__,
         pyarrow=pyarrow.__version__, scipy=scipy.__version__, statsmodels=statsmodels.__version__)
try:
    import torch, ultralytics; v.update(torch=torch.__version__, ultralytics=ultralytics.__version__)
except Exception:
    pass
print("versions", v)
EOF
}

setup_env_b() {   # setup B environment (merged-crack endpoint of experiment 18): versions of pod session 2
  [ -x "$WS/envB/bin/python" ] || uv venv -q -p 3.13.18 "$WS/envB"
  VIRTUAL_ENV="$WS/envB" uv pip install -q torch==2.15.0 torchvision --index-url https://download.pytorch.org/whl/cu132
  VIRTUAL_ENV="$WS/envB" uv pip install -q ultralytics==8.4.153 rfdetr==1.12.1 supervision==0.31.7 huggingface_hub \
    numpy==2.5.4 pandas==3.1.1 pyarrow==26.0.1 pillow pyyaml
  "$WS/envB/bin/python" -c "import torch, ultralytics; from importlib.metadata import version as v; assert torch.cuda.is_available(); print('setup B', ultralytics.__version__, v('rfdetr'), v('supervision'), torch.__version__)"
}

# ---------------------------------------------------------------- locked inputs
sha_ok() { [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ] || die "SHA-256 of $1 differs"; }

fetch_lock() {
  if [ ! -f "$LOCK/.ready" ]; then
    mkdir -p "$WS/lockdl" "$LOCK"
    python - <<EOF
from huggingface_hub import hf_hub_download
for f in ["pavement_detection_transfer_lock.tar.gz", "private_inputs.tar", "scoring_outputs.tar.gz"]:
    hf_hub_download("SreekarAditya/pavement-detection-transfer", f, repo_type="dataset", local_dir="$WS/lockdl")
EOF
    sha_ok "$WS/lockdl/pavement_detection_transfer_lock.tar.gz" "$LOCK_TAR_SHA"
    sha_ok "$WS/lockdl/private_inputs.tar" "$PRIVATE_SHA"
    sha_ok "$WS/lockdl/scoring_outputs.tar.gz" "$SCORING_SHA"
    # extract into an empty folder and take its single top-level folder (no pipe that can end early)
    local xd="$WS/lockdl/extract"; rm -rf "$xd"; mkdir -p "$xd"
    tar -xzf "$WS/lockdl/pavement_detection_transfer_lock.tar.gz" -C "$xd"
    local tops=("$xd"/*)
    [ "${#tops[@]}" -eq 1 ] && [ -d "${tops[0]}" ] || die "lock archive does not have one top-level folder"
    rmdir "$LOCK" 2>/dev/null || true
    mv "${tops[0]}" "$LOCK"
    tar -xf "$WS/lockdl/private_inputs.tar" -C "$LOCK"
    mkdir -p "$WS/scoring" && tar -xzf "$WS/lockdl/scoring_outputs.tar.gz" -C "$WS/scoring"
    touch "$LOCK/.ready"
  fi
  python "$LOCK/scripts/pdt_score.py" --root "$LOCK" --manifest_sha "$MANIFEST_SHA" --stage verify \
    --out "$WS/lock_verify"
  SCORE_JSON="$(find "$WS/scoring" -name score_results.json -print -quit)"
  [ -n "$SCORE_JSON" ] || die "score_results.json not found in the scoring outputs"
  export SCORE_JSON
}

# arguments of the lock and rule gates that every analysis script checks before it reads anything
GATE=(--rules "$RULES" --rules_sha "${RULES_SHA:-}" --params_sha "${PARAMS_SHA:-}" --manifest_sha "$MANIFEST_SHA")

# ---------------------------------------------------------------- data and weights (graphics-card pods)
fetch_rdd() {
  [ -n "$(find "$RUN/data" -name '*.jpg' -print -quit 2>/dev/null)" ] || \
    kaggle datasets download -d sreekaraditya/rdd2022-yolo-crackscan-v2 -p "$RUN/data" --unzip
  python "$LOCK/scripts/pdt_gate.py" prepare --data "$RUN/data" --out "$RUN/session1"
  python - <<EOF
import json
r = json.load(open("$RUN/session1/prepare_report.json"))
assert r["val_list_sha256"] == "$VAL_LIST_SHA", ("validation list differs from session 1", r["val_list_sha256"])
print("validation list matches session 1")
EOF
}

reproduction_gate() {   # the registered gate on this card; ACCEPT_GATE_DIFF=1 only on Aditya's instruction
  python "$LOCK/scripts/pdt_gate.py" gate --data "$RUN/data" --weights "$RUN/weights/yolo12s_seed0_best.pt" \
    --out "$RUN/session1"
  python - <<EOF
import json, os, sys
g = json.load(open("$RUN/session1/gate_results.json"))["primary"]["map50"]
print("gate mAP50", round(g, 4), "registered $GATE_MAP50")
if abs(g - $GATE_MAP50) > 0.001 and os.environ.get("ACCEPT_GATE_DIFF") != "1":
    sys.exit("STOP: reproduction gate differs by more than 0.001 on this card")
EOF
}

fetch_best() {
  mkdir -p "$RUN/weights"
  python - <<EOF
from huggingface_hub import snapshot_download
from pathlib import Path
d = snapshot_download("SreekarAditya/yolo-rdd2022-benchmark", allow_patterns=["yolo-rdd2022-benchmark/*.pt"],
                      local_dir="$RUN/hf")
for p in (Path(d) / "yolo-rdd2022-benchmark").glob("*.pt"):
    t = Path("$RUN/weights") / p.name
    if not t.exists():
        t.symlink_to(p)
EOF
  check_weights best "$RUN/weights"
}

check_weights() {   # check_weights best|last DIR : every panel file against weights_hashes.csv
  python - "$1" "$2" "$PR/run/weights_hashes.csv" <<'EOF'
import csv, hashlib, sys
from pathlib import Path
kind, d, table = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
bad = []
for r in csv.DictReader(open(table)):
    p = d / f"{r['checkpoint']}_{kind}.pt"
    if not p.exists():
        bad.append(f"{p.name} missing"); continue
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != r[f"{kind}_sha256"]:
        bad.append(f"{p.name} hash differs")
if bad:
    sys.exit("STOP: " + "; ".join(bad))
print(f"{kind}.pt: all panel files match weights_hashes.csv")
EOF
}

fetch_ext() {   # fetch_ext slug...  (download, checksums, preparation as in pod session 2)
  python "$LOCK/scripts/pdt_s2_external.py" fetch --inventory "$LOCK/docs/external_inventory.json" \
    --root "$RUN/ext" --only "$@"
  python "$LOCK/scripts/pdt_s2_external.py" prepare --root "$RUN/ext" --only "$@"
}

# ---------------------------------------------------------------- running and packing
# par NAME CMD... : start a job in the background with its own log; wait_all waits and stops on any failure
PIDS=(); NAMES=()
par() { local n="$1"; shift; log "start $n"; ( "$@" ) >"$LOGDIR/$n.log" 2>&1 & PIDS+=($!); NAMES+=("$n"); }
wait_all() {
  local fail=0 i
  for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then log "done ${NAMES[$i]}"; else log "FAILED ${NAMES[$i]} (see $LOGDIR/${NAMES[$i]}.log)"; fail=1; fi
  done
  PIDS=(); NAMES=()
  [ "$fail" = 0 ] || die "a parallel job failed"
}

start_task() {   # start_task NN : output folder, log folder, environment record
  TASK="$1"; OUT="$OUTBASE/$TASK"; LOGDIR="$OUT/logs"; mkdir -p "$OUT" "$LOGDIR"
  export PYTHONPATH="$S${PYTHONPATH:+:$PYTHONPATH}"
  log "task $TASK on $(hostname), $NCPU processors"
}

finish_task() {   # manifest of every output file, archive, and its hash for the Mac
  python -c "import sys; sys.path.insert(0, '$S'); import pr_common as PC; PC.env_record('$OUT'); PC.write_manifest('$OUT')"
  tar -czf "$OUTBASE/post_reg_${TASK}_outputs.tar.gz" -C "$OUTBASE" "$TASK"
  sha256sum "$OUTBASE/post_reg_${TASK}_outputs.tar.gz" | tee "$OUTBASE/post_reg_${TASK}_outputs.sha256"
  log "send to the Mac: runpodctl send $OUTBASE/post_reg_${TASK}_outputs.tar.gz"
  rm -f ~/.kaggle/kaggle.json   # sign-in removed at the end, as in pod sessions 1 and 2
}

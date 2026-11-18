#!/usr/bin/env bash
set -euo pipefail
umask 077
cd /workspace
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PR_WORKERS=8
export RULES_SHA=cf1db87134e8f147054df0e9534f41fccf5d46b34c0064d36eab1c172b4cef80
export PARAMS_SHA=2623e57590da09aea111c734a79d17540676ee996b205a1efe6e91343778fb31
Q=/workspace/post_reg/supplements/qualification
REPO=https://github.com/SreekarAditya/pavement-detection-transfer.git
CODE_TAG=post-reg-qualification-bharatpothole-v1
DECISION_TAG=post-reg-qualification-bharatpothole-commit-v1
LOG=/workspace/bharat-logs
export PYTHONPATH="/workspace/post_reg/scripts:$Q"
mkdir -p "$LOG" post_reg_out
stage() { printf '%s\n' "$1" > bharat.stage; }
pack() {
    local kind="$1"
    cp -r "$LOG" "$kind/logs"
    cp RELEASE_COMMIT INPUTS.sha256 "$kind/"
    python - "$kind" <<'PY'
from pathlib import Path
import sys
import pr_common as pc
out=Path(sys.argv[1])
pc.env_record(out)
pc.write_manifest(out)
PY
    tar -czf "post_reg_out/bharat-$kind.tar.gz" "$kind"
    sha256sum "post_reg_out/bharat-$kind.tar.gz" | cut -d' ' -f1 > "bharat-$kind.sha256"
}

read -r release < RELEASE_COMMIT
if [[ "$1" != score ]]; then
    stage "Verify inputs and install registered environment"
    sha256sum -c INPUTS.sha256
    test "$(git ls-remote --refs "$REPO" "refs/tags/$CODE_TAG" | cut -f1)" = "$release"
    tar -xzf post-reg-code.tar.gz
    tar -xzf original-rules.tar.gz
    mkdir -p locked-test
    tar -xzf locked-test-scripts.tar.gz -C locked-test
    python -m pip install -q uv
    uv python install 3.13.18
    source post_reg/run/common.sh
    device=cpu
    [[ "$1" != decide ]] || device=gpu
    setup_env "$device"
    sha_ok "$RULES/RULES.md" "$RULES_SHA"
    sha_ok "$RULES/params.yaml" "$PARAMS_SHA"
    sha_ok "$RULES/exp1_constants.yaml" b0e41e96804816b8e0894e371a6c8dae628086b6a40fd76352719d8e456311e5
    stage "Run synthetic checks"
    LOCK_SCRIPTS=/workspace/locked-test/scripts bash post_reg/run/run_tests.sh >"$LOG/original-tests.log" 2>&1
    python -m unittest discover -s "$Q" -p 'test_*.py' >"$LOG/supplement-tests.log" 2>&1
    stage "Verify the original lock"
    fetch_lock >"$LOG/lock.log" 2>&1
fi

case "$1" in
screen)
    tar -xzf reference-c0.tar.gz
    stage "Deduplicate and screen source recordings"
    python "$Q/bharat.py" screen --root "$LOCK" --rules "$RULES" --c0 /workspace/inputs/c0 \
        --sources /workspace/external_sources --out /workspace/screen --workers "$PR_WORKERS" 2>&1 | tee "$LOG/screen.log"
    if [ -d screen/split ]; then
        cat screen/split/{calibration,heldout}.txt > bharat-image-paths.txt
        tar -cf post_reg_out/bharat-images.tar -C /workspace/external_sources/bharat-images -T bharat-image-paths.txt
        sha256sum post_reg_out/bharat-images.tar | cut -d' ' -f1 > bharat-images.sha256
    fi
    pack screen
    stage "Screen archive ready"
    ;;
decide)
    test -f /workspace/.bharat-auth
    source /workspace/.bharat-auth
    unlink /workspace/.bharat-auth
    uv pip install -q six==1.17.0
    stage "Run registered reproduction gate"
    fetch_best >"$LOG/checkpoints.log" 2>&1
    fetch_rdd >"$LOG/rdd.log" 2>&1
    unset KAGGLE_KEY KAGGLE_USERNAME
    reproduction_gate >"$LOG/reproduction.log" 2>&1
    test -f calibration-inputs.tar.gz
    tar -xzf calibration-inputs.tar.gz
    test ! -e screen/split/private
    stage "Await verified image handoff"
    for ((i=0; i<360; i++)); do
        test ! -f bharat-images.tar.verified || break
        sleep 10
    done
    test -f bharat-images.tar.verified
    mkdir bharat-images
    tar -xf bharat-images.tar -C bharat-images
    for part in calibration heldout; do
        sed 's|^|/workspace/bharat-images/|' "screen/split/$part.txt" > "screen/split/$part-paths.txt"
    done
    stage "Run locked detector on images"
    expected=3fe3eaf8fa8f9300d11286859f229aa61f151a06f11d334d654f716d6a63d57d
    weight="$RUN/weights/yolo12m_seed0_best.pt"
    sha_ok "$weight" "$expected"
    python "$S/pr_infer.py" val --root "$LOCK" --out /workspace/predictions --work /workspace/infer \
       --weights "yolo12m_seed0=$weight" \
       --pool "calibration=/workspace/screen/split/calibration-paths.txt" "heldout=/workspace/screen/split/heldout-paths.txt" \
       >"$LOG/inference.log" 2>&1
    stage "Compute the calibration-only commitment"
    python "$Q/bharat.py" commit --root "$LOCK" --rules "$RULES" --out /workspace/decision \
       --split /workspace/screen/split --eligibility /workspace/screen/eligibility.json \
       --calibration /workspace/predictions/yolo12m_seed0/calibration.parquet \
       --heldout /workspace/predictions/yolo12m_seed0/heldout.parquet \
       --weight_sha "$expected" --code_commit "$release" >"$LOG/decision.log" 2>&1
    mv predictions decision/
    ln -s /workspace/decision/predictions predictions
    pack decision
    stage "Decision archive ready for publication"
    ;;
score)
    stage "Verify public commitment"
    git init -q public-commit
    git -C public-commit fetch -q --depth=1 "$REPO" "refs/tags/$DECISION_TAG:refs/tags/$DECISION_TAG"
    test "$(git -C public-commit rev-parse "$DECISION_TAG^{commit}")" = "$2"
    sha256sum -c unseal.sha256
    mkdir private-evaluation
    tar -xf bharat-holdout.tar -C private-evaluation
    source /workspace/venv/bin/activate
    stage "Score the committed heldout evaluation once"
    python "$Q/bharat.py" score --root /workspace/lock --rules /workspace/post_reg/rules --out /workspace/scoring \
       --split /workspace/screen/split --eligibility /workspace/screen/eligibility.json \
       --calibration /workspace/predictions/yolo12m_seed0/calibration.parquet \
       --heldout /workspace/predictions/yolo12m_seed0/heldout.parquet \
       --commitment /workspace/decision/commitment.json --commitment_sha "$3" \
       --labels /workspace/private-evaluation/heldout_labels.parquet \
       --approved_repo /workspace/public-commit --approved_tag "$DECISION_TAG" \
       --approved_path "$4" >"$LOG/scoring.log" 2>&1
    cp decision/commitment.json screen/eligibility.json scoring/
    pack scoring
    stage "Scoring archive ready"
    ;;
*) exit 2 ;;
esac

#!/usr/bin/env bash
# Confirm the published addendum, the frozen rule files and the original full lock.
set -euo pipefail
code="$1"
inputs="$2"
work="$3"
py="$4"
tag=post-reg-post-adaptation-audit-v1
repo=https://github.com/SreekarAditya/pavement-detection-transfer.git
dataset=https://huggingface.co/datasets/SreekarAditya/pavement-detection-transfer/resolve/main
mkdir -p "$work"
git -C "$work" init --quiet
git -C "$work" fetch --quiet --depth=1 "$repo" "refs/tags/$tag"
commit="$(git -C "$work" rev-parse 'FETCH_HEAD^{commit}')"
echo "Published addendum tag $tag resolves to $commit"
for name in ADDITIONAL_RULE.md score.py test_score.py verify_return.py requirements.txt run_on_pod.sh protocol_gate.sh; do
    git -C "$work" show "$commit:post_reg/supplements/post_adaptation_audit/$name" | cmp - "$code/$name"
done

(
    cd "$inputs/post_reg/rules"
    sha256sum -c <<'SUMS'
cf1db87134e8f147054df0e9534f41fccf5d46b34c0064d36eab1c172b4cef80  RULES.md
2623e57590da09aea111c734a79d17540676ee996b205a1efe6e91343778fb31  params.yaml
b0e41e96804816b8e0894e371a6c8dae628086b6a40fd76352719d8e456311e5  exp1_constants.yaml
SUMS
)
archives="${AUDIT_ARCHIVE_DIR:-$work/archives}"
mkdir -p "$archives"
for name in pavement_detection_transfer_lock.tar.gz private_inputs.tar scoring_outputs.tar.gz; do
    if [[ ! -f "$archives/$name" ]]; then
        curl --fail --location --retry 2 --connect-timeout 15 --max-time 360 \
            --output "$archives/$name.partial" "$dataset/$name"
        mv "$archives/$name.partial" "$archives/$name"
    fi
done
(
    cd "$archives"
    sha256sum -c <<'SUMS'
ffc1209fe446871a0f31cac86ce5baac4d488e6d6a5783c98e9d0d67afc1cfb1  pavement_detection_transfer_lock.tar.gz
1eda9d8c9f758d0d98026ee59224138392dedee468c4d3f13b039195d5e0088c  private_inputs.tar
87894c62b43a22ce75d272036d950e73925081d270b8fc52f44f7aee885f7d58  scoring_outputs.tar.gz
SUMS
)
mkdir "$work/lock"
tar -xzf "$archives/pavement_detection_transfer_lock.tar.gz" --strip-components=1 -C "$work/lock"
tar -xf "$archives/private_inputs.tar" -C "$work/lock"
"$py" "$work/lock/scripts/pdt_score.py" --root "$work/lock" \
    --manifest_sha ef49673bdc6301009223fb7eecd0aa71be7f53eb6220395d30262b64b19228c7 \
    --stage verify --out "$work/lock_verify"

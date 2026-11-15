#!/usr/bin/env bash
# Pod session 2: finish setup B and the session. Run when no setup B process is still running.
# Keeps every setup B run whose completion appears in any setup B log, removes unfinished files, runs whatever is
# still missing (all six models, skipping finished files), merges the records, checks completeness, then continues
# exactly as run_session2.sh (external image properties, packing).
set -euo pipefail
ROOT=/workspace/run
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$ROOT/outputs"
ENVB="$ROOT/envB/bin/python"
"$ENVB" -c "import torch, ultralytics, rfdetr; assert torch.cuda.is_available(); assert ultralytics.__version__ == '8.4.153'"
if pgrep -f pdt_s2_infer_b.py >/dev/null; then echo "STOP: a setup B process is still running"; exit 1; fi

echo "== setup B: keep logged runs only"
python - "$OUT/s2b/sealed_b" "$HERE"/session2_resume_b.log "$HERE"/b_g*.log <<'PYEOF'
import re, sys, pathlib
done = set()
for log in sys.argv[2:]:
    for line in open(log, errors="replace"):
        m = re.search(r"\]\s+(\S+) on (\S+): \d+ predictions", line)
        if m:
            done.add((m.group(1), m.group(2)))
removed = 0
for p in pathlib.Path(sys.argv[1]).rglob("*"):
    if p.is_file() and (p.name.endswith(".tmp") or (p.name[:-len(".parquet")], p.parent.name) not in done):
        p.unlink(); removed += 1
print(f"logged runs kept: {len(done)}; unfinished files removed: {removed}")
PYEOF
mkdir -p "$HERE/logs_b_earlier"; mv "$HERE"/b_g*.log "$HERE/logs_b_earlier/" 2>/dev/null || true
if [ -f "$OUT/s2b/b_runs.csv" ]; then mv "$OUT/s2b/b_runs.csv" "$OUT/s2b/b_runs_sequential.csv"; fi
for f in "$OUT"/s2b/b_runs_yolo*.csv "$OUT"/s2b/b_runs_rfdetr*.csv; do
  if [ -f "$f" ] && [[ "$f" != *_earlier.csv ]]; then mv "$f" "${f%.csv}_earlier.csv"; fi
done

echo "== setup B: remaining runs in four parallel groups"
ARGS=(--g4c "$ROOT/inputs/g4c" --g4 "$OUT/g4" --c0 "$ROOT/inputs/c0" --kaggle "$ROOT/data" --ext "$ROOT/ext" \
      --cache "$ROOT/hf_b" --out "$OUT/s2b")
"$ENVB" "$HERE/pdt_s2_infer_b.py" "${ARGS[@]}" --models yolov8n yolov8s > "$HERE/b_g1.log" 2>&1 & G1=$!
"$ENVB" "$HERE/pdt_s2_infer_b.py" "${ARGS[@]}" --models yolo26s yolo26m > "$HERE/b_g2.log" 2>&1 & G2=$!
"$ENVB" "$HERE/pdt_s2_infer_b.py" "${ARGS[@]}" --models rfdetr_nano > "$HERE/b_g3.log" 2>&1 & G3=$!
"$ENVB" "$HERE/pdt_s2_infer_b.py" "${ARGS[@]}" --models rfdetr_small > "$HERE/b_g4.log" 2>&1 & G4=$!
FAIL=0
for p in $G1 $G2 $G3 $G4; do wait $p || FAIL=1; done
tail -n 3 "$HERE"/b_g*.log
[ $FAIL -eq 0 ] || { echo "STOP: a setup B group failed (see b_g*.log)"; exit 1; }
"$ENVB" "$HERE/pdt_s2_infer_b.py" --out "$OUT/s2b" --finalize
python - "$OUT/s2b/sealed_b_manifest.json" "$OUT/g4/g4_g6_decision.json" <<'PYEOF'
import json, sys
man = json.load(open(sys.argv[1])); dec = json.load(open(sys.argv[2]))["collections"]
pools = 1 + sum(1 for v in dec.values() if v["eligible_four_class"] or v["eligible_pothole"])
assert len(man) == 6 * pools, (len(man), 6 * pools)
print(f"setup B complete: {len(man)} prediction files (6 models x {pools} pools)")
PYEOF

echo "== label-free properties of the external images (forecast inputs; reads no labels or predictions)"
python "$HERE/pdt_s2_props.py" --g4 "$OUT/g4" --ext "$ROOT/ext" --e2 "$ROOT/inputs/e2" \
  --weights "$ROOT/weights/yolo12s_seed0_best.pt" --out "$OUT/ext_props"

echo "== packing"
mkdir -p "$OUT/ext_labels"
for d in "$ROOT"/ext/*/pool; do s=$(basename "$(dirname "$d")"); mkdir -p "$OUT/ext_labels/$s"; cp "$d"/labels.parquet "$d"/images.csv "$OUT/ext_labels/$s/"; done
cp "$ROOT/ext/prepare_report.json" "$ROOT/ext/fetch_report.json" "$OUT/"
cp -r "$ROOT/ext/overlays" "$OUT/"
cp "$HERE"/session2.log "$OUT/" 2>/dev/null || true
tar -czf "$ROOT/session2_outputs.tar.gz" -C "$OUT" --exclude="session1/ultra" --exclude="*/ultra" \
  session1 g4 s2a s2b ext_props ext_labels overlays prepare_report.json fetch_report.json $( [ -f "$OUT/session2.log" ] && echo session2.log )
sha256sum "$ROOT/session2_outputs.tar.gz"
rm -f ~/.kaggle/kaggle.json
echo "== done: download $ROOT/session2_outputs.tar.gz, then terminate the pod"

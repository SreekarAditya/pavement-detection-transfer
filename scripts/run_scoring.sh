#!/usr/bin/env bash
# Scoring after the lock (protocol 3.3.10, Section 9, steps 8 and 9). Usage:
#   bash run_scoring.sh <lock root> <SHA-256 of lock_manifest.json as published> <output folder>
set -euo pipefail
ROOT="$1"; SHA="$2"; OUT="$3"; HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT"
python "$HERE/pdt_score.py" --root "$ROOT" --manifest_sha "$SHA" --stage verify --out "$OUT"
# step 8: S1 only, selection recorded before anything else is opened
python "$HERE/pdt_score.py" --root "$ROOT" --manifest_sha "$SHA" --stage s1 --out "$OUT"
sha256sum "$OUT/selection_s1.json"
# step 9: S2 and external scores
python "$HERE/pdt_score.py" --root "$ROOT" --manifest_sha "$SHA" --stage all --selection "$OUT/selection_s1.json" --out "$OUT"
python "$HERE/pdt_partitions.py" --e1 "$ROOT/e1" --e2 "$ROOT/e2" --g3 "$ROOT/g3" --out "$OUT/partitions.json"
python "$HERE/pdt_forecast.py" evaluate --forecasts "$ROOT/forecast/forecasts.json" --g4 "$ROOT/s2/g4" \
  --sealed_ext "$ROOT/s2/s2a/sealed_ext" --out "$OUT/forecast_evaluation.json"

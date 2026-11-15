# What do the country partitions of the Road Damage Dataset 2022 measure? Pre-registration and locked inputs

Authors: Sreekar Aditya Reddy Pesaru (corresponding), Sahasra Reddy Patlola, Dudam Bharath Kumar. Department of Civil Engineering, Chaitanya Bharathi Institute of Technology, Hyderabad, India.

This deposit is the pre-registration of the study (protocol version 3.3.10, `docs/protocol_v3_3.md`) together with every input that the confirmatory analysis uses, deposited before any score on the internal scoring folds (S1, S2), the external road collections or the DetectionBench scoring set was computed.

## Contents

- `docs/`: protocol 3.3.10; gate records G0, G3, G4 to G6, G4c, G7; rules fixed before the corresponding data were processed (G4 to G6, G7, alert rate for all classes, forecast).
- `scripts/`: every script used to produce the locked inputs, and the scoring scripts (`run_scoring.sh`, `pdt_score.py`, `pdt_core.py`, `pdt_partitions.py`, `pdt_forecast.py`).
- `e1/sealed/predictions/`: setup A predictions on the RDD2022 validation split (all candidates above confidence 0.001, original pixels).
- `s2/s2a/sealed_ext/`: setup A predictions on the external collections; `s2/s2b/sealed_b/`: setup B predictions (DetectionBench checkpoints) on its scoring set and the external collections. No labels were joined when these were produced.
- Fold lists and clusters (`g3/`, `g4c/`), external image pools, blocks and gate decisions (`s2/g4/`), label-free image properties (`e2/`, `s2/ext_props/`), the committed forecasts (`forecast/forecasts.json`) and the G7 outputs (`g7/`).
- `lock_manifest.json`: SHA-256 of every file of the scoring root, including files not distributed here (outcome files computed and sealed in pod session 1, and the processed label tables of the external collections, which the scripts regenerate from the public releases).

## Verification

`python scripts/pdt_score.py --root <root> --manifest_sha <SHA-256 of lock_manifest.json> --stage verify` recomputes every hash and stops on any difference.

## Licenses

Code: MIT. Documents: CC BY 4.0. Predictions are derived from RDD2022 (CC BY-SA 4.0) and from the external collections under their own licenses (listed in `docs/G4_G6_rule.md` and the inventory).

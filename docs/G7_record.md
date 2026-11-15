# Gate G7: record (development fold only)

Protocol version 3.3.8, Section 9, step 4. Run 15 November 2026. Rule fixed before any outcome was opened: `G7_rule.md` (first version 04:29 UTC; amended 04:32 UTC, before opening, SHA-256 c64fd9aecf8f32b03e69d6d6830ecc392e8a26024d2627f440b9a88987115efb). Script `pdt_g7.py` (SHA-256 bb21408c53373dd6005450f822256a0e4a11407d4b43c59f83f2068957775541).

Only development-fold pothole predictions were read: the script filters by file name and class while reading the sealed files, after checking each file against `sealed_manifest.json`. No S1, S2 or external outcome was loaded.

## 1. Alert rate

| Quantity | Value |
| --- | --- |
| Development images | 1,044 |
| Development pothole (D40) reference boxes | 165 |
| ***k*** (pothole boxes per 100 images) | **15.80** |
| Development quota, floor(15.80 × 1,044 / 100) | 164 |
| Underfill | 0 for all 20 runs |

For reference (label counts only): S1 has 241 pothole boxes in 1,568 images (15.37 per 100), S2 has 219 in 1,565 (13.99 per 100).

**Check.** The fast quota implementation was compared with a direct one (explicit image copies, global sort with copy index, selection, matching) for YOLOv5s and YOLOv12s seed 0 at the point estimate and on three bootstrap draws; matched counts, reference counts and quotas agreed exactly.

## 2. Development-fold pothole recall at the quota

| Architecture | Seeds | Recall | Bootstrap SD |
| --- | --- | --- | --- |
| YOLOv5s | 3 | 0.531 | 0.037 |
| YOLOv8s | 3 | 0.552 | 0.038 |
| YOLOv9s | 3 | 0.556 | 0.039 |
| YOLOv10s | 3 | 0.543 | 0.035 |
| YOLO11s | 3 | 0.558 | 0.036 |
| YOLOv12n | 1 | 0.509 | 0.043 |
| YOLOv12s | 3 | 0.564 | 0.038 |
| YOLOv12m | 1 | 0.594 | 0.043 |

Per-run counts are in `g7_dev_runs.csv`. Bootstrap: 2,000 cluster draws within partition strata, seed 2026.

## 3. Precision projection

- Mean bootstrap correlation between architectures: 0.86 (most sampling error is shared by all models and does not add spread).
- Sample variance of the eight values: 0.000610; part expected from sampling error: 0.000226; estimated spread of true benchmark recall: variance 0.000384 (SD 0.020).
- **Repeatability gate from the internal folds alone passes in 64% of simulated studies.**
- Share of simulated studies with an inconclusive outcome (two collections; 2,000 studies per cell):

| Potholes per collection | *R* = 0 | *R* = 0.5 | *R* = 1 |
| --- | --- | --- | --- |
| 150 | 0.82 | 0.88 | 0.73 |
| 500 | 0.75 | 0.70 | 0.66 |
| 1,500 | 0.72 | 0.66 | 0.65 |

Without extra architecture-by-collection noise; the variant with that noise gives 0.68 to 0.86. Full category shares are in `g7_projection.csv`.

## 4. Recorded before the lock

**A likely inconclusive outcome is recorded.** At *R* = 0.5 without extra noise, the share inconclusive exceeds 0.5 at every collection size considered (0.66 to 0.88), so the reading does not depend on which size G4 finds. The limiting factor is the internal denominator: with 241 and 219 pothole boxes in S1 and S2 and an architecture spread of about 0.02 in recall, Cov(*x*₁, *x*₂) is often not distinguishable from zero, and larger external collections do not remove that. This is the outcome anticipated in Section 11, limitation (a).

## 5. Limits of the projection

The spread is itself estimated from eight development values and is uncertain; fold covariances are scaled from the development covariance by pothole counts; the within-study bootstrap is approximated parametrically. The projection describes plausibility, not a forecast of the result.

## 6. Files

`06_pod_outputs/g7/`: `g7_dev_runs.csv` (fe260355…), `g7_dev_arch.csv` (7add7281…), `g7_dev_bootstrap_arch.csv` (334adfa7…), `g7_projection.csv` (be4b21a7…), `g7_summary.json` (ef73b94b…, with hashes of the inputs).

## 7. Addendum: all-class projection (15 November 2026)

Rule fixed before any all-class outcome was opened: `G7_rule_allclass.md` (05:06 UTC, SHA-256 8e342c38d13e9cedf6036e1b0284d081bdc92a91f9b0f0d45d122f69f0b4a00a). The script was generalized (`--classes pothole|all`; SHA-256 dc63d96ecba959a15a418554089711bcbc6b97dbb099e9e251d4a68f22527daa); rerun in pothole mode, it reproduces `g7_dev_runs.csv`, `g7_dev_arch.csv`, `g7_dev_bootstrap_arch.csv` and `g7_projection.csv` above byte for byte. The registered design is unchanged by this addendum.

| Quantity | Value |
| --- | --- |
| Development reference boxes, all classes | 1,592 (S1 2,228; S2 2,286) |
| *k*_all (boxes per 100 images) | 152.49 |
| Development quota | 1,591; no underfill |
| Mean bootstrap correlation between architectures | 0.93 |
| Spread of true benchmark recall | SD 0.016 (sample variance 0.000281; sampling part 0.000022) |
| Repeatability gate from internal folds alone | passes in 100% of simulated studies |

Development-fold all-class recall at the quota: YOLOv12m 0.609, YOLOv9s 0.600, YOLOv12s 0.589, YOLO11s 0.583, YOLOv8s 0.583, YOLOv5s 0.577, YOLOv12n 0.568, YOLOv10s 0.556 (bootstrap SD 0.016 to 0.019).

Share inconclusive: 0.000 to 0.002 in every scenario (500, 1,500 and 5,000 boxes per collection; *R* = 0, 0.5, 1; with and without extra noise). At *R* = 0.5 without extra noise the modal outcome is partial retention for 1,500 boxes or more (0.91 at 1,500; 0.99 at 5,000) and is split between partial retention and compatible with full retention at 500. Full shares: `06_pod_outputs/g7_allclass/g7_projection.csv`.

**Reading.** With all classes the internal denominator is well determined, and the test separates the outcome categories. This holds only if at least two external collections label all four classes with a defensible class mapping (rule item 4), which G4 checks from label files.

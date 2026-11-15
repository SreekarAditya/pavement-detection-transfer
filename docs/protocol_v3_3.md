# Protocol, version 3.3.10

Written 15 November 2026. Not yet registered; registration happens at step 7 (Section 9) as a public, timestamped Zenodo deposit with a DOI.

Design approval (Section 13) and empirical gate status (Section 10) are recorded separately. Approval of this text does not mean any gate has passed.

## Version history

Earlier drafts were revised after internal review before registration.

- **3.3.10.** Section 10 records gates G4 to G6 and G4c and the session 2 costs; Section 8 names the forecast rule file. No change to any design decision or analysis.
- **3.3.9.** The primary target becomes recall over the four RDD2022 damage classes (longitudinal, transverse and alligator cracks and potholes); pothole transfer becomes secondary item 6. Sections 1, 3 to 6, 7, 9, 10 and 11 changed accordingly; Section 11 discloses when and why. Registration is a Zenodo deposit instead of the Open Science Framework. Section 10 records gate G7.
- **3.3.8.** Section 5 adds a prespecified sensitivity with stricter leakage removal (descriptor threshold 0.65); secondary item 11 is also reported at 0.65; Section 11 discloses when and why it was added; Section 10 records gates G1 to G3. No change to the primary analysis.
- **3.3.7.** Clarifications found while testing the scripts, no change to any decision rule. Section 2: the matcher check (G2) is run in Ultralytics' own resized input space, where the recomputed flags must match Ultralytics' prediction by prediction; all outcomes are matched in original image pixels, the space that applies to every model family and input size (Ultralytics rounds the short side of non-square images when resizing, which shifts some overlaps near 0.5). Section 3: SSCD descriptors use the authors' DISC2021 evaluation preprocessing (short edge 288 pixels, aspect ratio preserved), and candidate pairs are formed from all pairs, not nearest-neighbor lists.
- **3.3.6.** Section 6 adds secondary items 9 to 11 (what the RDD2022 partitions measure). They use pod session 1 outputs, run only at step 9 after unblinding, and need no human labeling. Leakage exposure moves from the exploratory list (Section 8) to item 11.
- **3.3.5.** Section 3, G3 boundary check: pairs shown in random order without similarity values; a "can't tell" answer that does not count toward a bin's decision; at least 10 decided pairs per bin; the "can't tell" count reported.
- **3.3.4.** Binding reporting rules: an interpretation rule under the outcome table (Section 5) and reporting commitments (Section 11).
- **3.3.3.** Section 3: the threshold check covers every case (explicit bins, top bin to 1.00, sparse bins merged upward, pHash-only fallback if no higher bin passes).
- **3.3.2.** Section 3 and G3: cited basis and a fixed boundary check for the 0.75 descriptor threshold. Sections 4 and 5: the bootstrap copy index added to the tie-break.
- **3.3.1.** Section 6: the training-history comparison is descriptive, not a bound on selection optimism. Section 5: outcome categories corrected (amplification; compatible with full retention; negative slope). Sections 4, 5 and 9: density-matched sensitivity defined; the scoring script verifies every locked file hash against the manifest.
- **3.3 (from 3.2).** Primary defined as transfer from the benchmark used for checkpoint selection; one ranking pool per fold and collection with an integer quota and deterministic tie-break; symmetric ratio with a Fieller confidence set, a repeatability gate and mutually exclusive outcomes; an executable lock order; a verified scoring set required for setup B; a descriptive training-history comparison.

## 1. Question and estimand

**Question.** How much of the across-model road-damage recall advantage (the four RDD2022 classes: longitudinal crack D00, transverse crack D10, alligator crack D20 and pothole D40) measured on the RDD2022 benchmark, the same benchmark previously used to select these checkpoints, carries over to approved external road collections at the same alert workload?

**Estimand.** For a fixed panel of eligible architectures, the **retention ratio**

*R* = Cov(*x̄*, *z*) / Cov(*x*₁, *x*₂),

where, for each architecture, *x*₁ and *x*₂ are four-class recalls (matched reference boxes of all four classes divided by all reference boxes of those classes in the pool) on the two internal scoring folds S1 and S2, *x̄* is their mean, and *z* is the equal-weight mean four-class recall over the eligible external collections. Covariances are taken across architectures. If each fold measures a common benchmark performance plus independent fold error, Cov(*x*₁, *x*₂) estimates the spread of benchmark performance across architectures and *R* estimates the slope of external recall on benchmark performance, free of attenuation by fold sampling error. *R* = 1: benchmark differences carry over in full; *R* > 1: external differences are larger than benchmark differences (amplification); *R* = 0: no linear carry-over; *R* < 0: a negative across-model slope, which does not imply that every model pair reversed.

**Limits.** The claim covers the named external collections, the eligible model panel and the stated workload. A value near 0 describes weak linear retention, not what happened to every model pair; a value near 1 can coexist with rank changes or lower absolute recall on every external collection. No claim is made about physical damage instances, other road networks or population equivalence. The four classes are pooled, so cracks, which make up most reference boxes, dominate the recall; pothole transfer is reported separately (Section 6, item 6). Crack labeling conventions differ between collections, and the transfer measure includes that difference. Fold independence is an assumption: the folds share the July checkpoint selection, so their errors may be correlated.

## 2. Deployment budget and model panels

**Budget.** At most **69 billion operations per image** (scenario: a dedicated edge device, with the detector allotted about half of a 30-frame-per-second period; figures and sources in `G0_freeze.md`). The ceiling sets eligibility only. Survey throughput depends on the full pipeline and the required frame spacing; measured latency is reported for every model. Setup A operation counts come from the Ultralytics 8.4.24 tagged tables; setup B uses each family's official counting convention, cited at gate G0.

| Setup | Eligible panel | Internal scoring data |
| --- | --- | --- |
| **A, own checkpoints** (primary) | YOLOv5s (YOLOv5su row), YOLOv8s, YOLOv9s, YOLOv10s (end-to-end mode), YOLO11s, YOLOv12n, YOLOv12s, YOLOv12m. Seeds 0, 42 and 123 where they exist; YOLOv12n and YOLOv12m seed 0 only | Clean holdout of the Kaggle validation split; used for July checkpoint selection |
| **B, DetectionBench** (conditional replication) | YOLOv8n, YOLOv8s, YOLO26s, YOLO26m (inference mode read from checkpoint metadata), RF-DETR Nano and Small at released resolutions (384 and 512 pixels). Excluded by the budget: YOLOv8m, RF-DETR Medium | A scoring set verified against the training and checkpoint-selection data of every included checkpoint (gate G4c) |

**Setup B comparison basis.** RF-DETR has no official figures at 640 pixels, so setup B compares the released pipelines, including their preprocessing and input resolution, with the compute ceiling applied at each model's released resolution. If the verified scoring set is independent of checkpoint selection, setup B also tests whether the result depends on selection optimism.

**One recall value per architecture per data set.** In setup A, each architecture's value is its seed mean; single-seed architectures are disclosed, and a seed-0-only sensitivity is reported. Architecture pairs are never treated as independent observations.

**Candidate generation (frozen).** Each model's own postprocessing: confidence floor 0.001; suppression overlap 0.7 (YOLOv10s and YOLO26 in the mode recorded at G0); at most 300 detections per image. No 0.25 operating threshold applies in the primary; the alert quota replaces it.

**Matching and reference scores (frozen).** A prediction matches a reference box of the same class when their intersection over union is 0.5 or more; pairs are chosen greedily by overlap, one to one, exactly as Ultralytics does. All outcomes are matched in original image pixels. Mean average precision at 0.5 overlap (mAP50) is Ultralytics' 101-point interpolated average precision, averaged over the classes present in the reference labels, at input 640 pixels, confidence floor 0.001, suppression overlap 0.7, at most 300 detections per image, half precision. **Matcher check (gate G2):** during each validation run the reference boxes and predictions are recorded as Ultralytics holds them (its resized input space); this study's matcher, run in that space, must reproduce Ultralytics' true-positive flags and its mAP50 within 0.002. The difference between mAP50 in original pixels and Ultralytics' value is reported for every run.

## 3. Data and folds

**Near-duplicate rule (fixed now, before any data are seen).** Two images are linked if their perceptual hash (pHash, 64 bits) differs by 4 bits or fewer, or if their copy-detection descriptors (SSCD disc_mixup, computed with the authors' DISC2021 evaluation preprocessing: short edge resized to 288 pixels, aspect ratio preserved) have cosine similarity of 0.75 or more. Links are computed over all image pairs. Basis: the SSCD authors' recommended operating point, at which pairs from the DISC2021 copy-detection benchmark are copies with 90% precision (Pizzi et al., 2022, "A Self-Supervised Descriptor for Image Copy Detection", CVPR; sscd-copy-detection repository README). DISC2021 contains edited web images, not consecutive road frames, so the threshold is checked at gate G3: a person views a random sample of 20 linked or near-linked image pairs in each similarity bin, 0.65 to 0.70, 0.70 to 0.75, 0.75 to 0.80, 0.80 to 0.85, 0.85 to 0.90 and 0.90 to 1.00 (all pairs if a bin has fewer than 20; a bin with fewer than 5 pairs is merged into the next higher bin), using images only, no labels or detector outputs, with pairs shown in random order and without their similarity values, and records for each pair "same location" (the same stretch of pavement visible in both, judged from fixed details such as crack shapes, patches, fixtures, markings, road edges and background objects), "different location" or "can't tell". Links between nearby frames of the same location are intended. "Can't tell" pairs do not count toward a bin's decision; a bin needs at least 10 decided pairs, so further pairs are drawn from that bin, in random order, until it has 10 or none remain; a bin still under 10 is merged into the next higher bin for the decision (the top bin is decided on the pairs it has). The number of "can't tell" pairs per bin is reported. If more than half of the decided pairs in the 0.75 to 0.80 bin show different locations, the threshold is raised to the lowest of 0.80, 0.85 or 0.90 whose bin has at most half different-location pairs among its decided pairs; it is never lowered. If none of those bins passes, the descriptor criterion is dropped and the pHash criterion alone defines links; this is reported as a limitation of leakage removal. The check and any revision are recorded before any outcome is opened. The largest cluster's size and share of images are reported. If SSCD is unavailable the descriptor criterion is dropped (pHash only) rather than replaced. Connected components of links are clusters.

**Setup A internal data.** Validation images with no link to any training image form the clean holdout. Its clusters are assigned, stratified by RDD2022 partition, seed 2026, to a **development fold** (25%), **S1** (37.5%) and **S2** (37.5%). S1 and S2 are withheld from alert-policy tuning but were used for July checkpoint selection.

**Setup B internal data.** The verified scoring set (gate G4c), divided the same way.

**External collections.** Candidates: Rome version 2, SVRDD Beijing, N-RDD2024 Turkey and Georgia, EcuPDD Ecuador, IRD Baghdad, HRP4K, Kent, PDD2026 Indonesia, UAV-PDD2023 Tianjin (identifiers in version 3.1, Section 3). Each enters only through gates G4 to G6. All are unverified until those gates run. Collections from a country present in RDD2022 are reported as such.

**Target.** Reference boxes of the four classes D00, D10, D20 and D40, matched one to one, same class, at intersection over union of 0.5 or more. For each external collection the class mapping is written at G4 from its documentation and label files, before any detector is run on it: only classes whose released names state longitudinal crack, transverse crack, alligator crack or pothole (or the RDD2022 codes D00, D10, D20, D40) are mapped. Boxes of other classes (for example block or oblique cracks, patches, manholes) are not reference boxes; there are no ignore regions, so a prediction on such an object counts as an unmatched alert. Unique physical damage instances are never inferred from video identifiers or positions.

## 4. Alert policy

- **Pool.** One pool per internal scoring fold and one per external collection; *N* is the number of released images in the pool.
- **Quota.** Each detector takes up to floor(*k* × *N* / 100) of its predictions of the four classes in the pool, in one ranking by confidence; ties are broken by image file name, then by box coordinates, then by bootstrap copy index (0 outside resampling). The same rule applies internally and externally.
- **Alert rate.** *k* = 152.49 (development-fold reference boxes of the four classes per 100 images; gate G7). For the pothole analysis (Section 6, item 6), *k* = 15.80 and the quota covers pothole predictions only.
- **Reporting.** Per detector and pool: assigned slots, used slots, underfill, and eligible reference boxes per 100 released images, by class.
- **Density-matched sensitivity (defined now).** The same alert rate means a different workload per target when damage prevalence differs. In this sensitivity, each pool's quota is floor(*k* × (ρ / ρ_dev) × *N* / 100), where ρ is the pool's eligible reference boxes of the target classes per 100 released images and ρ_dev is the same quantity on the development fold. It uses box counts only, which may be viewed before the lock (Section 9, step 5); no detector score enters it.
- **Interpretation.** Retrospective review of released images. The external spacing workflow and the alert rate produced by a development-fold threshold are secondary analyses.
- **No external tuning.** *k* is fixed on the development fold (gate G7); nothing is chosen on external labels.

## 5. Primary analysis

1. For each architecture: *x*₁ (S1 recall), *x*₂ (S2 recall), *z_s* (recall on external collection *s*), *z* (equal-weight mean of *z_s*).
2. **Numerator** Cov(*x̄*, *z*) and **denominator** Cov(*x*₁, *x*₂), across the fixed panel. Their ratio is *R*.
3. **Resampling.** 2,000 bootstrap draws, seed 2026. Internal clusters are resampled within RDD2022 partition strata; external blocks are resampled within each collection. In each draw, the same units are used for every architecture, repeated draws of a unit are treated as distinct copies carrying a copy index used in the quota tie-break (Section 4), and the quota and matching are reapplied before the numerator and denominator are recomputed together.
4. **Repeatability gate.** The one-sided 95% lower bound of the denominator (bootstrap percentile) must be above 0.
5. **Confidence set.** Fieller's method at the 95% level with critical value 1.96, using the bootstrap covariance of numerator and denominator. The set may be a bounded interval [L, U], unbounded, or disjoint.

**Outcomes (mutually exclusive, declared now):**

| Outcome | Rule |
| --- | --- |
| Inconclusive | Repeatability gate fails, or the Fieller set is unbounded or disjoint, or L ≤ 0 and U ≥ 1 |
| Negative slope | U < 0. The across-model slope is negative; this does not mean every model pair reversed |
| No detectable retention | L ≤ 0 ≤ U and U < 1 |
| Partial retention | 0 < L and U < 1 |
| Compatible with full retention | 0 < L ≤ 1 ≤ U |
| Amplification | L > 1. External differences exceed benchmark differences |

The value 0.5 (half the benchmark advantage retained) is shown as a reference line in figures and is not a decision cutoff.

**Interpretation (binding on the paper).** Every outcome category describes the linear association of four-class recall across the model panel, between the benchmark and the named external collections, under the stated workload. None, including "compatible with full retention", is reported as evidence that a deployed system works on a new road network.

**Prespecified sensitivities reported with the primary:** each collection's own ratio; leave-one-architecture-out ratios; seed-0-only panel; density-matched quota; **strict leakage removal**: the clean holdout, clusters and development, S1 and S2 folds rebuilt with the same rule and seed but an SSCD threshold of 0.65 (the lowest bin checked at G3), with the primary analysis repeated on them and *k* unchanged (also for the pothole analysis).

**Setup B** repeats the analysis as a replication if gate G4c passes; otherwise its external results are descriptive (Section 8).

## 6. Secondary analyses

All are declared now and reported whatever they show. None can replace the primary.

1. **Selected versus default.** Rule frozen now: the default is the architecture with the highest published COCO accuracy within the budget (setup A: YOLOv12m; setup B: from official figures once YOLO26's mode is read at G0); the selected model is the one with the highest S1 four-class recall at the quota (ties to fewer operations, then fixed identifier). The selection is recorded after S1 is opened, with external scores still sealed. Paired external recall difference, selected minus default, with block-bootstrap intervals; if they coincide, the identical decision is reported with no substitute.
2. External spacing workflow on collections with verified spacing metadata (descriptive).
3. Alert rate produced on each external collection by the confidence threshold that gives *k* alerts per 100 images on the development fold.
4. Source-balanced benchmark recall (RDD2022 partitions weighted equally).
5. Merged-crack endpoint on collections whose crack definitions match.
6. **Pothole transfer.** The primary analysis repeated with pothole (D40) as the only target, *k* = 15.80, on every collection meeting the pothole criteria of G4 (at least 150 eligible pothole boxes and 20 independent blocks), with all Section 5 rules, outcome categories and sensitivities. Recorded at G7, before the lock, as a likely inconclusive outcome.
7. Spearman rank correlation between *x̄* and each collection's recall.
8. **Training-history comparison (descriptive).** Per architecture, best-epoch minus epoch 91 to 100 mean validation mAP50 (July Table 2), shown beside the benchmark recalls. It is not a bound on checkpoint-selection optimism or on bias in recall at the alert quota, and it is not used to correct any estimate.

**What the RDD2022 partitions measure (secondary, descriptive; added in 3.3.6).** Run at step 9 on the setup A clean holdout (development, S1 and S2 together) unless stated otherwise, by RDD2022 partition and class (four classes), with 2,000 cluster-bootstrap draws within partition strata, seed 2026. A reference box counts as **detected** by a checkpoint if a prediction of the same class with confidence 0.25 or more (the July operating threshold) matches it one to one at intersection over union of 0.5 or more (Ultralytics' matcher, as in E1), and as **reachable** if any prediction of the same class above the candidate floor 0.001 overlaps it at 0.5 or more (not one to one). The 20 checkpoints are the six three-seed architectures (YOLOv5s, YOLOv8s, YOLOv9s, YOLOv10s end-to-end, YOLO11s, YOLOv12s) and YOLOv12n and YOLOv12m (seed 0).

9. **Shared misses and oracle ceiling.** Share of reference boxes missed by all 20 checkpoints (shared misses) and share detected by at least one (oracle recall), against the best single checkpoint's recall.
10. **Attribution of the partition gap to measured properties.** Outcome: detection of each reference box by YOLOv12s at confidence 0.25, as the share of its three seeds; the 20-checkpoint mean is a sensitivity. One binomial model with a logit link, fitted on all partitions pooled, with these predictors fixed now: class; logarithm of box area at model input; box aspect ratio; horizontal and vertical box center (normalized); source image width and height; image lightness mean and standard deviation (CIELAB L*); logarithm of Laplacian variance (sharpness); edge density; entropy; number of reference boxes in the image. No partition indicators and no feature-space distance enter the model. For each partition, the gap between its observed recall and the pooled recall is split into the part reproduced by the model's mean predicted recall for that partition (explained by measured acquisition and object properties) and the remainder. The remainder contains annotation differences and genuine road-surface differences, which this analysis cannot separate; no annotation audit is run.
11. **Leakage exposure by partition.** Share of each partition's validation images linked to a training image under the final G3 rule and at the 0.65 sensitivity threshold (label-free, reported at G3), and recall at confidence 0.25 on linked versus clean validation images (step 9).

## 7. Fallbacks (declared now)

1. **Fewer than 2 collections eligible for the four-class primary but at least 2 meeting the pothole criteria.** The pothole analysis (Section 6, item 6) becomes the primary, with its G7 projection of a likely inconclusive outcome stated.
2. **Fewer than 2 collections meeting either criterion.** The paper becomes an internal decision-sensitivity study: how training seed, reference labels (N-RDD2024 on the verified image intersection) and matching protocol change benchmark recall differences, with split-half repeatability as the reference.
3. **Repeatability gate fails.** The primary is reported as inconclusive; secondary analyses are reported as declared.

## 8. Exploratory analyses

Run only if budget remains; labeled exploratory; cannot change a confirmatory conclusion (the forecast follows `forecast_rule.md`, fixed before fitting): resolution policies at the same quota (detail loss against a sham at 640 pixels, full-frame 1280 pixels, 2 × 2 tiling); reference substitution with N-RDD2024; a label-free forecast of each collection's recall, committed with the predictions; the remaining E1 checkpoints; other public RDD2022 detectors with unknown splits (external only); setup B external performance if gate G4c fails.

## 9. Execution order and what may be viewed

| Step | Action | May be viewed | Stays sealed |
| --- | --- | --- | --- |
| 1 | Freeze G0 (budget, panels, defaults, YOLO26 mode) | Official model-card figures | Everything below |
| 2 | Pod session 1: G1 reproduction, G2 matcher check, E1 predictions and outcome files, E2 properties, C0 descriptors and hashes | G1 aggregate mAP50 and the per-partition YOLOv12s values (both reproduce published July figures); G2 differences; run time and cost; label-free C0 counts | All per-object and per-image outcome files and any per-architecture recall (hashed on creation, not opened) |
| 3 | Laptop: apply the fixed near-duplicate rule; build clusters; assign development, S1, S2; hash the lists (G3) | Cluster counts and sizes (label-free) | S1 and S2 outcomes |
| 4 | Development fold only: fix *k* with a written rationale; precision projection from development-fold variance under stated external score and variance scenarios (G7) | Development-fold outcomes | S1, S2 and external outcomes |
| 5 | External eligibility and blocks (G4 to G6); setup B audit (G4c) | Licenses, hashes, overlap removals, reference box counts by class, class mappings, block structure | External detector scores |
| 6 | External inference (predictions only); write predictions and manifests | Prediction files (no labels joined) | External scores |
| 7 | G8 lock: commit protocol, scripts, fold lists, *k*, manifests and prediction hashes; publish them as a GitHub release and a Zenodo deposit (public, timestamped, DOI; files may be under embargo, record and date public) | — | S1, S2 and external scores |
| 8 | Open S1; record the secondary selection | S1 outcomes | S2 and external scores |
| 9 | Open S2 and external scores; compute primary, secondary and sensitivity analyses | All | — |

Before opening S1, S2 or external outcome files, the scoring script recomputes the SHA-256 hash of every locked input (protocol, scripts, fold lists, *k*, manifests, predictions) and compares it with the committed manifest; any missing file or mismatch stops the run.

## 10. Gates (empirical; status separate from design approval)

| Gate | Pass rule | Status |
| --- | --- | --- |
| G0 Budget and panels | 69 billion operations frozen with its basis; panels and defaults fixed; counting convention cited per family; YOLO26 mode read (missing mode blocks setup B only) | **Frozen 15 November 2026** (`G0_freeze.md`): YOLO26 end-to-end; defaults YOLOv12m (A) and RF-DETR Small (B) |
| G1 Reproduction | YOLOv12s seed 0 gives 0.6192 ± 0.002 mean average precision at overlap 0.5 on the 5,757 validation images | **Passed 15 November 2026**: 0.6182 (difference −0.0010) |
| G2 Matcher | In Ultralytics' input space, recomputed mean average precision within 0.002 of Ultralytics' value for every run (flag mismatches reported) | **Passed 15 November 2026**: all 27 runs, difference 0.0; flag mismatches 0 except one prediction in each of two runs |
| G3 Clusters and folds | Fixed near-duplicate rule applied; boundary check of the descriptor threshold recorded and any revision made by the Section 3 rule before any outcome is opened; largest cluster reported; development, S1 and S2 lists hashed | **Done 15 November 2026** (`G3_record.md`): threshold 0.75 kept; 1,580 of 5,757 validation images linked to training; clean holdout 4,177 (development 1,044, S1 1,568, S2 1,565); largest cluster 3 images |
| G4 Collection eligibility | Per collection: license, download integrity, images linked to RDD2022 by the near-duplicate rule removed, class mapping written, at least 20 independent blocks remaining. **Four-class primary:** at least 150 eligible reference boxes in each of D00, D10, D20 and D40. **Pothole analysis:** at least 150 eligible pothole boxes | **Done 15 November 2026** (`G4_G6_rule.md`, `G4_G6_record.md`): four-class eligible SVRDD Beijing, EcuPDD Ecuador, IRD Baghdad; pothole eligible all seven downloaded collections; N-RDD2024 and UAV-PDD2023 failed on released pothole counts; overlay check by the corresponding author passed |
| G4c Setup B scoring set | Every included checkpoint loaded; scoring set verified against its training and checkpoint-selection data by file name and content hash; near-duplicates removed; class order checked; enough images for development, S1 and S2. Failure: setup B descriptive | **Passed 15 November 2026** (`G4c_record.md`, `G4_G6_record.md`): scoring set 4,178 images (development 1,043, S1 1,568, S2 1,567), 0 content mismatches; RF-DETR label ids per DetectionBench's source (0 to 3) |
| G5 Blocks | Per collection: verified independent blocks (road segments, videos or image clusters); spacing metadata recorded where it exists | **Done 15 November 2026**: blocks by within-collection image links (pHash 4 or fewer, or SSCD cosine 0.65 or more); no release has verified frame-spacing metadata |
| G6 Minimum panel | At least 2 eligible collections from different releases and collection teams (checked separately for the four-class primary and the pothole analysis; Section 7) | **Passed 15 November 2026**: four-class 3 collections, pothole 7; the four-class primary is in force |
| G7 Alert rate and precision | *k* fixed on the development fold with rationale; precision projection recorded; a likely inconclusive outcome recorded before the lock | **Done 15 November 2026** (`G7_record.md`): pothole *k* = 15.80, projected share inconclusive 0.65 to 0.88, likely inconclusive recorded; four-class *k* = 152.49, projected share inconclusive 0.000 to 0.002 |
| G8 Lock | Step 7 completed; the scoring script's hash verification passes on all locked inputs | Not run |
| G9 Budget | Measured cost per pass from pod session 1; projected total within USD 30 | Pod session 1 about USD 1.73; G3 pair export about USD 0.06; session 2 about USD 11.35 (USD 2.59 first run, USD 8.76 rerun); total about USD 13.14 |

## 11. Disclosures

- **Published rankings.** The July manuscript's overall (four-class) mAP50 per architecture was known when this protocol was written. July manuscript, Table 3 (per-class AP50, full validation split), pothole column: YOLOv12m 0.5977, YOLOv12s 0.5820, YOLOv9s 0.5753, YOLO11s 0.5668, YOLOv8s 0.5662, YOLOv5s 0.5635, YOLOv12n 0.5527, YOLOv10s 0.5209. DetectionBench's model cards report overall test scores for setup B. Both were known when this protocol was written. This protocol measures recall at a fixed alert quota on cluster-clean folds, a different quantity; external transfer is the unresolved question.
- The July work selected checkpoints on the full validation split, which contains S1 and S2.
- The 0.65 sensitivity (Section 5) was added on 15 November 2026, after the label-free G3 check found that most decided pairs in the 0.65 to 0.75 bins showed the same location (16 of 20 and 16 of 19). No model prediction or outcome had been opened. The primary analysis keeps the threshold given by the registered rule (0.75).
- **Change of primary target (3.3.9, 15 November 2026).** Up to version 3.3.8 the primary target was pothole recall, and the all-class quota was secondary item 6. The precision projection at G7, computed on development-fold outcomes only, gave a 0.65 to 0.88 chance of an inconclusive pothole result, limited by the 241 and 219 pothole boxes in S1 and S2; the same projection for the four classes gave 0.000 to 0.002. The primary target was changed to the four classes after that projection and after the external collections' released class names and pothole counts had been read (dataset audit, 14 November; inventory, 15 November). No S1, S2 or external outcome had been opened. The pothole analysis is kept in full as a secondary analysis.
- A dataset audit (14 November 2026) opened label files of several external collections to count classes and boxes; no detector was run and no score computed.
- Steps completed before registration are listed with dates.

**Reporting commitments (binding on the paper).**

1. The primary result is stated only for the named external collections, the eligible panel and the stated workload.
2. The limitations section states, without mitigating language: (a) with 8 architectures in setup A and 6 in setup B, the ratio is imprecise and an inconclusive Fieller set is a realistic outcome, and the pothole analysis was projected before the lock to be likely inconclusive; (b) S1 and S2 share the July checkpoint selection, so the denominator does not isolate measurement noise; setup B, if gate G4c passes, is the only available check on selection optimism, and if G4c fails the primary result is conditional on that optimism; (c) crack labeling conventions differ between collections; (d) the primary target was changed after the G7 projection (Section 11, disclosures).
3. No claim of clean provenance is made unless gate G8 passes, including the scoring script's hash verification.

## 12. Points settled before registration

1. Closed in 3.3.2: the 0.75 threshold cites the SSCD authors' operating point and is checked on road images at G3 with a fixed revision rule.
2. Closed as a stated limitation: fold errors may be correlated through shared checkpoint selection. The question (Section 1) is defined as transfer from the benchmark used for selection, the assumption is stated in the limitations, and setup B, if gate G4c passes, is the test of whether the result depends on selection optimism (Section 2).

## 13. Design approval

| Author | Decision |
| --- | --- |
| Sreekar Aditya Reddy Pesaru (corresponding author) | Approved versions 3.3.8, 3.3.9 and 3.3.10 (15 November 2026) |
| Sahasra Reddy Patlola (co-author) | Approved versions 3.3.8, 3.3.9 and 3.3.10 (15 November 2026) |
| Dr. Dudam Bharath Kumar (co-author) | General written approval to proceed and submit; approval of versions 3.3.8, 3.3.9 and 3.3.10 confirmed by the corresponding author (15 November 2026) |

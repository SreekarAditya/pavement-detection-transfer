# Rule document for the analyses after registration

Status: draft for approval. It becomes binding when this file and `params.yaml` are copied to `post_reg/rules/`, together with the split lists (`post_reg/rules/split/`) and the experiment 5 run table, committed, and tagged `post-reg-rules-v1`. Every number the analysis code uses is read from `params.yaml`; this file states what each value means and why it was chosen. Where the two disagree, the analysis stops.

These analyses are not part of the registered study (protocol 3.3.10, doi:10.5281/zenodo.24246355). The registered results are unchanged and are reported as recorded.

## 0. Definitions used by every experiment

**Seed.** 2026 for every random step, combined with fixed stream keys in the code so that each experiment, collection and class set has its own reproducible stream.

**Matching and quota.** As in the locked scripts (`scripts/pdt_core.py`): one-to-one, same-class matching at intersection over union 0.5 or more; the alert quota is one confidence ranking over a pool, floor(k × N / 100) alerts, with the locked tie-break. Alert rates: k = 152.49 per 100 images (four classes), 15.80 (pothole), 136.69 (merged cracks: D00, D10 and D20 counted as one class).

**Detectors.** The procedure is run on the registered selection, YOLOv12m seed 0 (`yolo12m_seed0`). The other 19 setup A checkpoints are the sensitivity; the six setup B checkpoints are the second-family sensitivity where an experiment names them.

**Class sets.** Four classes and merged cracks on the three four-class collections (EcuPDD, IRD, SVRDD); pothole on all seven external collections.

**Calibration and held-out parts.** Each external collection is split into two parts made of whole similarity blocks (`s2/g4/blocks.csv`). The blocks are put in one random order (seed 2026) and added to the calibration part until it holds at least half of the collection's images; the rest form the held-out part. No image, and no block, is in both parts.

*Why image-balanced.* A plain random half of the blocks gave PDD2026 2,367 calibration images against 3,963 held-out ones, because one block holds 1,507 of its 6,330 images. Filling the calibration part by images keeps both parts at about half of every collection (2,043 and 2,043 for HRP4K, 3,165 and 3,165 for PDD2026; `real_split/split_draft_counts.csv` from the pre-checks) while still never splitting a block. The large PDD2026 block stays whole on one side, which is reported as a limitation.

*Why one half each.* The calibration part must support the largest sample sizes tested (n up to 500 images, M up to 800 boxes), and the held-out part must estimate true recall precisely enough to judge an error of 0.05; an even split serves both. Where a grid value exceeds what a calibration part holds, that value is skipped and reported (Section E3).

**Sample draws and bootstrap.** 2,000 repeated calibration or pilot samples per setting (Monte Carlo standard error of a coverage near 0.95 is about 0.005), and 2,000 bootstrap draws over blocks for every interval. The two sources of variation are reported separately.

**Order of claims.** Experiment 1 is primary, with two co-primary measures. Experiments 2, 3, 5 and 15 carry stated criteria and are secondary. Everything else is descriptive and carries no verdict.

## E1. The whole procedure against floor and ceiling (primary)

**Chain, on each collection and class set (four classes on three collections; pothole on seven), repeated over 2,000 draws:**
1. *Threshold:* the confidence giving k alerts per 100 images on a block sample of n unlabeled calibration images.
2. *Pilot:* calibration blocks in random order until at least M reference boxes are labeled; the recall at that threshold is estimated with the ratio estimator and its block-clustered interval (Section E3).
3. *Decision:* accept if the lower end of the two-sided 90% interval is at or above the recall floor; otherwise adapt. For the four-class set, "adapt" maps to the experiment 5 run of size 100 with draw = procedure draw modulo 3, its recalibrated threshold and its fresh-audit estimate (`exp1.adapt_size`). The pothole set has no adaptation run; its decisions are reported only.

**Constants n and M.** For each collection, the maximum, over the other collections of the same class set only, of the smallest grid value that met the experiment 2 or 3 criterion there (selected detector, block samples). If any other collection met the criterion at no grid value, the largest grid value is used and flagged "criterion not met elsewhere". A constant larger than the target's calibration part uses the whole part and is flagged. The constants file is computed after experiments 2 and 3, approved, committed and tagged `post-reg-exp1-constants` before experiment 1 runs. A separate value for an unseen network (the maximum over all collections) is kept for experiment 18.

**Recall floor.** A requirement, the same for every network: half the recall that the benchmark reports for the selected detector at the quota, averaged over the two clean-holdout scoring folds and rounded to the nearest 0.05 (the grid of experiment 3):

| Class set | Benchmark recall, fold S1 | Fold S2 | Half the mean | Floor |
| --- | --- | --- | --- | --- |
| Four classes | 0.6207 | 0.5809 | 0.3004 | 0.30 |
| Pothole | 0.5062 | 0.4977 | 0.2510 | 0.25 |
| Merged cracks (experiment 18 only) | 0.6452 | 0.6057 | 0.3127 | 0.30 |

Source: `score_results.json` (registered scoring, `A_four_class.primary`, `A_pothole.primary`, `secondary.merged_crack_A`, entries for YOLOv12m). *Why this anchor.* An agency adopting a benchmarked detector is told the benchmark recall; "keeps at least half of what the benchmark promised" is a requirement stated in those terms and uses no external result. External recalls of the registered study were known when the floor was set, so the decision share is also reported at every floor of the experiment 3 grid (0.10 to 0.60), computed from the stored interval ends in `e1_draws.parquet`.

**Floors.** (a) Development-fold threshold: the confidence giving k alerts per 100 development-fold images; stated recall = recall at that threshold on folds S1 and S2. (b) Confidence 0.25 (software default); stated recall = recall at 0.25 on S1 and S2. **Ceiling.** The threshold giving k alerts per 100 images over the whole collection; true recall on all its labels.

**Co-primary measures, on the held-out part.** Relative error of the alert volume against k; absolute error of the stated recall against true recall at the arm's threshold.

**Criterion.** The procedure is *validated on a network* if, over the 2,000 draws, at least 90% have the alert volume within ±10% of k and at least 90% have the recall error within ±0.05. The count of validated networks is reported, with the worst named. Uncertainty from the finite held-out part: held-out blocks resampled 2,000 times and both shares recomputed (`e1_bootstrap.parquet`).

**Secondary measures.** Recall under a pool-level cap (alerts above the threshold cut to the quota over the pool) and under a batch-level cap: images in random block order (one stream per arm and draw), batches of B = 100 images, alerts above the threshold reviewed in confidence order up to floor(k × B / 100) per batch, no carry-over; B = 25 and 400 as sensitivity. *Why 100:* a plausible daily review batch; 25 is the smallest batch that still allows pothole alerts (floor(15.80 × 25 / 100) = 3), and 400 is close to the pool-level cap.

**Check.** The procedure arm is recomputed with only calibration labels present and must reproduce every threshold, estimate and decision (`e1_checks.json`).

## E2. Threshold from an unlabeled calibration sample

n = 25, 50, 100, 250, 500 calibration images, sampled as whole blocks in random order until at least n images are included; 2,000 draws per n; the threshold is applied unchanged to the held-out part. No label is read (the code switches every label reader off). A value of n larger than the calibration part is skipped and reported (Kent: 358 calibration images, so n = 500 is skipped).

**Criterion.** The smallest n at which at least 90% of draws give held-out alert volume within ±10% of k, per collection, class set and checkpoint.

**Sensitivities.** (a) A contiguous sample in file-name order for IRD (`Dash_####` running numbers), Rome (date and time stamps) and SVRDD (time-stamped names), labeled as unverified order. EcuPDD is excluded (its time stamps are messaging-app sharing times, not capture times), as are Kent and PDD2026 (running numbers restart within class-named prefixes, so order follows the class folder). (b) The six setup B checkpoints on merged cracks at 136.69.

## E3. Pilot audit size, counted in reference boxes

M = 50, 100, 200, 400, 800 reference boxes of the class set; whole calibration blocks in random order until at least M boxes are labeled; 2,000 draws. A value of M larger than the calibration part's boxes is skipped and reported: from the image-balanced split, M = 800 is unavailable for pothole on EcuPDD (700), IRD (528), Kent (625), Rome (560) and SVRDD (490).

**Operating threshold** (`threshold_rule: calibration_quota`): the confidence giving k alerts per 100 images over the whole calibration part, so that the audit is judged separately from threshold-sampling noise. **True recall:** recall at that threshold on the held-out part.

**Estimators.** (a) Ratio estimator (matched over reference boxes) with a block-clustered linearization interval at 90% and 95%. (b) Prediction-powered inference, secondary: pseudo-reference = boxes on which at least half of the other 19 checkpoints agree (same class, overlap 0.5), each at its own quota threshold, the audited checkpoint excluded; PPI++ with the weight λ in [0, 1] chosen on the pilot sample; block-clustered variance; λ recorded per draw. *Why half:* a majority of the other checkpoints is the conventional consensus; a lower share fills the pseudo-reference with single-model false alarms.

**Decision rule.** Accept if the lower end of the two-sided 90% interval is at or above the floor, a one-sided 5% risk of accepting a detector exactly at the floor (the consumer's risk usual in acceptance sampling). Operating characteristic curves over floors 0.10 to 0.60 in steps of 0.05, with wrong-acceptance and wrong-rejection rates.

**Criteria.** Recall criterion: at least 90% of draws within ±0.05 of true recall; the smallest such M per collection and class set, with the design effect of block clustering. If no M meets it, the paper says that class cannot be audited to that margin with labels of this size. Prediction-powered inference is *useful* if the 95% interval's coverage is 0.93 to 0.97 and its median width is at least 10% smaller than estimator (a)'s at M = 200 in at least two of the three four-class collections.

## E4. Label-free forecast against pilot labels

The registered forecasts (`forecast/forecasts.json`; YOLOv12s seed 0 at confidence 0.25, as registered). Absolute error of the forecast against recall at 0.25 on the held-out part, beside the pilot estimate's error at each M. Appearance shift against label-density shift as a scatter for the seven pothole collections only; no correlation for the three four-class collections. Descriptive.

## E5. Local adaptation

**Run table** (`exp5_run_table.csv`, committed with this file): the selected checkpoint at 50, 100 and 200 images, three draws each, on each four-class collection (size curve); all 20 checkpoints at 100 images, one draw each (ranking check); the selected checkpoint on the whole calibration part (ceiling a). If the cost projected from the timing run (task 02) exceeds the budget, sizes are cut first, then the ranking check; never fewer than three draws for the selected checkpoint.

**Samples.** Whole calibration blocks in a seeded order: first the pilot (200 boxes), then further blocks up to the run's size; a fresh audit of 200 boxes from blocks not used for the pilot or training. *Why 200:* the middle of the experiment 3 grid, about ±0.06 at 95% for recall near 0.3 before clustering, and it fits every four-class calibration part beside the largest training size.

**Recipe** (fixed before training): 30 epochs, AdamW, learning rate 0.001, batch 16, input 640 pixels, no warm-up, no frozen layers, seed 0, final-epoch weights, validation off; every other setting at the Ultralytics default (including mosaic, switched off for the last 10 epochs). *Why no warm-up:* Ultralytics warms up for at least 100 iterations; at 50 images and batch 16 a 30-epoch run has about 120 iterations, so most of it would run at warm-up learning rates, with a bias learning rate of 0.1, on an already converged checkpoint. *Why 0.001 with AdamW:* close to the rate Ultralytics itself picks for short runs with four classes (0.002 × 5 / (4 + 4) = 0.00125), small enough to adapt a converged detector without discarding it.

**Threshold after adaptation:** the confidence giving k alerts per 100 images on the calibration images not used for training (unlabeled use only). **Outcome:** held-out recall at the quota, adapted minus unadapted, both predicted on the same card in the same task, with a block-bootstrap interval. **Criterion:** adaptation recovers recall on a collection if the lower end of the 95% interval of the gain is above 0. Also reported: optimism (recall on the training images against held-out recall), forgetting (S1 and S2 recall), the ranking check (Spearman correlation of seed-mean recall, unadapted against adapted), ceiling (a), and ceiling (b): the best unadapted checkpoint by held-out recall, labeled as a bound because it is chosen on held-out labels.

## E6. Epoch-100 check

The registered primary analysis rerun with the 20 `last.pt` (each checked to be the final weights of a 100-epoch run) beside the 20 `best.pt`, both predicted on the same A40 with the registered inference settings and scored with the registered code. The outcome category is reported whatever it is.

## E7. Setup B on cracks only

The six setup B checkpoints on D00, D10 and D20 merged, k = 136.69, selection redone on S1 crack recall; registered primary analysis and leave-one-checkpoint-out sensitivities. With six checkpoints, an unbounded or disjoint Fieller set means no information, not support.

## E8. Leakage effect on R

R and its Fieller set with internal recall from all validation images, linked images only, the clean holdout at 0.75 and the strict holdout at 0.65; recall on linked minus clean images against parameter count. Descriptive.

## E9. Four-zone figure

Each clean-holdout reference box in one zone: found by the reference checkpoint at the quota; found by another checkpoint at its own quota; matched only by predictions at confidence 0.001 or more; unreachable. *Why the quota* (`exp9.operating: quota`) *and not confidence 0.25:* every recall in the paper is read at the quota, so "found" means the same thing here. Reference checkpoint: the one with the highest clean-holdout recall (labeled as chosen on the same images), and YOLOv12s seed 0 as a fixed version. Chance baseline: every prediction moved to a random position in its image, 20 repeats, keeping its confidence and quota membership. Zone shares with block-bootstrap intervals by partition and class. Descriptive.

## E10. Class-mix standardization

At confidence 0.25: external per-class recall reweighted to the RDD2022 class mix and benchmark recall reweighted to each external mix, weights and recalls from the same set; explained share with interval per collection, no pooled value; the quota version beside the registered density-matched quota. Descriptive.

## E11. What the review queue contains

**Part A.** Each unmatched alert of the selected detector at the four-class quota is classed, in order: wrong subclass (overlap 0.5 or more with a reference box of another mapped class); near miss (overlap 0.2 to 0.5 with a reference box of the same class); unmapped road object (at least half of the alert's own area inside a box of an unmapped class); background. *Why area of the alert:* patches and manholes are usually far larger than an alert on them, so intersection over union would call an alert lying wholly on a patch "background". Agreement count: other checkpoints with a same-class prediction at overlap 0.5 inside their own quota.

**Part B.** 150 unmatched alerts per four-class collection (a 90% half-width of about 0.07 on a proportion), stratified into confidence tertiles of that collection's unmatched alerts, 50 per tertile. *Why tertiles:* fixed confidence cut points would leave strata empty or tiny because each detector's confidences at the quota sit in a different range. Two raters (Aditya, Sahasra) blind to confidence, class, checkpoint and each other; categories: distress of a mapped class missing from the labels (with the class), other distress, road object, not damage, cannot tell. Cohen's kappa with a bootstrap interval; disagreements adjudicated by Dr. Bharath Kumar. Recall and the share of useful alerts recomputed with adjudicated missing distress counted as found, as a sensitivity; the high-agreement stratum reported apart. No model output or suggestion is shown to the raters.

## E12. Recall across alert quotas

Recall and R at 25, 50, 75, 100, 152.49, 200, 250 and 300 alerts per 100 images, where the sealed predictions allow (the pre-checks found enough predictions for every quota). Descriptive.

## E13. Recall by class

Per-class recall at the quota on each external collection, all 20 checkpoints and seed means, with intervals. Descriptive.

## E14. Detection against defect size at model input

Width, shortest side and square root of area of each reference box after the letterbox resize to 640 pixels, binned at 0, 8, 16, 32, 64, 128, 256 and 640 pixels; recall and shared-miss share per bin and class, clean holdout and each external collection. *Why these bins:* powers of two, matching the strides of the detector heads (8, 16 and 32 pixels), with 8 added so that the thinnest cracks are not pooled with 8 to 16 pixel boxes. Physical size only where a collection documents its camera geometry. Descriptive.

## E15. Norway resolution test

On the Norway clean-holdout images, the selected checkpoint and YOLOv12s seed 0, with the prediction pipeline for every variant (the untiled baseline included): untiled at 640 pixels; lower half of the frame as two side-by-side tiles; 2 × 2 tiles; whole frame at 1,280 pixels. Tiles overlap by 20%, boxes are mapped back to original pixels and duplicates merged by class-wise suppression at overlap 0.7 (the detectors' own setting).

*Why two tiles for the crop.* Norway frames are about twice as wide as high (3643 × 2041, 3650 × 2044 and 4040 × 2035 pixels; pre-check `image_sizes.csv`). Cropping the whole lower half keeps the long side, so the letterbox would shrink it by the same factor and nothing at model input would change. Two half-width tiles of the lower half double the scale at model input over the road surface, at half the cost of 2 × 2 tiles. *Why 20% overlap:* cracks are long, and a crack cut at a tile edge becomes two partial boxes that fail the 0.5 overlap test; 20% overlap keeps most defects whole in at least one tile.

**Comparison** at the same alert volume: the four-class quota applied per variant over the Norway pool; paired differences against untiled 640 with block-bootstrap intervals; recall at confidence 0.25 also reported, labeled as not matched in alert volume. **Reverse test:** United States and Japan clean-holdout images shrunk to Norway's median scale at model input (ratio of median scales, capped at 1) and placed on a grey canvas of the original size, beside the same images unchanged. **Criterion:** resolution is called a cause only if a Norway variant's recall at matched alerts rises with a 95% interval above 0 and the reverse test shows a loss.

## E16. Confidence calibration (supplement)

Expected calibration error with 15 equal-width bins and risk-coverage curves for the selected detector, predictions at confidence 0.001 or more, clean holdout and each external collection. Descriptive.

## E18. Networks never opened before

Candidates and their download hashes are committed in `exp18_candidates.csv` before any image is downloaded. Preparation, geometry and the frozen class mapping (`exp18.mapping`: only names that state longitudinal, transverse or alligator cracking or potholes, or carry D00, D10, D20 or D40; a bare "crack" stays unmapped) as in the gate G4 rule. **Inputs.** A release may come in several parts (for example train and valid folders; `images_dir` and `labels_dir` take `;`-separated paths) and in YOLO, Pascal VOC or COCO format; a COCO image listed without annotations counts as labeled with no boxes. Where the folder names inside an archive cannot be known before download, the COCO layout is fixed by rule instead of by path (`auto`): every JSON file in the archive that holds images, annotations and categories is one part, taken in sorted path order, and its images are looked up in that file's own folder; image files listed in no COCO file are counted and not used. Release checksums may be SHA-256 or, where the host publishes only MD5, `md5:<hex>`. **Copies inside a release are collapsed before anything is counted.** Two images are one if any of the seven non-identity rotations or reflections of one has a perceptual hash within 4 bits of the other, or if `copy_regex` maps both file names to the same source name (export tools such as Roboflow write augmented copies as `<source>_jpg.rf.<hash>.jpg`). Only the image with the smallest name in each group is kept. Reason: augmented copies inflate box counts and, once split, put the same road surface on both sides of the split; Tianjin's release falls from a raw pothole count above the threshold to 65 after this step. **Crops of one photograph share a block.** Where file names carry a parent-photograph identifier, `parent_regex` links all crops of that parent before blocks are formed, so they cannot land on both sides of the split. Overlap removal by the registered rule (perceptual hash 4 bits or fewer, or SSCD cosine 0.75 or more) against gate G4's own RDD2022 reference set and the seven collections already used; blocks at cosine 0.65.

**Endpoints**, any that apply, each with at least 20 blocks: four classes (150 boxes of each class), merged cracks (450 crack boxes in total), pothole (150 boxes). Split as in Section 0; held-out labels written to `private/`, hashed, and not read before phase D. One draw of the procedure with the unseen-network constants and the floor of the endpoint's class set; for the selected checkpoint's four-class decision "adapt", one adaptation (experiment 5 recipe, 100 images, fresh audit). Thresholds, stated recalls with intervals, decisions and adaptation are committed (`exp18_commitments.json`) before scoring; scoring recomputes the procedure and stops if any committed value differs. **Reported:** per network and endpoint, whether the alert volume landed within ±10% of k and the stated recall within ±0.05 of true recall; the count of networks where both hold, every failure named; sensitivity checkpoints apart. **Candidates at the time of writing** (`exp18_candidates.csv`; `check_candidates` stops on any blank or `?` field): Georgia and Türkiye (N-RDD2024 version 5, Mendeley Data 10.17632/27c8pwsd6v.5, CC BY 4.0, COCO exports) and Tianjin (UAV-PDD2023 version 4, Zenodo 8429208, CC BY 4.0, Pascal VOC). The Tianjin archive's member list (read from the zip directory alone, `exp18_tianjin_members.csv`) shows 2440 images named `lr_`, `tb_` or `r180_` plus a photograph number and a quadrant; the three prefixes share their photograph numbers, so each of about 237 drone photographs appears as left-right, top-bottom and 180-degree versions, each cut into up to four crops. No untransformed original is released. `parent_regex` puts every crop and version of one photograph in one block; the dihedral hash rule then keeps one version of each crop (the `lr_` file, by name order), so about one third of the images should remain. If more than 976 images (40%) remain after the collapse, it is reported as failed for Tianjin and the network is dropped rather than the rule changed after the fact. Mendeley offers neither a member list nor partial downloads, so Georgia and Türkiye use the `auto` layout; the parts found are recorded in `exp18_prepare.json`. Expected endpoint for all three: merged cracks only; whether each clears 450 crack boxes and 20 blocks is decided by the eligibility record after collapse and overlap removal, not by the raw counts. Georgia and Türkiye come from one team and one labeling protocol, so they are reported as two networks but are not independent evidence about labeling practice. Limitation: the N-RDD2024 exports were stretched to 640 × 640 pixels by the export tool, so object geometry at model input differs from the native photographs. **Excluded, with reasons:** CNRDD (signed access form), RDD2024_SA (available on request, license not verified), RoadDamageVision pooled release (contains RDD2022 China_Drone images; Spain subset provenance not established), CrackDataset_DL_HY (class aliases not in the frozen mapping; noncommercial license), RM-RDD and SM-RDD (license not verified; authors label potholes as D00, which conflicts with the frozen mapping), Bangladesh pothole set (no downloadable payload).

## E19. Image-level endpoint

At the quota: share of images with at least one reference box that receive at least one alert, and share of images without reference boxes that receive one. Descriptive.

## Budget

USD 30 in total. Spent: USD 13.46 on the 15 November pods (itemized pod history, all counted against the cap although some of it preceded the lock) and USD 0.25 on tasks 01 and 02 (account balance USD 16.86 before, USD 16.61 after), leaving **USD 16.29**. USD 2.00 is held back for experiment 18 (phase B inference and any adaptation) and the processor tasks (USD 0.06 per hour on the task 01 pod).

**Projection from the timing run** (task 02, `timing_with_projection.json`): on the A40, 30 epochs on 100 EcuPDD calibration images took 95.3 seconds for YOLOv12m and 46.5 seconds for YOLOv5s; the registered inference pipeline ran at 51.6 images per second. Scaled by image count, checkpoint size and the square root of source pixels, the full run table (87 runs: 27 for the size curve, 57 further ranking runs, 3 ceilings) plus its inference costs USD 5.50 (task 07), and experiments 6 and 15 cost USD 1.94 (task 06); USD 7.44 together, USD 5.99 to 11.62 if source pixels are taken as not mattering or as mattering linearly. RDD2022 staging, pod start-up and transfer are not in these figures. Even the upper figure leaves the reserve intact, so **the full run table is kept** (`exp5_run_table.csv`, 87 runs); nothing is cut.

**Stops.** Task 07 runs two fine-tuning processes on the card (YOLOv12m used 67% of it and 12 GB of its 46 GB in the timing run) and stops each at a run boundary after 14 hours (USD 8.75, above the upper projection of USD 8.73); task 06 is stopped by hand after 5 hours (USD 3.13). If either stop is reached, the remaining runs are listed as a deviation and the cut order of Section E5 applies before any further spending.

## Deviations

Any departure from this document (a skipped grid value, a shortfall, a stop) is listed in the task report and in the paper, with its reason. A change to this document after the tag is a new tagged version with the change and its reason recorded, and no result computed under the old version is discarded.

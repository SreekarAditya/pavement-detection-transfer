# Revision analyses: rules addendum

Status: approved by the corresponding author on 19 November 2026; corrected before tagging on the same day so that R1 states the registered experiment 1 shortfall rule (whole calibration part used and flagged), not the skip rule of experiments 2 and 3. This file becomes binding when it and `params_revision.yaml` are copied to `post_reg/rules/`, committed together with the revision code (`post_reg/revision/`), and tagged `post-reg-revision-v1`. No revision analysis reads data before that tag exists. Every constant is read from `params_revision.yaml`. Where this file and the YAML file disagree, the analysis stops.

## Standing of these analyses

These analyses were added after the registered results were read, in response to an internal review. The networks they use have open held-out labels, so they are **post hoc sensitivity analyses**: not confirmatory, and they replace no registered result. Every registered result is still reported under its original definition. Each analysis states below, before it runs, what it reports and how its result changes the text of the paper.

**No new decision threshold is introduced.** Each analysis reuses one of three registered values without change:
- the 90% share of draws within tolerance;
- the tolerances themselves, ±0.05 recall and ±10% alert volume;
- the 5% consumer's risk.

These were study choices in the registered rules. Only the 5% risk follows a common convention (acceptance sampling). The paper says this. Two further kinds of number appear below, and neither is a decision threshold:
- 95% Monte Carlo intervals, which use the standard normal quantile;
- grids that are swept and reported in full.

**Monte Carlo error.** Every share, rate and coverage that comes from resampling is reported with its Monte Carlo standard error (Morris, White and Crowther, 2019, *Statistics in Medicine* 38(11):2074–2102). For quantities pooled across splits, the split is the independent unit: the standard error is the standard deviation of the per-split values divided by the square root of the number of splits. A registered criterion is reported as **met**, **not met**, or **not resolved by the simulation** (when its 95% Monte Carlo interval contains the registered value).

**Frozen code** is imported and never edited. This covers `post_reg/scripts/` and `post_reg/supplements/qualification/` at the tag `qualification-evaluator-v1`. Where a frozen routine reads a constant that this addendum varies (the seed of the draw streams, or the box-matching threshold), the driver passes a modified copy of the constant in memory and logs it. The frozen files are not changed.

Before reading any data, the revision driver checks the SHA-256 of:
- the registered `RULES.md`, `params.yaml` and `exp1_constants.yaml`;
- this file and `params_revision.yaml`.

## R1. Experiment 1 over many block splits

**Splits.** For each seed in `r1.split_seeds` (3001 to 3050), every external collection is split by the frozen `make_split(files, blocks, share=0.5, seed, balance="images")` with the registered blocks file. The split lists of every seed are written out and hashed. The registered split (seed 2026) is run alongside as the reproduction gate and reported separately.

**Chain.** For each split, the frozen experiment 1 chain runs for the selected checkpoint (`yolo12m_seed0`) on every registered network and class set:
- at the registered constants: the procedure's n and M, the alert target k, the recall floor, the benchmark operating points, the ceiling, the pilot rule and the interval method;
- with `r1.draws` draws;
- with the draw streams seeded by the split seed.

Where n or M is larger than a calibration part, the whole part is used and the shortfall is flagged, as the registered experiment 1 rules require (registered rules, "Constants n and M"). The skip rule of experiments 2 and 3 does not apply to the chain. The two networks opened after registration (Georgia and Türkiye, merged-crack class set) are included as additional rows, using their committed unseen-network constants. They are labeled post hoc because their held-out labels are now open.

**Reproduction gate.** Seed 2026 with 2,000 draws must reproduce the registered `selected/e1_summary.csv` exactly. If it does not, the run stops and nothing else is reported.

**Reported per network and class set.**
- The pooled share of draws within tolerance over the 50 splits, for recall and for alert volume, each with its Monte Carlo standard error and 95% interval.
- The median per-split share and its 10th and 90th percentiles.
- The number of splits on which the registered criterion holds.
- The median pilot size in images, and the sampling fractions n / calibration images and M / calibration boxes.
- The share of draws whose pilot is the whole calibration part, and the share whose threshold sample is the whole calibration part.
- The within-tolerance share recomputed on draws whose pilot is **not** the whole calibration part, with the number of such draws.
- The number of splits on which n or M exceeds the calibration part, so that the whole part is used.

**Criterion.** The registered criterion is applied to the pooled draws: met if at least 90% are within tolerance, with the Monte Carlo rule above.

**Consequence for the text.**
- The abstract reports how many networks and class sets meet the criterion across splits, how many do not, and how many are not resolved. The registered-split count stays in the main results table.
- Where every draw's pilot is the whole calibration part, the pilot result is described as a comparison of calibration-part recall with held-out recall, not as a pilot success.
- Where every draw uses the same threshold sample, the alert-volume result is described as degenerate.

## R2. Recall-controlled threshold

**Arm.** In each R1 draw, the arm uses the same pilot blocks and labels as the procedure. Candidate thresholds are the distinct confidence values of the detector's predictions on the pilot images. The arm picks the highest candidate at which the lower end of the frozen two-sided 90% block-clustered interval of pilot recall is at or above the network's registered floor; that lower end is a one-sided 95% bound. If no candidate qualifies, the arm returns "no qualifying threshold".

**Reported.**
- The share of draws in which held-out recall at the arm's threshold is at or above the floor, beside the nominal 95%.
- The held-out alert volume relative to the target.
- The same two quantities for the procedure arm.
- The share of draws with no qualifying threshold.
- Monte Carlo standard errors throughout.

**Consequence for the text.** The comparison applies where the procedure's realized false-accept rate (R4) exceeds the 5% consumer's risk beyond Monte Carlo error. If, on such an endpoint, the recall-controlled arm is consistent with its nominal 95%, the discussion states that choosing the threshold for alert volume first cost the recall guarantee there.

## R3. Whole calibration part as the unlabeled sample

**Arm.** One threshold per split, network and class set, set from all calibration images by the frozen quota rule.

**Reported.** Held-out alert error, its distribution over splits, and the result beside the n = 500 result.

**Consequence for the text.** Where the alert error stays outside ±10% with the whole calibration part on most splits, the text states that more unlabeled images did not bring alert volume within tolerance there. It reports the number of splits.

## R4. Interval target, coverage and realized risk

**Estimand.** Recall on the network at the operating threshold; the held-out part stands in for the network.

**Reported per network, class set and M**, on the registered split and pooled over the R1 splits:
- coverage of the frozen 90% and 95% intervals against **calibration-part census recall** at the same threshold, computed from all calibration labels;
- agreement of the same intervals with held-out recall;
- both, without and with the finite-population correction for sampling blocks without replacement (variance multiplied by 1 − pilot blocks / calibration blocks), always reported side by side;
- the realized false-accept rate: among draws whose held-out recall is below the floor, the share accepted;
- decision accuracy split by whether held-out recall lies within the registered recall tolerance (±0.05) of the floor or farther away, at the registered floor and at every value of the registered floor grid (`exp3.floor_grid`).

Every coverage and rate has its Monte Carlo standard error.

**Consequence for the text.** The abstract reports two kinds of departure, each beyond Monte Carlo error:
- a coverage whose 95% Monte Carlo interval excludes its nominal level, at the M the procedure actually used;
- a realized false-accept rate above the 5% consumer's risk.

## R5. The frozen evaluator on clauses it was not built for

**Freeze.** The evaluator, schema and operators that produced the registered 12-case transfer result are tagged `qualification-evaluator-v1` before any new clause is read for encoding. They are not changed during R5. Any later change gets a new version, and every fixture, old and new, is rerun.

**Sources.** Listed in `params_revision.yaml` (`r5.sources`), chosen by the selection rules given there. None overlaps with the sources of the registered conformance and transfer cases. Each source file is pinned by SHA-256. Copyrighted sources stay outside the public repository; only their hashes and clause locators are published.

**Clause inventory, before encoding.** Every clause in the listed sections that states a measured quantity with a threshold, tolerance or time limit, and an action, is entered in `clause_inventory.csv` with its locator. The inventory is committed and tagged `post-reg-revision-clauses` before any fixture is written.

**Fixtures and expected answers.** Each clause gets test fixtures (an input record and a question), written without running the evaluator. An author enters every expected answer from the source alone, without seeing evaluator output. The same author marks any clause that needs interpretation beyond its text; such clauses are listed and counted but not scored. Answers fall into two groups, reported separately:
- **Group A.** The source states the outcome, in a worked example or an explicit decision rule. The answer cites page and table.
- **Group B.** No published outcome exists, but the clause wording admits one answer. The answer quotes the governing words.

The completed answer file is hashed, and its hash is committed and tagged `post-reg-revision-answers` before the evaluator runs on any new fixture. The paper reports that an author wrote the answers.

**Outcomes per clause.**
- Expressible unchanged.
- Expressible only with a new operator.
- Correctly refused (missing evidence or mismatched units).
- Not representable.

Agreement between the evaluator's answer and the expected answer is reported per fixture and per group.

**Consequence for the text.** The abstract reports the share of scored clauses expressible unchanged, with its count. The registered 12-case result moves to the body as a development result.

## R6. Comparison with an existing constraint language

**Language.** The World Wide Web Consortium Shapes Constraint Language (SHACL), validated with `pyshacl`. The same R5 clauses and the 12 registered cases are encoded. The shapes are written and committed before the answers tag.

**Reported per clause and representation.**
- Whether it can express the clause.
- Whether it can refuse for missing evidence.
- Whether it can carry the uncertainty rule (lower interval bound against a floor).
- The number of fields or shapes needed.

**Consequence for the text.** The paper claims only the capabilities that the comparison shows the decision record has and SHACL lacks. If there are none, the record is described as a domain profile, not a new representation.

## R7. Sensitivity to the box-matching threshold

**Analysis.** On the registered split, the four-class networks are rescored at every same-class intersection-over-union threshold in `r7.match_iou_sweep` (0.1 to 0.5 in steps of 0.1; 0.5 is the registered value). Held-out recall, the pilot estimates and the benchmark operating-point arms are all rescored. No single alternative threshold is preferred; the full sweep is reported.

**Reproduction gate.** At 0.5, the same driver must reproduce the registered held-out recalls exactly.

**Consequence for the text.**
- If, at any swept threshold below 0.5, a four-class benchmark arm's overstatement of recall falls within the registered recall tolerance (±0.05), the abstract states that the overstatement depends on the box-matching convention and names the threshold.
- Otherwise the text states that the overstatement exceeds the tolerance at every swept threshold.

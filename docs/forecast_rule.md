# Label-free forecast of each external collection's recall: rule (fixed before fitting)

Protocol version 3.3.9, Section 8 (exploratory: "a label-free forecast of each collection's recall, committed with the predictions"). Written 15 November 2026 before any forecast model was fitted. Fitting uses development-fold outcomes only; external outcomes stay sealed until the lock.

**Question.** Can the recall a detector will reach on a new road collection be forecast from that collection's images alone, before any of its labels are used?

**Detector and outcome.** YOLOv12s, three seeds. A reference box is detected if a prediction of the same class with confidence 0.25 or more matches it one to one at intersection over union of 0.5 or more (E1 definition). Collection recall = detected boxes / boxes, averaged over the three seeds. Two targets: four-class recall (D00, D10, D20, D40; the three four-class collections) and pothole recall (D40; all seven collections).

**Forecasters (all fixed now).**
- F0, benchmark value: development-fold recall, the same for every collection (what the benchmark alone would predict).
- F1, image properties and feature distance: a binomial model with a logit link fitted on development-fold images that contain at least one reference box of the target classes, weighted by that number of boxes. Outcome per image: seed-mean share of its target boxes detected. Predictors, standardized with development-fold means and standard deviations: logarithm of source width and height, CIELAB lightness mean and standard deviation, logarithm of (Laplacian variance + 1), edge density, grey-level entropy, mean cosine distance of YOLOv12s backbone features to the 10 nearest of 10,000 RDD2022 training images. Forecast for a collection: mean predicted probability over all images in its pool (labels are not used, so images cannot be weighted by box count).
- F2, feature distance only: as F1 with the feature distance as the only predictor.

**Evaluation (after unblinding, descriptive).** For each target: forecast error per collection; mean absolute error of F0, F1 and F2; Spearman rank correlation between each forecast and the observed recall across collections (pothole target, seven collections; four-class target, three collections, error only). The forecast is called useful only if F1 or F2 has a lower mean absolute error than F0 on both targets; this is exploratory and cannot change any confirmatory conclusion.

**Committed at the lock:** this rule, the fitting script, the fitted coefficients and the forecasts (`forecasts.json`).

# Gate G7: rule for the alert rate and the precision projection (fixed before any outcome is opened)

Written 15 November 2026 under protocol version 3.3.8 (Sections 4 and 9, step 4), before any development-fold prediction or outcome was opened.

## 1. Alert rate *k* (label counts only)

*k* = the number of eligible pothole (D40) reference boxes per 100 images in the development fold (main folds, threshold 0.75), rounded to two decimals.

**Rationale.** A reviewer is given as many alerts as there are potholes in the fold, on average. At this workload a perfect detector could find every pothole and no more, so recall equals precision (the R-precision point of information retrieval). The choice uses reference-box counts only; no detector score or outcome enters it, so it cannot be tuned toward any result. The same *k* is used for every detector, every internal fold, every external collection and the 0.65 sensitivity folds.

## 2. Use of development-fold outcomes

Development-fold outcomes are used only for the precision projection below. They do not change *k*, the panel, the folds or any analysis rule.

## 3. Precision projection (gate G7)

1. For each run, pothole recall at the quota floor(*k* × *N* / 100) on the development pool (*N* = development images), with the protocol's ranking, tie-break and one-to-one matching at intersection over union of 0.5 in original pixels.
2. Per architecture: the seed mean (single checkpoint for YOLOv12n and YOLOv12m).
3. Sampling covariance of the eight architecture values: 2,000 cluster-bootstrap draws within partition strata, seed 2026, quota and matching reapplied in each draw.
4. Spread of true benchmark recall across architectures: the sample variance of the eight development values minus the part expected from sampling error, tr(*H*Σ*H*)/7, where Σ is the bootstrap covariance and *H* the centering matrix (this accounts for errors shared by all architectures, which do not add spread); floored at zero. The true benchmark values are the development values shrunk toward their mean so that their sample variance equals this spread (all zero if the spread is zero).
5. Fold sampling covariance for S1 and S2: the development covariance scaled by the development-to-fold pothole-count ratio (165/241 for S1, 165/219 for S2).
6. External scenarios: two eligible collections; per-collection sampling covariance equal to the development covariance scaled by 165 / (collection potholes), for 150, 500 and 1,500 potholes per collection; true retention *R* = 0, 0.5 and 1; external architecture effects equal to *R* times the true benchmark deviations plus, in a second variant, independent architecture-by-collection noise with variance equal to the benchmark spread.
7. For each scenario, 2,000 simulated studies, seed 2026. Within each study the bootstrap is approximated parametrically (400 replicates of the fold and collection errors around the simulated observed values), giving the repeatability gate, the covariance of numerator and denominator, and the Fieller 95% set; report the share of each outcome category. The scenario closest to the expected collection sizes is read once the G4 pothole box counts are known (box counts only); a likely inconclusive outcome (share inconclusive above 0.5 in that scenario at *R* = 0.5 without extra noise) is recorded before the lock.
8. Also reported: the share of studies passing the repeatability gate from the internal folds alone, which does not depend on external data.

Amended 15 November 2026, before any outcome was opened: items 4, 6 and 7 made specific (sampling-error correction for shared errors, construction of the true values, parametric approximation of the bootstrap within simulated studies); item 8 added. *k* unchanged.

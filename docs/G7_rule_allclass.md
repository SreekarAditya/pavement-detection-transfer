# Gate G7, addendum: all-class precision projection (fixed before any all-class outcome was opened)

Written 15 November 2026 under protocol version 3.3.8, after the pothole projection (`G7_record.md`) and before any development-fold outcome of the other three classes was opened. Purpose: to see whether making the all-class alert quota (Section 6, secondary item 6) the primary target would make the transfer test decisive. This addendum changes nothing in the registered design; any change would be a separate, disclosed amendment made before the lock.

1. **Target.** All four classes (longitudinal crack, transverse crack, alligator crack, pothole). A prediction matches a reference box of the same class at intersection over union of 0.5 or more, one to one, by Ultralytics' greedy rule, in original pixels. Recall = matched reference boxes / all reference boxes in the pool.
2. **Alert rate.** *k*_all = development-fold reference boxes of all classes per 100 images, rounded to two decimals (label counts only). Quota floor(*k*_all × *N* / 100) over all predictions of all classes, one ranking by confidence, same tie-break as Section 4.
3. **Projection.** Identical to `G7_rule.md` items 1 to 8, with box counts of all classes in place of pothole counts. External collection sizes: 500, 1,500 and 5,000 reference boxes per collection.
4. **Feasibility note, recorded now.** An all-class primary needs external collections labeled for all four classes with a class mapping; several candidate collections label potholes only. Which collections qualify is read at G4 from label files only.

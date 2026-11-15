# Corrections made after the lock

The lock (protocol 3.3.10; Zenodo doi:10.5281/zenodo.24246355; `lock_manifest.json` SHA-256 ef49673bdc6301009223fb7eecd0aa71be7f53eb6220395d30262b64b19228c7) is unchanged. The registered scripts remain in the Zenodo deposit and in the commit tagged `v3.3.10-lock`. Corrections are listed here with their date and effect.

## 1. Secondary item 11, linked share at threshold 0.75 (15 November 2026)

**Error.** In `scripts/pdt_partitions.py`, the variable `cl` holding the clean holdout is narrowed (line 86) to clean images that contain at least one reference box, for items 9 and 10. Item 11 reused `cl` to decide which validation images are clean. Clean images without reference boxes were therefore counted as linked to training, which inflated the linked share at threshold 0.75 for China_Drone, Czech, India, Japan and Norway.

**Fix.** Item 11 now uses `clean`, the full clean holdout read before line 86. One line changed.

**Effect.** Only `item11_leakage/threshold_0.75/<partition>/linked_share` changes. The corrected shares equal the label-free shares recorded at gate G3 before any outcome was opened (`g3/main/leakage_by_partition.csv`). Not affected: the linked share at threshold 0.65 (a separate variable), the recall on linked versus clean images (images without reference boxes contribute no boxes), items 9 and 10, and every confirmatory and other secondary analysis.

**Verification.** The corrected script was rerun on the locked inputs and its output compared with the registered output value by value on 15 November 2026: exactly 5 values changed (the threshold 0.75 linked share of China_Drone, Czech, India, Japan and Norway), each now equal to its G3 value; every other value in the output is identical.

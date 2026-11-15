# Corrections made after the lock

The lock (protocol 3.3.10; Zenodo doi:10.5281/zenodo.24246355; `lock_manifest.json` SHA-256 ef49673bdc6301009223fb7eecd0aa71be7f53eb6220395d30262b64b19228c7) is unchanged. The registered scripts remain in the Zenodo deposit and in the commit tagged `v3.3.10-lock`. Corrections are listed here with their date and effect.

## 1. Secondary item 11, linked share at threshold 0.75 (15 November 2026)

**Error.** In `scripts/pdt_partitions.py`, the variable `cl` holding the clean holdout is narrowed (line 86) to clean images that contain at least one reference box, for items 9 and 10. Item 11 reused `cl` to decide which validation images are clean. Clean images without reference boxes were therefore counted as linked to training, which inflated the linked share at threshold 0.75 for China_Drone, Czech, India, Japan and Norway.

**Fix.** Item 11 now uses `clean`, the full clean holdout read before line 86. One line changed.

**Effect.** Only `item11_leakage/threshold_0.75/<partition>/linked_share` changes. The corrected shares equal the label-free shares recorded at gate G3 before any outcome was opened (`g3/main/leakage_by_partition.csv`). Not affected: the linked share at threshold 0.65 (a separate variable), the recall on linked versus clean images (images without reference boxes contribute no boxes), items 9 and 10, and every confirmatory and other secondary analysis.

**Verification.** The corrected script was rerun on the locked inputs and its output compared with the registered output value by value on 15 November 2026: exactly 5 values changed (the threshold 0.75 linked share of China_Drone, Czech, India, Japan and Norway), each now equal to its G3 value; every other value in the output is identical.

## 2. Gate G4c, class semantics of the setup B labels (15 November 2026)

**Finding.** Gate G4c required the class order of the DetectionBench labels (Hugging Face dataset `dronefreak/RDD2022`) to be checked before the lock. At the gate this was checked by class name only. After unblinding, a box-level comparison on the 475 images covered by both the DetectionBench labels and the RDD2022 labels showed that DetectionBench categories 0, 1 and 2 match the RDD2022 crack classes D00, D10 and D20 box for box (593, 241 and 230 boxes at overlap 0.9 or more). Category 3, released under the name `pothole`, matches none of the 103 RDD2022 pothole (D40) boxes on those images. Its 131 boxes mark crosswalks, lane markings, painted road text, manholes and patches, which are RDD2022 classes outside the four scored ones. The RDD2022 potholes are absent from these labels.

**Consequence.** Setup B was scored internally against these labels, so its fourth class meant road markings, manholes and patches on S1 and S2 and potholes on the external collections. Under the protocol's G4c failure rule, setup B is reported as descriptive and not as a replication of the primary analysis, and the setup B pothole analysis (secondary item 6) is not interpreted. The setup A analyses, which use the RDD2022 labels, are not affected.

**What did not change.** No script and no output changed. The setup B values are reported as computed by the registered scripts.

**Verification of inference.** To rule out an inference or conversion error, the DetectionBench YOLOv8s and YOLO26s checkpoints (same SHA-256 as those registered) were rerun independently on 24 EcuPDD images. Every confident box from the registered run was reproduced (42 of 42 and 30 of 30).

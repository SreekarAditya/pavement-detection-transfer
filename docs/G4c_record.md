# Gate G4c: setup B scoring set (record, part 1)

Protocol version 3.3.9. Run 15 November 2026 with `pdt_g4c.py` (label-free). Part 2 (content hashes against the Kaggle files, checkpoints, class order) runs in pod session 2 before any setup B inference (`pdt_s2_infer_b.py`).

## 1. DetectionBench's split lists

Source: Hugging Face `dronefreak/RDD2022`, parquet conversion (SHA-256 in `external_inventory.json`; train 89930eda…, validation f84d5662…, test 6d7636d4…).

| Check | Result |
| --- | --- |
| Images per list | train 26,869; validation 5,758; test 5,758 |
| Shared file names between lists | 0 for every pair |
| Names outside the Kaggle release used in setup A | 0 |
| Kaggle images not in any DetectionBench list | 0 (DetectionBench re-split the 38,385 annotated images) |
| Origin of the DetectionBench test images in the Kaggle release | 4,923 from its training part, 835 from its validation part |
| Git LFS SHA-256 listed (`db_lfs_sha256.csv`) | every image; the 13 files without a hash are `metadata.jsonl` files |

DetectionBench's checkpoints were trained on its train list and selected on its validation list, so its test list is independent of their training and selection. It is not independent of setup A's training data, which is irrelevant to setup B.

## 2. Near-duplicate removal and folds

Rule: pHash distance 4 or fewer, or SSCD cosine 0.75 or more (C0 descriptors), against every DetectionBench train and validation image.

| | Threshold 0.75 (main) | Threshold 0.65 (strict sensitivity) |
| --- | --- | --- |
| Test images linked to train or validation | 1,580 of 5,758 (27.4%) | 2,834 (49.2%) |
| Scoring set | 4,178 | 2,924 |
| Clusters (largest) | 4,128 (4 images) | 2,854 (4 images) |
| Development / S1 / S2 images | 1,043 / 1,568 / 1,567 | 732 / 1,097 / 1,095 |

Share of DetectionBench test images linked at 0.75, by partition: China_MotorBike 0.990, China_Drone 0.422, India 0.376, Czech 0.346, Norway 0.261, United States 0.113, Japan 0.087.

Fold lists (SHA-256): main development ebfc1484…, S1 7c258d4c…, S2 7185969a…; sensitivity development 39b609bf…, S1 e5c9b888…, S2 1ed1a382…. Summary: `g4c_summary.json` (7224ab42…).

## 3. Note for the paper

DetectionBench's published test scores were computed on a test list of which 27.4% of images are near-copies of its training or validation images by this study's rule. This is a label-free observation and is reported with the leakage results.

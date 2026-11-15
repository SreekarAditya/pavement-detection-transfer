# Gates G4 to G6 and G4c part 2: record

Protocol version 3.3.9; rules in `G4_G6_rule.md`. Pod session 2, 15 November 2026. Outputs: `06_pod_outputs/session2/session2_outputs.tar.gz` (SHA-256 0b9b1e855db6f96fd60b4a5cf3f228e317d7ca6f1dbd9997723de524fd5e02a8). No external score, recall or setup B internal score has been computed; the prediction files are sealed (`sealed_ext_manifest.json` 6879b0da…, 140 files; `sealed_b_manifest.json` b3349121…, 48 files).

## 1. Environment check

Reproduction gate repeated in session 2: YOLOv12s seed 0 mAP50 0.6180 (session 1: 0.6182; tolerance 0.001). Setup A: Ultralytics 8.4.24, PyTorch 2.15.0+cu132. Setup B: Ultralytics 8.4.153, rfdetr 1.12.1, supervision 0.31.7, PyTorch 2.15.0+cu132.

## 2. G4 and G5 by collection

| Collection | Images in pool | Removed (linked to RDD2022) | D00 | D10 | D20 | D40 | Blocks | Largest block share | Four-class | Pothole |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Rome version 2 | 2,009 | 0 | – | – | – | 1,261 | 871 | 0.023 | no | yes |
| SVRDD Beijing | 7,995 | 5 | 4,659 | 3,401 | 1,727 | 918 | 5,542 | 0.013 | yes | yes |
| EcuPDD Ecuador | 1,291 | 3 | 1,194 | 644 | 878 | 1,450 | 1,203 | 0.004 | yes | yes |
| IRD Baghdad | 4,349 | 3 | 1,974 | 1,551 | 2,320 | 903 | 1,258 | 0.042 | yes | yes |
| HRP4K | 4,086 | 0 | – | – | – | 4,748 | 2,804 | 0.010 | no | yes |
| Kent | 713 | 0 | – | – | – | 1,152 | 529 | 0.011 | no | yes |
| PDD2026 Indonesia | 6,330 | 2 | – | – | – | 1,688 | 3,013 | 0.238 | no | yes |

EcuPDD: one image could not be decoded (truncated file) and is not in the pool (1,295 released). Boxes are counts after the geometry rule and after removal.

## 3. G6

- Four-class primary: **passes** (SVRDD Beijing, EcuPDD Ecuador, IRD Baghdad: three releases, three collection teams). The four-class analysis is the primary in force.
- Pothole analysis (Section 6, item 6): **passes** (all seven collections).

## 4. G4c part 2

All 38,385 DetectionBench images were compared with the Kaggle files of the same name: 0 SHA-256 mismatches; all 4,178 scoring-set images checked, 0 mismatches. Checkpoint revisions and hashes in `g4c_part2.json` (5d634d03…). RF-DETR label ids: offset 0 from DetectionBench's source (categories 0 to 3); on the development fold RF-DETR Nano produced id 4 four times in 313,000 predictions (the spare output slot), RF-DETR Small never. **G4c passes**: setup B is a replication.

## 5. Overlay check (human)

A label-only sample of 12 images per collection was viewed by the corresponding author on 15 November 2026. Boxes sit on the labeled damage in all seven collections; no collection is excluded. Two observations were resolved: in IRD, the boxes on speed bumps belong to the unscored SpeedBump class (no IRD pothole box overlaps any of its 541 speed-bump boxes); in PDD2026, most boxes are unscored classes (depression, rutting, upheaval, patching) and the images were released resized to 1,024 × 1,024, while the pothole boxes sit on potholes.

## 6. July per-country rerun (confidence 0.25)

YOLOv12s seed 0, Ultralytics 8.4.24, the July setting for per-country mAP50 (July Section 3.6):

| Partition | Images | mAP50 | July mAP50 | Precision | July | Recall | July |
| --- | --- | --- | --- | --- | --- | --- | --- |
| China_MotorBike | 280 | 0.9466 | 0.9265 | 0.9037 | 0.9109 | 0.9136 | 0.9209 |
| China_Drone | 379 | 0.7120 | 0.6053 | 0.7165 | 0.7633 | 0.6709 | 0.6721 |
| United States | 744 | 0.6623 | 0.5514 | 0.6257 | 0.6486 | 0.6468 | 0.6326 |
| Japan | 1,538 | 0.6044 | 0.4767 | 0.6046 | 0.6232 | 0.5700 | 0.5676 |
| Czech | 406 | 0.4849 | 0.2738 | 0.6244 | 0.6379 | 0.3200 | 0.3200 |
| Norway | 1,206 | 0.4360 | 0.2197 | 0.5990 | 0.6485 | 0.2593 | 0.2597 |
| India | 1,204 | 0.4085 | 0.2808 | 0.4607 | 0.4723 | 0.3426 | 0.3357 |

Recall reproduces (differences of 0.007 or less); precision mostly within 0.02 (Norway 0.05, China_Drone 0.05); the July per-country mAP50 values do not reproduce. The July headline spread (0.2197 to 0.9265, "70.7 points") is therefore not used in the new paper.

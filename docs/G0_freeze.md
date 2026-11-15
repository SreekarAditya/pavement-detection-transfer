# Gate G0: budget, panels and defaults (frozen)

Frozen 15 November 2026 under protocol version 3.3.4 (Section 2 and gate G0). Sources: the official model cards and benchmark pages listed in Section 7 (read 15 November 2026) and the checkpoint metadata read below. No detector was run and no RDD2022 or external outcome was opened.

## 1. Budget

**69 billion operations per image.** Scenario: a dedicated edge device (NVIDIA Jetson Orin Nano Super) with the detector allotted about half of a 30-frame-per-second period, which is about 16.67 milliseconds. Anchor: Ultralytics' published 640-pixel TensorRT FP16 model-only latency on that device, 13.50 milliseconds for YOLO11m (inside) and 17.48 milliseconds for YOLO11l (outside). The ceiling sets eligibility only; measured latency is reported for every model.

## 2. Counting conventions

| Family | Source of operations per image | Input size |
| --- | --- | --- |
| Ultralytics YOLO (setups A and B) | Ultralytics v8.4.24 tagged model tables (multiply–accumulates doubled) | 640 |
| RF-DETR (setup B) | RF-DETR authors' computation table (arXiv 2511.09554v2, Appendix G) | Released resolution: Nano 384, Small 512 |

## 3. Panels

COCO values are box AP50–95 (percent) from the official cards; operations in billions per image.

**Setup A (own checkpoints, primary).**

| Architecture | COCO AP | Operations | Seeds |
| --- | ---: | ---: | --- |
| YOLOv5s (YOLOv5su row) | 43.0 | 24.0 | 0, 42, 123 |
| YOLOv8s | 44.9 | 28.6 | 0, 42, 123 |
| YOLOv9s | 46.8 | 26.7 | 0, 42, 123 |
| YOLOv10s (end-to-end) | 46.3 | 21.6 | 0, 42, 123 |
| YOLO11s | 47.0 | 21.5 | 0, 42, 123 |
| YOLOv12n | 40.6 | 6.5 | 0 |
| YOLOv12s | 48.0 | 21.4 | 0, 42, 123 |
| YOLOv12m | 52.5 | 67.5 | 0 |

**Setup B (DetectionBench, conditional replication).**

| Architecture | COCO AP | Operations | Mode or resolution |
| --- | ---: | ---: | --- |
| YOLOv8n | 37.3 | 8.7 | 640 |
| YOLOv8s | 44.9 | 28.6 | 640 |
| YOLO26s | 47.8 | 20.7 | 640, end-to-end |
| YOLO26m | 52.5 | 68.2 | 640, end-to-end |
| RF-DETR Nano | 48.4 | 31.9 | 384 |
| RF-DETR Small | 53.0 | 59.8 | 512 |

Excluded by the budget: YOLOv8m (78.9), RF-DETR Medium (78.8).

## 4. YOLO26 inference mode (read from checkpoint metadata)

Read by parsing the pickle opcodes of `best.pt` (no code executed). Both checkpoints store `end2end: True` in the model definition, so they run end-to-end without suppression, and the end-to-end COCO AP applies (YOLO26s 47.8, YOLO26m 52.5).

| Repository | Revision | best.pt SHA-256 | Saved by Ultralytics | Model file | end2end |
| --- | --- | --- | --- | --- | --- |
| dronefreak/rdd2022-yolo26s | 2f1b3d34e1f56bdcbdd1933b46e13b33d7bdc064 | b4e45aaacc0a1e14368ddadabeec3edb8694b77a91df163ea46fdb904f29e8c7 | 8.4.153 | yolo26s.yaml | True |
| dronefreak/rdd2022-yolo26m | a57f517eb5614a8954737cd0c4fa300826f79892 | eaa5869b6f4bc4abcc49fe7efceb04c2ef172127a51b6234dbadbd46deb65aa6 | 8.4.153 | yolo26m.yaml | True |

## 5. Defaults (rule: highest published COCO AP within the budget; ties to fewer operations, then fixed identifier)

- **Setup A: YOLOv12m** (52.5 AP, 67.5 billion operations), seed 0.
- **Setup B: RF-DETR Small** (53.0 AP at 512 pixels, 59.8 billion operations), ahead of YOLO26m end-to-end (52.5).

**Aggregation.** The selected-versus-default comparison (protocol Section 6.1) uses the same per-architecture value as the primary: the seed mean where seeds exist, the single checkpoint otherwise. YOLOv12m and YOLOv12n are single checkpoints. If the selected architecture is the default, the difference is zero by construction and reported as such.

## 6. Carried forward to later gates (not part of G0)

- **Software versions.** Setup A runs on Ultralytics 8.4.24 (the version recorded in its checkpoints). Setup B's YOLO checkpoints were saved by Ultralytics 8.4.153 and are run in a separate environment with that version; the RF-DETR package version is pinned at G4c.
- **Gate G4c.** DetectionBench's training settings (`args.yaml`, both YOLO26 repositories) record `split: val`, 50 epochs and early stopping with patience 10, so their best checkpoints were selected on their validation split. G4c must establish which RDD2022 images formed that split and verify that the setup B scoring set excludes them.

## 7. Sources (read 15 November 2026)

- Ultralytics v8.4.24 model tables (COCO AP, operations, parameters): [YOLOv5](https://github.com/ultralytics/ultralytics/blob/v8.4.24/docs/en/models/yolov5.md), [YOLOv8](https://github.com/ultralytics/ultralytics/blob/v8.4.24/docs/en/models/yolov8.md), [YOLOv9](https://github.com/ultralytics/ultralytics/blob/v8.4.24/docs/en/models/yolov9.md), [YOLOv10](https://github.com/ultralytics/ultralytics/blob/v8.4.24/docs/en/models/yolov10.md), [YOLO11](https://github.com/ultralytics/ultralytics/blob/v8.4.24/docs/en/models/yolo11.md), [YOLO12](https://github.com/ultralytics/ultralytics/blob/v8.4.24/docs/en/models/yolo12.md), [YOLO26 model data](https://github.com/ultralytics/ultralytics/blob/v8.4.24/docs/model_data.py); operation-counting convention in [torch_utils.py](https://github.com/ultralytics/ultralytics/blob/v8.4.24/ultralytics/utils/torch_utils.py).
- Edge-device latency anchor: [Ultralytics v8.4.33 NVIDIA Jetson guide, Orin Nano Super](https://github.com/ultralytics/ultralytics/blob/v8.4.33/docs/en/guides/nvidia-jetson.md#nvidia-jetson-orin-nano-super-developer-kit).
- RF-DETR COCO AP at released resolutions: [Roboflow benchmarks, pinned revision](https://github.com/roboflow/rf-detr/blob/b49df291cdd6b8d19fb722ab402e9a018d2edcd4/docs/learn/benchmarks.md); operation counts: [RF-DETR authors, arXiv 2511.09554v2, Appendix G](https://arxiv.org/html/2511.09554v2#A7).
- DetectionBench checkpoints: [publisher collection](https://huggingface.co/collections/dronefreak/rdd2022-object-detection-model-zoo), at the revisions listed in Section 4.

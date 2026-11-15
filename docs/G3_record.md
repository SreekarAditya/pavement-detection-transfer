# Gate G3: near-duplicate rule, clusters and folds (label-free)

Completed 15 November 2026 under protocol version 3.3.7, Section 3. Inputs: C0 outputs from pod session 1 (SSCD disc_mixup, short edge 288), the pair check by Sreekar Aditya Reddy Pesaru on the viewing page, and the validation file list. No label, prediction or outcome was read.

## 1. Boundary check and threshold decision

| Similarity bin | Pairs in total | Answered | Same | Different | Can't tell | Decided | Share different |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.65-0.70 | 5,487 | 20 | 16 | 4 | 0 | 20 | 0.200 |
| 0.70-0.75 | 3,336 | 20 | 16 | 3 | 1 | 19 | 0.158 |
| 0.75-0.80 | 2,366 | 20 | 15 | 4 | 1 | 19 | 0.211 |
| 0.80-0.85 | 1,539 | 20 | 20 | 0 | 0 | 20 | 0.000 |
| 0.85-0.90 | 938 | 20 | 20 | 0 | 0 | 20 | 0.000 |
| 0.90-1.00 | 745 | 20 | 19 | 0 | 1 | 19 | 0.000 |

**Decision:** 0.75 kept: at most half of the decided pairs in 0.75-0.80 are different locations. Links: pHash distance of 4 bits or fewer, or SSCD cosine similarity of 0.75 or more. No bin had fewer than 5 pairs, so no bins were merged; no reserve pairs were needed.

**Observation recorded before any outcome is opened.** In the two bins below the threshold (0.65 to 0.75), most decided pairs were also judged to show the same location (16 of 20 and 16 of 19). The rule never lowers the threshold, so these pairs remain unlinked in the primary analysis; residual leakage below 0.75 is a limitation of leakage removal.

## 2. Leakage, clusters and folds

- Validation images: 5,757. Linked to at least one training image: 1,580 (27.4%). Clean holdout: 4,177.
- Clusters among clean images: 4,137; largest cluster 3 images (0.07% of the clean holdout).
- Folds (seed 2026, stratified by partition): development 1,044, S1 1,568, S2 1,565 images.

**Leakage by partition (secondary item 11, label-free part):**

| Partition | Validation images | Linked to training | Share |
| --- | ---: | ---: | ---: |
| China_Drone | 379 | 163 | 43.0% |
| China_MotorBike | 280 | 276 | 98.6% |
| Czech | 406 | 148 | 36.5% |
| India | 1,204 | 442 | 36.7% |
| Japan | 1,538 | 143 | 9.3% |
| Norway | 1,206 | 343 | 28.4% |
| United_States | 744 | 65 | 8.7% |

**Clean-holdout images per fold and partition:**

| Partition | Development | S1 | S2 |
| --- | ---: | ---: | ---: |
| China_Drone | 54 | 81 | 81 |
| China_MotorBike | 1 | 2 | 1 |
| Czech | 64 | 97 | 97 |
| India | 190 | 286 | 286 |
| Japan | 349 | 523 | 523 |
| Norway | 216 | 324 | 323 |
| United_States | 170 | 255 | 254 |

## 3. Files and SHA-256

| File | SHA-256 |
| --- | --- |
| fold_dev.txt | 0901a1c20d43d7e8ea526d57046f594d221653490ad239af5bd317bb63a0b856 |
| fold_S1.txt | 712e9154f56e8b27c5b1d208beb5ae329db673e398022aa6624020697962ee22 |
| fold_S2.txt | 863fb179de6904b840c1f2675d39fe77a4e423d6920e8f0e39f5d3235200ff3d |
| clusters.csv | 9f4ce814112565027b2026c2ed589e91a7ec806baf465ed7732e54cc4294df96 |
| g3_decision.json | 940471effd43e1336dee2ea59826bea627a30a0255fe2e874f204fe8f265950b |
| g3_summary.json | 3bf8a4770fc49016871ac3fe5711eff437223ff45bf3095e73bd39476119231a |
| leakage_by_partition.csv | fc6a147403ec0515d67b407c44a872488fe8400cbe527a33b25c40675317b7b5 |
| g3_answers.json (pair check) | 0434f45a2f1ccee39849fc5d483d8b819350a9da2e198627c4ce6ccd885fca49 |
| manifest.json (pairs shown) | 19c6e22f860008ed0f2020f81b01225369a84ebe3058c3a016e37bdfe8f9b15b |

## 4. Sensitivity folds at SSCD threshold 0.65 (protocol 3.3.8, Section 5)

Same rule, code and seed; only the descriptor threshold changes. Built 15 November 2026, before any outcome was opened.

- Linked to training: 2,858 of 5,757 validation images (49.6%). Clean holdout: 2,899. Clusters: 2,856; largest 3 images.
- Folds: development 723, S1 1,090, S2 1,086 images.

| Partition | Linked at 0.75 | Linked at 0.65 |
| --- | ---: | ---: |
| China_Drone | 43.0% | 62.8% |
| China_MotorBike | 98.6% | 99.3% |
| Czech | 36.5% | 67.0% |
| India | 36.7% | 72.7% |
| Japan | 9.3% | 20.8% |
| Norway | 28.4% | 56.7% |
| United_States | 8.7% | 25.7% |

Fold list SHA-256 at 0.65: development 2822cbac81cf6440abef95b7fe91e03ed23ac3e49f2789357b9c2fbfc1325cd8; S1 f830a99ba14b835db2f34f97a4281a2678568e5aff7ed82e148ef67fd71cb351; S2 0b77f8e7b6dd79ff4e5d46b6ca76abc8a73eb19b31d39c817593ed4be32a021f.

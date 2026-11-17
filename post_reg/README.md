# Analyses after registration

Code for the analyses that follow the registered study (protocol 3.3.10, Zenodo doi:10.5281/zenodo.24246355). The locked tree (`docs/`, `scripts/`, `e1/`, `e2/`, `g3/`, `g4c/`, `g7/`, `s2/`, `forecast/`) is read-only; nothing here changes it.

| Folder | Contents |
| --- | --- |
| `scripts/` | Analysis code (`pr_*.py`). Every script checks the lock manifest and the rule document's hashes before it reads anything (`--skip_gate` exists for synthetic tests only). |
| `rules/` | The rule document (`RULES.md`, `params.yaml`, split lists, run tables, constants), committed and tagged before the analyses it governs. |
| `rules_draft/` | Draft `params.yaml`; nothing in it is final until it is copied to `rules/`, committed and tagged `post-reg-rules-v1`. |
| `run/` | Pod runners, one per task (`task01.sh` to `task09.sh`), shared settings (`common.sh`), expected weight hashes, and `run_tests.sh`. |
| `tests/` | Synthetic data generators and tests; no real data. |

## Running a task on a pod

1. Clone the repository on the pod and set the hashes from the task header, for example
   `export RULES_SHA=... PARAMS_SHA=...`.
2. Kaggle and Hugging Face sign-ins are made by Aditya on the pod; no token is written into any file here.
3. `bash post_reg/run/taskNN.sh [part]`. The runner installs the pinned versions (Python 3.13.18; numpy 2.5.4, pandas 3.1.1, pyarrow 26.0.1, scipy 1.18.1, statsmodels 0.15.0; on graphics-card pods Ultralytics 8.4.24 and PyTorch 2.15.0+cu132), downloads and checks the locked inputs, runs the steps (in parallel where they are independent), writes `MANIFEST.sha256`, and packs `post_reg_<task>_outputs.tar.gz` with its SHA-256 for the Mac.
4. Before the first real step on any new pod: `bash post_reg/run/run_tests.sh` (synthetic, a few minutes; `GPU=1` adds inference on the card).

| Task | Runner | Pod | Runs at the same time |
| --- | --- | --- | --- |
| 01 pre-checks | `task01.sh` | processor | the four table steps and the weight download |
| 02 fine-tuning timing | `task02.sh` (needs `SPLIT_DRAFT`) | A40 | data download and weights |
| 03 experiments 2, 3, 4 | `task03.sh` | processor | experiment 2 beside experiments 3 and 4 |
| 04 experiments 7, 8, 9, 10, 12, 13, 14, 16, 19 and the code-path check | `task04.sh` | processor | every part |
| 05 experiment 11 | `task05.sh a`, `b`, then `c ANSWERS1 ANSWERS2 [ADJUDICATION]` | processor | — |
| 06 experiments 6 and 15 | `task06.sh` (last.pt sent to `/workspace/run/weights_last/`) | A40 | best.pt, last.pt and variant inference share the card; then both scorings |
| 07 experiment 5 | `task07.sh` (`RUN_TABLE_SHA`, optional `WORKERS`, `MAX_HOURS`) | A40 | `WORKERS` fine-tuning processes share the card; resumable |
| 08 experiment 1 | `task08.sh constants`, then `task08.sh chain` (`T03`, `T07`, `CONSTANTS_SHA`) | processor | selected checkpoint beside the 19 others |
| 09 experiment 18 | `task09.sh A`, `B`, `C`, [`C_adapt`], `C_commit`, `D` | processor; A40 for B and C_adapt | setup A and setup B inference in phase B |

Human judgments (experiment 11 ratings and adjudication) are made by people; the packets carry no pre-filled answer, ranking or hint.

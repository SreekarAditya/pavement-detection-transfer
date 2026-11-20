"""Confirm the seven required datasets, split seals, and original commit before allocating."""

import csv
import json
from pathlib import Path
import subprocess
from urllib.error import HTTPError
from urllib.request import urlopen
from preflight import digest

NAMES = {"highrpd", "dashcam", "uneti", "m2s", "attain", "wasit", "capstone"}
SECONDARY = "5c8e6779c1ebc8fc5824ebe509f49c05eb07b03c"
BASELINE = "6441d5693914c54f2f0fad73f5d72eef62125ef89f1086563e345fc611b33a3b"


def check():
    post = Path.cwd() / "06_pod_outputs/post_reg"
    root = post / "aei_qualification/03_all_retry/attempt_02"
    archive = root / "qualification-nine-cpu.tar.gz"
    if digest(archive) != BASELINE:
        raise ValueError("The retry archive differs")
    protocol = root / ".control/protocol"
    subject = subprocess.check_output(["git", "-C", str(protocol), "show", "-s", "--format=%s", SECONDARY],
                                      text=True).strip()
    if subject != "2026-11-19: Freeze secondary within-release eligibility rule":
        raise ValueError("The approved secondary rule commit is missing")
    verified = root / "verified"
    rule = subprocess.check_output(["git", "-C", str(protocol), "show", SECONDARY + ":SECONDARY_RULE.md"])
    if rule != (verified / "frozen/SECONDARY_RULE.md").read_bytes():
        raise ValueError("The archived rule differs from the original commit")
    with (verified / "nine_dataset_outcomes.csv").open(newline="") as source:
        if {row["id"] for row in csv.DictReader(source) if row["secondary_outcome"] == "eligible"} != NAMES:
            raise ValueError("The retry archive does not identify exactly the approved seven datasets")
    for name in sorted(NAMES):
        split = verified / name / "split"
        for relative, expected in json.loads((split / "split_seal.json").read_text())["sha256"].items():
            if digest(split / relative) != expected:
                raise ValueError("Split byte seal differs: " + name + "/" + relative)
    constants = post / "08_constants/exp1_constants.yaml"
    if digest(constants) != "b0e41e96804816b8e0894e371a6c8dae628086b6a40fd76352719d8e456311e5":
        raise ValueError("Unseen-pothole constants checksum differs")
    print("Secondary eligibility rule commit:", SECONDARY, flush=True)
    print("Seven split seals and constants verified. No label content read.", flush=True)
    # The prior authorization covered one export and its pod has been terminated.
    # A private staged copy can be supplied without sharing a login in chat.
    capstone = post / "revision/remaining_experiments/private/capstone-v11-coco.zip"
    if not capstone.is_file():
        try:
            with urlopen("https://api.roboflow.com/capstone-9ndxs/potholes-detection-8wpmd/11/coco", timeout=20):
                pass
        except HTTPError as error:
            raise SystemExit(f"STOP: Capstone v11 COCO export HTTP {error.code}. The exact archived ZIP is not staged.") from None
        raise SystemExit("STOP: Capstone v11 COCO export is not staged for archive verification.")
    if digest(capstone) != "a7ff11fd5bb1e7439eb56a810e8c04a91e86d0d4f663f5983ee53b97aff35788":
        raise SystemExit("STOP: Capstone v11 staged export differs from the retry archive.")


if __name__ == "__main__":
    check()

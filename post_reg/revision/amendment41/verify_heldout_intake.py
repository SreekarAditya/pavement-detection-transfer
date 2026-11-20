"""Validate source receipts and hold the inventory tag if source spans are unresolved."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

REVISION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVISION / "qualification_v2"))
from duplicates import template_flag

STATES = tuple(
    "Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|"
    "Maryland|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|"
    "New Hampshire|New Jersey|New Mexico|New York|North Carolina|North Dakota|Ohio|Oklahoma|"
    "Oregon|Pennsylvania|Rhode Island|South Carolina|South Dakota|Tennessee|Texas|Utah|"
    "Vermont|Virginia|Washington|West Virginia|Wisconsin|Wyoming".split("|")
)
NHAI = re.compile(r"(?im)^\s*(?:schedule\s*[-–]?\s*[ke]|-[ke])\s*$")


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verify_dispositions(rows, downloads, repo):
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "post-reg-revision-v4.1^{commit}"],
                                     text=True).strip()
    if commit != "c1aaa1a5abb2b98b20184150df4dee3034ae7aee":
        raise ValueError("Amendment 4.1 intake tag differs")
    if tuple(row["candidate"] for row in rows if row["group"] == "state") != STATES:
        raise ValueError("State dispositions are incomplete, duplicated, or out of order")
    national = [row for row in rows if row["group"] == "nhai"]
    if [row["search_rank"] for row in national] != list(range(1, len(national) + 1)):
        raise ValueError("NHAI search sequence differs")
    if sum(row["disposition"] == "included" for row in national) > 20:
        raise ValueError("Included NHAI schedules exceed the frozen limit")
    with (REVISION / "clauses/source_manifest.csv").open(newline="") as file:
        development = {row["sha256"] for row in csv.DictReader(file)}
    included = set()
    for row in rows:
        if not row["reason"] or row["disposition"] not in {"included", "skipped"}:
            raise ValueError("Missing source disposition or reason: " + row["candidate"])
        if row["disposition"] == "skipped":
            continue
        source = downloads / row["file"]
        if sha(source) != row["source_sha256"] or source.read_bytes()[:5] != b"%PDF-":
            raise ValueError("Included source PDF differs: " + row["candidate"])
        if row["source_sha256"] in development | included:
            raise ValueError("An included source repeats an earlier file: " + row["candidate"])
        included.add(row["source_sha256"])
        source_text = downloads / row["text_file"]
        if sha(source_text) != row["text_sha256"]:
            raise ValueError("Included source text differs: " + row["candidate"])
        full = source_text.read_text()
        if row["group"] == "nhai" and not NHAI.search(full):
            raise ValueError("No eligible maintenance schedule heading: " + row["candidate"])
        if row["group"] == "state" and not row.get("acceptance_locator"):
            raise ValueError("No cited numeric crack-data acceptance criterion: " + row["candidate"])
        row["federal_template_flag"] = template_flag(full)
    return rows


def verify_inventory(rows, inventory_path):
    """Check coverage receipts before a tag. Counts alone cannot certify an inventory."""
    if not inventory_path.is_file():
        raise ValueError("No complete mechanically verified inventory of operative clause spans")
    document = json.loads(inventory_path.read_text())
    included = {row["candidate"]: row for row in rows if row["disposition"] == "included"}
    if set(document["sources"]) != set(included):
        raise ValueError("Clause inventory does not cover every included source")
    for key, coverage in document["sources"].items():
        if coverage["source_sha256"] != included[key]["source_sha256"] or coverage["unresolved_spans"]:
            raise ValueError("Unverified operative clause span in " + key)
        if not coverage["span_receipts"]:
            raise ValueError("No operative clause spans verified for " + key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dispositions", "downloads", "repo", "inventory", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    rows = verify_dispositions(json.loads(args.dispositions.read_text()), args.downloads, args.repo)
    args.out.write_text(json.dumps(rows, indent=2) + "\n")
    print("All source dispositions and included source checksums verified")
    verify_inventory(rows, args.inventory)


if __name__ == "__main__":
    main()

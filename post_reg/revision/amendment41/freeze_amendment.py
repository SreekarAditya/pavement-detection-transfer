"""Commit the approved rule and its private transcript seal before retrieving held-out sources."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys


def freeze(repo, transcript):
    data = transcript.read_bytes()
    rows = json.loads(data)
    if len(rows) != 564 or len({row["clause_id"] for row in rows}) != 564:
        raise ValueError("Private transcript count differs")
    for row in rows:
        if hashlib.sha256(row["text"].encode()).hexdigest() != row["text_sha256"]:
            raise ValueError("A transcript row checksum differs")
    tag = "post-reg-revision-v4.1"
    git = ["git", "-C", str(repo)]
    if subprocess.check_output(git + ["tag", "-l", tag]).strip():
        raise ValueError("The Amendment 4.1 tag already exists")
    approved = subprocess.check_output(git + ["rev-parse", "qualification-evaluator-v2^{commit}"]).decode().strip()
    if approved != "d7ccfbc7fcaca0a9fa2267a5848c74700d90d7e3":
        raise ValueError("Frozen evaluator tag differs")
    rel = "post_reg/rules/RULES_revision_amendment_v4_1.md"
    with (repo / rel).open("x") as output:
        output.write(
            "# Revision analyses: amendment 4.1\n\n"
            "Approved by the corresponding author on November 20, 2026.\n\n"
            "For a development row marked span unverified, a held-out clause that matches the "
            "development row on measurement, comparison operator, threshold value and unit "
            "after the frozen unit normalization is treated as a duplicate, reported, and excluded "
            "from the held-out test.\n\n"
            f"Private development transcript SHA-256: `{hashlib.sha256(data).hexdigest()}`.\n")
    subprocess.run(git + ["add", "--", rel], check=True)
    subprocess.run(git + ["commit", "-m", "Freeze additional duplicate rule before held-out intake", "--", rel], check=True)
    subprocess.run(git + ["tag", tag], check=True)
    print(subprocess.check_output(git + ["rev-parse", tag]).decode().strip())


if __name__ == "__main__":
    freeze(Path(sys.argv[1]), Path(sys.argv[2]))

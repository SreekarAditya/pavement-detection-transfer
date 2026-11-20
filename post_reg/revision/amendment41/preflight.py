"""Shared local readiness checks. No pod is allocated by this command."""

import argparse
import hashlib
import json
from pathlib import Path

MANIFEST = "ef49673bdc6301009223fb7eecd0aa71be7f53eb6220395d30262b64b19228c7"


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verify(root):
    path = root / "lock_manifest.json"
    if digest(path) != MANIFEST:
        raise ValueError("Locked manifest checksum differs")
    files = json.loads(path.read_text())
    if len(files) != 451:
        raise ValueError("Expected 451 locked files")
    mismatches = [name for name, expected in files.items() if not (root / name).is_file()
                  or digest(root / name) != expected]
    if mismatches:
        raise ValueError("Locked file check failed: " + ", ".join(mismatches))
    return len(files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(f"Lock verified: {verify(args.root)} files")

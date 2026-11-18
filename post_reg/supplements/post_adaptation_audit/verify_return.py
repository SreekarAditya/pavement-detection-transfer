"""Verify the downloaded result and its input/code identity using the Python standard library."""

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from statistics import NormalDist


CODE = {"score.py", "test_score.py", "run_on_pod.sh", "requirements.txt", "verify_return.py",
        "protocol_gate.sh", "ADDITIONAL_RULE.md"}
SCORING_FILES = {"fits.csv", "audit_blocks.csv", "summary.json", "MANIFEST.sha256"}
OUTER_FILES = {"environment.json", "test.log", "score.log", "protocol.log", "MANIFEST.sha256"}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def valid_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"unsafe archive or manifest path: {name!r}")
    return path


def unpack(archive, expected, target):
    if not re.fullmatch("[0-9a-fA-F]{64}", expected) or digest(archive) != expected.lower():
        raise ValueError(f"archive SHA-256 disagrees with expected value: {archive}")
    target.mkdir()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            path = target / valid_name(member.name)
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                path.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, path.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
            else:
                raise ValueError(f"unexpected tar member type: {member.name}")


def all_paths(root):
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}


def check_manifest(root):
    manifest_name = "MANIFEST.sha256"
    seen = set()
    for row in (root / manifest_name).read_text().splitlines():
        match = re.fullmatch(r"([0-9a-fA-F]{64})  (.+)", row)
        if not match:
            raise ValueError(f"malformed manifest entry in {root}")
        sha, raw = match.groups()
        name = valid_name(raw).as_posix()
        if name in seen or digest(root / name) != sha.lower():
            raise ValueError(f"duplicate or bad checksum: {name}")
        seen.add(name)
    if seen != all_paths(root) - {manifest_name}:
        raise ValueError(f"file inventory differs from manifest in {root}")


def check_results(root, input_sha, code):
    results = root / "results"
    check_manifest(results)
    if all_paths(results) != SCORING_FILES:
        raise ValueError("unexpected set of scoring outputs")
    paths = all_paths(root) - {"MANIFEST.sha256"}
    expected = (OUTER_FILES - {"MANIFEST.sha256"}) | {f"results/{p}" for p in SCORING_FILES}
    if paths != expected:
        raise ValueError("unexpected set of run outputs")
    metadata = json.loads((root / "environment.json").read_text())
    if metadata["input_archive_sha256"] != input_sha.lower():
        raise ValueError("returned input identity differs from the trusted local input")
    if metadata["code_sha256"] != {name: digest(code / name) for name in CODE}:
        raise ValueError("returned code identity differs from the trusted local code")
    check_counts(results)


def check_counts(results):
    def rows(name):
        with (results / name).open(newline="") as stream:
            return list(csv.DictReader(stream))

    def number(row, name):
        return float(row[name]) if row[name] else math.nan

    def equal(actual, expected):
        return (math.isnan(actual) and math.isnan(expected)) or math.isclose(
            actual, expected, rel_tol=0.0, abs_tol=1e-12)

    fits = rows("fits.csv")
    blocks = rows("audit_blocks.csv")
    identities = {f"r{i:03}": (coll, d) for coll, start in
                  [("ecupdd", 3), ("ird", 32), ("svrdd", 61)]
                  for d, i in enumerate(range(start, start + 3))}
    ids = [f["run_id"] for f in fits]
    if len(fits) != 9 or set(ids) != set(identities) or any(b["run_id"] not in ids for b in blocks):
        raise ValueError("Expected all nine distinct fits")
    for fit in fits:
        rid = fit["run_id"]
        if identities[rid] != (fit["collection"], int(fit["draw"])) or number(fit, "recall_floor") != 0.30:
            raise ValueError(f"{rid}: identity or floor mismatch")
        cluster = [b for b in blocks if b["run_id"] == rid]
        if len({b["block"] for b in cluster}) != len(cluster):
            raise ValueError(f"{rid}: duplicate block")
        y = [int(b["matched"]) for b in cluster]
        x = [int(b["refs"]) for b in cluster]
        if any(m < 0 or m > n for m, n in zip(y, x)):
            raise ValueError(f"{rid}: invalid block counts")
        n = len(x)
        p = sum(y) / sum(x) if sum(x) else math.nan
        se = math.sqrt(n/(n-1) * sum((m - p*v)**2 for m, v in zip(y, x))) / sum(x) if n > 1 and sum(x) else math.nan
        radius = NormalDist().inv_cdf(.95) * se
        refs, matched = number(fit, "heldout_refs"), number(fit, "heldout_matched")
        truth = matched / refs if refs > 0 else math.nan
        if math.isfinite(refs) and not (0 <= matched <= refs):
            raise ValueError(f"{rid}: invalid held-out counts")
        checks = {"audit_recall": p, "audit_se": se, "audit_lo90": p-radius, "audit_hi90": p+radius,
                  "heldout_recall": truth, "heldout_alerts_per_100":
                  100 * number(fit, "heldout_alerts") / int(fit["heldout_images"])}
        if fit["audit_status"] != "no_recalibrated_threshold":
            checks.update(audit_matched=sum(y), audit_refs=sum(x), audit_blocks=n,
                          audit_images=sum(int(b["images"]) for b in cluster))
        if any(not equal(number(fit, key), value) for key, value in checks.items()):
            raise ValueError(f"{rid}: counts or interval disagree with fit table")
        for rule, estimate in [("point", p), ("interval", p-radius)]:
            # The frozen scorer decides from its full-precision float. Use its
            # round-trippable CSV value for a bound at machine precision of the floor.
            field = "audit_recall" if rule == "point" else "audit_lo90"
            if abs(estimate - .30) <= 1e-12:
                estimate = number(fit, field)
            accept = estimate >= .30
            outcome = ("unscorable" if not math.isfinite(truth) else
                       ("correct_accept" if accept else "unnecessary_adaptation") if truth >= .30 else
                       ("false_accept" if accept else "correct_adapt"))
            if fit[f"{rule}_action"] != ("accept" if accept else "adapt") or fit[f"{rule}_outcome"] != outcome:
                raise ValueError(f"{rid}: {rule} action or held-out outcome differs")


def pack(root, archive, input_sha, code):
    check_results(root, input_sha, code)
    paths = sorted(all_paths(root) - {"MANIFEST.sha256"})
    (root / "MANIFEST.sha256").write_text("".join(f"{digest(root / p)}  {p}\n" for p in paths))
    check_manifest(root)
    with tarfile.open(archive, "x:gz") as tar:
        for name in sorted(all_paths(root)):
            tar.add(root / name, arcname=name, recursive=False)
    checksum = f"{digest(archive)}  {archive.name}\n"
    archive.with_name(archive.name + ".sha256").write_text(checksum)
    print(checksum, end="")


def main():
    args = sys.argv[1:]
    if args[:1] == ["--unpack-inputs"] and len(args) == 4:
        _, archive, sha, target = args
        unpack(Path(archive), sha, Path(target))
        check_manifest(Path(target))
        print("Input archive and complete member manifest verified.")
        return
    if args[:1] == ["--pack-return"] and len(args) == 5:
        _, root, archive, sha, code = args
        pack(Path(root), Path(archive), sha, Path(code))
        return
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("return_archive", type=Path)
    p.add_argument("return_sha256", help="SHA-256 recorded on the pod")
    p.add_argument("input_sha256", help="SHA-256 from the local handoff")
    p.add_argument("--code-dir", type=Path, default=Path(__file__).resolve().parent)
    parsed = p.parse_args()
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch) / "return"
        unpack(parsed.return_archive, parsed.return_sha256, root)
        check_manifest(root)
        check_results(root, parsed.input_sha256, parsed.code_dir)
    print("Return archive, manifests, input and scoring code identities verified.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, tarfile.TarError) as error:
        raise SystemExit(f"Verification stopped: {error}") from error

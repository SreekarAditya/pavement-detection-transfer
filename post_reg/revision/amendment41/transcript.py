"""Extract conservative, verbatim development clause spans from pinned PDFs."""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile

REVISION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVISION / "qualification_v2"))
from text_checks import m3, threshold_values

SPEC = json.loads((REVISION / "measures/source_map.json").read_text())
PAGE = re.compile(r"\bpp?\.\s*(\d+)(?:[–-](\d+))?")
SECTION = re.compile(r"§\s*(\d+(?:\.\d+)*)")
ACTION = re.compile(r"\b(?:shall|must|required|should|qualif\w*|accept\w*|reject\w*|"
                    r"record\w*|report\w*|classif\w*|repair\w*|remov\w*|replace\w*|"
                    r"retain\w*|flag\w*|class|severity|condition|threshold|tolerance)\b", re.I)
WORDS = re.compile(r"[a-z]{4,}")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def exact(text):
    return re.sub(r"\s+", "", text)


def locator_text(row, text):
    source_id = row["source_id"]
    locator = row["locator"]
    if SPEC[source_id].get("section_scoped"):
        section = SECTION.search(locator)
        if section is None:
            return "", "No section number at the locator"
        target = re.compile(r"(?m)^\s*" + re.escape(section[1]) + r"\.?\s+")
        starts = list(target.finditer(text))
        if len(starts) != 1:
            return "", "The numbered source section is not uniquely located"
        start = starts[0]
        end = re.search(r"(?m)^\s*\d+(?:\.\d+)+\.?\s+", text[start.end():])
        return text[start.start():start.end() + end.start()] if end else text[start.start():], ""
    page = PAGE.search(locator)
    if page is None:
        if row["clause_id"] != "state_program_1-profile-sampling":
            return "", "No PDF page can be resolved from the locator"
        first, last = 15, 16
    else:
        first, last = int(page[1]), int(page[2] or page[1])
    offset = SPEC[source_id]["page_offset"]
    pages = text.split("\f")
    start, finish = first + offset - 1, last + offset
    if not 0 <= start < finish <= len(pages):
        return "", "The locator falls outside the pinned PDF"
    return "\f".join(pages[start:finish]), ""


def candidates(passage, excerpt):
    # Extract whole layout blocks, never a window of words or a rewritten sentence.
    # A unique, pinned author-form excerpt identifies which intact block to consider.
    needle = exact(excerpt)
    if len(needle) < 15:
        return []
    return [block.strip() for block in re.split(r"\f|\n[ \t]*\n+", passage)
            if needle in exact(block)]


def evaluate_span(row, span, passage, comparison):
    check_a = bool(span) and exact(span) in exact(passage)
    bounds = list(threshold_values(comparison))
    consistency = m3(dict(comparison=comparison), span) if span else {"status": "not_applicable"}
    check_b = bool(bounds) and consistency["status"] == "passed"
    reasons = []
    if not check_a:
        reasons.append("No unique full block matched the cited source and recorded excerpt")
    if not bounds:
        reasons.append("The frozen v2 development row has no comparable value and unit")
    elif not check_b:
        reasons.append("The span does not support every frozen v2 threshold and unit")
    if span:
        quantity = set(WORDS.findall(row["quantity"].casefold()))
        action_context = span
        if len(span.split()) > 240 or not ACTION.search(action_context):
            reasons.append("The source block does not isolate an operative requirement")
        elif not quantity.intersection(WORDS.findall(span.casefold())):
            reasons.append("The source block does not identify the encoded measurement")
    return check_a, check_b, reasons


def by_id(rows):
    indexed = {r["clause_id"]: r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate development clause ID")
    return indexed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("sources", "restricted-pdf", "forms", "development", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if read_hash(args.development) != "66c8eac50c5dd2640f11e42d25c015f3c488cd6a2b86fa9e15693ea82fcb3cfd":
        raise ValueError("The verified v2 development archive differs")
    with (REVISION / "clauses/clause_inventory.csv").open(newline="") as source:
        inventory = by_id(list(csv.DictReader(source)))
    with args.forms.open(newline="") as source, tarfile.open(args.development) as archive:
        anchors = by_id(list(csv.DictReader(source)))
        with io.TextIOWrapper(archive.extractfile("report/development/development_clauses.csv")) as handle:
            encodings = by_id(list(csv.DictReader(handle)))
    if len(inventory) != 564 or set(inventory) != set(encodings) or set(inventory) != set(anchors):
        raise ValueError("Development transcript inputs must cover the frozen 564 IDs")

    documents = {}
    with (REVISION / "clauses/source_manifest.csv").open(newline="") as handle:
        for receipt in csv.DictReader(handle):
            key = receipt["source_id"]
            filename = receipt["source_file"]
            source = (args.restricted_pdf if key == "aashto_r85" else
                      args.sources / Path(SPEC[key]["file"]).with_suffix(".pdf").parent / filename)
            if read_hash(source) != receipt["sha256"]:
                raise ValueError("Pinned source PDF differs: " + key)
            data = subprocess.check_output(["pdftotext", "-layout", "-enc", "UTF-8", str(source), "-"])
            documents[key] = (receipt["sha256"], data.decode())
    transcript = []
    for key, row in inventory.items():
        source_sha, document = documents[row["source_id"]]
        passage, locator_reason = locator_text(row, document)
        blocks = candidates(passage, anchors[key]["source_excerpt"])
        span = blocks[0] if len(blocks) == 1 else ""
        encoded = encodings[key]
        comparison = json.loads(encoded["comparison"]) if encoded["comparison"] else None
        check_a, check_b, reasons = evaluate_span(row, span, passage, comparison)
        if locator_reason:
            reasons.insert(0, locator_reason)
        transcript.append(dict(clause_id=key, source_id=row["source_id"], locator=row["locator"],
                               source_sha256=source_sha, text=span, text_sha256=digest(span.encode()),
                               check_a=check_a, check_b=check_b, status="span unverified" if reasons else "verified",
                               reason="; ".join(reasons), measurement=row["quantity"], unit=row["unit"],
                               threshold_or_limit=row["threshold_or_limit"], comparison=comparison))
    with args.out.open("x") as out:
        json.dump(transcript, out, ensure_ascii=False, indent=2)
        out.write("\n")
    print(f"Transcript verified file SHA-256 {read_hash(args.out)}")
    print("Rows:", len(transcript), "Span unverified:", sum(row["status"] == "span unverified" for row in transcript))


if __name__ == "__main__":
    main()

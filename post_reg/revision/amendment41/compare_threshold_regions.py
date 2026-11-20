"""Find independently read symbolic thresholds that require clause-level review."""

import argparse
from collections import defaultdict
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

import fitz

from stitch_disputes import retry_ocr


NUMBER = r"(?:\d+(?:\.\d+)?|\.\d+)"
FORMULA = re.compile(
    rf"(?<![\w.])(?P<variable>[A-Za-z])\s*(?P<operator><=|>=|[<>=≤≥])\s*"
    rf"(?P<start>{NUMBER})(?:\s*(?:[-–−]|\bto\b)\s*(?P<end>{NUMBER}))?\s*"
    r"(?P<unit>mm|cm|km|m|%)(?!\w|\s*[/²³])"
)


def text_lines(page):
    pdf, ocr = defaultdict(list), defaultdict(list)
    projection = fitz.Matrix(page["render_matrix"])
    x, y = page["render_bounds"][:2]
    for word in page["pdf_words"]:
        region = fitz.Rect(word[:4]) * projection - fitz.Rect(x, y, x, y)
        pdf[tuple(word[5:7])].append((word[4], region))
    for row in csv.DictReader(page["ocr_tsv"].splitlines(), delimiter="\t"):
        if row["level"] != "5" or not row["text"].strip():
            continue
        left, top, width, height = (int(row[key]) for key in ("left", "top", "width", "height"))
        line = tuple(row[key] for key in ("block_num", "par_num", "line_num"))
        ocr[line].append((row["text"], fitz.Rect(left, top, left + width, top + height)))
    return pdf.values(), ocr.values()


def comparison(match):
    operator = {"≤": "<=", "≥": ">="}.get(match["operator"], match["operator"])
    return (
        (match["variable"].casefold(), operator),
        (Decimal(match["start"]), Decimal(match["end"]) if match["end"] else None, match["unit"]),
    )


def locate_formulas(lines):
    for line in lines:
        text = " ".join(word for word, _ in line)
        for match in FORMULA.finditer(text):
            region = fitz.Rect()
            position = 0
            for word, box in line:
                end = position + len(word)
                if position < match.end() and end > match.start():
                    region |= box
                position = end + 1
            key, values = comparison(match)
            yield key, values, match[0], region


def review(path, downloads):
    manifest = json.loads((path / "MANIFEST.json").read_text())
    counts, candidates, unread = [], [], []
    for source, entry in manifest.items():
        source_file = downloads / entry["source"]
        if hashlib.sha256(source_file.read_bytes()).hexdigest() != entry["source_sha256"]:
            raise ValueError(f"Original source differs: {source_file.name}")
        for filename, expected in entry["readings"].items():
            raw = (path / filename).read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError(f"Page receipt differs: {filename}")
            page = json.loads(raw)
            pdf, ocr = map(lambda lines: list(locate_formulas(lines)), text_lines(page))
            located = dict(source=source, file=entry["source"], page=page["page"], receipt_sha256=expected)
            comparisons = 0
            for variable, values, expression, box in pdf:
                matches = [(reading, text, region) for key, reading, text, region in ocr
                           if key == variable and box.intersects(region)]
                crop = []
                if not matches:
                    crop = retry_ocr(source_file, page, {"box": box}, remove_edge_rules=True)
                    for reading in crop:
                        for item in FORMULA.finditer(reading["text"]):
                            key, bounds = comparison(item)
                            if key == variable:
                                matches.append((bounds, item[0], None))
                distinct = {item[0] for item in matches}
                if len(distinct) != 1:
                    unread.append(located | dict(pdf_text=expression, same_region_ocr_matches=len(matches),
                                                 pdf_pixels=list(box), crop_ocr=crop))
                    continue
                bounds, ocr_text, ocr_region = matches[0]
                comparisons += 1
                if bounds != values:
                    candidates.append(located | dict(pdf_text=expression, ocr_text=ocr_text,
                                                     pdf_pixels=list(box), crop_ocr=crop,
                                                     ocr_pixels=list(ocr_region) if ocr_region is not None else None))
            counts.append(located | dict(layer_expressions=len(pdf), ocr_expressions=len(ocr),
                                          compared=comparisons))
    return dict(page_coverage=counts, review_candidates=candidates, without_paired_ocr=unread)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readings", type=Path, required=True)
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.readings, args.downloads)
    with args.out.open("x") as file:
        json.dump(result, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()

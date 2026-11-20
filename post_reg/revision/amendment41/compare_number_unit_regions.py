"""Locate review candidates for number-and-unit phrases in the saved readings."""

import argparse
from collections import defaultdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import sys

import fitz

from compare_threshold_regions import text_lines

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qualification_v2"))
from text_checks import TABLES


def unit_aliases():
    aliases = defaultdict(set)
    for unit, options in TABLES["units"].items():
        for option in {unit, *options}:
            aliases[option.casefold()].add(unit)
    return {key: next(iter(values)) for key, values in aliases.items() if len(values) == 1}


ALIASES = unit_aliases()
UNITS = "|".join(re.escape(name) for name in sorted(ALIASES, key=lambda name: (-len(name), name)))
NUMBER = r"(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
QUANTITY = re.compile(rf"(?<![\w.])(?P<number>[-−]?{NUMBER})\s*(?P<unit>{UNITS})(?![\w/])", re.I)


def phrases(lines):
    for words in lines:
        line = " ".join(word for word, _ in words)
        for match in QUANTITY.finditer(line):
            box = fitz.Rect()
            offset = 0
            for word, region in words:
                end = offset + len(word)
                if offset < match.end() and end > match.start():
                    box |= region
                offset = end + 1
            yield (
                (Decimal(match["number"].replace("−", "-").replace(",", "")), ALIASES[match["unit"].casefold()]),
                match[0], line, box,
            )


def review(readings):
    entries = json.loads((readings / "MANIFEST.json").read_text())
    comparisons, candidates = [], []
    for source, entry in entries.items():
        for name, digest in entry["readings"].items():
            data = (readings / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"Page receipt differs: {name}")
            page = json.loads(data)
            location = dict(source=source, file=entry["source"], page=page["page"], receipt_sha256=digest)
            primary, secondary = map(lambda lines: list(phrases(lines)), text_lines(page))
            paired = 0
            for quantity, phrase, line, box in primary:
                matches = [item for item in secondary
                           if (box & item[3]).get_area() >= 0.6 * max(box.get_area(), item[3].get_area())]
                if len(matches) != 1:
                    continue
                paired += 1
                other, ocr_phrase, ocr_line, ocr_box = matches[0]
                if quantity != other:
                    candidates.append(
                        location | dict(pdf_text=phrase, ocr_text=ocr_phrase,
                                        pdf_line=line, ocr_line=ocr_line,
                                        pdf_pixels=list(box), ocr_pixels=list(ocr_box)),
                    )
            comparisons.append(location | dict(pdf_phrases=len(primary), ocr_phrases=len(secondary), paired=paired))
    return dict(page_coverage=comparisons, review_candidates=candidates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readings", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.out.open("x") as file:
        json.dump(review(args.readings), file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()

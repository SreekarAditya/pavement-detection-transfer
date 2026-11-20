"""Extract the cited locator ranges before running the clause measures."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent


def text_at(row, text, spec):
    if spec.get("section_scoped"):
        number = re.search(r"§([\d.]+)", row["locator"])[1].rstrip(".")
        match = re.search(r"(?m)^\s*" + re.escape(number) + r"\.\s+", text)
        if match is None:
            raise ValueError("Missing section " + row["clause_id"])
        following = text[match.end():]
        next_section = re.search(r"(?m)^\s*\d+(?:\.\d+)*\.\s+", following)
        return following[:next_section.start()] if next_section else following
    match = re.search(r"\bpp?\.\s*(\d+)(?:[–-](\d+))?", row["locator"])
    if match:
        start, end = map(int, (match[1], match[2] or match[1]))
    elif row["clause_id"] == "state_program_1-profile-sampling":
        start, end = 15, 16
    else:
        raise ValueError("Missing page locator for " + row["clause_id"])
    pages = text.split("\f")
    offset = spec["page_offset"]
    first, last = start + offset - 1, end + offset
    if first < 0 or last > len(pages) or first >= last:
        raise ValueError("Invalid page locator for " + row["clause_id"])
    body = "\n".join(pages[first:last])
    # Page labels, running titles and date headers cannot establish measurability.
    lines = [line for line in body.splitlines() if not re.fullmatch(r"\s*\d+\s*", line)]
    lines = [line for line in lines if not re.search(
        r"November 10, 2020|23 CFR Ch\.|Federal Highway Administration, DOT|4-Laning of Barhi-Koderma", line)]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    for key in ("sources", "private", "inventory", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    mapping = json.loads((HERE / "source_map.json").read_text())
    manifest = []
    result = {}
    with args.inventory.open(newline="") as source:
        for row in csv.DictReader(source):
            spec = mapping[row["source_id"]]
            path = (args.private if row["source_id"] == "aashto_r85" else args.sources) / spec["file"]
            raw = path.read_bytes()
            body = text_at(row, raw.decode(), spec)
            if not body.strip():
                raise ValueError("Empty passage " + row["clause_id"])
            result[row["clause_id"]] = dict(source_id=row["source_id"],locator=row["locator"],text=body,
                                           extracted_source_sha256=hashlib.sha256(raw).hexdigest())
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

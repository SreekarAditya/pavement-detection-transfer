"""Collect independent PDF text and OCR readings without encoding clauses."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

import fitz


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.rename(path)


def collect_page(args):
    source, page_number, output = args
    os.environ["OMP_THREAD_LIMIT"] = "1"
    with fitz.open(source) as document:
        page = document[page_number - 1]
        blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES)["blocks"]
        lines = [line for block in blocks for line in block.get("lines", [])]
        directions = Counter()
        for line in lines:
            angle = math.degrees(math.atan2(*reversed(line["dir"])))
            directions[round(angle / 90) * 90] += sum(len(span["text"]) for span in line["spans"])
        dominant = directions.most_common(1)[0][0] if directions else -page.rotation
        rotation = (-dominant - page.rotation) % 360
        matrix = fitz.Matrix(300 / 72, 300 / 72).prerotate(rotation)
        pixels = page.get_pixmap(matrix=matrix, colorspace=fitz.csGRAY, alpha=False)
        ocr = subprocess.run(
            ["tesseract", "stdin", "stdout", "--dpi", "300", "-l", "eng", "--psm", "3", "tsv"],
            input=pixels.tobytes("png"), capture_output=True, check=True, timeout=180,
        )
        words = page.get_text("words")
        write_json(
            output, {
                "page": page_number,
                "page_rotation": page.rotation,
                "ocr_rotation": rotation,
                "render_bounds": list(pixels.irect),
                "render_matrix": list(page.rotation_matrix * matrix),
                "pdf_words": words,
                "ocr_tsv": ocr.stdout.decode("utf-8"),
            },
        )
    return output.name


def scope(source, row):
    with fitz.open(source) as document:
        page_count = document.page_count
    if row["group"] == "state":
        return range(1, page_count + 1)
    first, last = map(int, re.findall(r"\d+", row["schedule_locator"]))
    if not 1 <= first <= last <= page_count:
        raise ValueError(f"Schedule page range outside PDF: {source.name}")
    return range(first, last + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dispositions", type=Path, required=True)
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    sources = json.loads(args.dispositions.read_text())
    args.out.mkdir(mode=0o700, exist_ok=False)
    tasks, receipts = [], {}
    for row in sources:
        if row["disposition"] != "included":
            continue
        source = args.downloads / row["file"]
        if sha(source) != row["source_sha256"]:
            raise ValueError(f"Held-out source checksum differs: {source.name}")
        pages = scope(source, row)
        files = [args.out / f"{source.stem}-{page:04d}.json" for page in pages]
        tasks.extend((source, page, file) for page, file in zip(pages, files, strict=True))
        receipts[row["candidate"]] = {
            "source": source.name,
            "source_sha256": row["source_sha256"],
            "pages": list(pages),
            "readings": [file.name for file in files],
        }
    print(f"Collecting two readings for {len(tasks)} pages in {len(receipts)} sources", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for _ in executor.map(collect_page, tasks, chunksize=1):
            pass
    for receipt in receipts.values():
        receipt["readings"] = {name: sha(args.out / name) for name in receipt["readings"]}
    write_json(args.out / "MANIFEST.json", receipts)
    print("Page readings and checksums saved", flush=True)


if __name__ == "__main__":
    main()

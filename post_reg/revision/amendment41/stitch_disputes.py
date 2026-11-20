"""Verify both readings of the recurring seal-and-stitch condition."""

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess

import fitz


CONDITION = re.compile(r"\bseal,? and stitch if L\s*>\s*(?P<value>[a-z0-9.]+?)\s*(?P<unit>mm|cm|m)\b", re.I)


def candidates(lines):
    for words in lines.values():
        text = " ".join(word for word, _ in words)
        for match in CONDITION.finditer(text):
            box = fitz.Rect()
            offset = 0
            for word, rect in words:
                end = offset + len(word)
                if offset < match.end() and end > match.start():
                    box |= rect
                offset = end + 1
            yield dict(text=match[0], value=match["value"], unit=match["unit"], box=box)


def reads(page):
    pdf, ocr = defaultdict(list), defaultdict(list)
    matrix = fitz.Matrix(page["render_matrix"])
    x, y, _, _ = page["render_bounds"]
    shift = fitz.Point(x, y)
    for word in page["pdf_words"]:
        pdf[tuple(word[5:7])].append(
            (word[4], fitz.Rect(word[:4]) * matrix - fitz.Rect(shift, shift))
        )
    for row in csv.DictReader(page["ocr_tsv"].splitlines(), delimiter="\t"):
        if row["level"] == "5" and row["text"]:
            left, top, width, height = (int(row[key]) for key in ("left", "top", "width", "height"))
            line = tuple(row[key] for key in ("block_num", "par_num", "line_num"))
            ocr[line].append((row["text"], fitz.Rect(left, top, left + width, top + height)))
    return list(candidates(pdf)), list(candidates(ocr))


def retry_ocr(source, page, reference, *, remove_edge_rules=False):
    """Read a narrow crop when page OCR misses the condition."""
    # Crop around the full matched phrase, not the disputed character alone.
    x, y, _, _ = page["render_bounds"]
    matrix = fitz.Matrix(page["render_matrix"])
    clip = (reference["box"] + fitz.Rect(x - 4, y - 4, x + 8, y + 4)) * ~matrix
    observations = []
    with fitz.open(source) as pdf:
        sheet = pdf[page["page"] - 1]
        clip *= sheet.rotation_matrix
        for dpi in (300, 450, 600):
            render = fitz.Matrix(dpi / 72, dpi / 72).prerotate(page["ocr_rotation"])
            pixels = sheet.get_pixmap(matrix=render, clip=clip, colorspace=fitz.csGRAY)
            if remove_edge_rules:
                import numpy as np
                from PIL import Image

                image = np.frombuffer(pixels.samples, dtype=np.uint8).reshape(pixels.height, pixels.width).copy()
                columns = np.arange(pixels.width)
                at_edge = (columns < 0.08 * pixels.width) | (columns > 0.92 * pixels.width)
                vertical_rule = (image < 96).mean(axis=0) > 0.85
                image[:, at_edge & vertical_rule] = 255
                with io.BytesIO() as buffer:
                    Image.fromarray(image).save(buffer, format="PNG")
                    png = buffer.getvalue()
            else:
                png = pixels.tobytes("png")
            result = subprocess.run(
                ["tesseract", "stdin", "stdout", "--dpi", str(dpi), "-l", "eng", "--psm", "7"],
                input=png, capture_output=True, check=True, timeout=30,
            )
            observations.append(dict(dpi=dpi, text=result.stdout.decode().strip()))
    return observations


def verify_pages(root, downloads):
    manifest = json.loads((root / "MANIFEST.json").read_text())
    disputes, unmatched = [], []
    for source_id, source in manifest.items():
        for filename, digest in source["readings"].items():
            raw = (root / filename).read_bytes()
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError(f"Page receipt differs: {filename}")
            page = json.loads(raw)
            pdf, ocr = reads(page)
            values = dict(source=source_id, source_file=source["source"], page=page["page"],
                          source_sha256=source["source_sha256"], readings_sha256=digest)
            for primary in pdf:
                matches = [second for second in ocr if primary["box"].intersects(second["box"])]
                retry = []
                if not matches:
                    source_file = downloads / source["source"]
                    if hashlib.sha256(source_file.read_bytes()).hexdigest() != source["source_sha256"]:
                        raise ValueError(f"Retry source differs: {source_file.name}")
                    retry = retry_ocr(source_file, page, primary)
                    matches = [m.groupdict() | {"text": m[0]} for reading in retry
                               for m in CONDITION.finditer(reading["text"])]
                interpretations = {(match["value"], match["unit"]) for match in matches}
                if len(interpretations) != 1:
                    unmatched.append(values | dict(pdf_text=primary["text"], same_region_ocr_matches=len(matches),
                                                  crop_ocr=retry))
                    continue
                if (primary["value"], primary["unit"]) not in interpretations:
                    disputes.append(values | dict(status="threshold disputed", excluded=True,
                                                   pdf_text=primary["text"], ocr_text=matches[0]["text"],
                                                   crop_ocr=retry))
    return dict(disputes=disputes, unmatched_readings=unmatched, scope="recurring seal-and-stitch condition")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readings", type=Path, required=True)
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify_pages(args.readings, args.downloads)
    with args.out.open("x") as destination:
        json.dump(result, destination, indent=2)
        destination.write("\n")


if __name__ == "__main__":
    main()

"""Additional duplicate rule for unverified development spans only."""

import argparse
from collections import defaultdict
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qualification_v2"))
from text_checks import TABLES, threshold_values


def form(value):
    return " ".join(str(value).casefold().split())


def normalized_unit(unit):
    text = form(unit)
    matches = {canonical for canonical, variants in TABLES["units"].items()
               if text == form(canonical) or text in map(form, variants)}
    for canonical in matches:
        if text == form(canonical):
            return canonical
    return next(iter(matches)) if len(matches) == 1 else text


def bound_key(value):
    try:
        parsed = Decimal(str(value))
        if parsed.is_finite():
            return format(parsed.normalize(), "f")
    except InvalidOperation:
        pass
    return form(value)


def signature(row):
    comparison = row.get("comparison")
    if comparison:
        if comparison["type"] == "compound":
            semantics = (comparison["type"], comparison["mode"], comparison.get("k"),
                         tuple(signature(row | {"comparison": child}) for child in comparison["conditions"]))
        elif comparison["type"] == "membership":
            semantics = ("membership", comparison["mode"], tuple(sorted(map(bound_key, comparison["values"]))),
                         normalized_unit(comparison["unit"]))
        else:
            atoms = tuple((operator, bound_key(bound), normalized_unit(unit))
                          for bound, unit, operator in threshold_values(comparison))
            if not atoms:
                return None
            semantics = (comparison["type"], comparison.get("mode", ""), atoms)
    else:
        threshold = str(row.get("threshold_or_limit", "")).strip()
        match = re.fullmatch(r"(<=|>=|!=|=|<|>)\s*(.+)", threshold)
        if match:
            value = bound_key(match[2])
            semantics = (
                "time_limit" if row["unit"] in {"hour", "day", "week", "month", "year", "date", "month-day"}
                else "scalar",
                {"date": "date", "month-day": "month_day"}.get(
                    row["unit"], "elapsed" if row["unit"] in {"hour", "day", "week", "month", "year"} else ""),
                ((match[1], value, normalized_unit(row["unit"])),))
        elif threshold:
            # Preserve the full operator and threshold expression when v2 cannot encode it.
            semantics = ("literal", re.sub(r"\s+", "", threshold.casefold()))
        else:
            return None
    return form(row["measurement"]), normalized_unit(row["unit"]), semantics


def match(transcript, heldout):
    by_signature = defaultdict(list)
    for row in transcript:
        if row["status"] != "span unverified":
            continue
        key = signature(row)
        if key is not None:
            by_signature[key].append(row["clause_id"])
    duplicates = []
    for row in heldout:
        key = signature(row)
        if key is not None and key in by_signature:
            duplicates.append(dict(heldout_clause_id=row["clause_id"],
                                   development_clause_ids=sorted(by_signature[key]),
                                   excluded_by="amendment 4.1"))
    return duplicates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--transcript-sha", required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = args.transcript.read_bytes()
    if hashlib.sha256(data).hexdigest() != args.transcript_sha:
        raise ValueError("Private transcript checksum differs")
    transcript = json.loads(data)
    if len(transcript) != 564 or len({row["clause_id"] for row in transcript}) != 564:
        raise ValueError("Private transcript lacks a unique record for each development clause")
    heldout = json.loads(args.heldout.read_text())
    if len({row["clause_id"] for row in heldout}) != len(heldout):
        raise ValueError("Held-out clause IDs are repeated")
    matches = match(transcript, heldout)
    with args.out.open("x") as output:
        json.dump({"development_spans_unverified": sum(row["status"] == "span unverified" for row in transcript),
                   "heldout_clauses_removed": len(matches), "duplicates": matches}, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()

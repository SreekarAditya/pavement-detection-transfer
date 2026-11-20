"""Build v2 records and fixtures from a sealed, structured clause inventory."""

import calendar
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import json
import re

from evaluator import SECONDS, add_months

NUMBER = r"-?(?:\d+(?:\.\d+)?|\.\d+)"
OP = r"<=|>=|!=|<|>|="
SCALAR = re.compile(rf"({OP})\s*({NUMBER})")
BAND = re.compile(rf"([\[(])\s*({NUMBER})\s*,\s*({NUMBER})\s*([\])])")
ATOMIC = re.compile(rf"([a-z_]+)\s*({OP})\s*({NUMBER})")
PRESET_FIELDS = {"iri": "in/mi", "cracking": "percent", "rutting": "in",
                 "faulting": "in", "good": "count", "poor": "count",
                 "width": "px", "height": "px"}


def scalar(field, unit, op, bound):
    return dict(type="scalar", field=field, unit=unit, op=op, bound=bound)


def band(field, unit, bounds, outside=False):
    left, lower, upper, right = bounds
    if outside:
        return dict(type="range_exclusion", field=field, unit=unit, lower=lower, upper=upper,
                    include_lower=left == "[", include_upper=right == "]")
    return dict(type="compound", mode="all", conditions=[
        scalar(field, unit, ">=" if left == "[" else ">", lower),
        scalar(field, unit, "<=" if right == "]" else "<", upper)])


def compound(text):
    parts = re.fullmatch(r"at least (\d+) of (.+)", text)
    if parts:
        count, body = parts.groups()
        mode, separator = "at_least", r",\s*"
    else:
        body = text
        mode = "any" if " or " in body else "all"
        separator = " or " if mode == "any" else " and "
    atoms = re.split(separator, body)
    if len(atoms) < 2:
        return None
    conditions = []
    for atom in atoms:
        found = ATOMIC.fullmatch(atom)
        if found and found[1] in PRESET_FIELDS:
            field, comparison, bound = found.groups()
            conditions.append(scalar(field, PRESET_FIELDS[field], comparison, bound))
            continue
        field, sep, interval = atom.partition(" in ")
        match = BAND.fullmatch(interval)
        if not sep or field not in PRESET_FIELDS or not match:
            return None
        conditions.append(band(field, PRESET_FIELDS[field], match.groups()))
    result = dict(type="compound", mode=mode, conditions=conditions)
    if parts:
        result["k"] = int(count)
    return result


def encode(row):
    if row.get("comparison_json"):
        return json.loads(row["comparison_json"]), ""
    limit = row["threshold_or_limit"].strip()
    unit = row["unit"]
    quantity = row["quantity"].casefold()
    if row.get("group_guess") == "interpretation":
        return None, "requires an interpretation of the source wording"
    if row.get("missing_capability"):
        return None, row["missing_capability"]
    found = SCALAR.fullmatch(limit)
    if found and "end minus begin milepoint" in quantity:
        return dict(type="cross_record", mode="section_length", begin="begin_mi", end="end_mi",
                    unit=unit, op=found[1], bound=found[2]), ""
    if "matching iri route and endpoints" in quantity:
        return dict(type="cross_record", mode="coincident_boundaries", begin="begin_mi",
                    end="end_mi", link_field="route", linked_records="reference_iri_sections",
                    unit="mi"), ""
    if found and unit in {*SECONDS, "month", "year"} and "year minus reporting year" not in quantity:
        bound = str(Decimal(found[2]) * 12) if unit == "year" else found[2]
        return dict(type="time_limit", mode="elapsed", start="event_at", end="action_at",
                    unit="month" if unit == "year" else unit, op=found[1], bound=bound,
                    source_bound=found[2], source_unit=unit), ""
    if found:
        return scalar("value", unit, *found.groups()), ""
    if unit in {"date", "month-day"}:
        fmt = r"\d{4}-\d{2}-\d{2}" if unit == "date" else r"\d{2}-\d{2}"
        match = re.fullmatch(rf"({OP})\s*({fmt})", limit)
        if match:
            return dict(type="time_limit", mode="date" if unit == "date" else "month_day",
                        field="value", op=match[1], bound=match[2]), ""
    outside = limit.startswith("outside ")
    bounds = BAND.fullmatch(limit.removeprefix("outside "))
    if bounds:
        return band("value", unit, bounds.groups(), outside), ""
    members = re.fullmatch(r"(in|outside) \{(.+)\}", limit)
    if members:
        return dict(type="membership", mode=members[1], field="value", unit=unit,
                    values=[int(item) if re.fullmatch(r"-?\d+", item.strip()) else item.strip()
                            for item in members[2].split(",")]), ""
    multiple = compound(limit)
    if multiple:
        return multiple, ""
    if " of reference" in limit or "half slab width" in limit:
        return None, "comparison to a derived reference-dependent threshold"
    return None, "comparison semantics cannot be determined from the inventory"


def fields(rule):
    if rule is None:
        return ["value"]
    if rule["type"] == "compound":
        return list(dict.fromkeys(field for item in rule["conditions"] for field in fields(item)))
    if rule["type"] == "time_limit" and rule["mode"] == "elapsed":
        return [rule["start"], rule["end"]]
    if rule["type"] == "cross_record":
        base = [rule["begin"], rule["end"]]
        return base if rule["mode"] == "section_length" else base + [rule["link_field"], rule["linked_records"]]
    return [rule["field"]]


def boundary(bound):
    point = Decimal(bound)
    step = Decimal(1).scaleb(min(-2, point.as_tuple().exponent - 1))
    return [str(point - step), str(point), str(point + step)]


def cases(rule):
    if rule is None:
        return [{"value": "0"}]
    kind = rule["type"]
    if kind == "scalar":
        values = boundary(rule["bound"])
        if rule["unit"] in {"count", "code"}:
            bound = int(rule["bound"])
            values = [bound - 1, bound, bound + 1]
        return [{rule["field"]: value, "units": {rule["field"]: rule["unit"]}} for value in values]
    if kind == "range_exclusion":
        return [{rule["field"]: value} for limit in (rule["lower"], rule["upper"])
                for value in boundary(limit)]
    if kind == "membership":
        probe = 0 if all(isinstance(value, int) for value in rule["values"]) else "unlisted"
        while probe in rule["values"]:
            probe = probe + 1 if isinstance(probe, int) else probe + "_"
        return [{rule["field"]: value} for value in [*rule["values"], probe]]
    if kind == "time_limit":
        mode = rule["mode"]
        if mode in {"date", "month_day"}:
            prefix = "" if mode == "date" else "2000-"
            target = date.fromisoformat(prefix + rule["bound"])
            return [{rule["field"]: (target + timedelta(days=offset)).isoformat()[len(prefix):]}
                    for offset in (-1, 0, 1)]
        start = datetime(2028, 1, 31, tzinfo=timezone.utc)
        span = Decimal(rule["bound"])
        end = (add_months(start, int(span)) if rule["unit"] == "month"
               else start + timedelta(seconds=float(span * SECONDS[rule["unit"]])))
        return [{rule["start"]: start.isoformat(),
                 rule["end"]: (end + timedelta(seconds=offset)).isoformat()}
                for offset in (-1, 0, 1) if end + timedelta(seconds=offset) >= start]
    if kind == "cross_record":
        if rule["mode"] == "section_length":
            return [{rule["begin"]: "0", rule["end"]: value} for value in boundary(rule["bound"])
                    if Decimal(value) > 0]
        first = {rule["begin"]: "0", rule["end"]: "1", rule["link_field"]: "example"}
        return [first | {rule["linked_records"]: [first]},
                first | {rule["linked_records"]: [first | {rule["end"]: "2"}]}]
    if kind == "compound":
        alternatives = [cases(child) for child in rule["conditions"]]
        base = {}
        for options in alternatives:
            base = combine(base, options[1 if len(options) > 1 else 0])
        return [combine(base, item) for options in alternatives for item in options]
    raise ValueError(f"Unknown fixture type {kind!r}")


def combine(first, second):
    result = first | second
    if "units" in first or "units" in second:
        result["units"] = first.get("units", {}) | second.get("units", {})
    return result


def make_requirement(row, rule, missing):
    spatial = rule is not None and rule.get("mode") == "coincident_boundaries"
    cross_length = rule is not None and rule.get("mode") == "section_length"
    return dict(id=row["clause_id"], source=row["source_id"], locator=row["locator"],
                measurement=row["quantity"], unit=row["unit"], required_evidence=fields(rule),
                comparison=rule, missing_capability=missing,
                when_true="length_violation" if cross_length else "no_violation" if spatial else row["action"],
                when_false="coincidence_violation" if spatial else "no_violation" if cross_length else "condition_not_met")


def build(row, rule, missing):
    req = make_requirement(row, rule, missing)
    variants = [(f"boundary-{index:02}", {"unit": row["unit"]} | evidence)
                for index, evidence in enumerate(cases(rule), 1)]
    variants.append(("box-only", dict(unit="px", image_box=[24, 28, 118, 103])))
    variants.append(("units", variants[0][1] | {"unit": "incompatible-test-unit"}))
    for suffix, evidence in variants:
        identifier = row["clause_id"] + "--" + suffix
        yield dict(fixture_id=identifier, clause_id=row["clause_id"], kind=suffix.split("-")[0]
                   if suffix.startswith("boundary-") else suffix,
                   record=dict(schema_version="qualification/2.0", requirement=req, evidence=evidence))

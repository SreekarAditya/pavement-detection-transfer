"""Evaluate evidence against comparison types fixed by revision amendment 4."""

import calendar
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import operator

RELATIONS = {"<": operator.lt, "<=": operator.le, ">": operator.gt,
             ">=": operator.ge, "=": operator.eq, "!=": operator.ne}
SECONDS = {"hour": 3600, "day": 86400, "week": 604800}
TYPES = {"scalar", "time_limit", "membership", "compound", "range_exclusion", "cross_record"}


class MissingEvidence(ValueError):
    pass


class UnitMismatch(ValueError):
    pass


def number(value):
    if isinstance(value, bool):
        raise ValueError("Expected a numeric measurement")
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("Expected a finite measurement")
    return result


def required(record, field):
    if field not in record or record[field] is None:
        raise MissingEvidence(field)
    return record[field]


def measurement(record, field, unit):
    value = required(record, field)
    if isinstance(value, dict):
        measured = required(value, "value")
        observed_unit = required(value, "unit")
    else:
        measured = value
        units = record.get("units", {})
        observed_unit = units.get(field, record.get("unit"))
    if observed_unit != unit:
        raise UnitMismatch(f"{field}: {observed_unit!r}, expected {unit!r}")
    return measured


def instant(value):
    stamp = datetime.fromisoformat(value)
    if stamp.utcoffset() is None:
        raise ValueError("A time zone is required for elapsed-time evidence")
    return stamp.astimezone(timezone.utc)


def add_months(start, months):
    year, month = divmod(start.year * 12 + start.month - 1 + months, 12)
    return start.replace(year=year, month=month + 1,
                         day=min(start.day, calendar.monthrange(year, month + 1)[1]))


def time_limit(rule, evidence):
    mode = rule["mode"]
    if mode in {"date", "month_day"}:
        unit = "date" if mode == "date" else "month-day"
        value = measurement(evidence, rule["field"], unit)
        prefix = "" if mode == "date" else "2000-"
        return RELATIONS[rule["op"]](date.fromisoformat(prefix + value),
                                     date.fromisoformat(prefix + rule["bound"]))
    if mode != "elapsed":
        raise ValueError("Unknown time comparison mode")
    start = instant(required(evidence, rule["start"]))
    end = instant(required(evidence, rule["end"]))
    if end < start:
        raise ValueError("Action precedes event")
    limit = number(rule["bound"])
    if limit < 0:
        raise ValueError("Duration is negative")
    unit = rule["unit"]
    if unit == "month":
        if limit != limit.to_integral_value():
            raise ValueError("Calendar month limits must be integral")
        # Calendar arithmetic follows the event's local calendar.
        local_start = datetime.fromisoformat(evidence[rule["start"]])
        deadline = add_months(local_start, int(limit)).astimezone(timezone.utc)
        return RELATIONS[rule["op"]](end, deadline)
    delta = end - start
    elapsed = Decimal(delta.days * 86400 + delta.seconds) + Decimal(delta.microseconds) / 1000000
    return RELATIONS[rule["op"]](elapsed, limit * SECONDS[unit])


def section(record, rule):
    lo = number(required(record, rule["begin"]))
    hi = number(required(record, rule["end"]))
    if lo >= hi:
        raise ValueError("Section begin must precede its end")
    return lo, hi


def cross_record(rule, evidence):
    current = section(evidence, rule)
    if rule["mode"] == "section_length":
        if evidence.get("unit") != rule["unit"]:
            raise UnitMismatch("Section boundary units differ")
        return RELATIONS[rule["op"]](current[1] - current[0], number(rule["bound"]))
    if rule["mode"] != "coincident_boundaries":
        raise ValueError("Unknown cross-record comparison mode")
    linked = required(evidence, rule["linked_records"])
    if not isinstance(linked, list) or not linked:
        raise MissingEvidence(rule["linked_records"])
    key = rule["link_field"]
    identifier = required(evidence, key)
    matches = [section(row, rule) for row in linked if required(row, key) == identifier]
    return current in matches


def predicate(rule, evidence):
    kind = rule["type"]
    if kind == "scalar":
        observed = number(measurement(evidence, rule["field"], rule["unit"]))
        return RELATIONS[rule["op"]](observed, number(rule["bound"]))
    if kind == "time_limit":
        return time_limit(rule, evidence)
    if kind == "membership":
        observed = measurement(evidence, rule["field"], rule["unit"])
        values = rule["values"]
        if not values:
            raise ValueError("A membership set cannot be empty")
        present = any(type(observed) is type(value) and observed == value for value in values)
        if rule["mode"] not in {"in", "outside"}:
            raise ValueError("Unknown membership mode")
        return present if rule["mode"] == "in" else not present
    if kind == "compound":
        conditions = rule["conditions"]
        if not conditions:
            raise ValueError("A compound comparison requires conditions")
        # Require evidence for every declared condition.
        values = [predicate(child, evidence) for child in conditions]
        mode = rule["mode"]
        if mode == "all":
            return all(values)
        if mode == "any":
            return any(values)
        count = rule.get("k")
        if mode != "at_least" or type(count) is not int or not 1 <= count <= len(values):
            raise ValueError("Invalid compound comparison count")
        return sum(values) >= count
    if kind == "range_exclusion":
        observed = number(measurement(evidence, rule["field"], rule["unit"]))
        lo, hi = number(rule["lower"]), number(rule["upper"])
        if lo > hi:
            raise ValueError("Range bounds are reversed")
        above = observed >= lo if rule["include_lower"] else observed > lo
        below = observed <= hi if rule["include_upper"] else observed < hi
        return not (above and below)
    if kind == "cross_record":
        return cross_record(rule, evidence)
    raise ValueError(f"Unsupported comparison type {kind!r}")


def evaluate(record):
    requirement = record["requirement"]
    evidence = record["evidence"]
    try:
        for field in requirement["required_evidence"]:
            required(evidence, field)
        if requirement["unit"] != evidence.get("unit") and "unit" in evidence:
            raise UnitMismatch("Record evidence units differ")
        rule = requirement.get("comparison")
        if rule is None:
            return {"action": "unrepresentable", "reason": requirement["missing_capability"]}
        present = predicate(rule, evidence)
        key = "when_true" if present else "when_false"
        return {"action": requirement[key], "predicate": present}
    except MissingEvidence as error:
        return {"action": "insufficient_measurement", "reason": str(error)}
    except UnitMismatch as error:
        return {"action": "unit_error", "reason": str(error)}
    except (InvalidOperation, ValueError, TypeError) as error:
        return {"action": "invalid_measurement", "reason": str(error)}

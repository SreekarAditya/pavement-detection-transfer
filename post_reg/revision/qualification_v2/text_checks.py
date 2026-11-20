"""Mechanical numeric and unit matching, with committed normalization tables."""
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
import unicodedata

TABLES = json.loads(Path(__file__).with_name("normalization.json").read_text())


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"(?<=[a-z])-\s*\n\s*(?=[a-z])", "", text)
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    # Join hyphenated number words while keeping signed numerical tokens.
    text = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", text)
    words = TABLES["number_words"]
    scales = TABLES["scales"]
    vocabulary = "|".join(sorted((*words, *scales), key=len, reverse=True))
    pattern = r"\b(?:" + vocabulary + r")(?:\s+(?:and\s+)?(?:" + vocabulary + r"))*\b"

    def number(match):
        total = current = 0
        for token in match[0].split():
            if token in words:
                current += words[token]
            elif token in scales:
                if scales[token] == 100:
                    current = max(current, 1) * 100
                else:
                    total += max(current, 1) * scales[token]
                    current = 0
        return str(total + current)
    text = re.sub(pattern, number, text)
    for month, index in TABLES["months"].items():
        text = re.sub(r"\b" + month + r"\s+(\d{1,2})(?:st|nd|rd|th)?\b",
                      lambda match: f"{index}-{match[1]}", text)
    return text


def numbers(text):
    result = set()
    value = normalize(text)
    value = re.sub(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}-\d{1,2}\b",
                   lambda match: match[0].replace("-", " "), value)
    for match in re.finditer(r"(?<![\w.])([+-]?(?:\d+(?:\.\d+)?|\.\d+))(?![\w])", value):
        try:
            result.add(Decimal(match[1]))
        except InvalidOperation:
            pass
    for match in re.finditer(r"±\s*(\d+(?:\.\d+)?)", value):
        result.update((Decimal(match[1]), -Decimal(match[1])))
    return result


def measurable(text):
    return bool(numbers(text))


def has_unit(text, unit):
    value = normalize(text)
    for candidate in TABLES["units"].get(unit, [unit]):
        candidate = normalize(candidate)
        if re.search(r"(?<![a-z])" + re.escape(candidate) + r"(?![a-z])", value):
            return True
    if unit not in TABLES["units"]:
        parts = re.split(r"/| per | over ", unit)
        if len(parts) > 1:
            return all(has_unit(text, part.strip()) for part in parts)
    return False


def encoded_values(requirement):
    if "criterion" in requirement:
        return numbers(str(requirement["criterion"]))
    return {Decimal(str(requirement[key])) for key in ("lower", "upper", "floor")
            if key in requirement}


def consistency(requirement, source):
    wanted = encoded_values(requirement)
    absent = sorted(wanted - numbers(source))
    units = [value.strip() for value in requirement["unit"].split(",")]
    missing_units = [unit for unit in units if not has_unit(source, unit)]
    return dict(passed=bool(wanted) and not absent and not missing_units,
                missing_values=[str(value) for value in absent], missing_units=missing_units)


def physical(quantity, unit):
    return unit in TABLES["physical_units"] or any(
        term in quantity.casefold() for term in TABLES["physical_quantity_terms"])


def threshold_values(rule):
    """Yield comparable bounds with their units and encoded directions."""
    if rule is None:
        return
    kind = rule["type"]
    if kind == "compound":
        for condition in rule["conditions"]:
            yield from threshold_values(condition)
    elif kind == "range_exclusion":
        yield rule["lower"], rule["unit"], "<" if rule["include_lower"] else "<="
        yield rule["upper"], rule["unit"], ">" if rule["include_upper"] else ">="
    elif kind == "membership":
        for value in rule["values"]:
            yield str(value), rule["unit"], None
    elif kind != "cross_record" or rule["mode"] == "section_length":
        unit = rule.get("unit") or ("date" if rule["mode"] == "date" else "month-day")
        yield rule.get("source_bound", rule["bound"]), rule.get("source_unit", unit), rule["op"]


def dates(text, annual=False):
    raw = unicodedata.normalize("NFKC", text).casefold()
    months = TABLES["months"]
    result = set()
    pattern = "|".join(sorted(months, key=len, reverse=True))
    forms = (
        (rf"\b(?P<month>{pattern})\.?\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(?P<year>\d{{4}}))?\b", True),
        (rf"\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<month>{pattern})\.?(?:\s+(?P<year>\d{{4}}))?\b", True),
        (r"(?<![\d-])(?:(?P<year>\d{4})-)?(?P<month>\d{2})-(?P<day>\d{2})(?![\d-])", False),
    )
    from datetime import date
    for expression, named in forms:
        for match in re.finditer(expression, raw):
            day, month, year = match['day'], match['month'], match['year']
            month = months[month] if named else month
            if annual or year:
                try:
                    iso = date(int(year or 2000), int(month), int(day)).isoformat()
                    result.add(iso[5:] if annual else iso)
                except ValueError:
                    continue
    return result


def source_numbers(text):
    """Read exact decimal and simple fractional values without treating list commas as digits."""
    observed = numbers(text)
    for match in re.finditer(r"(?<![\w/])(\d+)\s*/\s*(\d+)(?![\w/])", text):
        numerator, denominator = map(Decimal, match.groups())
        if denominator and denominator % 1 == 0:
            value = numerator / denominator
            if value * denominator == numerator:
                observed.add(value)
    return observed


def m3(requirement, source):
    observations = source_numbers(source)
    components = list(threshold_values(requirement["comparison"]))
    exceptions = []
    for raw, unit, _ in components:
        bound = str(raw)
        if unit in {"date", "month-day"}:
            present = bound in dates(source, annual=unit == "month-day")
            unit_present = present
        elif unit == "code" and not re.fullmatch(r"-?\d+", bound):
            present = bool(re.search(r"(?<!\w)" + re.escape(bound.casefold()) + r"(?!\w)", source.casefold()))
            unit_present = has_unit(source, unit)
        else:
            present = Decimal(bound) in observations
            unit_present = has_unit(source, unit)
        if not present or not unit_present:
            exceptions.append(dict(bound=bound, unit=unit, missing_value=not present, missing_unit=not unit_present))
    if not components:
        status = "not_applicable"
    else:
        status = "failed" if exceptions else "passed"
    return dict(status=status, exceptions=exceptions, components=len(components))


LEXICON = json.loads(Path(__file__).with_name("direction_lexicon.json").read_text())
PHRASES = {phrase: relation for relation, phrases in LEXICON.items() for phrase in phrases}
PHRASE = "|".join(map(re.escape, sorted(PHRASES, key=len, reverse=True)))
N = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
PREFIX = re.compile(rf"(?<![\w])(?P<phrase>{PHRASE})\s*(?:(?:of|the|a)\s+)*(?P<number>{N})(?![\d.])")
SUFFIX = re.compile(rf"(?<![\w.])(?P<number>{N})(?![\d.])\s*(?:%|[a-z]+(?:\s+[a-z]+)?)?\s+(?P<phrase>or more)\b")


def direction_observations(source):
    normalized = normalize(source)
    output = {}
    for pattern in (PREFIX, SUFFIX):
        for match in pattern.finditer(normalized):
            if re.search(r"\b(?:no|not)\s+$", normalized[max(0, match.start() - 9):match.start()]):
                continue
            output.setdefault(Decimal(match["number"]), set()).add(PHRASES[match["phrase"]])
    return output


def m3b(components, source):
    observed = direction_observations(source)
    output = []
    for bound, unit, expected in components:
        try:
            directions = observed.get(Decimal(str(bound)), set())
        except InvalidOperation:
            directions = set()
        if not expected or len(directions) != 1:
            status = "direction not determinable"
        else:
            status = "passed" if expected in directions else "direction failure"
        output.append(dict(bound=str(bound), unit=unit, expected=expected, source_directions=sorted(directions),
                           status=status))
    if not output:
        output.append(dict(bound="", unit="", expected=None, source_directions=[],
                           status="direction not determinable"))
    statuses = {row["status"] for row in output}
    aggregate = ("direction failure" if "direction failure" in statuses else
                 "direction not determinable" if "direction not determinable" in statuses else "passed")
    return dict(status=aggregate, comparisons=output)


def original_directions(requirement):
    criterion = str(requirement.get("criterion", ""))
    unit = requirement.get("unit", "")
    numeric = r"-?(?:\d+(?:\.\d+)?|\.\d+)"
    pieces = [(m[2], unit, m[1]) for m in re.finditer(rf"(<=|>=|!=|<|>|=)\s*({numeric})(?![\d.-])", criterion)]
    for left, lo, hi, right in re.findall(rf"([\[(])({numeric}),({numeric})([\])])", criterion):
        if criterion.startswith("outside"):
            pieces.extend([(lo, unit, "<" if left == "[" else "<="),
                           (hi, unit, ">" if right == "]" else ">=")])
        else:
            pieces.extend([(lo, unit, ">=" if left == "[" else ">"),
                           (hi, unit, "<=" if right == "]" else "<")])
    if not pieces:
        for key, direction in (("lower", ">="), ("upper", "<="), ("floor", ">=")):
            if key in requirement:
                pieces.append((requirement[key], unit, direction))
    return pieces

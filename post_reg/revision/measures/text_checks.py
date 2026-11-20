"""Mechanical numeric and unit matching, with committed normalization tables."""
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
import unicodedata

TABLES = json.loads(Path(__file__).with_name("normalization.json").read_text())


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
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

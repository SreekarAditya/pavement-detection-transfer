"""Derive a private author review from clause forms and supplied observations."""
import argparse
import csv
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
import json
import operator
from pathlib import Path
import re

LOCATOR_FIELDS = ("clause_id", "source_id", "locator", "source_excerpt", "excerpt_locator")
FORM_FIELDS = (
    "measured_quantity", "unit", "evidence_field", "comparison", "threshold",
    "lower_included", "upper_included", "denominator_or_aggregation", "required_evidence",
    "action_when_true", "action_when_false", "time_limit", "document_scope",
    "answer_basis", "interpretation_needed", "interpretation_reason",
)
REQUIRED_FORM = (
    "measured_quantity", "unit", "comparison", "denominator_or_aggregation",
    "required_evidence", "action_when_true", "action_when_false", "document_scope", "answer_basis",
)
OPERATORS = {
    "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge,
    "=": operator.eq, "!=": operator.ne,
}
COMPARISONS = (*OPERATORS, "between", "outside", "in", "not_in", "all", "any", "at_least", "deadline")
WORKED_EXAMPLE = re.compile(r"hif18022-(?:iri-length|rut-length|rut-coincidence)--route-aaa-\d+\Z")


class FormError(ValueError):
    """An author field does not determine a comparison."""


@dataclass(frozen=True)
class Outcome:
    status: str
    answer: str = ""
    detail: str = ""


def display(value):
    return str(value).strip() if value is not None else ""


def field_value(evidence, field):
    result = evidence
    for part in field.split("."):
        if not isinstance(result, dict) or part not in result:
            return None
        result = result[part]
    return result


def present(value):
    return value is not None and value != "" and value != [] and value != {}


def read_flag(value, name):
    token = display(value).casefold()
    if token not in {"yes", "no"}:
        raise FormError(f"{name} must be yes or no")
    return token == "yes"


def read_threshold(value):
    if not isinstance(value, str):
        return value
    value = value.strip()
    try:
        return json.loads(value, parse_float=Decimal)
    except json.JSONDecodeError:
        return value


def comparable(value):
    if isinstance(value, bool):
        return "boolean", value
    if isinstance(value, (int, float, Decimal)):
        try:
            amount = Decimal(str(value))
        except InvalidOperation as error:
            raise FormError("invalid number") from error
        if not amount.is_finite():
            raise FormError("non-finite number")
        return "number", amount
    if not isinstance(value, str):
        raise FormError("comparison operands must be scalar")
    try:
        number = Decimal(value.strip())
    except InvalidOperation:
        number = None
    if number is not None:
        if not number.is_finite():
            raise FormError("non-finite number")
        return "number", number
    if re.fullmatch(r"(?:\d{4}-)?\d{2}-\d{2}", value):
        try:
            if len(value) == 5:
                return "month-day", date.fromisoformat(f"2000-{value}")
            return "date", date.fromisoformat(value)
        except ValueError as error:
            raise FormError("invalid calendar value") from error
    return "text", value


def compare(value, comparison, target):
    kind, actual = comparable(value)
    expected_kind, expected = comparable(target)
    if kind != expected_kind:
        raise FormError("comparison operands have different value types")
    if comparison in {"<", "<=", ">", ">="} and kind in {"text", "boolean"}:
        raise FormError("ordered comparisons require numbers or dates")
    return OPERATORS[comparison](actual, expected)


def check_rule(rule, evidence):
    comparison = rule.get("comparison")
    target = rule.get("threshold")
    if comparison in {"all", "any", "at_least"}:
        clauses = target.get("conditions") if comparison == "at_least" and isinstance(target, dict) else target
        if not isinstance(clauses, list) or not clauses or any(not isinstance(item, dict) for item in clauses):
            raise FormError("composite threshold needs a nonempty JSON array of conditions")
        outcomes = [check_rule(item, evidence) for item in clauses]
        if comparison == "all":
            return all(outcomes)
        if comparison == "any":
            return any(outcomes)
        minimum = target.get("minimum") if isinstance(target, dict) else None
        if type(minimum) is not int or not 1 <= minimum <= len(outcomes):
            raise FormError("at_least needs a valid integer minimum")
        return sum(outcomes) >= minimum

    field = display(rule.get("evidence_field"))
    value = field_value(evidence, field)
    if not field or not present(value):
        raise FormError("comparison needs an observed evidence_field")
    if comparison in OPERATORS:
        return compare(value, comparison, target)
    if comparison in {"between", "outside"}:
        if not isinstance(target, list) or len(target) != 2:
            raise FormError("interval threshold needs a JSON pair")
        lower, upper = target
        if compare(lower, ">", upper):
            raise FormError("interval bounds are reversed")
        use_lower = read_flag(rule.get("lower_included"), "lower_included")
        use_upper = read_flag(rule.get("upper_included"), "upper_included")
        inside = compare(value, ">=" if use_lower else ">", lower) and compare(value, "<=" if use_upper else "<", upper)
        return inside if comparison == "between" else not inside
    if comparison in {"in", "not_in"}:
        if not isinstance(target, list) or not target:
            raise FormError("membership threshold needs a nonempty JSON array")
        matches = [compare(value, "=", item) for item in target]
        return any(matches) if comparison == "in" else not any(matches)
    raise FormError("unknown comparison")


def derive(form, evidence):
    """Return an author-specified action, a refusal, or a request for direct entry."""
    if not any(display(form.get(field)) for field in FORM_FIELDS):
        return Outcome("awaiting_form")
    try:
        interpretation = read_flag(form.get("interpretation_needed"), "interpretation_needed")
        if interpretation:
            if not display(form.get("interpretation_reason")) or not display(form.get("answer_basis")):
                raise FormError("an interpretation case needs its reason and basis")
            return Outcome("interpretation_needed", detail=display(form.get("interpretation_reason")))

        missing_form = [field for field in REQUIRED_FORM if not display(form.get(field))]
        if missing_form:
            raise FormError("complete the form fields " + ", ".join(missing_form))
        comparison = display(form["comparison"])
        if comparison not in COMPARISONS:
            raise FormError("unknown comparison")
        unit = display(form["unit"])
        time_limit = display(form.get("time_limit"))
        rule = dict(form)
        rule["comparison"] = comparison
        rule["threshold"] = read_threshold(display(form.get("threshold")))
        if comparison == "deadline":
            parts = time_limit.split(maxsplit=1)
            if len(parts) != 2 or parts[1] != unit or comparable(parts[0])[0] != "number":
                raise FormError("deadline needs a numeric time_limit followed by the form unit")
            inclusive = read_flag(form.get("upper_included"), "upper_included")
            rule.update(comparison="<=" if inclusive else "<", threshold=parts[0])
        elif not display(form.get("threshold")):
            raise FormError("the chosen comparison needs a threshold")
        fields = [field.strip() for field in form["required_evidence"].split(",") if field.strip()]
        if not fields:
            raise FormError("required_evidence must list one or more fields")
        missing = [field for field in fields if not present(field_value(evidence, field))]
        if missing:
            return Outcome("derived", "cannot assess", "Missing evidence " + ", ".join(missing))
        seen_unit = display(evidence.get("unit"))
        if not seen_unit:
            return Outcome("derived", "cannot assess", "Measurement unit is missing")
        if seen_unit != unit:
            return Outcome("derived", "unit mismatch", f"Expected {unit}, received {seen_unit}")

        result = check_rule(rule, evidence)
        answer = display(form["action_when_true" if result else "action_when_false"])
        if "{time_limit}" in answer and not time_limit:
            raise FormError("the action mentions a time limit that is missing")
        return Outcome("derived", answer.replace("{time_limit}", time_limit))
    except FormError as error:
        return Outcome("form_insufficient", "form insufficient", str(error))


def load_rows(path, key):
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    identifiers = [row.get(key) for row in rows]
    if not rows or not all(identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{path.name} must have distinct, nonempty {key} values")
    return rows


def write_review(forms_path, fixtures_path, output):
    forms = {row["clause_id"]: row for row in load_rows(forms_path, "clause_id")}
    fixtures = load_rows(fixtures_path, "fixture_id")
    if set(forms) != {fixture["clause_id"] for fixture in fixtures}:
        raise ValueError("forms and fixtures cover different clauses")
    positions = {clause_id: position for position, clause_id in enumerate(forms)}
    fixtures.sort(key=lambda row: positions[row["clause_id"]])
    columns = (
        *LOCATOR_FIELDS, *FORM_FIELDS, "fixture_id", "question", "evidence_available",
        "fixture_observations", "provenance", "derivation_status", "derived_answer",
        "derivation_detail", "author_confirmed", "override_or_direct_answer",
    )
    review = []
    for fixture in fixtures:
        form = forms[fixture["clause_id"]]
        evidence = json.loads(fixture["input_record_json"])["evidence"]
        result = derive(form, evidence)
        source = "worked example" if WORKED_EXAMPLE.fullmatch(fixture["fixture_id"]) else "rule text"
        review.append({
            **{field: form.get(field, "") for field in (*LOCATOR_FIELDS, *FORM_FIELDS)},
            **{field: fixture[field] for field in ("fixture_id", "question", "evidence_available")},
            "fixture_observations": json.dumps(evidence, ensure_ascii=False, separators=(",", ":")),
            "provenance": source, "derivation_status": result.status,
            "derived_answer": result.answer, "derivation_detail": result.detail,
            "author_confirmed": "", "override_or_direct_answer": "",
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    # A refreshed sheet has a new location so it cannot overwrite author edits.
    with output.open("x", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(review)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for parameter in ("forms", "fixtures", "out"):
        parser.add_argument("--" + parameter, type=Path, required=True)
    args = parser.parse_args()
    try:
        write_review(args.forms, args.fixtures, args.out)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()

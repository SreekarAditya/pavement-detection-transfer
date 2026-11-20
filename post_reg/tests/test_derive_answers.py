"""Test the amendment's author-form conventions with invented examples."""
import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "revision"))
from derive_answers import FORM_FIELDS, LOCATOR_FIELDS, derive, write_review


FORM = {
    "measured_quantity": "Example physical length",
    "unit": "m",
    "evidence_field": "length",
    "comparison": "<=",
    "threshold": "2",
    "denominator_or_aggregation": "One observation",
    "required_evidence": "length",
    "action_when_true": "is within limit",
    "action_when_false": "requires follow-up",
    "document_scope": "Example program",
    "answer_basis": "Example clause 1",
    "interpretation_needed": "no",
}


class DeriveAnswersTests(unittest.TestCase):
    def decide(self, value, **fields):
        return derive(FORM | fields, {"length": value, "unit": "m"})

    def test_every_scalar_comparison(self):
        for comp, accepted, declined in (
            ("<", 1, 2), ("<=", 2, 3), (">", 3, 2), (">=", 2, 1), ("=", 2, 3), ("!=", 3, 2),
        ):
            with self.subTest(comparison=comp):
                self.assertEqual(self.decide(accepted, comparison=comp).answer, "is within limit")
                self.assertEqual(self.decide(declined, comparison=comp).answer, "requires follow-up")

    def test_included_and_excluded_interval_boundaries(self):
        base = {"comparison": "between", "threshold": "[1,2]", "lower_included": "yes", "upper_included": "no"}
        self.assertEqual(self.decide(1, **base).answer, "is within limit")
        self.assertEqual(self.decide(2, **base).answer, "requires follow-up")
        self.assertEqual(self.decide(2, **(base | {"comparison": "outside"})).answer, "is within limit")
        self.assertEqual(self.decide(2, **(base | {"upper_included": "yes"})).answer, "is within limit")

    def test_membership(self):
        self.assertEqual(self.decide(2, comparison="in", threshold="[1,2]").answer, "is within limit")
        self.assertEqual(self.decide(3, comparison="in", threshold="[1,2]").answer, "requires follow-up")
        self.assertEqual(self.decide(3, comparison="not_in", threshold="[1,2]").answer, "is within limit")
        self.assertEqual(self.decide(1, comparison="not_in", threshold="[1,2]").answer, "requires follow-up")

    def test_composite_comparisons(self):
        conditions = [
            {"evidence_field": "length", "comparison": "<=", "threshold": 2},
            {"evidence_field": "observations", "comparison": ">=", "threshold": 3},
        ]
        evidence = {"unit": "m", "length": 2, "observations": 2}
        form = FORM | {"required_evidence": "length,observations", "threshold": json.dumps(conditions)}
        self.assertEqual(derive(form | {"comparison": "all"}, evidence).answer, "requires follow-up")
        self.assertEqual(derive(form | {"comparison": "any"}, evidence).answer, "is within limit")
        threshold = json.dumps({"minimum": 1, "conditions": conditions})
        self.assertEqual(derive(form | {"comparison": "at_least", "threshold": threshold}, evidence).answer,
                         "is within limit")

    def test_unit_mismatch(self):
        self.assertEqual(derive(FORM, {"length": 2, "unit": "px"}).answer, "unit mismatch")
        self.assertEqual(derive(FORM, {"length": 2}).answer, "cannot assess")

    def test_box_only_and_missing_required_evidence(self):
        evidence = {"image_box": [2, 4, 6, 8], "unit": "px"}
        self.assertEqual(derive(FORM, evidence).answer, "cannot assess")
        self.assertEqual(derive(FORM, {"length": None, "unit": "m"}).answer, "cannot assess")
        self.assertEqual(derive(FORM, {"length": 0, "unit": "m"}).answer, "is within limit")

    def test_time_limit_and_action(self):
        form = FORM | {
            "unit": "hour", "measured_quantity": "time elapsed", "comparison": "deadline",
            "evidence_field": "elapsed", "required_evidence": "elapsed", "threshold": "",
            "time_limit": "4 hour", "upper_included": "yes",
            "action_when_true": "Complete within {time_limit}", "action_when_false": "Refer overdue incident",
        }
        self.assertEqual(derive(form, {"elapsed": 4, "unit": "hour"}).answer, "Complete within 4 hour")
        self.assertEqual(derive(form | {"upper_included": "no"}, {"elapsed": 4, "unit": "hour"}).answer,
                         "Refer overdue incident")
        self.assertEqual(self.decide(1, action_when_true="Inspect within {time_limit}", time_limit="4 hours").answer,
                         "Inspect within 4 hours")

    def test_calendars(self):
        form = FORM | {"unit": "month-day", "threshold": "06-01", "comparison": "<"}
        self.assertEqual(derive(form, {"length": "05-31", "unit": "month-day"}).answer, "is within limit")
        self.assertEqual(derive(form, {"length": "06-01", "unit": "month-day"}).answer, "requires follow-up")

    def test_form_insufficient(self):
        for fields in (
            {"action_when_true": ""}, {"comparison": "unencoded rule"}, {"threshold": ""},
            {"comparison": "between", "threshold": "[2,1]", "lower_included": "yes", "upper_included": "yes"},
            {"evidence_field": "unknown"}, {"comparison": "<", "threshold": "unspecified"},
        ):
            with self.subTest(fields=fields):
                outcome = self.decide(1, **fields)
                self.assertEqual((outcome.status, outcome.answer), ("form_insufficient", "form insufficient"))
        self.assertEqual(derive({}, {}).status, "awaiting_form")

    def test_author_interpretation(self):
        form = FORM | {"interpretation_needed": "yes", "interpretation_reason": "Definition unavailable"}
        outcome = derive(form, {"length": 2, "unit": "m"})
        self.assertEqual((outcome.status, outcome.answer), ("interpretation_needed", ""))
        self.assertEqual(derive(form | {"interpretation_reason": ""}, {}).status, "form_insufficient")

    def test_review_keeps_form_and_observations_beside_the_answer(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            forms, fixtures, review = (root / name for name in ("forms.csv", "fixtures.csv", "review.csv"))
            with forms.open("w", newline="") as file:
                writer = csv.DictWriter(file, (*LOCATOR_FIELDS, *FORM_FIELDS))
                writer.writeheader()
                writer.writerow(FORM | {"clause_id": "one", "locator": "Example p. 2"})
            fixture = {
                "clause_id": "one", "fixture_id": "one--test", "question": "What action applies?",
                "evidence_available": "One measured length",
                "input_record_json": json.dumps({"evidence": {"length": 2, "unit": "m"}}),
            }
            with fixtures.open("w", newline="") as file:
                writer = csv.DictWriter(file, tuple(fixture))
                writer.writeheader()
                writer.writerow(fixture)
            write_review(forms, fixtures, review)
            with review.open(newline="") as file:
                row = next(csv.DictReader(file))
            self.assertEqual(row["answer_basis"], FORM["answer_basis"])
            self.assertEqual(row["derived_answer"], "is within limit")
            self.assertEqual(row["override_or_direct_answer"], "")
            self.assertEqual(row["provenance"], "rule text")
            with self.assertRaises(FileExistsError):
                write_review(forms, fixtures, review)


if __name__ == "__main__":
    unittest.main()

from datetime import datetime
import unittest

from duplicates import groups, normalized, template_flag
from evaluator import add_months, evaluate, predicate
from fixtures import encode, build
from text_checks import m3, m3b, threshold_values


def scalar(op, bound="3", unit="m"):
    return dict(type="scalar", field="value", unit=unit, op=op, bound=bound)


class Operators(unittest.TestCase):
    def test_all_scalar_operators_at_boundary(self):
        for op, expected in (("<", False), ("<=", True), (">", False), (">=", True),
                             ("=", True), ("!=", False)):
            with self.subTest(op=op):
                self.assertIs(predicate(scalar(op), {"value": 3, "unit": "m"}), expected)

    def test_time_units_and_calendar(self):
        rule = dict(type="time_limit", mode="elapsed", start="start", end="end", op="<=")
        evidence = dict(start="2028-02-01T00:00:00Z", end="2028-02-08T00:00:00Z")
        for unit, bound in (("week", 1), ("day", 7), ("hour", 168)):
            self.assertTrue(predicate(rule | dict(unit=unit, bound=bound), evidence))
        self.assertFalse(predicate(rule | dict(unit="hour", bound=167), evidence))
        value = datetime.fromisoformat("2028-01-31T00:00:00+05:30")
        self.assertEqual(add_months(value, 1).isoformat(), "2028-02-29T00:00:00+05:30")
        self.assertTrue(predicate(rule | dict(unit="month", bound=1),
                                  dict(start=value.isoformat(), end="2028-02-29T00:00:00+05:30")))
        self.assertFalse(predicate(rule | dict(unit="month", bound=1),
                                   dict(start=value.isoformat(), end="2028-03-01T00:00:00+05:30")))
        for mode, unit, bound in (("date", "date", "2028-06-15"), ("month_day", "month-day", "06-15")):
            self.assertTrue(predicate(dict(type="time_limit", mode=mode, field="value", op="<=", bound=bound),
                                      dict(unit=unit, value=bound)))

    def test_membership_modes_and_scope(self):
        rule = dict(type="membership", field="value", unit="code", values=[2, 5])
        self.assertTrue(predicate(rule | {"mode": "in"}, {"value": 2, "unit": "code"}))
        self.assertTrue(predicate(rule | {"mode": "outside"}, {"value": 3, "unit": "code"}))
        self.assertFalse(predicate(rule | {"mode": "in"}, {"value": "2", "unit": "code"}))

    def test_compound_modes(self):
        rule = dict(type="compound", conditions=[scalar(">"), scalar("<"), scalar("=")])
        self.assertFalse(predicate(rule | {"mode": "all"}, {"value": 3, "unit": "m"}))
        self.assertTrue(predicate(rule | {"mode": "any"}, {"value": 3, "unit": "m"}))
        self.assertFalse(predicate(rule | {"mode": "at_least", "k": 2}, {"value": 3, "unit": "m"}))

    def test_range_exclusion_boundary(self):
        rule = dict(type="range_exclusion", field="value", unit="m", lower="3", upper="4",
                    include_lower=True, include_upper=False)
        self.assertFalse(predicate(rule, {"value": 3, "unit": "m"}))
        self.assertTrue(predicate(rule, {"value": 4, "unit": "m"}))
        self.assertTrue(predicate(rule, {"value": 2, "unit": "m"}))

    def test_cross_record_lengths_and_matching_endpoints(self):
        rule = dict(type="cross_record", begin="begin", end="end", unit="mi")
        a = dict(begin="0.5", end="0.6", unit="mi", route="R")
        self.assertTrue(predicate(rule | dict(mode="section_length", op="=", bound="0.10"), a))
        matched = rule | dict(mode="coincident_boundaries", linked_records="other", link_field="route")
        self.assertTrue(predicate(matched, a | dict(other=[a])))
        self.assertFalse(predicate(matched, a | dict(other=[a | dict(route="S")])))
        self.assertFalse(predicate(matched, a | dict(other=[a | dict(end="0.61")])))

    def test_fixtures_refuse_missing_or_wrong_units(self):
        item = dict(clause_id="sample", source_id="test", locator="p. 1", action="Accept",
                    quantity="pothole width", unit="m", threshold_or_limit=">=3")
        rows = {row["kind"]: row for row in build(item, *encode(item))}
        self.assertEqual(evaluate(rows["box-only"]["record"])["action"], "insufficient_measurement")
        self.assertEqual(evaluate(rows["units"]["record"])["action"], "unit_error")


class SourceChecks(unittest.TestCase):
    def test_direction_compares_strictness_and_binds_number(self):
        terms = list(threshold_values(scalar(">=")))
        self.assertEqual(m3b(terms, "At least 3 m and below 10 m")["status"], "passed")
        self.assertEqual(m3b(terms, "More than 3 m")["status"], "direction failure")
        self.assertEqual(m3b(terms, "3 m or more")["status"], "passed")
        self.assertEqual(m3b(terms, "No more than 3 m")["status"], "direction not determinable")
        self.assertEqual(m3b(terms, "At least 3 m or less than 3 m")["status"], "direction not determinable")
        self.assertEqual(m3b([("3", "m", "<=")], "Not more than 3 m")["status"], "passed")

    def test_source_values_units_and_dates(self):
        compound = dict(type="compound", mode="all", conditions=[scalar(">=", "95", "percent"),
                                                                scalar("<=", "170", "percent")])
        self.assertEqual(m3({"comparison": compound}, "95 to 170 percent")["status"], "passed")
        self.assertEqual(m3({"comparison": scalar(">=", "0.125", "in")}, "1/8 inch")["status"], "passed")
        date = dict(type="time_limit", mode="month_day", field="value", op="<", bound="06-01")
        self.assertEqual(m3({"comparison": date}, "before [1st June] every year")["status"], "passed")
        self.assertEqual(m3({"comparison": date}, "before 1st July every year")["status"], "failed")

    def test_duplicates_preserve_threshold_and_flag_template(self):
        a = "(iii) EXAMPLE: Fill potholes within 24 hours."
        b = "6.4.1 fill potholes within 24 h."
        self.assertEqual(normalized(a.replace("EXAMPLE:", "EXAMPLE"), ["Example"]), normalized(b))
        self.assertNotEqual(normalized(b), normalized(b.replace("24", "48")))
        self.assertTrue(template_flag("FHWA-HIF-18-032"))
        self.assertTrue(template_flag("adapted from the FHWA sample quality management plan"))
        self.assertFalse(template_flag("FHWA pavement guide"))
        records = groups([dict(clause_id="a", source_id="a", clause_text=b)],
                         [dict(clause_id="b", source_id="b", clause_text=b)])
        self.assertEqual(len(records), 1)
        self.assertEqual(len(next(iter(records.values()))), 2)


if __name__ == "__main__":
    unittest.main()

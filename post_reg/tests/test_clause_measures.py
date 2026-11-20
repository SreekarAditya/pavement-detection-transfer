"""Synthetic tests for source checks and clause measure accounting."""
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "revision/measures"))
import build_passages as B
import run_measures as R
import text_checks as T


class ClauseMeasureTests(unittest.TestCase):
    def test_m0_digit_word_time_and_no_value(self):
        for text in ("threshold 12 mm", "twenty four hours", "ninety-five percent", "one half of the width"):
            self.assertTrue(T.measurable(text))
        self.assertFalse(T.measurable("Inspect the pavement for distress"))

    def test_normalized_numbers(self):
        from decimal import Decimal
        self.assertIn(Decimal(24), T.numbers("twenty-four hours"))
        self.assertIn(Decimal(250), T.numbers("two hundred and fifty"))
        self.assertEqual(T.numbers("±0.08 inches"), {Decimal(".08"), Decimal("-.08")})
        self.assertIn(Decimal(".1"), T.numbers("Limit .1 in."))
        self.assertTrue(T.numbers("06-15") <= T.numbers("June 15"))

    def test_m3_equivalent_units_and_failure_details(self):
        request = dict(unit="mm", criterion=">=25")
        self.assertTrue(T.consistency(request, "twenty five millimeters")["passed"])
        missing = T.consistency(request, "24 pixels")
        self.assertFalse(missing["passed"])
        self.assertEqual(missing["missing_values"], ["25"])
        self.assertEqual(missing["missing_units"], ["mm"])
        self.assertFalse(T.consistency(request, "0.025 m")["passed"])

    def test_source_locator_discards_page_labels(self):
        row = dict(clause_id="toy", locator="p. 1")
        self.assertEqual(B.text_at(row, "Inspect the pavement\n1\n", dict(page_offset=0)), "Inspect the pavement")
        section = "6.4.1. Twelve percent is required\n6.4.2. Next criterion"
        text = B.text_at(dict(clause_id="toy",locator="§6.4.1"), section, dict(section_scoped=True))
        self.assertIn("Twelve percent",text)
        self.assertNotIn("Next criterion",text)

    def test_m1_capability_categories(self):
        self.assertEqual(R.classify(dict(operator="maximum"), True)[0], "expressible_unchanged")
        self.assertEqual(R.classify(dict(operator="extra",extension_operator="extra"), True)[0], "requires_new_operator")
        self.assertEqual(R.classify(dict(operator="extra"), False)[0], "not_representable")

    def test_box_only_and_unit_mismatch_keep_raw_result(self):
        fixture = dict(fixture_id="toy--box-only",clause_id="toy",
                       input_record_json=json.dumps({"evidence":dict(image_box=[1,2,3,4])}))
        result = R.measure_fixture(fixture, lambda record: {"action":"unrepresentable"}, True, True)
        self.assertEqual(result["action"], "unrepresentable")
        self.assertEqual(result["raw_action"], "unrepresentable")

    def test_m0_to_m5_denominators_and_exceptions(self):
        inventory = [dict(clause_id="a",source_id="s"),dict(clause_id="b",source_id="s")]
        coverage = [dict(classification="expressible_unchanged"),dict(classification="requires_new_operator")]
        coverage = [r|dict(clause_id=c) for r,c in zip(coverage,("a","b"))]
        outcomes = [
            dict(clause_id="a",measurable=True,physical_quantity=True,fixture_kind="box_only",action="cannot assess"),
            dict(clause_id="b",measurable=False,physical_quantity=True,fixture_kind="box_only",action="unrepresentable"),
            dict(clause_id="a",measurable=True,physical_quantity=True,fixture_kind="unit_mismatch",action="unit mismatch"),
            dict(clause_id="b",measurable=False,physical_quantity=True,fixture_kind="unit_mismatch",action="unrepresentable"),
        ]
        consistency=[dict(clause_id=c,encoded=True,passed=True) for c in ("a","b")]
        published=[dict(source_id="worked",agreement=True),dict(source_id="development",agreement=False)]
        result=R.summarize(inventory,dict(a=True,b=False),coverage,outcomes,consistency,published)
        self.assertEqual(next(r for r in result if r["measure"]=="M2")["N"],1)
        self.assertEqual(next(r for r in result if r["measure"]=="M3")["N"],1)
        self.assertEqual(next(r for r in result if r["measure"]=="M4")["N"],2)
        self.assertEqual(next(r for r in result if r.get("category")=="requires_new_operator" and r["source_id"]=="all")["N"],1)

    def test_shacl_checks_only_selected_shape_with_nested_evidence(self):
        try:
            import pyshacl
        except ImportError:
            self.skipTest("SHACL library is installed in the dedicated validation environment")
        with TemporaryDirectory() as directory:
            shape=Path(directory)/"toy.ttl"
            shape.write_text(
                '@prefix sh: <http://www.w3.org/ns/shacl#> . '
                '@prefix ex: <urn:qualification:revision:> . '
                'ex:toy-shape a sh:NodeShape ; sh:targetClass ex:toy ; '
                'sh:property [ sh:path ex:reference ; sh:minCount 1 ; '
                'sh:node [ sh:property [ sh:path ex:length ; sh:minCount 1 ; sh:minInclusive 2 ] ] ] .')
            task=("item","toy",dict(reference=dict(length=3)),str(shape),"toy")
            self.assertTrue(R.validate_shape(task)["shacl_conforms"])
            self.assertFalse(R.validate_shape(("missing","toy",{},str(shape),"toy"))["shacl_conforms"])

    def test_seal_mismatch_stops_before_evaluator_imports(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            passage=root/"passages.json"
            passage.write_text("{}")
            seals=dict(files={},passages_sha256=hashlib.sha256(b"different").hexdigest())
            seals_path=root/"input_seals.json"
            seals_path.write_text(json.dumps(seals))
            with patch.object(R,"HERE",root):
                with self.assertRaisesRegex(ValueError,"source passages differ"):
                    R.verify_inputs(root,passage)


if __name__ == "__main__":
    unittest.main()

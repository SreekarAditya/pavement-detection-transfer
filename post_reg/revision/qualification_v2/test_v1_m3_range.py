import unittest

from m3_v1_range_audit import corrected_check, original


class V1RangeRegression(unittest.TestCase):
    def test_range_matches_and_wrong_value_fails(self):
        requirement = {"criterion": "[95,170]", "unit": "in/mi"}
        self.assertFalse(original.consistency(requirement, "95 to 170 in/mi")["passed"])
        self.assertTrue(corrected_check(requirement, "95 to 170 in/mi")["passed"])
        result = corrected_check(requirement, "95 to 171 in/mi")
        self.assertFalse(result["passed"])
        self.assertEqual(result["missing_values"], ["170"])
        self.assertEqual(requirement["criterion"], "[95,170]")

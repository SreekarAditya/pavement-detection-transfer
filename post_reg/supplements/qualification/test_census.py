"""Small deterministic checks of the census comparison and inherited tie rule."""
import unittest

import numpy as np

import audit_policy as A
import census
import pr_common as PC


class CensusTests(unittest.TestCase):
    def test_floor_and_missing_references(self):
        self.assertEqual(census.side(.30,.30),"pass")
        self.assertEqual(census.side(.29999,.30),"below")
        self.assertEqual(census.side(census.ratio(0,0),.25),"unscorable")

    def test_decision_change(self):
        for triple, answer in ((("below","pass","pass"),"corrected"),
                               (("pass","below","pass"),"introduced"),
                               (("below","below","pass"),"disagreement_persisted"),
                               (("pass","pass","pass"),"agreement_unchanged"),
                               (("pass","unscorable","pass"),"unscorable")):
            self.assertEqual(census.transition(*triple),answer)

    def test_ties_and_band_boundaries(self):
        confidences = np.array([.8,.7,.7,.5])
        threshold = PC.threshold_for_rate(confidences,100,2)
        self.assertEqual(threshold,.7)
        self.assertEqual(int((confidences>=threshold).sum()),3)
        self.assertEqual(census.band(90,100,.1),"within")
        self.assertEqual(census.band(110,100,.1),"within")

    def test_shortfall(self):
        view = A.look(np.array([0,1]),np.array([40,80]),np.array([1,1]),np.array([3,4]),200)
        self.assertEqual((view["images"],view["blocks"],view["boxes"]),(120,2,7))
        self.assertEqual(A.three_way(view,95,.25),("unresolved","shortfall"))

    def test_signed_decompositions(self):
        np.testing.assert_allclose(census.decomposition(.35,.25,.4),[.1,-.15])
        np.testing.assert_allclose(census.workload_decomposition(15,18,13,16),[-1,3,-5])


if __name__ == "__main__":
    unittest.main()

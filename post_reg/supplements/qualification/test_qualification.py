import unittest
import numpy as np

from audit_policy import look, ordered_blocks, policies, three_way
from conformance import fhwa, specifications, verdict
from external import clip_box, parent
from transfer import cases, extended


class QualificationTests(unittest.TestCase):
    def test_fhwa_development(self):
        rec, key = next(fhwa())
        result = verdict(rec)
        self.assertAlmostEqual(result["difference"], 2.29, places=2)
        self.assertAlmostEqual(result["p_governing"], 0.086, places=3)
        self.assertEqual(result["action"], key["decision"])

    def test_original_specifications_and_extension(self):
        for rec, want in specifications():
            self.assertEqual(verdict(rec)["action"], want)
            self.assertEqual(extended(rec), verdict(rec))
        for rec, want, hours in cases():
            result = extended(rec)
            self.assertEqual((result["action"], result.get("allowance_hours")), (want, hours))

    def test_equal_boundary_and_unresolved(self):
        interval = dict(valid=True, lo95=.25, hi95=.4)
        self.assertEqual(three_way(interval,95,.25)[0], "accept")
        interval["lo95"]=.20
        self.assertEqual(three_way(interval,95,.25)[0], "unresolved")
        interval["hi95"]=.25
        self.assertEqual(three_way(interval,95,.25)[0], "unresolved")
        interval["hi95"]=.24
        self.assertEqual(three_way(interval,95,.25)[0], "adapt")

    def test_common_prefix_and_zero_refs(self):
        order=ordered_blocks(4,[2026,302,0,0,0])
        images=np.full(4,50)
        refs=np.zeros(4,dtype=int)
        m=np.zeros(4,dtype=int)
        result, first, full=policies(order,images,m,refs,.25)
        self.assertEqual((first["blocks"],full["blocks"]),(2,4))
        self.assertEqual(result["selective_95"]["images"],200)
        self.assertTrue(result["selective_95"]["second_audit"])
        self.assertEqual(result["selective_95"]["action"],"unresolved")
        self.assertEqual(result["point_100"]["action"],"adapt")
        self.assertEqual(look(order,images,m,refs,100)["evidence_reason"], first["evidence_reason"])
        result, first, full=policies(np.array([0]),np.array([220]),np.array([1]),np.array([4]),.25)
        self.assertTrue(result["selective_95"]["no_distinct_second"])
        self.assertFalse(result["selective_95"]["second_audit"])
        self.assertEqual(result["selective_95"]["images"],220)

    def test_parent_and_box_geometry(self):
        self.assertEqual(parent("DJI_20230707094536_0001_640_00_05.jpg"),("dji_20230707094536_0001",0,5))
        self.assertEqual(parent("1x00001_640_08_12.jpg"),("1x00001",8,12))
        self.assertIsNone(parent("unpaired.jpg"))
        np.testing.assert_allclose(clip_box(.0,.5,.25,.25,640),[0,240,80,400])


if __name__=="__main__":
    unittest.main()

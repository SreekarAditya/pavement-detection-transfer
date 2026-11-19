"""Synthetic checks for revision-specific behavior."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"revision"))
import pr_rev as R


class RevisionTests(unittest.TestCase):
    def test_split_is_deterministic_disjoint_and_whole(self):
        root=SimpleNamespace(pool=lambda _:list("abcdef"),blocks=lambda _:dict(zip("abcdef","aabbcc")))
        with TemporaryDirectory() as temp:
            base=Path(temp)
            for out in (base/"one",base/"two"):
                R.make_lists(root,["test"],2026,out)
            a,b=R.PC.read_split(base/"one","test")
            self.assertEqual((a,b),R.PC.read_split(base/"two","test"))
            self.assertFalse(set(a)&set(b))
            self.assertFalse({root.blocks(None)[f] for f in a}&{root.blocks(None)[f] for f in b})

    def test_registered_seed_calls_frozen_chain_with_original_settings(self):
        params=dict(seed=2026,sample_draws=2000,selected="detector")
        a=SimpleNamespace(root="lock",constants="approved",e5="scores",samples="samples",pred5="predictions")
        with TemporaryDirectory() as temp, patch.object(R.E1,"chain") as frozen, patch.object(R.pd,"read_parquet"):
            R.chain(a,R.frozen_seed(params,2026),Path(temp),True)
            settings,supplied=frozen.call_args.args
            self.assertEqual(supplied,params)
            self.assertEqual(settings.e5,"scores")
            self.assertFalse(settings.sensitivity)

    def test_r2_highest_and_no_threshold(self):
        conf=np.array([.9,.8,.7,.6])
        delta=np.array([1,1,1,1])
        groups=np.array([0,1,0,1])
        refs=np.array([2,2])
        self.assertEqual(R.r2_candidates(conf,delta,groups,refs,.5,1.645),.8)
        self.assertIsNone(R.r2_candidates(conf,delta,groups,refs,1.01,1.645))

    def test_coverage_includes_ordinary_and_finite_correction(self):
        result=R.coverage_for([1,3],[4,4],4,.5,.82,[.90,.95])
        self.assertEqual(result["fpc_variance_factor"],.5)
        self.assertEqual(result["cover90_ordinary_census"],1)
        self.assertEqual(result["cover90_fpc_heldout"],0)
        self.assertEqual(result["cover95_fpc_heldout"],1)

    def test_split_level_mcse_and_classification(self):
        rows=[]
        cases={"met":[.95,.96,.97],"not met":[.81,.82,.83],"not resolved":[.89,.90,.91]}
        for coll,values in cases.items():
            for seed,value in enumerate(values,1):
                rows.append(dict(collection=coll,class_set="pothole",M=100,post_hoc_network=False,
                                 metric="recall_within",value=value,numerator=100*value,denominator=100,
                                 target=.90,seed=seed))
        conf=dict(monte_carlo=dict(interval=.95),r1=dict(split_seeds=dict(first=1,last=3)))
        got=R.summarize_metrics(pd.DataFrame(rows),conf).set_index("collection")
        for expected,values in cases.items():
            self.assertAlmostEqual(got.loc[expected,"pooled_value"],np.mean(values))
            self.assertAlmostEqual(got.loc[expected,"mcse"],np.std(values,ddof=1)/np.sqrt(3))
            self.assertEqual(got.loc[expected,"status"],expected)

    def test_seed_override_never_writes_the_rules(self):
        with TemporaryDirectory() as temp:
            path=Path(temp)/"params.yaml"
            original=b"seed: 2026\n"
            path.write_bytes(original)
            params=R.yaml.safe_load(path.read_text())
            revised=R.frozen_seed(params,3001)
            self.assertEqual(revised["seed"],3001)
            self.assertEqual(params["seed"],2026)
            self.assertEqual(path.read_bytes(),original)

    def test_clause_rejects_mismatched_answers_before_evaluation(self):
        with TemporaryDirectory() as temp, patch.object(R,"HERE",Path(temp)):
            clause=Path(temp)/"clauses"
            clause.mkdir()
            answer=Path(temp)/"answers.csv"
            answer.write_text("frozen answer fixture\n")
            digest=R.PC.sha(answer)
            (clause/"ANSWERS_SHA256.txt").write_text(digest+"\n")
            answer.write_text("changed answer fixture\n")
            with patch.dict(os.environ,ANSWERS_SHA=digest), self.assertRaisesRegex(SystemExit,"SHA-256 differs"):
                R.clause_command(SimpleNamespace(answers=answer))


if __name__=="__main__":
    unittest.main()

"""Run the locked pdt_score.py on a synthetic root (verification skipped: synthetic files are not in the manifest)."""
import sys, types
from pathlib import Path
root, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, str(Path(root) / "scripts"))
import pdt_score as S
S.verify = lambda r, m: print("verify skipped (synthetic root)")
a = types.SimpleNamespace(root=root, manifest_sha="x", stage="s1", selection="", out=out)
S.main(a)
a.stage = "all"; a.selection = str(Path(out) / "selection_s1.json")
S.main(a)

"""Input failures must be listed together before the revision gate."""
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

path = Path(__file__).resolve().parents[1] / "revision/preflight_inputs.py"
spec = importlib.util.spec_from_file_location("revision_inputs", path)
inputs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inputs)


class InputPreflightTests(unittest.TestCase):
    def test_lists_each_dangling_link_and_missing_file(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            t07, t08 = root / "07", root / "08"
            t07.mkdir()
            t08.mkdir()
            (t07 / "broken-one").symlink_to(root / "missing-one")
            (t08 / "broken-two").symlink_to(root / "missing-two")
            errors = inputs.inspect_inputs(root / "lock", root / "rules", t07, t08,
                                           root / "ext18", root / "pred18")
            self.assertTrue(any("Dangling symlink:" in error and "broken-one" in error for error in errors))
            self.assertTrue(any("Dangling symlink:" in error and "broken-two" in error for error in errors))
            self.assertTrue(any("e5_scores.csv" in error for error in errors))
            self.assertTrue(any("blocks.csv" in error for error in errors))
            self.assertTrue(any("labels_final_ecupdd.parquet" in error for error in errors))

    def test_valid_symlinks_are_accepted(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            t07, t08 = root / "07", root / "08"
            t07.mkdir()
            t08.mkdir()
            target = root / "prediction.parquet"
            target.write_bytes(b"fixture")
            (t07 / "valid-link").symlink_to(target)
            errors = inputs.inspect_inputs(root / "lock", root / "rules", t07, t08,
                                           root / "ext18", root / "pred18")
            self.assertFalse(any("Dangling symlink:" in error for error in errors))


if __name__ == "__main__":
    unittest.main()

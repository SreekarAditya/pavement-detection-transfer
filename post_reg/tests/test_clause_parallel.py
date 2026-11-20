"""Scheduling tests using synthetic data only."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1]/'run/clause_parallel.py'
spec = importlib.util.spec_from_file_location('clause_parallel', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ParallelTests(unittest.TestCase):
    def test_alternating_partition_is_disjoint_and_complete(self):
        for size in (0, 1, 2, 7, 2586):
            left, right = module.partition(list(range(size)))
            self.assertFalse({i for i, _ in left} & {i for i, _ in right})
            self.assertEqual(module.merge(size, left, right), list(range(size)))

    def test_original_order_restored(self):
        self.assertEqual(module.merge(4, [[2, 'c'], [0, 'a']], [[3, 'd'], [1, 'b']]),
                         ['a', 'b', 'c', 'd'])

    def test_missing_or_duplicate_work_stops(self):
        for left, right in [([[0, 'a']], []), ([[0, 'a']], [[0, 'b']]),
                            ([[0, 'a']], [[2, 'c']])]:
            with self.assertRaises(ValueError):
                module.merge(2, left, right)


if __name__ == '__main__':
    unittest.main()

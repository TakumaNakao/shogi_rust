import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_halfkp_s11_listwise_1m import compare_positions


def positions():
    return [{"id": i, "sfen": str(i), "successful": True,
             "depth": 1, "search_failed": False} for i in range(1000)]


class ComparisonTests(unittest.TestCase):
    def test_small_inconclusive_difference_is_recorded_not_vetoed(self):
        a, b = positions(), positions()
        for row in b[:3]:
            row.update(successful=False, depth=0)
        result = compare_positions(a, b)
        self.assertTrue(result["within_budget_tolerance"])
        self.assertEqual(result["lost_successes"], 3)
        self.assertEqual(result["common_conclusive"], 997)

    def test_more_than_ten_losses_requires_review(self):
        a, b = positions(), positions()
        for row in b[:11]:
            row["successful"] = False
        self.assertFalse(compare_positions(a, b)["within_budget_tolerance"])

    def test_mismatched_positions_and_failures_rejected(self):
        a, b = positions(), positions()
        b[0]["sfen"] = "different"
        with self.assertRaises(RuntimeError):
            compare_positions(a, b)
        b = positions()
        b[0]["search_failed"] = True
        with self.assertRaises(RuntimeError):
            compare_positions(a, b)
        with self.assertRaises(RuntimeError):
            compare_positions(a, b[:-1])


if __name__ == "__main__":
    unittest.main()

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_halfkp_s11_confirmation import select_openings


class OpeningTests(unittest.TestCase):
    def test_unused_unique_and_same_order(self):
        lines = ["", "used", "used"] + [f"position-{i}" for i in range(30)]
        result = select_openings(lines + lines, {"used", "position-0"})
        self.assertEqual(result, [f"position-{i}" for i in range(1, 26)])

    def test_insufficient_starts_refused(self):
        with self.assertRaises(RuntimeError):
            select_openings(["duplicate"] * 30, set())


if __name__ == "__main__":
    unittest.main()

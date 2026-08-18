import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_mmto_root_audit", ROOT / "scripts" / "analyze_mmto_root_audit.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(root_key, game_key, loss, regret, pairs=2):
    return {
        "root_key": root_key,
        "game_key": game_key,
        "mean_pair_loss": loss,
        "regret_cp": regret,
        "pairs": pairs,
        "agreement": True,
        "best_move": "7g7f",
        "teacher_move": "7g7f",
    }


class MmtoAuditAnalyzerTest(unittest.TestCase):
    def test_game_equal_bootstrap_uses_all_roots_within_each_game(self):
        parent = {
            "r1": row("r1", "g1", 1.0, 10.0),
            "r2": row("r2", "g1", 3.0, 12.0),
            "r3": row("r3", "g2", 2.0, 20.0),
        }
        candidate = {
            "r1": row("r1", "g1", 0.0, 9.0),
            "r2": row("r2", "g1", 2.0, 11.0),
            "r3": row("r3", "g2", 1.0, 18.0),
        }
        result = MODULE.compare(parent, candidate, seed=1, bootstrap_resamples=100)
        self.assertEqual(result["roots"], 3)
        self.assertEqual(result["games"], 2)
        self.assertAlmostEqual(result["loss_delta_mean_game_equal"], -1.0)
        self.assertAlmostEqual(result["regret_delta_mean_cp_game_equal"], -1.5)

    def test_root_set_mismatch_is_rejected(self):
        parent = {"r1": row("r1", "g1", 1.0, 1.0)}
        candidate = {"r2": row("r2", "g1", 1.0, 1.0)}
        with self.assertRaises(SystemExit):
            MODULE.compare(parent, candidate, seed=1, bootstrap_resamples=10)

if __name__ == "__main__":
    unittest.main()

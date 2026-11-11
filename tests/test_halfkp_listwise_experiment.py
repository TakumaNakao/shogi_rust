from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("listwise_experiment", ROOT / "scripts/run_halfkp_listwise_experiment.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def games(directory, opening, reason="Resign"):
    destination = directory / "games"
    destination.mkdir(parents=True, exist_ok=True)
    for index, color in enumerate(("black", "white")):
        result = "NewWin" if index == 0 else "BaselineWin"
        (destination / f"game_{index:03d}.usi").write_text(
            f"result {result}\nreason {reason}\nnew_as {color}\nstart_sfen {opening}\nposition startpos moves 7g7f\n")


class ExperimentTests(unittest.TestCase):
    def test_hkcl_preflight_rejects_old_headers_before_any_stage_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.hkcl"
            path.write_bytes(struct.pack("<8sIIIQI", b"HKCL0002", 2, 64, 105480, 2000, 0))
            experiment = runner.Experiment(Path(temporary) / "run")
            with mock.patch.object(runner, "EXACT", path), mock.patch.object(experiment, "command") as command:
                with self.assertRaisesRegex(RuntimeError, "incompatible HKCL header"):
                    experiment.prepare()
                command.assert_not_called()

    def test_exact_preflight_requires_current_attack_and_defense_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "suite.hkcl"
            metadata = {"lineage_id": "halfkp64-scratch-v1", "generator_config": {
                "rule_version": 3, "targets": {"mate3": 500, "defense_pairs": 500}}}
            def header():
                payload = json.dumps(metadata).encode()
                path.write_bytes(struct.pack("<8sIIIQI", b"HKCL0004", 4, 64, 105480, 1000, len(payload)) + payload)
            header()
            self.assertEqual(runner.validate_hkcl_header(path, exact=True)["records"], 1000)
            metadata["generator_config"]["targets"]["defense_pairs"] = 0
            header()
            with self.assertRaisesRegex(RuntimeError, "attacking and defending proofs"):
                runner.validate_hkcl_header(path, exact=True)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(RuntimeError, "truncated HKCL metadata"):
                runner.validate_hkcl_header(path)

    def test_completed_pair_resumes_without_replaying_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment = runner.Experiment(Path(temporary))
            for name in ("usi_engine", "usi_benchmark"):
                (experiment.tools / name).write_bytes(b"frozen tool")
            candidate = experiment.base / "candidate.bin"
            baseline = experiment.base / "baseline.bin"
            candidate.write_bytes(b"candidate")
            baseline.write_bytes(b"baseline")
            runner.write(experiment.base / "openings.json", {"sets": {"test": ["startpos"]}})

            def play(name, command):
                directory = Path(command[command.index("--record-dir") + 1]).parent
                games(directory, "startpos")

            with mock.patch.object(experiment, "command", side_effect=play) as command:
                first = experiment.match("test", candidate, baseline, "test")
                second = experiment.match("test", candidate, baseline, "test")
                self.assertEqual(first, second)
                self.assertEqual(command.call_count, 1)
                self.assertEqual(first["score"], 0.5)
                record = experiment.base / "matches/test/pair_0000/games/game_000.usi"
                record.write_text(record.read_text() + "corruption\n")
                with self.assertRaisesRegex(RuntimeError, "completed USI pair changed"):
                    experiment.match("test", candidate, baseline, "test")

    def test_pair_rejects_unsafe_termination_and_wrong_opening(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            games(directory, "startpos", "IllegalMove")
            with self.assertRaisesRegex(RuntimeError, "unsafe game termination"):
                runner.parse_pair(directory, "startpos")
            games(directory, "startpos")
            with self.assertRaisesRegex(RuntimeError, "opening differs"):
                runner.parse_pair(directory, "different position")

    def test_frozen_snapshot_is_never_replaced_by_later_outer(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            run = base / "run"
            (run / "metrics").mkdir(parents=True)
            (run / "model.bin").write_bytes(b"outer2")
            (run / "optimizer.bin").write_bytes(b"optimizer2")
            runner.write(run / "checkpoint.json", {"completed_outer": 2})
            runner.write(run / "metrics/outer_0002.json", {"outer": 2,
                "model_sha256": runner.sha(run / "model.bin"), "optimizer_sha256": runner.sha(run / "optimizer.bin")})
            frozen = base / "outer2"
            runner.snapshot(run, frozen, 2)
            (run / "model.bin").write_bytes(b"outer8")
            runner.write(run / "checkpoint.json", {"completed_outer": 8})
            runner.snapshot(run, frozen, 2)
            self.assertEqual((frozen / "model.bin").read_bytes(), b"outer2")
            with self.assertRaisesRegex(RuntimeError, "earlier unsaved snapshot"):
                runner.snapshot(run, base / "missing-outer4", 4)

    def test_s09_saved_snapshot_survives_script_reentry(self):
        source = (ROOT / "scripts/run_halfkp_s09_2m_unattended.sh").read_text()
        function = "save_pass() {" + source.split("save_pass() {", 1)[1].split("\n}\n", 1)[0] + "\n}\n"
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ("model.bin", "optimizer.bin", "metric.json"):
                (directory / name).write_bytes(name.encode())
            (directory / "SHA256SUMS").write_text("".join(f"{runner.sha(directory/name)}  {name}\n" for name in ("model.bin", "optimizer.bin", "metric.json")))
            subprocess.run(["bash", "-c", 'set -euo pipefail\n' + function + '\nsave_pass /nonexistent "$1" 2', "bash", str(directory)], check=True)
            self.assertEqual((directory / "model.bin").read_bytes(), b"model.bin")


if __name__ == "__main__":
    unittest.main()

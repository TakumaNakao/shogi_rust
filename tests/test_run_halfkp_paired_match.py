#!/usr/bin/env python3

from __future__ import annotations

import dataclasses
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "run_halfkp_paired_match.py"
SPEC = importlib.util.spec_from_file_location("run_halfkp_paired_match", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def make_opening(index: int) -> Any:
    record = {
        "schema": "test_opening_v1",
        "split": "valid",
        "opening_family": f"family-{index}",
        "sfen": f"test-sfen-{index}",
        "ply": 8 + index,
    }
    raw = json.dumps(record, separators=(",", ":")).encode("utf-8")
    return runner.SelectedOpening(
        pair_index=index,
        source_line=index + 1,
        raw_json=raw,
        opening_family=record["opening_family"],
        sfen=record["sfen"],
        ply=record["ply"],
    )


def fingerprint_inputs(pairs: int) -> dict[str, Any]:
    return {
        "schema": runner.FINGERPRINT_SCHEMA,
        "lineage_id": runner.LINEAGE_ID,
        "promotion_match_binary_sha256": "b" * 64,
        "candidate_sha256": "c" * 64,
        "candidate_manifest_sha256": "d" * 64,
        "parent_sha256": "p" * 64,
        "parent_manifest_sha256": "q" * 64,
        "registry_sha256": "r" * 64,
        "openings_sha256": "o" * 64,
        "pairs": pairs,
        "nodes": 20_000,
        "max_depth": 64,
        "max_plies": 256,
        "bootstrap": {
            "method": "opening_pair_percentile",
            "resamples": 2_000,
            "seed": 12345,
        },
        "selected_openings": [],
    }


def make_child_report(
    opening: Any,
    inputs: dict[str, Any],
    results: tuple[str, str],
    safety_failure: str | None = None,
) -> dict[str, Any]:
    colors = ("black", "white")
    counts = {key: 0 for key in runner.COUNT_KEYS}
    games = []
    for color, result in zip(colors, results):
        counts[
            {
                "candidate_win": "candidate_wins",
                "parent_win": "parent_wins",
                "draw": "draws",
            }[result]
        ] += 1
        if result == "candidate_win":
            counts[f"candidate_{color}_wins"] += 1
        termination = "max_plies" if result == "draw" else "no_legal_move"
        if termination == "max_plies":
            counts["max_ply_draws"] += 1
        games.append(
            {
                "pair": 0,
                "opening_family": opening.opening_family,
                "opening_ply": opening.ply,
                "candidate_color": color,
                "result": result,
                "termination": termination,
                "played_plies": 10,
            }
        )
    if safety_failure is not None:
        if results[0] != "draw":
            raise AssertionError("safety failure fixture requires a draw first game")
        counts["max_ply_draws"] -= 1
        counts[safety_failure] += 1
        games[0]["termination"] = {
            "search_failures": "search_failure",
            "missing_root_results": "missing_root",
            "illegal_moves": "illegal_move",
            "non_finite_scores": "non_finite_score",
        }[safety_failure]
    score = (counts["candidate_wins"] + 0.5 * counts["draws"]) / 2.0
    lower, _ = runner.wilson_interval(
        counts["candidate_wins"] + 0.5 * counts["draws"], 2
    )
    safety = all(counts[key] == 0 for key in runner.SAFETY_COUNT_KEYS)
    return {
        "schema": "halfkp_promotion_match_v1",
        "lineage_id": runner.LINEAGE_ID,
        "candidate_sha256": inputs["candidate_sha256"],
        "parent_sha256": inputs["parent_sha256"],
        "openings_sha256": opening.input_sha256,
        "node_budget_per_move": inputs["nodes"],
        "max_depth": inputs["max_depth"],
        "max_plies": inputs["max_plies"],
        "pairs": 1,
        "games": 2,
        "score": score,
        "score_wilson_95_lower": lower,
        "passes_point_score": score >= 0.5,
        "passes_noninferiority": lower >= 0.475,
        "passes_safety": safety,
        "promotion_gate_passed": score >= 0.5 and lower >= 0.475 and safety,
        "counts": counts,
        "game_reports": games,
    }


def make_artifact(
    opening: Any,
    inputs: dict[str, Any],
    results: tuple[str, str],
    fingerprint: str = "f" * 64,
    safety_failure: str | None = None,
) -> dict[str, Any]:
    child = make_child_report(opening, inputs, results, safety_failure)
    return runner.make_pair_artifact(opening, fingerprint, inputs, child)


class AggregateTests(unittest.TestCase):
    def test_aggregate_preserves_wdl_colors_safety_and_global_pair_indices(self) -> None:
        inputs = fingerprint_inputs(2)
        openings = [make_opening(0), make_opening(1)]
        artifacts = [
            make_artifact(
                openings[0], inputs, ("candidate_win", "candidate_win")
            ),
            make_artifact(openings[1], inputs, ("parent_win", "draw")),
        ]
        metadata = [
            {"pair_index": index, "path": f"pairs/pair_{index:04d}.json", "sha256": str(index) * 64, "bytes": 1}
            for index in range(2)
        ]

        report = runner.aggregate_pair_artifacts(
            artifacts, "f" * 64, inputs, metadata
        )

        self.assertEqual(report["pairs"], 2)
        self.assertEqual(report["games"], 4)
        self.assertEqual(report["score"], 0.625)
        self.assertEqual(report["counts"]["candidate_wins"], 2)
        self.assertEqual(report["counts"]["parent_wins"], 1)
        self.assertEqual(report["counts"]["draws"], 1)
        self.assertEqual(report["counts"]["candidate_black_wins"], 1)
        self.assertEqual(report["counts"]["candidate_white_wins"], 1)
        self.assertTrue(report["passes_safety"])
        self.assertEqual(
            report["paired_score_ci95_lower"],
            report["opening_pair_bootstrap"]["ci95_lower"],
        )
        self.assertEqual(
            report["paired_score_ci95_upper"],
            report["opening_pair_bootstrap"]["ci95_upper"],
        )
        self.assertEqual(
            [game["pair"] for game in report["game_reports"]], [0, 0, 1, 1]
        )
        self.assertEqual(
            report["pair_score_distribution"],
            {"0.00": 0, "0.25": 1, "0.50": 0, "0.75": 0, "1.00": 1},
        )

    def test_aggregate_propagates_child_safety_failure(self) -> None:
        inputs = fingerprint_inputs(1)
        opening = make_opening(0)
        artifact = make_artifact(
            opening,
            inputs,
            ("draw", "draw"),
            safety_failure="search_failures",
        )
        metadata = [
            {"pair_index": 0, "path": "pairs/pair_0000.json", "sha256": "0" * 64, "bytes": 1}
        ]

        report = runner.aggregate_pair_artifacts(
            [artifact], "f" * 64, inputs, metadata
        )

        self.assertFalse(report["passes_safety"])
        self.assertFalse(report["promotion_gate_passed"])
        self.assertEqual(report["counts"]["search_failures"], 1)


class BootstrapTests(unittest.TestCase):
    def test_bootstrap_is_seed_deterministic_and_pair_order_invariant(self) -> None:
        scores = [0.0, 0.25, 0.5, 0.75, 1.0]
        expected = runner.paired_percentile_bootstrap(scores, 5_000, 9876)
        self.assertEqual(
            runner.paired_percentile_bootstrap(list(reversed(scores)), 5_000, 9876),
            expected,
        )
        self.assertEqual(
            runner.paired_percentile_bootstrap(scores, 5_000, 9876), expected
        )
        self.assertLess(expected[0], 0.5)
        self.assertGreater(expected[1], 0.5)

    def test_constant_pair_scores_have_exact_zero_width_interval(self) -> None:
        self.assertEqual(
            runner.paired_percentile_bootstrap([0.75] * 20, 100_000, 123),
            (0.75, 0.75),
        )


class ProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.binary = self.root / "halfkp_promotion_match"
        self.binary.write_bytes(b"binary")
        self.binary.chmod(0o755)
        self.candidate = self.root / "candidate.bin"
        self.parent = self.root / "parent.bin"
        self.candidate.write_bytes(b"candidate")
        self.parent.write_bytes(b"parent")
        self.candidate_manifest = self.root / "candidate.json"
        self.parent_manifest = self.root / "parent.json"
        self.candidate_manifest.write_text(
            json.dumps(
                {
                    "lineage_id": runner.LINEAGE_ID,
                    "model": {"sha256": runner.sha256_file(self.candidate)},
                }
            ),
            encoding="utf-8",
        )
        self.parent_manifest.write_text(
            json.dumps(
                {
                    "lineage_id": runner.LINEAGE_ID,
                    "model": {"sha256": runner.sha256_file(self.parent)},
                }
            ),
            encoding="utf-8",
        )
        self.registry = self.root / "registry.json"
        self.registry.write_text(
            json.dumps(
                {
                    "lineage_id": runner.LINEAGE_ID,
                    "accepted": [
                        {"model_sha256": runner.sha256_file(self.parent)}
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.openings_path = self.root / "openings.jsonl"
        self.openings_path.write_bytes(make_opening(0).input_bytes)
        self.inputs = runner.RunInputs(
            binary=self.binary,
            candidate=self.candidate,
            candidate_manifest=self.candidate_manifest,
            parent=self.parent,
            parent_manifest=self.parent_manifest,
            registry=self.registry,
            openings=self.openings_path,
            output_dir=self.root / "output",
            pairs=1,
            nodes=20_000,
            max_depth=64,
            max_plies=256,
            bootstrap_resamples=100_000,
            bootstrap_seed=20260806,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_fingerprint_is_stable_and_changes_with_match_semantics(self) -> None:
        openings = runner.select_openings(self.openings_path, 1)
        payload, fingerprint = runner.build_provenance(self.inputs, openings)
        payload_again, fingerprint_again = runner.build_provenance(
            self.inputs, openings
        )
        changed_payload, changed_fingerprint = runner.build_provenance(
            dataclasses.replace(self.inputs, nodes=20_001), openings
        )

        self.assertEqual(payload, payload_again)
        self.assertEqual(fingerprint, fingerprint_again)
        self.assertNotEqual(payload, changed_payload)
        self.assertNotEqual(fingerprint, changed_fingerprint)

    def test_existing_run_requires_exact_fingerprint_and_payload(self) -> None:
        path = self.root / "run.json"
        payload, fingerprint = runner.build_provenance(
            self.inputs, runner.select_openings(self.openings_path, 1)
        )
        runner.initialize_or_validate_run(path, payload, fingerprint, {})
        runner.initialize_or_validate_run(path, payload, fingerprint, {})

        with self.assertRaisesRegex(runner.MatchRunnerError, "fingerprint"):
            runner.initialize_or_validate_run(path, payload, "x" * 64, {})

        changed = dict(payload)
        changed["nodes"] = 1
        with self.assertRaisesRegex(runner.MatchRunnerError, "payload"):
            runner.initialize_or_validate_run(path, changed, fingerprint, {})


class ResumeRangeTests(unittest.TestCase):
    def test_scan_returns_only_missing_pair_indices(self) -> None:
        inputs = fingerprint_inputs(4)
        fingerprint = "f" * 64
        openings = [make_opening(index) for index in range(4)]
        with tempfile.TemporaryDirectory() as temporary:
            pair_dir = Path(temporary) / "pairs"
            for index in (0, 2):
                runner.atomic_create_json(
                    runner.pair_path(pair_dir, index),
                    make_artifact(
                        openings[index],
                        inputs,
                        ("candidate_win", "parent_win"),
                        fingerprint,
                    ),
                )

            artifacts, missing = runner.scan_pair_artifacts(
                pair_dir, openings, fingerprint, inputs
            )

            self.assertEqual(sorted(artifacts), [0, 2])
            self.assertEqual(missing, [1, 3])

    def test_scan_fails_closed_on_wrong_fingerprint_or_out_of_range_pair(self) -> None:
        inputs = fingerprint_inputs(2)
        fingerprint = "f" * 64
        openings = [make_opening(index) for index in range(2)]
        with tempfile.TemporaryDirectory() as temporary:
            pair_dir = Path(temporary) / "pairs"
            runner.atomic_create_json(
                runner.pair_path(pair_dir, 0),
                make_artifact(
                    openings[0],
                    inputs,
                    ("candidate_win", "parent_win"),
                    "z" * 64,
                ),
            )
            with self.assertRaisesRegex(runner.MatchRunnerError, "provenance"):
                runner.scan_pair_artifacts(pair_dir, openings, fingerprint, inputs)

        with tempfile.TemporaryDirectory() as temporary:
            pair_dir = Path(temporary) / "pairs"
            pair_dir.mkdir(parents=True)
            (pair_dir / "pair_9999.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(runner.MatchRunnerError, "unexpected"):
                runner.scan_pair_artifacts(pair_dir, openings, fingerprint, inputs)

    def test_full_runner_restarts_at_first_missing_pair_and_completed_run_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "halfkp_promotion_match"
            binary.write_bytes(b"fake-binary")
            binary.chmod(0o755)
            candidate = root / "candidate.bin"
            parent = root / "parent.bin"
            candidate.write_bytes(b"candidate")
            parent.write_bytes(b"parent")
            candidate_manifest = root / "candidate.json"
            parent_manifest = root / "parent.json"
            candidate_manifest.write_text(
                json.dumps(
                    {
                        "lineage_id": runner.LINEAGE_ID,
                        "model": {"sha256": runner.sha256_file(candidate)},
                    }
                ),
                encoding="utf-8",
            )
            parent_manifest.write_text(
                json.dumps(
                    {
                        "lineage_id": runner.LINEAGE_ID,
                        "model": {"sha256": runner.sha256_file(parent)},
                    }
                ),
                encoding="utf-8",
            )
            registry = root / "registry.json"
            registry.write_text(
                json.dumps(
                    {
                        "lineage_id": runner.LINEAGE_ID,
                        "accepted": [
                            {"model_sha256": runner.sha256_file(parent)}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            openings_path = root / "openings.jsonl"
            openings_path.write_bytes(
                b"".join(make_opening(index).input_bytes for index in range(3))
            )
            run_inputs = runner.RunInputs(
                binary=binary,
                candidate=candidate,
                candidate_manifest=candidate_manifest,
                parent=parent,
                parent_manifest=parent_manifest,
                registry=registry,
                openings=openings_path,
                output_dir=root / "output",
                pairs=3,
                nodes=20_000,
                max_depth=64,
                max_plies=256,
                bootstrap_resamples=100,
                bootstrap_seed=7,
            )
            first_calls = []

            def interrupt_on_second(_run_inputs, opening, inputs):
                first_calls.append(opening.pair_index)
                if opening.pair_index == 1:
                    raise runner.MatchRunnerError("injected interruption")
                return make_child_report(
                    opening, inputs, ("candidate_win", "parent_win")
                )

            with mock.patch.object(
                runner, "run_child_pair", side_effect=interrupt_on_second
            ):
                with redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(
                        runner.MatchRunnerError, "injected interruption"
                    ):
                        runner.run(run_inputs)

            self.assertEqual(first_calls, [0, 1])
            self.assertTrue((run_inputs.output_dir / "pairs/pair_0000.json").is_file())
            self.assertFalse((run_inputs.output_dir / "pairs/pair_0001.json").exists())
            self.assertFalse((run_inputs.output_dir / "completed.json").exists())

            resumed_calls = []

            def finish_missing(_run_inputs, opening, inputs):
                resumed_calls.append(opening.pair_index)
                return make_child_report(
                    opening, inputs, ("candidate_win", "parent_win")
                )

            with mock.patch.object(
                runner, "run_child_pair", side_effect=finish_missing
            ):
                with redirect_stdout(io.StringIO()):
                    report = runner.run(run_inputs)

            self.assertEqual(resumed_calls, [1, 2])
            self.assertEqual(report["score"], 0.5)
            self.assertTrue((run_inputs.output_dir / "report.json").is_file())
            self.assertTrue((run_inputs.output_dir / "completed.json").is_file())

            with mock.patch.object(
                runner,
                "run_child_pair",
                side_effect=AssertionError("completed run invoked a child"),
            ):
                with redirect_stdout(io.StringIO()):
                    completed_report = runner.run(run_inputs)
            self.assertEqual(completed_report["score"], 0.5)


if __name__ == "__main__":
    unittest.main()

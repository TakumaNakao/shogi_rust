#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "halfkp_mmto_unattended.py"
SPEC = importlib.util.spec_from_file_location("halfkp_mmto_unattended", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
unattended = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = unattended
SPEC.loader.exec_module(unattended)


def base_config(workspace: Path) -> dict[str, Any]:
    run_root = workspace / "run"
    placeholder = {"path": "placeholder", "sha256": "a" * 64}
    dataset = {
        "train": dict(placeholder),
        "valid": dict(placeholder),
        "manifest": dict(placeholder),
    }
    return {
        "schema": unattended.CONFIG_SCHEMA,
        "workspace": str(workspace),
        "run_root": str(run_root),
        "lineage_id": unattended.LINEAGE_ID,
        "forbid_kpp_and_v250": True,
        "service": {
            "heartbeat_seconds": 60,
            "max_attempts": 3,
            "retry_seconds": 60,
            "minimum_free_gib": 20,
        },
        "trainer": {
            "jobs": 8,
            "search_block_roots": 1024,
            "pair_batch_size": 8192,
            "output_learning_rate_ratio": 0.1,
            "gradient_clip_norm": 1.0,
            "tau_cp": 36.63,
            "pv_probe_roots": 1024,
            "max_feature_delta_per_outer": 0.02,
            "max_output_delta_per_outer": 0.004,
            "extra_args": [],
        },
        "audit": {
            "input": dict(placeholder),
            "parent_jsonl": dict(placeholder),
            "parent_manifest": dict(placeholder),
            "analyzer": dict(placeholder),
            "roots": 6099,
            "seed": 20260806,
            "jobs": 8,
            "block_roots": 1024,
        },
        "paired_match": {
            "wrapper": dict(placeholder),
            "openings": dict(placeholder),
            "openings_manifest": dict(placeholder),
            "pairs": 250,
            "nodes": 20000,
            "max_depth": 64,
            "max_plies": 256,
            "bootstrap_resamples": 100000,
            "bootstrap_seed": 20260806,
        },
        "phases": {
            "A": {
                "candidate_model": dict(placeholder),
                "gate": {"catastrophic_ci_upper_below": 0.45},
            },
            "B": {
                "start_policy": "accepted_generation2_zero_adagrad",
                "output_dir": str(run_root / "phase-b-primary"),
                "fallback_output_dir": str(run_root / "phase-b-fallback"),
                "dataset": json.loads(json.dumps(dataset)),
                "learning_rate": 0.0003,
                "fallback_learning_rate": 0.0001,
                "max_train_roots": 100000,
                "max_valid_roots": 10000,
                "gate": {"score_below": 0.475, "and_ci_upper_below": 0.5},
            },
            "C": {
                "start_policy": "accepted_generation2_zero_adagrad",
                "output_dir": str(run_root / "phase-c"),
                "dataset": json.loads(json.dumps(dataset)),
                "max_train_roots": None,
                "max_valid_roots": 90759,
                "max_outer": 4,
                "gate": {
                    "catastrophic_ci_upper_below": 0.45,
                    "score_below": 0.475,
                    "and_ci_upper_below": 0.5,
                },
                "extension_gate": {"min_score": 0.525},
            },
        },
    }


class FakeProcess:
    def __init__(self, return_code: int, pid: int):
        self.returncode = return_code
        self.pid = pid

    def poll(self) -> int:
        return self.returncode

    def wait(self, timeout: int | None = None) -> int:
        return self.returncode


class SupervisorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        (self.workspace / ".git").mkdir()
        self.config = base_config(self.workspace)
        self.config_path = self.workspace / "config.json"
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        self.supervisor = unattended.Supervisor(self.config_path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_decision(self, variant: str) -> Path:
        path = self.supervisor.run_root / "phase-b-learning-rate-decision.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        rate = 0.0003 if variant == "primary" else 0.0001
        reason = (
            "primary_completed_safely"
            if variant == "primary"
            else "primary_parameter_or_outer_delta_safety_limit"
        )
        path.write_text(
            json.dumps(
                {
                    "schema": "halfkp_mmto_phase_b_rate_decision_v1",
                    "decided_at": "test",
                    "config_sha256": self.supervisor.config_sha,
                    "variant": variant,
                    "learning_rate": rate,
                    "reason": reason,
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_retry_command_rechecks_checkpoint_and_uses_separate_attempt_logs(self) -> None:
        checkpoint = self.supervisor.run_root / "trainer/checkpoint.json"
        commands: list[list[str]] = []

        def command() -> list[str]:
            result = ["trainer"]
            if checkpoint.exists():
                result.append("--resume")
            return result

        def popen(command_line, **kwargs):
            commands.append(list(command_line))
            kwargs["stdout"].write(f"attempt {len(commands)}\n")
            kwargs["stdout"].flush()
            if len(commands) == 1:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_text("{}", encoding="utf-8")
                return FakeProcess(1, 101)
            return FakeProcess(0, 102)

        with mock.patch.object(unattended.subprocess, "Popen", side_effect=popen), mock.patch.object(
            unattended.time, "sleep"
        ):
            log = self.supervisor.run_command("trainer", command)

        self.assertEqual(commands, [["trainer"], ["trainer", "--resume"]])
        self.assertTrue(log.name.endswith("attempt-02.log"))
        self.assertIn("attempt 1", (log.parent / "trainer.attempt-01.log").read_text())
        self.assertIn("attempt 2", log.read_text())

    def test_primary_rate_decision_is_immutable(self) -> None:
        decision_path = self.write_decision("primary")
        original = decision_path.read_bytes()
        failure_log = self.workspace / "failure.log"
        failure_log.write_text("outer delta limit exceeded", encoding="utf-8")
        failure = unattended.CommandFailure("train", 1, failure_log, True)

        with mock.patch.object(
            self.supervisor, "run_training_outer", side_effect=failure
        ):
            with self.assertRaisesRegex(
                unattended.TerminalBlocked, "immutable primary"
            ):
                self.supervisor.phase_b({})
        self.assertEqual(decision_path.read_bytes(), original)

    def test_fallback_rate_decision_resumes_only_fallback(self) -> None:
        self.write_decision("fallback")
        training = {"model_sha256": "m"}
        audit = {"offline_improves": True}
        match = {
            "passes_safety": True,
            "score": 0.5,
            "paired_score_ci95_upper": 0.6,
        }
        with mock.patch.object(
            self.supervisor, "run_training_outer", return_value=training
        ) as run_training, mock.patch.object(
            self.supervisor, "archive_outer", return_value=self.workspace / "archive"
        ), mock.patch.object(
            self.supervisor, "evaluate_outer", return_value=(audit, match)
        ):
            rate, _, _ = self.supervisor.phase_b({})

        self.assertEqual(rate, 0.0001)
        call = run_training.call_args.args
        self.assertEqual(call[1], "B")
        self.assertEqual(call[4], 1)
        self.assertEqual(call[5], 0.0001)
        self.assertEqual(call[6], "fallback")

    def test_phase_c_live_outer_three_uses_prior_archives_then_current_checkpoint(self) -> None:
        output_dir = Path(self.config["phases"]["C"]["output_dir"])
        output_dir.mkdir(parents=True)
        (output_dir / "checkpoint.json").write_text(
            json.dumps({"completed_outer": 3}), encoding="utf-8"
        )
        archive_root = self.supervisor.run_root / "archives/phase-c"
        for outer in (1, 2):
            (archive_root / f"outer_{outer:04d}").mkdir(parents=True)

        def evaluation(_identity, _archive, _phase, outer):
            scores = {1: 0.53, 2: 0.54, 3: 0.50}
            return (
                {"offline_improves": True, "offline_strong_improvement": True},
                {
                    "passes_safety": True,
                    "score": scores[outer],
                    "paired_score_ci95_upper": 0.6,
                },
            )

        with mock.patch.object(self.supervisor, "verify_archive") as verify_archive, mock.patch.object(
            self.supervisor, "run_training_outer", return_value={"model_sha256": "x"}
        ) as run_training, mock.patch.object(
            self.supervisor,
            "archive_outer",
            return_value=archive_root / "outer_0003",
        ), mock.patch.object(
            self.supervisor, "evaluate_outer", side_effect=evaluation
        ):
            self.supervisor.phase_c({}, 0.0003)

        self.assertEqual(
            [call.args[2] for call in verify_archive.call_args_list], [1, 2]
        )
        self.assertEqual(len(run_training.call_args_list), 1)
        self.assertEqual(run_training.call_args.args[4], 3)
        self.assertEqual(
            self.supervisor.status["terminal_reason"],
            "phase_C_outer_3_no_extension",
        )

    def test_markerless_audit_is_deleted_and_fully_regenerated(self) -> None:
        archive = self.workspace / "archive"
        archive.mkdir()
        model = archive / "model.bin"
        model.write_bytes(b"candidate")
        audit_jsonl, sidecar_comparison, marker = self.supervisor.audit_paths("B", 1)
        sidecar = audit_jsonl.with_name(audit_jsonl.name + ".manifest.json")
        audit_jsonl.parent.mkdir(parents=True)
        for path in (audit_jsonl, sidecar, sidecar_comparison):
            path.write_text("stale", encoding="utf-8")

        def run_command(step, _command, *_args, **_kwargs):
            if "analyze_audit" not in step:
                self.assertFalse(audit_jsonl.exists())
                self.assertFalse(sidecar.exists())
                self.assertFalse(sidecar_comparison.exists())
                audit_jsonl.write_text("{}\n", encoding="utf-8")
                sidecar.write_text(
                    json.dumps(
                        {
                            "schema": "halfkp_mmto_root_audit_v1",
                            "model_sha256": unattended.sha256_file(model),
                            "input_sha256": "a" * 64,
                            "tau_cp": 36.63,
                            "roots": 6099,
                            "games": 6099,
                            "rejected_bounds": 0,
                        }
                    ),
                    encoding="utf-8",
                )
            else:
                sidecar_comparison.write_text(
                    json.dumps(
                        {
                            "comparisons": {
                                "candidate": {
                                    "summary": {
                                        "roots": 6099,
                                        "games": 6099,
                                        "rejected_bounds": 0,
                                        "qnodes_per_root": 123.0,
                                    },
                                    "versus_parent": {
                                        "roots": 6099,
                                        "loss_delta_mean": -0.01,
                                        "loss_delta_bootstrap_95": [-0.02, -0.001],
                                    },
                                }
                            }
                        }
                    ),
                    encoding="utf-8",
                )
            return self.workspace / "log"

        with mock.patch.object(self.supervisor, "_verify_declared"), mock.patch.object(
            self.supervisor, "binary", return_value=Path("/bin/true")
        ), mock.patch.object(
            self.supervisor, "run_command", side_effect=run_command
        ), mock.patch.object(self.supervisor, "verify_archive"):
            result = self.supervisor.run_audit({}, archive, "B", 1)

        self.assertTrue(marker.is_file())
        self.assertTrue(result["offline_strong_improvement"])
        self.assertEqual(len(result["files"]), 3)

        tampered = json.loads(marker.read_text())
        tampered["offline_improves"] = False
        marker.write_text(json.dumps(tampered), encoding="utf-8")
        with self.assertRaisesRegex(unattended.TerminalBlocked, "semantics"):
            self.supervisor.verify_audit_marker(marker, model, "B", 1)

    def test_existing_archive_must_match_live_training(self) -> None:
        archive = self.supervisor.run_root / "archives/phase-b/outer_0001"
        archive.mkdir(parents=True)
        existing = {
            "model_sha256": "old-model",
            "optimizer_sha256": "old-optimizer",
            "files": [],
        }
        training = {
            "model_sha256": "new-model",
            "optimizer_sha256": "new-optimizer",
            "path_sha256": {},
        }
        with mock.patch.object(
            self.supervisor, "verify_archive", return_value=existing
        ):
            with self.assertRaisesRegex(
                unattended.TerminalBlocked, "differs from live checkpoint"
            ):
                self.supervisor.archive_outer("B", 1, training)

    def test_frozen_policy_rejects_outer_expansion(self) -> None:
        self.supervisor.validate_frozen_policy()
        self.supervisor.config["phases"]["C"]["max_outer"] = 5
        with self.assertRaisesRegex(unattended.TerminalBlocked, "max_outer"):
            self.supervisor.validate_frozen_policy()

    def test_shutdown_status_without_stop_marker_resumes(self) -> None:
        self.supervisor.save_status(status="stopped_by_user")
        with mock.patch.object(
            self.supervisor, "validate_preflight", return_value={}
        ), mock.patch.object(
            self.supervisor, "ensure_identity", return_value={}
        ), mock.patch.object(
            self.supervisor, "phase_a"
        ) as phase_a, mock.patch.object(
            self.supervisor,
            "phase_b",
            return_value=(0.0003, {}, {}),
        ), mock.patch.object(self.supervisor, "phase_c"):
            self.supervisor.run_workflow()
        phase_a.assert_called_once()


if __name__ == "__main__":
    unittest.main()

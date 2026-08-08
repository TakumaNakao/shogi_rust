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

    def make_recovery_supervisor(self) -> unattended.Supervisor:
        """Return a fresh-run supervisor for the fixed 1e-4 Phase C recovery."""
        evidence_names = (
            "status",
            "phase_b_archive_manifest",
            "phase_b_model",
            "phase_b_audit_marker",
            "phase_b_match_report",
            "phase_c_failure_log",
            "phase_c_checkpoint",
            "phase_c_train_lock",
        )
        self.config["workflow"] = {
            "mode": "phase_c_scaled_recovery",
            "prior_run_root": str(self.workspace / "prior-run"),
            "evidence": {
                name: {"path": f"prior-run/{name}", "sha256": "a" * 64}
                for name in evidence_names
            },
        }
        self.config["service"]["unit_file"] = (
            "scripts/systemd/halfkp-mmto-full-recovery-20260808.service"
        )
        phase_c = self.config["phases"]["C"]
        phase_c["learning_rate"] = 0.0001
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        return unattended.Supervisor(self.config_path)

    def make_recovery_evidence_supervisor(
        self,
    ) -> tuple[unattended.Supervisor, dict[str, Path]]:
        """Build a small, internally consistent prior-run evidence fixture."""
        prior = self.workspace / "prior-failed-run"
        archive = prior / "archives/phase-b/outer_0001"
        evaluation = prior / "evaluations/phase-b-outer-0001"
        phase_c = prior / "phase-c-full"
        logs = prior / "logs"
        for directory in (archive, evaluation / "match", phase_c, logs):
            directory.mkdir(parents=True, exist_ok=True)

        parent_model = self.workspace / "accepted-generation-2.bin"
        parent_model.write_bytes(b"accepted scratch generation 2")
        parent_sha = unattended.sha256_file(parent_model)
        incumbent_model = archive / "model.bin"
        incumbent_model.write_bytes(b"phase B incumbent diagnostic candidate")
        incumbent_sha = unattended.sha256_file(incumbent_model)
        old_config_sha = "b" * 64
        dataset_sha = self.config["phases"]["C"]["dataset"]["manifest"]["sha256"]

        archive_manifest = archive / "archive_manifest.json"
        archive_manifest.write_text(
            json.dumps(
                {
                    "schema": unattended.ARCHIVE_SCHEMA,
                    "config_sha256": old_config_sha,
                    "phase": "B",
                    "outer": 1,
                    "model_sha256": incumbent_sha,
                    "optimizer_sha256": "c" * 64,
                    "files": [
                        {
                            "path": "model.bin",
                            "sha256": incumbent_sha,
                            "bytes": incumbent_model.stat().st_size,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        audit_marker = evaluation / "audit_completed.json"
        audit_marker.write_text(
            json.dumps(
                {
                    "schema": unattended.AUDIT_MARKER_SCHEMA,
                    "config_sha256": old_config_sha,
                    "phase": "B",
                    "outer": 1,
                    "model_sha256": incumbent_sha,
                    "loss_delta_mean": -0.03795726736700742,
                    "loss_delta_ci95_upper": -0.03581002362663455,
                    "offline_improves": True,
                    "offline_strong_improvement": True,
                    "qnodes_per_root": 204886.45892769308,
                }
            ),
            encoding="utf-8",
        )
        match_report = evaluation / "match/report.json"
        match_report.write_text(
            json.dumps(
                {
                    "schema": "halfkp_resumable_paired_match_v1",
                    "lineage_id": unattended.LINEAGE_ID,
                    "candidate_sha256": incumbent_sha,
                    "parent_sha256": parent_sha,
                    "openings_sha256": self.config["paired_match"]["openings"][
                        "sha256"
                    ],
                    "pairs": 250,
                    "games": 500,
                    "score": 0.572,
                    "paired_score_ci95_lower": 0.531,
                    "paired_score_ci95_upper": 0.612,
                    "passes_safety": True,
                }
            ),
            encoding="utf-8",
        )
        failure_log = logs / "phase_c_full_train_outer_0001.attempt-01.log"
        failure_log.write_text(
            "outer delta limit exceeded: feature=0.033063922 "
            "hidden=0.019872695 output=0.0027041882\n",
            encoding="utf-8",
        )
        checkpoint = phase_c / "checkpoint.json"
        checkpoint.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "completed_outer": 0,
                    "init_model_sha256": parent_sha,
                    "dataset_manifest_sha256": dataset_sha,
                    "previous_model_sha256": parent_sha,
                    "model_sha256": parent_sha,
                    "optimizer_sha256": (
                        "f4f97edc7e2815cffbcd54911d35831df6ca4f0f0000a6708808465a7d264e7b"
                    ),
                }
            ),
            encoding="utf-8",
        )
        train_lock = phase_c / "unattended_train_lock.json"
        train_lock.write_text(
            json.dumps(
                {
                    "schema": unattended.TRAIN_LOCK_SCHEMA,
                    "config_sha256": old_config_sha,
                    "phase": "C",
                    "parent_generation": 2,
                    "parent_model_sha256": parent_sha,
                    "dataset": self.config["phases"]["C"]["dataset"],
                    "hyperparameters": {
                        "learning_rate": 0.0003,
                        "max_feature_delta_per_outer": 0.02,
                        "max_output_delta_per_outer": 0.004,
                    },
                }
            ),
            encoding="utf-8",
        )
        (phase_c / "model.bin").write_bytes(parent_model.read_bytes())
        (phase_c / "optimizer.bin").write_bytes(b"rolled-back zero AdaGrad state")
        status = prior / "status.json"
        last_error = f"step phase_c_full_train_outer_0001 failed; log={failure_log}"
        status.write_text(
            json.dumps(
                {
                    "schema": unattended.STATUS_SCHEMA,
                    "config_sha256": old_config_sha,
                    "status": "blocked",
                    "current_phase": "C",
                    "current_outer": 1,
                    "terminal_reason": last_error,
                    "last_error": last_error,
                }
            ),
            encoding="utf-8",
        )

        paths = {
            "status": status,
            "phase_b_archive_manifest": archive_manifest,
            "phase_b_model": incumbent_model,
            "phase_b_audit_marker": audit_marker,
            "phase_b_match_report": match_report,
            "phase_c_failure_log": failure_log,
            "phase_c_checkpoint": checkpoint,
            "phase_c_train_lock": train_lock,
        }
        self.config["workflow"] = {
            "mode": "phase_c_scaled_recovery",
            "prior_run_root": str(prior),
            "evidence": {
                name: {
                    "path": str(path),
                    "sha256": unattended.sha256_file(path),
                }
                for name, path in paths.items()
            },
        }
        self.config["phases"]["C"]["learning_rate"] = 0.0001
        self.config["service"]["unit_file"] = (
            "scripts/systemd/halfkp-mmto-full-recovery-20260808.service"
        )
        self.config["parent"] = {
            "generation": 2,
            "model": {"path": str(parent_model), "sha256": parent_sha},
            "model_manifest": {"path": str(parent_model), "sha256": parent_sha},
            "promotion_registry": {"path": str(parent_model), "sha256": parent_sha},
            "gate_report": {"path": str(parent_model), "sha256": parent_sha},
        }
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        return unattended.Supervisor(self.config_path), paths

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

    def test_recovery_workflow_skips_a_and_b_and_uses_only_fixed_rate(self) -> None:
        supervisor = self.make_recovery_supervisor()
        identity = {"parent_sha256": "accepted-generation-2"}
        with mock.patch.object(
            supervisor, "validate_preflight", return_value={}
        ), mock.patch.object(
            supervisor, "ensure_identity", return_value=identity
        ), mock.patch.object(
            supervisor, "verify_recovery_evidence", return_value={"verified": True}
        ) as verify_evidence, mock.patch.object(
            supervisor, "completed_step"
        ), mock.patch.object(
            supervisor, "phase_a"
        ) as phase_a, mock.patch.object(
            supervisor, "phase_b"
        ) as phase_b, mock.patch.object(
            supervisor, "phase_c"
        ) as phase_c:
            supervisor.run_workflow()

        phase_a.assert_not_called()
        phase_b.assert_not_called()
        verify_evidence.assert_called_once_with()
        phase_c.assert_called_once_with(identity, 0.0001)

    def test_recovery_policy_rejects_any_learning_rate_sweep(self) -> None:
        supervisor = self.make_recovery_supervisor()
        supervisor.validate_frozen_policy()

        supervisor.config["phases"]["C"]["learning_rate"] = 0.00015
        with self.assertRaisesRegex(unattended.TerminalBlocked, "learning_rate"):
            supervisor.validate_frozen_policy()

        supervisor.config["phases"]["C"]["learning_rate"] = 0.0001
        supervisor.config["phases"]["C"]["fallback_learning_rate"] = 0.00005
        with self.assertRaisesRegex(unattended.TerminalBlocked, "fallback"):
            supervisor.validate_frozen_policy()

    def test_recovery_training_failure_blocks_without_trying_another_rate(self) -> None:
        supervisor = self.make_recovery_supervisor()
        failure_log = self.workspace / "recovery-failure.log"
        failure_log.write_text(
            "outer delta limit exceeded: feature=0.021 hidden=0.010 output=0.001\n",
            encoding="utf-8",
        )
        failure = unattended.CommandFailure("phase_c", 1, failure_log, True)
        identity = {"parent_sha256": "accepted-generation-2"}
        with mock.patch.object(
            supervisor, "validate_preflight", return_value={}
        ), mock.patch.object(
            supervisor, "ensure_identity", return_value=identity
        ), mock.patch.object(
            supervisor, "verify_recovery_evidence", return_value={"verified": True}
        ), mock.patch.object(
            supervisor, "completed_step"
        ), mock.patch.object(
            supervisor, "phase_c", side_effect=failure
        ) as phase_c:
            supervisor.run_workflow()

        phase_c.assert_called_once_with(identity, 0.0001)
        self.assertEqual(supervisor.status["status"], "blocked")
        self.assertIn("phase_c", supervisor.status["terminal_reason"])

    def test_recovery_evidence_sha_and_semantics_are_both_fail_closed(self) -> None:
        supervisor, paths = self.make_recovery_evidence_supervisor()
        optimizer = paths["phase_c_checkpoint"].parent / "optimizer.bin"
        actual_sha256_file = unattended.sha256_file

        def evidence_sha(path: Path) -> str:
            if Path(path) == optimizer:
                return "f4f97edc7e2815cffbcd54911d35831df6ca4f0f0000a6708808465a7d264e7b"
            return actual_sha256_file(Path(path))

        with mock.patch.object(unattended, "sha256_file", side_effect=evidence_sha):
            supervisor.verify_recovery_evidence()

            paths["phase_c_failure_log"].write_text(
                "outer delta limit exceeded: feature=0.010 hidden=0.010 output=0.001\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(unattended.TerminalBlocked, "SHA-256"):
                supervisor.verify_recovery_evidence()

            workflow = supervisor.config["workflow"]
            workflow["evidence"]["phase_c_failure_log"]["sha256"] = evidence_sha(
                paths["phase_c_failure_log"]
            )
            with self.assertRaisesRegex(
                unattended.TerminalBlocked, "evidence|delta|failure"
            ):
                supervisor.verify_recovery_evidence()

    def test_recovery_phase_c_uses_new_output_not_failed_run_artifacts(self) -> None:
        supervisor, _ = self.make_recovery_evidence_supervisor()
        failed_run = Path(supervisor.config["workflow"]["prior_run_root"])
        output_dir = Path(supervisor.config["phases"]["C"]["output_dir"])
        identity = {"binaries": {"trainer": {"path": "/bin/true"}}}
        command = supervisor.trainer_command(
            identity,
            supervisor.config["phases"]["C"],
            output_dir,
            1,
            0.0001,
            False,
        )
        init_index = command.index("--init")
        self.assertEqual(
            command[init_index + 1],
            str(Path(supervisor.config["parent"]["model"]["path"]).resolve()),
        )
        self.assertNotIn("--resume", command)
        self.assertNotIn(str(failed_run), command)

        training = {"model_sha256": "fresh"}
        audit = {"offline_improves": True, "offline_strong_improvement": True}
        match = {
            "passes_safety": True,
            "score": 0.56,
            "paired_score_ci95_upper": 0.6,
        }
        with mock.patch.object(
            supervisor, "run_training_outer", return_value=training
        ) as run_training, mock.patch.object(
            supervisor, "archive_outer", return_value=self.workspace / "archive"
        ), mock.patch.object(
            supervisor, "evaluate_outer", return_value=(audit, match)
        ):
            supervisor.phase_c({}, 0.0001)

        first_call = run_training.call_args_list[0].args
        self.assertEqual(first_call[1], "C")
        self.assertEqual(first_call[3], output_dir)
        self.assertEqual(first_call[5], 0.0001)
        self.assertNotIn(str(failed_run), repr(run_training.call_args_list))

    def test_phase_c_outer_four_must_pass_final_quality_gate(self) -> None:
        supervisor = self.make_recovery_supervisor()
        scores = {1: 0.53, 2: 0.54, 3: 0.55, 4: 0.52}

        def evaluation(_identity, _archive, _phase, outer):
            return (
                {"offline_improves": True, "offline_strong_improvement": True},
                {
                    "passes_safety": True,
                    "score": scores[outer],
                    "paired_score_ci95_upper": 0.6,
                },
            )

        with mock.patch.object(
            supervisor, "run_training_outer", return_value={"model_sha256": "x"}
        ), mock.patch.object(
            supervisor, "archive_outer", return_value=self.workspace / "archive"
        ), mock.patch.object(
            supervisor, "evaluate_outer", side_effect=evaluation
        ) as evaluate_outer:
            supervisor.phase_c({}, 0.0001)

        self.assertEqual(
            [call.args[3] for call in evaluate_outer.call_args_list], [1, 2, 3, 4]
        )
        self.assertEqual(supervisor.status["status"], "completed")
        self.assertEqual(
            supervisor.status["terminal_reason"], "phase_C_outer_4_no_extension"
        )

    def test_phase_c_outer_four_records_complete_only_after_final_quality_gate(self) -> None:
        supervisor = self.make_recovery_supervisor()
        scores = {1: 0.53, 2: 0.54, 3: 0.55, 4: 0.56}

        def evaluation(_identity, _archive, _phase, outer):
            return (
                {"offline_improves": True, "offline_strong_improvement": True},
                {
                    "passes_safety": True,
                    "score": scores[outer],
                    "paired_score_ci95_upper": 0.6,
                },
            )

        with mock.patch.object(
            supervisor, "run_training_outer", return_value={"model_sha256": "x"}
        ), mock.patch.object(
            supervisor, "archive_outer", return_value=self.workspace / "archive"
        ), mock.patch.object(
            supervisor, "evaluate_outer", side_effect=evaluation
        ):
            supervisor.phase_c({}, 0.0001)

        self.assertEqual(supervisor.status["status"], "completed")
        self.assertEqual(
            supervisor.status["terminal_reason"], "phase_C_outer_4_complete"
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
        with mock.patch.object(self.supervisor, "_verify_declared"):
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

    def test_repeated_command_success_does_not_erase_crash_loop_count(self) -> None:
        self.supervisor.completed_step("match_command")
        self.supervisor.save_status(supervisor_crash_count=2)
        self.supervisor.completed_step("match_command")
        self.assertEqual(self.supervisor.status["supervisor_crash_count"], 2)

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

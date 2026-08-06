#!/usr/bin/env python3
"""Fail-closed supervisor for the long-running scratch HalfKP MMTO workflow."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


CONFIG_SCHEMA = "halfkp_mmto_unattended_config_v1"
STATUS_SCHEMA = "halfkp_mmto_unattended_status_v1"
IDENTITY_SCHEMA = "halfkp_mmto_unattended_identity_v1"
TRAIN_LOCK_SCHEMA = "halfkp_mmto_unattended_train_lock_v1"
ARCHIVE_SCHEMA = "halfkp_mmto_outer_archive_v1"
AUDIT_MARKER_SCHEMA = "halfkp_mmto_unattended_audit_v1"
LINEAGE_ID = "halfkp64-scratch-v1"


class SupervisorError(RuntimeError):
    """Base class for controlled supervisor errors."""


class TerminalBlocked(SupervisorError):
    """A deterministic integrity or execution failure requiring review."""


class GateStop(SupervisorError):
    """A valid experiment result that forbids automatic continuation."""


class StopRequested(SupervisorError):
    """The operator or service manager requested a clean stop."""


class CommandFailure(SupervisorError):
    def __init__(self, step: str, return_code: int, log_path: Path, deterministic: bool):
        super().__init__(
            f"step {step} failed with exit {return_code}; log={log_path}"
        )
        self.step = step
        self.return_code = return_code
        self.log_path = log_path
        self.deterministic = deterministic


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TerminalBlocked(f"cannot read valid JSON from {path}: {error}") from error
    if not isinstance(value, dict):
        raise TerminalBlocked(f"expected a JSON object in {path}")
    return value


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_file(path: Path) -> None:
    with path.open("rb") as source:
        os.fsync(source.fileno())


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    atomic_bytes(path, payload)


def atomic_copy(source: Path, destination: Path, mode: int | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as input_file, os.fdopen(descriptor, "wb") as output:
            shutil.copyfileobj(input_file, output, 1024 * 1024)
            if mode is not None:
                os.fchmod(output.fileno(), mode)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def require_finite(value: Any, label: str = "value") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TerminalBlocked(f"non-finite {label}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            require_finite(item, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite(item, f"{label}.{key}")


def artifact_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise TerminalBlocked(f"artifact is missing: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def command_output(command: Sequence[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(
            list(command), cwd=cwd, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise TerminalBlocked(f"command failed: {' '.join(command)}: {error}") from error


class Supervisor:
    def __init__(self, config_path: Path):
        self.config_path = config_path.resolve()
        self.config = load_json(self.config_path)
        if self.config.get("schema") != CONFIG_SCHEMA:
            raise TerminalBlocked("unsupported unattended config schema")
        self.config_sha = sha256_file(self.config_path)
        workspace_value = self.config.get("workspace")
        if not isinstance(workspace_value, str):
            raise TerminalBlocked("config.workspace must be a path")
        self.workspace = Path(workspace_value).resolve()
        if not (self.workspace / ".git").exists():
            raise TerminalBlocked(f"workspace is not a Git repository: {self.workspace}")
        run_value = self.config.get("run_root")
        if not isinstance(run_value, str):
            raise TerminalBlocked("config.run_root must be a path")
        self.run_root = self.resolve(run_value)
        try:
            self.run_root.relative_to(self.workspace)
        except ValueError as error:
            raise TerminalBlocked("run_root must be inside workspace") from error
        if self.config.get("lineage_id") != LINEAGE_ID:
            raise TerminalBlocked("unattended config has wrong lineage")
        service = self.section("service")
        self.heartbeat_seconds = self.positive_int(
            service.get("heartbeat_seconds"), "heartbeat_seconds"
        )
        self.max_attempts = self.positive_int(
            service.get("max_attempts"), "max_attempts"
        )
        self.retry_seconds = self.positive_int(
            service.get("retry_seconds"), "retry_seconds"
        )
        self.status_path = self.run_root / "status.json"
        self.identity_path = self.run_root / "run_identity.json"
        self.stop_path = self.run_root / "STOP_REQUESTED.json"
        self.lock_path = self.run_root / "supervisor.lock"
        self._lock_descriptor: int | None = None
        self.child: subprocess.Popen[str] | None = None
        self.received_signal: int | None = None
        self.status = self._load_status()

    def resolve(self, value: str | Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self.workspace / path
        return path.resolve()

    def section(self, name: str) -> dict[str, Any]:
        value = self.config.get(name)
        if not isinstance(value, dict):
            raise TerminalBlocked(f"config.{name} must be an object")
        return value

    @staticmethod
    def positive_int(value: Any, label: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise TerminalBlocked(f"{label} must be a positive integer")
        return value

    @staticmethod
    def exact_number(
        mapping: Mapping[str, Any], key: str, expected: float, label: str
    ) -> None:
        value = mapping.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) != expected
        ):
            raise TerminalBlocked(f"{label}.{key} must be exactly {expected}")

    @staticmethod
    def exact_integer(
        mapping: Mapping[str, Any], key: str, expected: int, label: str
    ) -> None:
        value = mapping.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value != expected:
            raise TerminalBlocked(f"{label}.{key} must be exactly {expected}")

    def validate_frozen_policy(self) -> None:
        """Reject config drift that could silently expand the unattended run."""
        service = self.section("service")
        for key, expected in (
            ("heartbeat_seconds", 60),
            ("max_attempts", 3),
            ("retry_seconds", 60),
            ("minimum_free_gib", 20),
        ):
            self.exact_integer(service, key, expected, "service")

        trainer = self.section("trainer")
        for key, expected in (
            ("jobs", 8),
            ("search_block_roots", 1024),
            ("pair_batch_size", 8192),
            ("pv_probe_roots", 1024),
        ):
            self.exact_integer(trainer, key, expected, "trainer")
        for key, expected in (
            ("output_learning_rate_ratio", 0.1),
            ("gradient_clip_norm", 1.0),
            ("tau_cp", 36.63),
            ("max_feature_delta_per_outer", 0.02),
            ("max_output_delta_per_outer", 0.004),
        ):
            self.exact_number(trainer, key, expected, "trainer")
        if trainer.get("extra_args") != []:
            raise TerminalBlocked("trainer.extra_args must remain empty")

        audit = self.section("audit")
        for key, expected in (
            ("roots", 6099),
            ("seed", 20260806),
            ("jobs", 8),
            ("block_roots", 1024),
        ):
            self.exact_integer(audit, key, expected, "audit")

        paired = self.section("paired_match")
        for key, expected in (
            ("pairs", 250),
            ("nodes", 20000),
            ("max_depth", 64),
            ("max_plies", 256),
            ("bootstrap_resamples", 100000),
            ("bootstrap_seed", 20260806),
        ):
            self.exact_integer(paired, key, expected, "paired_match")

        phases = self.section("phases")
        if set(phases) != {"A", "B", "C"}:
            raise TerminalBlocked("only phases A, B, and C may run unattended")
        phase_a = phases["A"]
        if set(phase_a.get("gate", {})) != {"catastrophic_ci_upper_below"}:
            raise TerminalBlocked("phase A gate shape changed")
        self.exact_number(
            phase_a["gate"], "catastrophic_ci_upper_below", 0.45, "phases.A.gate"
        )

        phase_b = phases["B"]
        phase_c = phases["C"]
        for name, phase in (("B", phase_b), ("C", phase_c)):
            if phase.get("start_policy") != "accepted_generation2_zero_adagrad":
                raise TerminalBlocked(f"phase {name} must start from zero-state AdaGrad")
        self.exact_number(phase_b, "learning_rate", 0.0003, "phases.B")
        self.exact_number(
            phase_b, "fallback_learning_rate", 0.0001, "phases.B"
        )
        self.exact_integer(phase_b, "max_train_roots", 100000, "phases.B")
        self.exact_integer(phase_b, "max_valid_roots", 10000, "phases.B")
        if set(phase_b.get("gate", {})) != {"score_below", "and_ci_upper_below"}:
            raise TerminalBlocked("phase B gate shape changed")
        self.exact_number(phase_b["gate"], "score_below", 0.475, "phases.B.gate")
        self.exact_number(
            phase_b["gate"], "and_ci_upper_below", 0.5, "phases.B.gate"
        )

        if phase_c.get("max_train_roots") is not None:
            raise TerminalBlocked("phase C must use all frozen train roots")
        self.exact_integer(phase_c, "max_valid_roots", 90759, "phases.C")
        self.exact_integer(phase_c, "max_outer", 4, "phases.C")
        if set(phase_c.get("gate", {})) != {
            "catastrophic_ci_upper_below",
            "score_below",
            "and_ci_upper_below",
        }:
            raise TerminalBlocked("phase C gate shape changed")
        self.exact_number(
            phase_c["gate"], "catastrophic_ci_upper_below", 0.45, "phases.C.gate"
        )
        self.exact_number(phase_c["gate"], "score_below", 0.475, "phases.C.gate")
        self.exact_number(
            phase_c["gate"], "and_ci_upper_below", 0.5, "phases.C.gate"
        )
        if set(phase_c.get("extension_gate", {})) != {"min_score"}:
            raise TerminalBlocked("phase C extension gate shape changed")
        self.exact_number(
            phase_c["extension_gate"],
            "min_score",
            0.525,
            "phases.C.extension_gate",
        )

        output_paths = [
            self.resolve(phase_b["output_dir"]),
            self.resolve(phase_b["fallback_output_dir"]),
            self.resolve(phase_c["output_dir"]),
        ]
        if len(set(output_paths)) != len(output_paths):
            raise TerminalBlocked("phase output directories must be distinct")
        for path in output_paths:
            try:
                path.relative_to(self.run_root)
            except ValueError as error:
                raise TerminalBlocked(
                    "phase output directories must remain inside run_root"
                ) from error

    def _load_status(self) -> dict[str, Any]:
        if not self.status_path.exists():
            return {
                "schema": STATUS_SCHEMA,
                "config_path": str(self.config_path),
                "config_sha256": self.config_sha,
                "status": "not_started",
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "current_phase": None,
                "current_step": None,
                "current_outer": None,
                "child_pid": None,
                "attempts": {},
                "completed_steps": [],
                "artifacts": {},
                "warnings": [],
                "last_error": None,
                "terminal_reason": None,
                "supervisor_crash_count": 0,
            }
        status = load_json(self.status_path)
        if (
            status.get("schema") != STATUS_SCHEMA
            or status.get("config_sha256") != self.config_sha
        ):
            raise TerminalBlocked("existing status has a different config provenance")
        return status

    def save_status(self, **updates: Any) -> None:
        self.status.update(updates)
        self.status["updated_at"] = utc_now()
        atomic_json(self.status_path, self.status)

    def record_artifact(self, name: str, path: Path) -> None:
        artifacts = dict(self.status.get("artifacts", {}))
        artifacts[name] = artifact_metadata(path)
        self.save_status(artifacts=artifacts)

    def completed_step(self, step: str, details: Mapping[str, Any] | None = None) -> None:
        completed = list(self.status.get("completed_steps", []))
        is_new = not any(
            entry.get("step") == step for entry in completed if isinstance(entry, dict)
        )
        if is_new:
            completed.append(
                {"step": step, "completed_at": utc_now(), "details": dict(details or {})}
            )
        updates: dict[str, Any] = {
            "completed_steps": completed,
            "last_error": None,
        }
        # A repeated no-op success must not erase crash-loop accounting.
        if is_new:
            updates["supervisor_crash_count"] = 0
        self.save_status(**updates)

    def acquire_lock(self) -> None:
        self.run_root.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(descriptor)
            raise TerminalBlocked("another unattended supervisor holds the run lock") from error
        self._lock_descriptor = descriptor

    def close(self) -> None:
        if self._lock_descriptor is not None:
            os.close(self._lock_descriptor)
            self._lock_descriptor = None

    def handle_signal(self, signum: int, _frame: Any) -> None:
        self.received_signal = signum
        self.terminate_child()

    def terminate_child(self) -> None:
        child = self.child
        if child is None or child.poll() is not None:
            return
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def check_stop(self) -> None:
        if self.stop_path.exists() or self.received_signal is not None:
            raise StopRequested("stop requested")

    def _git_revision(self) -> str:
        return command_output(["git", "rev-parse", "HEAD"], self.workspace)

    def _git_clean(self) -> bool:
        return not command_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            self.workspace,
        )

    def _require_git_tracked(self, path: Path, label: str) -> None:
        try:
            relative = path.resolve().relative_to(self.workspace)
        except ValueError as error:
            raise TerminalBlocked(f"{label} must be inside the Git workspace") from error
        command_output(
            ["git", "ls-files", "--error-unmatch", "--", str(relative)],
            self.workspace,
        )

    def _verify_declared(self, entry: Mapping[str, Any], label: str) -> Path:
        path_value = entry.get("path")
        expected = entry.get("sha256")
        if not isinstance(path_value, str) or not isinstance(expected, str):
            raise TerminalBlocked(f"{label} needs path and sha256")
        path = self.resolve(path_value)
        if not path.is_file() or sha256_file(path) != expected:
            raise TerminalBlocked(f"{label} SHA-256 mismatch: {path}")
        return path

    def _check_ac_power(self) -> dict[str, Any]:
        supplies = Path("/sys/class/power_supply")
        mains: list[dict[str, Any]] = []
        if supplies.exists():
            for supply in supplies.iterdir():
                try:
                    kind = (supply / "type").read_text().strip()
                except OSError:
                    continue
                if kind != "Mains":
                    continue
                try:
                    online = int((supply / "online").read_text().strip())
                except (OSError, ValueError):
                    online = -1
                mains.append({"name": supply.name, "online": online})
        if mains and not any(entry["online"] == 1 for entry in mains):
            raise TerminalBlocked("AC power is not online")
        return {"detected": bool(mains), "supplies": mains}

    def validate_preflight(self, require_clean: bool = True) -> dict[str, Any]:
        if self.config.get("forbid_kpp_and_v250", True) is not True:
            raise TerminalBlocked("KPP/v2.5 exclusion must remain enabled")
        self.validate_frozen_policy()
        plan = self._verify_declared(self.section("plan"), "execution plan")
        parent = self.section("parent")
        parent_paths = {
            name: self._verify_declared(parent[name], f"parent.{name}")
            for name in ("model", "model_manifest", "promotion_registry", "gate_report")
            if isinstance(parent.get(name), dict)
        }
        if set(parent_paths) != {
            "model",
            "model_manifest",
            "promotion_registry",
            "gate_report",
        }:
            raise TerminalBlocked("parent artifact declarations are incomplete")
        if parent.get("generation") != 2:
            raise TerminalBlocked("only accepted generation 2 is permitted")
        manifest = load_json(parent_paths["model_manifest"])
        if (
            manifest.get("lineage_id") != LINEAGE_ID
            or manifest.get("generation") != 2
            or manifest.get("model", {}).get("sha256")
            != parent["model"]["sha256"]
        ):
            raise TerminalBlocked("parent model manifest is inconsistent")
        registry = load_json(parent_paths["promotion_registry"])
        if registry.get("lineage_id") != LINEAGE_ID or not any(
            isinstance(entry, dict)
            and entry.get("model_sha256") == parent["model"]["sha256"]
            and entry.get("generation") == 2
            for entry in registry.get("accepted", [])
        ):
            raise TerminalBlocked("parent is not accepted generation 2")

        audit = self.section("audit")
        for name in ("input", "parent_jsonl", "parent_manifest", "analyzer"):
            self._verify_declared(audit[name], f"audit.{name}")
        paired = self.section("paired_match")
        for name in ("wrapper", "openings", "openings_manifest"):
            self._verify_declared(paired[name], f"paired_match.{name}")
        openings_manifest = load_json(self.resolve(paired["openings_manifest"]["path"]))
        opening_entries = {
            entry.get("path"): entry
            for entry in openings_manifest.get("output_files", [])
            if isinstance(entry, dict)
        }
        opening_overlap = openings_manifest.get("stats", {}).get("output_overlap")
        if (
            openings_manifest.get("schema") != "halfkp_mmto_dev_openings_v1_manifest"
            or len(openings_manifest.get("output_files", [])) != 1
            or openings_manifest.get("derivation", {})
            .get("request", {})
            .get("seed")
            != 20260731
            or openings_manifest.get("derivation", {})
            .get("request", {})
            .get("pairs")
            != paired["pairs"]
            or opening_entries.get("openings.jsonl", {}).get("sha256")
            != paired["openings"]["sha256"]
            or opening_entries.get("openings.jsonl", {}).get("records")
            != paired["pairs"]
            or openings_manifest.get("stats", {}).get("selected", {}).get("pairs")
            != paired["pairs"]
            or openings_manifest.get("stats", {})
            .get("selected", {})
            .get("unique_exact_sfen")
            != paired["pairs"]
            or openings_manifest.get("stats", {})
            .get("selected", {})
            .get("unique_opening_families")
            != paired["pairs"]
            or not isinstance(opening_overlap, dict)
            or set(opening_overlap)
            != {
                "train_exact_sfen",
                "test_exact_sfen",
                "final_reserve_exact_sfen",
            }
            or any(value != 0 for value in opening_overlap.values())
        ):
            raise TerminalBlocked("paired opening manifest is inconsistent")
        phases = self.section("phases")
        phase_a = phases.get("A")
        if not isinstance(phase_a, dict):
            raise TerminalBlocked("phase A config is missing")
        self._verify_declared(phase_a["candidate_model"], "phase A candidate")
        for phase_name in ("B", "C"):
            phase = phases.get(phase_name)
            if not isinstance(phase, dict):
                raise TerminalBlocked(f"phase {phase_name} config is missing")
            dataset = phase.get("dataset")
            if not isinstance(dataset, dict):
                raise TerminalBlocked(f"phase {phase_name} dataset is missing")
            for name in ("train", "valid", "manifest"):
                self._verify_declared(dataset[name], f"phase {phase_name} dataset.{name}")
            declared_manifest = load_json(self.resolve(dataset["manifest"]["path"]))
            outputs = {
                entry.get("path"): entry
                for entry in declared_manifest.get("output_files", [])
                if isinstance(entry, dict)
            }
            if (
                declared_manifest.get("schema") != "rank_value_dataset_v4_manifest"
                or declared_manifest.get("min_player_rate") != 4000
                or declared_manifest.get("min_opponent_rate") != 4000
                or outputs.get("train.jsonl", {}).get("sha256")
                != dataset["train"]["sha256"]
                or outputs.get("valid.jsonl", {}).get("sha256")
                != dataset["valid"]["sha256"]
            ):
                raise TerminalBlocked(f"phase {phase_name} dataset manifest mismatch")

        build = self.section("build")
        expected_build = [
            "cargo",
            "build",
            "--release",
            "--locked",
            "--features",
            "halfkp64,training-tools",
            "--bin",
            "halfkp_mmto_train",
            "--bin",
            "halfkp_mmto_audit",
            "--bin",
            "halfkp_promotion_match",
        ]
        if build.get("command") != expected_build or build.get("artifacts") != {
            "trainer": "target/release/halfkp_mmto_train",
            "audit": "target/release/halfkp_mmto_audit",
            "match": "target/release/halfkp_promotion_match",
        }:
            raise TerminalBlocked("release build policy changed")
        if build.get("git_revision") != "lock_on_first_run":
            raise TerminalBlocked("build.git_revision must be lock_on_first_run")
        critical_paths = {
            "unattended config": self.config_path,
            "unattended supervisor": Path(__file__).resolve(),
            "paired match wrapper": self.resolve(paired["wrapper"]["path"]),
            "service manager": self.workspace
            / "scripts/manage_halfkp_mmto_unattended.sh",
            "persistent service unit": self.workspace
            / "scripts/systemd/halfkp-mmto-unattended-20260806.service",
        }
        for label, path in critical_paths.items():
            self._require_git_tracked(path, label)
        if require_clean and build.get("require_clean_git", True) and not self._git_clean():
            raise TerminalBlocked("tracked Git worktree is not clean")
        minimum_gib = self.positive_int(
            self.section("service").get("minimum_free_gib"), "minimum_free_gib"
        )
        usage = shutil.disk_usage(self.run_root.parent)
        if usage.free < minimum_gib * 1024**3:
            raise TerminalBlocked(
                f"free disk is below {minimum_gib} GiB: {usage.free / 1024**3:.1f} GiB"
            )
        power = self._check_ac_power()
        return {
            "schema": "halfkp_mmto_unattended_preflight_v1",
            "checked_at": utc_now(),
            "config_sha256": self.config_sha,
            "plan": str(plan),
            "git_revision": self._git_revision(),
            "git_clean": self._git_clean(),
            "free_gib": usage.free / 1024**3,
            "ac_power": power,
            "parent_sha256": parent["model"]["sha256"],
        }

    def _log_tail_contains(self, path: Path, patterns: Sequence[str]) -> bool:
        if not path.exists():
            return False
        with path.open("rb") as source:
            source.seek(0, os.SEEK_END)
            size = source.tell()
            source.seek(max(0, size - 256 * 1024))
            text = source.read().decode("utf-8", errors="replace")
        return any(pattern in text for pattern in patterns)

    def run_command(
        self,
        step: str,
        command: Sequence[str] | Callable[[], Sequence[str]],
        phase: str | None = None,
        outer: int | None = None,
        deterministic_patterns: Sequence[str] = (),
        environment: Mapping[str, str] | None = None,
    ) -> Path:
        logs = self.run_root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        safe_step = step.replace("/", "_").replace(" ", "_")
        attempts = dict(self.status.get("attempts", {}))
        while attempts.get(step, 0) < self.max_attempts:
            self.check_stop()
            attempt = attempts.get(step, 0) + 1
            log_path = logs / f"{safe_step}.attempt-{attempt:02d}.log"
            attempt_command = list(command() if callable(command) else command)
            if not attempt_command or not all(
                isinstance(argument, str) for argument in attempt_command
            ):
                raise TerminalBlocked(f"step {step} produced an invalid command")
            attempts[step] = attempt
            self.save_status(
                status="running",
                current_phase=phase,
                current_step=step,
                current_outer=outer,
                attempts=attempts,
                last_error=None,
            )
            env = os.environ.copy()
            if environment:
                env.update(environment)
            # One file contains one attempt only.  Deterministic failure
            # classification must never match text from an older attempt.
            with log_path.open("w", encoding="utf-8") as log:
                log.write(
                    f"[{utc_now()}] attempt={attempt} command="
                    + json.dumps(attempt_command, ensure_ascii=False)
                    + "\n"
                )
                log.flush()
                os.fsync(log.fileno())
                try:
                    self.child = subprocess.Popen(
                        attempt_command,
                        cwd=self.workspace,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        start_new_session=True,
                    )
                except OSError as error:
                    self.child = None
                    raise TerminalBlocked(f"cannot start {step}: {error}") from error
                self.save_status(child_pid=self.child.pid)
                while self.child.poll() is None:
                    try:
                        self.child.wait(timeout=self.heartbeat_seconds)
                    except subprocess.TimeoutExpired:
                        self.save_status(child_pid=self.child.pid)
                        if self.stop_path.exists() or self.received_signal is not None:
                            self.terminate_child()
                return_code = self.child.returncode
                self.child = None
                self.save_status(child_pid=None)
            if return_code == 0:
                attempts[step] = 0
                self.save_status(attempts=attempts)
                self.completed_step(
                    step,
                    {
                        "command": attempt_command,
                        "log": str(log_path),
                        "attempt": attempt,
                    },
                )
                return log_path
            if self.stop_path.exists() or self.received_signal is not None:
                raise StopRequested("command interrupted by stop request")
            deterministic = self._log_tail_contains(log_path, deterministic_patterns)
            message = f"{step} exit={return_code} attempt={attempt}/{self.max_attempts}"
            self.save_status(last_error=message)
            if deterministic or attempt >= self.max_attempts:
                raise CommandFailure(step, return_code, log_path, deterministic)
            time.sleep(self.retry_seconds)
        raise TerminalBlocked(f"attempt accounting exhausted for {step}")

    def ensure_identity(self) -> dict[str, Any]:
        build = self.section("build")
        artifacts_config = build.get("artifacts")
        if not isinstance(artifacts_config, dict):
            raise TerminalBlocked("build.artifacts is missing")
        if self.identity_path.exists():
            identity = load_json(self.identity_path)
            if (
                identity.get("schema") != IDENTITY_SCHEMA
                or identity.get("config_sha256") != self.config_sha
                or identity.get("git_revision") != self._git_revision()
            ):
                raise TerminalBlocked("run identity does not match config or Git revision")
            declared_binaries = identity.get("binaries")
            if (
                not isinstance(declared_binaries, dict)
                or set(declared_binaries) != set(artifacts_config)
            ):
                raise TerminalBlocked("locked binary identity has the wrong file set")
            locked_dir = self.run_root / "locked_binaries"
            for name, value in declared_binaries.items():
                if not isinstance(value, dict):
                    raise TerminalBlocked("invalid binary identity")
                path_value = value.get("path")
                if not isinstance(path_value, str):
                    raise TerminalBlocked("invalid locked binary path")
                declared_path = Path(path_value)
                if not declared_path.is_absolute():
                    declared_path = self.workspace / declared_path
                expected_path = locked_dir / Path(artifacts_config[name]).name
                if declared_path != expected_path or declared_path.is_symlink():
                    raise TerminalBlocked(f"unsafe locked binary path: {name}")
                path = declared_path
                if (
                    not path.is_file()
                    or not os.access(path, os.X_OK)
                    or stat.S_IMODE(path.stat().st_mode) != 0o555
                    or sha256_file(path) != value.get("sha256")
                    or path.stat().st_size != value.get("bytes")
                ):
                    raise TerminalBlocked(f"locked binary changed: {name}")
            supervisor = identity.get("supervisor")
            if (
                not isinstance(supervisor, dict)
                or supervisor.get("sha256") != sha256_file(Path(__file__).resolve())
                or supervisor.get("bytes") != Path(__file__).resolve().stat().st_size
            ):
                raise TerminalBlocked("unattended supervisor changed after identity lock")
            return identity

        command = build.get("command")
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            raise TerminalBlocked("build.command must be a string array")
        revision_before = self._git_revision()
        if not self._git_clean():
            raise TerminalBlocked("Git worktree changed before release build")
        self.run_command(
            "build_release_binaries",
            command,
            environment={"RUST_FONTCONFIG_DLOPEN": "1"},
        )
        if self._git_revision() != revision_before or not self._git_clean():
            raise TerminalBlocked("Git revision or worktree changed during release build")
        binaries: dict[str, Any] = {}
        locked_dir = self.run_root / "locked_binaries"
        locked_dir.mkdir(parents=True, exist_ok=True)
        for name, path_value in artifacts_config.items():
            if not isinstance(path_value, str):
                raise TerminalBlocked(f"build artifact {name} is not a path")
            path = self.resolve(path_value)
            metadata = artifact_metadata(path)
            if not os.access(path, os.X_OK):
                raise TerminalBlocked(f"build artifact is not executable: {path}")
            locked_path = locked_dir / path.name
            if locked_path.exists():
                if (
                    locked_path.is_symlink()
                    or not locked_path.is_file()
                    or sha256_file(locked_path) != metadata["sha256"]
                    or not os.access(locked_path, os.X_OK)
                    or stat.S_IMODE(locked_path.stat().st_mode) != 0o555
                ):
                    raise TerminalBlocked(f"stale locked binary differs: {locked_path}")
            else:
                atomic_copy(path, locked_path, 0o555)
            source_after = artifact_metadata(path)
            locked_metadata = artifact_metadata(locked_path)
            if (
                source_after["sha256"] != metadata["sha256"]
                or source_after["bytes"] != metadata["bytes"]
                or locked_metadata["sha256"] != metadata["sha256"]
                or locked_metadata["bytes"] != metadata["bytes"]
            ):
                raise TerminalBlocked(f"build artifact changed while locking: {path}")
            binaries[name] = locked_metadata
        identity = {
            "schema": IDENTITY_SCHEMA,
            "created_at": utc_now(),
            "config_path": str(self.config_path),
            "config_sha256": self.config_sha,
            "git_revision": self._git_revision(),
            "supervisor": artifact_metadata(Path(__file__).resolve()),
            "binaries": binaries,
        }
        atomic_json(self.identity_path, identity)
        self.record_artifact("run_identity", self.identity_path)
        return identity

    def binary(self, identity: Mapping[str, Any], name: str) -> Path:
        try:
            return self.resolve(identity["binaries"][name]["path"])
        except (KeyError, TypeError) as error:
            raise TerminalBlocked(f"run identity lacks binary {name}") from error

    def parent_path(self, name: str) -> Path:
        return self.resolve(self.section("parent")[name]["path"])

    def verify_phase_inputs(
        self, phase_name: str, phase: Mapping[str, Any]
    ) -> None:
        parent = self.section("parent")
        for name in ("model", "model_manifest", "promotion_registry", "gate_report"):
            self._verify_declared(parent[name], f"parent.{name}")
        dataset = phase.get("dataset")
        if not isinstance(dataset, dict):
            raise TerminalBlocked(f"phase {phase_name} dataset is missing")
        for name in ("train", "valid", "manifest"):
            self._verify_declared(dataset[name], f"phase {phase_name} dataset.{name}")

    def write_candidate_manifest(
        self,
        path: Path,
        model: Path,
        stage: str,
        outer: int | None,
        declared_model_path: Path | None = None,
    ) -> None:
        parent = self.section("parent")
        model_metadata = artifact_metadata(model)
        if declared_model_path is not None:
            model_metadata["path"] = str(declared_model_path)
        value = {
            "schema": "halfkp_mmto_candidate_v1",
            "lineage_id": LINEAGE_ID,
            "generation": 3,
            "status": "diagnostic_candidate",
            "stage": stage,
            "completed_outer": outer,
            "parent_model_sha256": parent["model"]["sha256"],
            "model": model_metadata,
        }
        if path.exists():
            existing = load_json(path)
            if existing != value:
                raise TerminalBlocked(f"candidate manifest mismatch: {path}")
            return
        atomic_json(path, value)

    def train_lock_payload(
        self,
        identity: Mapping[str, Any],
        phase_name: str,
        phase: Mapping[str, Any],
        learning_rate: float,
    ) -> dict[str, Any]:
        trainer = self.section("trainer")
        return {
            "schema": TRAIN_LOCK_SCHEMA,
            "config_sha256": self.config_sha,
            "git_revision": identity["git_revision"],
            "trainer_sha256": identity["binaries"]["trainer"]["sha256"],
            "phase": phase_name,
            "parent_generation": self.section("parent")["generation"],
            "parent_model_sha256": self.section("parent")["model"]["sha256"],
            "dataset": phase["dataset"],
            "hyperparameters": {
                "learning_rate": learning_rate,
                "output_learning_rate_ratio": trainer["output_learning_rate_ratio"],
                "gradient_clip_norm": trainer["gradient_clip_norm"],
                "tau_cp": trainer["tau_cp"],
                "search_block_roots": trainer["search_block_roots"],
                "jobs": trainer["jobs"],
                "pair_batch_size": trainer["pair_batch_size"],
                "pv_probe_roots": trainer["pv_probe_roots"],
                "max_feature_delta_per_outer": trainer[
                    "max_feature_delta_per_outer"
                ],
                "max_output_delta_per_outer": trainer[
                    "max_output_delta_per_outer"
                ],
                "max_train_roots": phase.get("max_train_roots"),
                "max_valid_roots": phase["max_valid_roots"],
            },
            "outer_policy": "absolute_target_one_milestone_at_a_time",
        }

    def ensure_train_lock(
        self,
        output_dir: Path,
        payload: Mapping[str, Any],
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "unattended_train_lock.json"
        if path.exists():
            if load_json(path) != payload:
                raise TerminalBlocked(f"training resume configuration changed: {path}")
            return
        if any((output_dir / name).exists() for name in ("model.bin", "optimizer.bin", "checkpoint.json")):
            raise TerminalBlocked("training artifacts exist without unattended train lock")
        atomic_json(path, payload)

    def trainer_command(
        self,
        identity: Mapping[str, Any],
        phase: Mapping[str, Any],
        output_dir: Path,
        target_outer: int,
        learning_rate: float,
        resume: bool,
    ) -> list[str]:
        dataset = phase["dataset"]
        trainer = self.section("trainer")
        command = [
            str(self.binary(identity, "trainer")),
            "--train",
            str(self.resolve(dataset["train"]["path"])),
            "--valid",
            str(self.resolve(dataset["valid"]["path"])),
            "--dataset-manifest",
            str(self.resolve(dataset["manifest"]["path"])),
            "--expected-dataset-manifest-sha256",
            dataset["manifest"]["sha256"],
            "--init",
            str(self.parent_path("model")),
            "--model-manifest",
            str(self.parent_path("model_manifest")),
            "--promotion-registry",
            str(self.parent_path("promotion_registry")),
            "--gate-report",
            str(self.parent_path("gate_report")),
            "--parent-generation",
            str(self.section("parent")["generation"]),
            "--output-dir",
            str(output_dir),
            "--outer-iterations",
            str(target_outer),
            "--search-block-roots",
            str(trainer["search_block_roots"]),
            "--jobs",
            str(trainer["jobs"]),
            "--pair-batch-size",
            str(trainer["pair_batch_size"]),
            "--learning-rate",
            str(learning_rate),
            "--output-learning-rate-ratio",
            str(trainer["output_learning_rate_ratio"]),
            "--gradient-clip-norm",
            str(trainer["gradient_clip_norm"]),
            "--tau-cp",
            str(trainer["tau_cp"]),
            "--max-feature-delta-per-outer",
            str(trainer["max_feature_delta_per_outer"]),
            "--max-output-delta-per-outer",
            str(trainer["max_output_delta_per_outer"]),
            "--max-valid-roots",
            str(phase["max_valid_roots"]),
            "--pv-probe-roots",
            str(trainer["pv_probe_roots"]),
        ]
        if phase.get("max_train_roots") is not None:
            command.extend(["--max-train-roots", str(phase["max_train_roots"])])
        for argument in trainer.get("extra_args", []):
            command.append(str(argument))
        if resume:
            command.append("--resume")
        return command

    def verify_training_output(
        self,
        output_dir: Path,
        target_outer: int,
        dataset_sha: str,
        phase: Mapping[str, Any],
    ) -> dict[str, Any]:
        checkpoint_path = output_dir / "checkpoint.json"
        metrics_path = output_dir / "metrics" / f"outer_{target_outer:04d}.json"
        run_manifest_path = output_dir / "run_manifest.json"
        model_path = output_dir / "model.bin"
        optimizer_path = output_dir / "optimizer.bin"
        checkpoint = load_json(checkpoint_path)
        metrics = load_json(metrics_path)
        run_manifest = load_json(run_manifest_path)
        model_sha = sha256_file(model_path)
        optimizer_sha = sha256_file(optimizer_path)
        dataset_manifest = load_json(self.resolve(phase["dataset"]["manifest"]["path"]))
        dataset_outputs = {
            entry.get("path"): entry
            for entry in dataset_manifest.get("output_files", [])
            if isinstance(entry, dict)
        }
        expected_train_roots = phase.get("max_train_roots")
        if expected_train_roots is None:
            expected_train_roots = dataset_outputs.get("train.jsonl", {}).get("records")
        expected_valid_roots = phase["max_valid_roots"]
        expected_parent = self.section("parent")["model"]["sha256"]
        if (
            checkpoint.get("schema_version") != 1
            or checkpoint.get("completed_outer") != target_outer
            or checkpoint.get("init_model_sha256") != expected_parent
            or checkpoint.get("dataset_manifest_sha256") != dataset_sha
            or checkpoint.get("model_sha256") != model_sha
            or checkpoint.get("optimizer_sha256") != optimizer_sha
        ):
            raise TerminalBlocked(f"checkpoint verification failed: {output_dir}")
        if (
            metrics.get("outer") != target_outer
            or metrics.get("model_sha256") != model_sha
            or metrics.get("optimizer_sha256") != optimizer_sha
            or metrics.get("train", {}).get("roots") != expected_train_roots
            or metrics.get("valid", {}).get("roots") != expected_valid_roots
            or not isinstance(metrics.get("train", {}).get("pairs"), int)
            or metrics.get("train", {}).get("pairs", 0) <= 0
            or not isinstance(metrics.get("valid", {}).get("pairs"), int)
            or metrics.get("valid", {}).get("pairs", 0) <= 0
            or metrics.get("train", {}).get("rejected_bounds") != 0
            or metrics.get("valid", {}).get("rejected_bounds") != 0
        ):
            raise TerminalBlocked(f"outer metrics verification failed: {metrics_path}")
        if (
            run_manifest.get("completed_outer") != target_outer
            or run_manifest.get("dataset_manifest_sha256") != dataset_sha
            or run_manifest.get("output_model", {}).get("sha256") != model_sha
            or run_manifest.get("optimizer", {}).get("sha256") != optimizer_sha
        ):
            raise TerminalBlocked(f"run manifest verification failed: {run_manifest_path}")
        require_finite(metrics, "metrics")
        require_finite(run_manifest, "run_manifest")
        paths = {
            "model.bin": model_path,
            "optimizer.bin": optimizer_path,
            "checkpoint.json": checkpoint_path,
            f"metrics/outer_{target_outer:04d}.json": metrics_path,
            "run_manifest.json": run_manifest_path,
        }
        return {
            "checkpoint": checkpoint,
            "metrics": metrics,
            "run_manifest": run_manifest,
            "model_sha256": model_sha,
            "optimizer_sha256": optimizer_sha,
            "paths": paths,
            "path_sha256": {
                name: sha256_file(path) for name, path in paths.items()
            },
        }

    def run_training_outer(
        self,
        identity: Mapping[str, Any],
        phase_name: str,
        phase: Mapping[str, Any],
        output_dir: Path,
        target_outer: int,
        learning_rate: float,
        run_variant: str = "primary",
    ) -> dict[str, Any]:
        lock = self.train_lock_payload(identity, phase_name, phase, learning_rate)
        self.ensure_train_lock(output_dir, lock)
        checkpoint_path = output_dir / "checkpoint.json"
        if checkpoint_path.exists():
            checkpoint = load_json(checkpoint_path)
            completed_outer = checkpoint.get("completed_outer")
            if not isinstance(completed_outer, int) or completed_outer > target_outer:
                raise TerminalBlocked("checkpoint is ahead of requested outer")
        else:
            completed_outer = 0
        if completed_outer < target_outer:
            def command() -> Sequence[str]:
                # A crash during the initial model/optimizer write can leave
                # files without a commit checkpoint.  Such files have no
                # resumable state and are safe to regenerate under the frozen
                # unattended train lock.
                if not checkpoint_path.exists():
                    removed = False
                    for name in ("model.bin", "optimizer.bin"):
                        path = output_dir / name
                        if path.exists() or path.is_symlink():
                            if not path.is_file() or path.is_symlink():
                                raise TerminalBlocked(
                                    f"unsafe uncommitted trainer artifact: {path}"
                                )
                            path.unlink()
                            removed = True
                    if removed:
                        fsync_directory(output_dir)
                return self.trainer_command(
                    identity,
                    phase,
                    output_dir,
                    target_outer,
                    learning_rate,
                    checkpoint_path.exists(),
                )

            self.run_command(
                f"phase_{phase_name.lower()}_{run_variant}_train_outer_{target_outer:04d}",
                command,
                phase_name,
                target_outer,
                deterministic_patterns=(
                    "outer delta limit exceeded",
                    "parameter bound touched",
                    "provenance mismatch",
                    "SHA mismatch",
                ),
            )
        self.verify_phase_inputs(phase_name, phase)
        return self.verify_training_output(
            output_dir,
            target_outer,
            phase["dataset"]["manifest"]["sha256"],
            phase,
        )

    def verify_archive(self, archive_dir: Path, phase: str, outer: int) -> dict[str, Any]:
        manifest_path = archive_dir / "archive_manifest.json"
        manifest = load_json(manifest_path)
        if (
            manifest.get("schema") != ARCHIVE_SCHEMA
            or manifest.get("config_sha256") != self.config_sha
            or manifest.get("phase") != phase
            or manifest.get("outer") != outer
        ):
            raise TerminalBlocked(f"archive provenance mismatch: {archive_dir}")
        files = manifest.get("files")
        if not isinstance(files, list) or not files:
            raise TerminalBlocked("archive manifest has no files")
        expected_names = {
            "model.bin",
            "optimizer.bin",
            "checkpoint.json",
            f"metrics/outer_{outer:04d}.json",
            "run_manifest.json",
            "model.bin.manifest.json",
        }
        if (
            len(files) != len(expected_names)
            or {entry.get("path") for entry in files if isinstance(entry, dict)}
            != expected_names
        ):
            raise TerminalBlocked("archive manifest file set is incomplete or unexpected")
        for entry in files:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise TerminalBlocked("invalid archive file metadata")
            relative = Path(entry["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise TerminalBlocked("unsafe archive relative path")
            path = archive_dir / relative
            if (
                not path.is_file()
                or path.stat().st_size != entry.get("bytes")
                or sha256_file(path) != entry.get("sha256")
            ):
                raise TerminalBlocked(f"archive artifact mismatch: {path}")
        model_sha = sha256_file(archive_dir / "model.bin")
        optimizer_sha = sha256_file(archive_dir / "optimizer.bin")
        if (
            manifest.get("model_sha256") != model_sha
            or manifest.get("optimizer_sha256") != optimizer_sha
        ):
            raise TerminalBlocked("archive top-level model or optimizer SHA mismatch")
        phase_config = self.section("phases").get(phase)
        if not isinstance(phase_config, dict) or not isinstance(
            phase_config.get("dataset"), dict
        ):
            raise TerminalBlocked("archive phase has no frozen dataset")
        dataset_sha = phase_config["dataset"]["manifest"]["sha256"]
        checkpoint = load_json(archive_dir / "checkpoint.json")
        metrics = load_json(archive_dir / f"metrics/outer_{outer:04d}.json")
        run_manifest = load_json(archive_dir / "run_manifest.json")
        parent_sha = self.section("parent")["model"]["sha256"]
        expected_previous_sha = parent_sha
        if outer > 1:
            previous_archive = archive_dir.parent / f"outer_{outer - 1:04d}"
            self.verify_archive(previous_archive, phase, outer - 1)
            expected_previous_sha = sha256_file(previous_archive / "model.bin")
        if (
            checkpoint.get("schema_version") != 1
            or checkpoint.get("completed_outer") != outer
            or checkpoint.get("init_model_sha256") != parent_sha
            or checkpoint.get("dataset_manifest_sha256") != dataset_sha
            or checkpoint.get("previous_model_sha256") != expected_previous_sha
            or checkpoint.get("model_sha256") != model_sha
            or checkpoint.get("optimizer_sha256") != optimizer_sha
        ):
            raise TerminalBlocked("archive checkpoint semantics mismatch")
        if (
            metrics.get("outer") != outer
            or metrics.get("model_sha256") != model_sha
            or metrics.get("optimizer_sha256") != optimizer_sha
            or metrics.get("train", {}).get("rejected_bounds") != 0
            or metrics.get("valid", {}).get("rejected_bounds") != 0
        ):
            raise TerminalBlocked("archive metrics semantics mismatch")
        if (
            run_manifest.get("completed_outer") != outer
            or run_manifest.get("dataset_manifest_sha256") != dataset_sha
            or run_manifest.get("output_model", {}).get("sha256") != model_sha
            or run_manifest.get("optimizer", {}).get("sha256") != optimizer_sha
        ):
            raise TerminalBlocked("archive run manifest semantics mismatch")
        require_finite(metrics, "archive_metrics")
        require_finite(run_manifest, "archive_run_manifest")
        candidate_manifest = load_json(archive_dir / "model.bin.manifest.json")
        if (
            candidate_manifest.get("lineage_id") != LINEAGE_ID
            or candidate_manifest.get("stage") != phase
            or candidate_manifest.get("completed_outer") != outer
            or candidate_manifest.get("parent_model_sha256") != parent_sha
            or candidate_manifest.get("model", {}).get("sha256")
            != model_sha
            or candidate_manifest.get("model", {}).get("bytes")
            != (archive_dir / "model.bin").stat().st_size
            or candidate_manifest.get("model", {}).get("path")
            != str(archive_dir / "model.bin")
        ):
            raise TerminalBlocked("archive candidate manifest mismatch")
        return manifest

    def archive_outer(
        self,
        phase: str,
        outer: int,
        training: Mapping[str, Any],
    ) -> Path:
        archive_dir = self.run_root / "archives" / f"phase-{phase.lower()}" / f"outer_{outer:04d}"
        if archive_dir.exists():
            existing = self.verify_archive(archive_dir, phase, outer)
            if (
                existing.get("model_sha256") != training.get("model_sha256")
                or existing.get("optimizer_sha256")
                != training.get("optimizer_sha256")
            ):
                raise TerminalBlocked("existing archive differs from live checkpoint")
            archived_files = {
                entry["path"]: entry["sha256"] for entry in existing["files"]
            }
            expected_hashes = training.get("path_sha256")
            if not isinstance(expected_hashes, dict) or any(
                archived_files.get(name) != digest
                for name, digest in expected_hashes.items()
            ):
                raise TerminalBlocked(
                    "existing archive artifacts differ from live checkpoint"
                )
            return archive_dir
        archive_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{archive_dir.name}.tmp-", dir=archive_dir.parent)
        )
        try:
            copied: list[dict[str, Any]] = []
            for relative_name, source in training["paths"].items():
                destination = staging / relative_name
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source.open("rb") as input_file, destination.open("xb") as output:
                    shutil.copyfileobj(input_file, output, 1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
                fsync_directory(destination.parent)
                copied_sha = sha256_file(destination)
                if copied_sha != training["path_sha256"][relative_name]:
                    raise TerminalBlocked(
                        f"training artifact changed during archive copy: {source}"
                    )
                copied.append(
                    {
                        "path": relative_name,
                        "sha256": copied_sha,
                        "bytes": destination.stat().st_size,
                    }
                )
            candidate_path = staging / "model.bin.manifest.json"
            self.write_candidate_manifest(
                candidate_path,
                staging / "model.bin",
                phase,
                outer,
                archive_dir / "model.bin",
            )
            copied.append(
                {
                    "path": candidate_path.name,
                    "sha256": sha256_file(candidate_path),
                    "bytes": candidate_path.stat().st_size,
                }
            )
            manifest = {
                "schema": ARCHIVE_SCHEMA,
                "created_at": utc_now(),
                "config_sha256": self.config_sha,
                "phase": phase,
                "outer": outer,
                "model_sha256": training["model_sha256"],
                "optimizer_sha256": training["optimizer_sha256"],
                "files": sorted(copied, key=lambda entry: entry["path"]),
            }
            atomic_json(staging / "archive_manifest.json", manifest)
            fsync_directory(staging)
            os.replace(staging, archive_dir)
            fsync_directory(archive_dir.parent)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        self.verify_archive(archive_dir, phase, outer)
        self.record_artifact(
            f"phase_{phase.lower()}_outer_{outer:04d}_archive",
            archive_dir / "archive_manifest.json",
        )
        return archive_dir

    def audit_paths(self, phase: str, outer: int) -> tuple[Path, Path, Path]:
        directory = self.run_root / "evaluations" / f"phase-{phase.lower()}-outer-{outer:04d}"
        return directory / "audit.jsonl", directory / "comparison.json", directory / "audit_completed.json"

    def verify_audit_marker(
        self, marker_path: Path, archive_model: Path, phase: str, outer: int
    ) -> dict[str, Any]:
        marker = load_json(marker_path)
        if (
            marker.get("schema") != AUDIT_MARKER_SCHEMA
            or marker.get("config_sha256") != self.config_sha
            or marker.get("phase") != phase
            or marker.get("outer") != outer
            or marker.get("model_sha256") != sha256_file(archive_model)
        ):
            raise TerminalBlocked("audit completed marker provenance mismatch")
        entries = marker.get("files")
        expected_names = {
            "audit.jsonl",
            "audit.jsonl.manifest.json",
            "comparison.json",
        }
        if not isinstance(entries, list) or len(entries) != len(expected_names):
            raise TerminalBlocked("audit completed marker file set is incomplete")
        seen: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise TerminalBlocked("audit completed marker file entry is invalid")
            name = entry.get("path")
            if (
                not isinstance(name, str)
                or not name
                or Path(name).name != name
                or name not in expected_names
                or name in seen
            ):
                raise TerminalBlocked("audit completed marker file path is unsafe")
            seen.add(name)
            path = marker_path.parent / name
            if (
                not path.is_file()
                or sha256_file(path) != entry.get("sha256")
                or path.stat().st_size != entry.get("bytes")
            ):
                raise TerminalBlocked(f"audit artifact mismatch: {path}")
        if seen != expected_names:
            raise TerminalBlocked("audit completed marker file set is incomplete")
        audit_config = self.section("audit")
        for name in ("input", "parent_jsonl", "parent_manifest", "analyzer"):
            self._verify_declared(audit_config[name], f"audit.{name}")
        sidecar = load_json(marker_path.parent / "audit.jsonl.manifest.json")
        sidecar_tau = sidecar.get("tau_cp")
        if (
            sidecar.get("schema") != "halfkp_mmto_root_audit_v1"
            or sidecar.get("model_sha256") != sha256_file(archive_model)
            or sidecar.get("input_sha256") != audit_config["input"]["sha256"]
            or sidecar.get("roots") != audit_config["roots"]
            or sidecar.get("games") != audit_config["roots"]
            or sidecar.get("rejected_bounds") != 0
            or not isinstance(sidecar_tau, (int, float))
            or isinstance(sidecar_tau, bool)
            or float(sidecar_tau) != float(self.section("trainer")["tau_cp"])
        ):
            raise TerminalBlocked("committed audit sidecar semantics mismatch")
        comparison = load_json(marker_path.parent / "comparison.json")
        candidate = comparison.get("comparisons", {}).get("candidate")
        if not isinstance(candidate, dict):
            raise TerminalBlocked("committed audit comparison lacks candidate")
        summary = candidate.get("summary", {})
        versus = candidate.get("versus_parent", {})
        interval = versus.get("loss_delta_bootstrap_95")
        if (
            summary.get("roots") != audit_config["roots"]
            or summary.get("games") != audit_config["roots"]
            or summary.get("rejected_bounds") != 0
            or versus.get("roots") != audit_config["roots"]
            or not isinstance(interval, list)
            or len(interval) != 2
        ):
            raise TerminalBlocked("committed audit comparison is incomplete")
        require_finite(comparison, "committed_audit_comparison")
        loss_delta = float(versus["loss_delta_mean"])
        ci_upper = float(interval[1])
        if (
            marker.get("loss_delta_mean") != loss_delta
            or marker.get("loss_delta_ci95_upper") != ci_upper
            or marker.get("offline_improves") is not (loss_delta < 0.0)
            or marker.get("offline_strong_improvement") is not (ci_upper < 0.0)
            or marker.get("qnodes_per_root") != summary.get("qnodes_per_root")
        ):
            raise TerminalBlocked("audit completed marker semantics mismatch")
        return marker

    def run_audit(
        self,
        identity: Mapping[str, Any],
        archive_dir: Path,
        phase: str,
        outer: int,
    ) -> dict[str, Any]:
        archive_model = archive_dir / "model.bin"
        audit_jsonl, comparison_path, marker_path = self.audit_paths(phase, outer)
        audit_jsonl.parent.mkdir(parents=True, exist_ok=True)
        if marker_path.exists():
            return self.verify_audit_marker(marker_path, archive_model, phase, outer)
        audit_config = self.section("audit")
        for name in ("input", "parent_jsonl", "parent_manifest", "analyzer"):
            self._verify_declared(audit_config[name], f"audit.{name}")
        sidecar = audit_jsonl.with_name(audit_jsonl.name + ".manifest.json")
        # The Rust sidecar does not attest the output JSONL.  The completed
        # marker is the sole commit point, so marker-less files are partial.
        for path in (audit_jsonl, sidecar, comparison_path):
            if path.exists():
                path.unlink()
        command = [
            str(self.binary(identity, "audit")),
            "--model",
            str(archive_model),
            "--input",
            str(self.resolve(audit_config["input"]["path"])),
            "--output",
            str(audit_jsonl),
            "--tau-cp",
            str(self.section("trainer")["tau_cp"]),
            "--block-roots",
            str(audit_config["block_roots"]),
            "--jobs",
            str(audit_config["jobs"]),
            "--max-roots",
            str(audit_config["roots"]),
        ]
        self.run_command(
            f"phase_{phase.lower()}_audit_outer_{outer:04d}",
            command,
            phase,
            outer,
            deterministic_patterns=("bound leaves", "invalid", "mismatch"),
        )
        declared = load_json(sidecar)
        declared_tau = declared.get("tau_cp")
        if (
            declared.get("schema") != "halfkp_mmto_root_audit_v1"
            or declared.get("model_sha256") != sha256_file(archive_model)
            or declared.get("input_sha256") != audit_config["input"]["sha256"]
            or not isinstance(declared_tau, (int, float))
            or isinstance(declared_tau, bool)
            or not math.isfinite(float(declared_tau))
            or float(declared_tau) != float(self.section("trainer")["tau_cp"])
            or declared.get("roots") != audit_config["roots"]
            or declared.get("games") != audit_config["roots"]
            or declared.get("rejected_bounds") != 0
        ):
            raise TerminalBlocked("fresh audit sidecar verification failed")
        analyzer_command = [
            "python3",
            str(self.resolve(audit_config["analyzer"]["path"])),
            "--parent",
            str(self.resolve(audit_config["parent_jsonl"]["path"])),
            "--candidate",
            "candidate",
            str(audit_jsonl),
            "--output",
            str(comparison_path),
            "--seed",
            str(audit_config["seed"] + outer),
        ]
        self.run_command(
            f"phase_{phase.lower()}_analyze_audit_outer_{outer:04d}",
            analyzer_command,
            phase,
            outer,
        )
        comparison = load_json(comparison_path)
        candidate = comparison.get("comparisons", {}).get("candidate")
        if not isinstance(candidate, dict):
            raise TerminalBlocked("audit comparison lacks candidate")
        summary = candidate.get("summary", {})
        versus = candidate.get("versus_parent", {})
        interval = versus.get("loss_delta_bootstrap_95")
        if (
            summary.get("roots") != audit_config["roots"]
            or summary.get("games") != audit_config["roots"]
            or summary.get("rejected_bounds") != 0
            or versus.get("roots") != audit_config["roots"]
            or not isinstance(interval, list)
            or len(interval) != 2
        ):
            raise TerminalBlocked("audit comparison is incomplete")
        require_finite(comparison, "audit_comparison")
        loss_delta = float(versus["loss_delta_mean"])
        ci_upper = float(interval[1])
        for name in ("input", "parent_jsonl", "parent_manifest", "analyzer"):
            self._verify_declared(audit_config[name], f"audit.{name}")
        self.verify_archive(archive_dir, phase, outer)
        for path in (audit_jsonl, sidecar, comparison_path):
            fsync_file(path)
        fsync_directory(audit_jsonl.parent)
        marker = {
            "schema": AUDIT_MARKER_SCHEMA,
            "completed_at": utc_now(),
            "config_sha256": self.config_sha,
            "phase": phase,
            "outer": outer,
            "model_sha256": sha256_file(archive_model),
            "loss_delta_mean": loss_delta,
            "loss_delta_ci95_upper": ci_upper,
            "offline_improves": loss_delta < 0.0,
            "offline_strong_improvement": ci_upper < 0.0,
            "qnodes_per_root": summary.get("qnodes_per_root"),
            "files": [
                {
                    "path": audit_jsonl.name,
                    "sha256": sha256_file(audit_jsonl),
                    "bytes": audit_jsonl.stat().st_size,
                },
                {
                    "path": sidecar.name,
                    "sha256": sha256_file(sidecar),
                    "bytes": sidecar.stat().st_size,
                },
                {
                    "path": comparison_path.name,
                    "sha256": sha256_file(comparison_path),
                    "bytes": comparison_path.stat().st_size,
                },
            ],
        }
        atomic_json(marker_path, marker)
        self.record_artifact(
            f"phase_{phase.lower()}_outer_{outer:04d}_audit", marker_path
        )
        return marker

    def run_match(
        self,
        identity: Mapping[str, Any],
        candidate_model: Path,
        candidate_manifest: Path,
        label: str,
        phase: str,
        outer: int | None,
    ) -> dict[str, Any]:
        paired = self.section("paired_match")
        for name in ("wrapper", "openings", "openings_manifest"):
            self._verify_declared(paired[name], f"paired_match.{name}")
        for name in ("model", "model_manifest", "promotion_registry"):
            self._verify_declared(self.section("parent")[name], f"parent.{name}")
        candidate_sha_before = sha256_file(candidate_model)
        output_dir = self.run_root / "evaluations" / label / "match"
        command = [
            "python3",
            str(self.resolve(paired["wrapper"]["path"])),
            "--binary",
            str(self.binary(identity, "match")),
            "--candidate",
            str(candidate_model),
            "--candidate-manifest",
            str(candidate_manifest),
            "--parent",
            str(self.parent_path("model")),
            "--parent-manifest",
            str(self.parent_path("model_manifest")),
            "--registry",
            str(self.parent_path("promotion_registry")),
            "--openings",
            str(self.resolve(paired["openings"]["path"])),
            "--pairs",
            str(paired["pairs"]),
            "--nodes",
            str(paired["nodes"]),
            "--max-depth",
            str(paired["max_depth"]),
            "--max-plies",
            str(paired["max_plies"]),
            "--bootstrap-resamples",
            str(paired["bootstrap_resamples"]),
            "--bootstrap-seed",
            str(paired["bootstrap_seed"]),
            "--output-dir",
            str(output_dir),
        ]
        self.run_command(
            f"{label.replace('-', '_')}_paired_match",
            command,
            phase,
            outer,
            deterministic_patterns=("provenance", "mismatch", "invalid"),
        )
        report_path = output_dir / "report.json"
        completed_path = output_dir / "completed.json"
        report = load_json(report_path)
        completed = load_json(completed_path)
        candidate_sha = sha256_file(candidate_model)
        if candidate_sha != candidate_sha_before:
            raise TerminalBlocked("candidate model changed during paired match")
        for name in ("wrapper", "openings", "openings_manifest"):
            self._verify_declared(paired[name], f"paired_match.{name}")
        for name in ("model", "model_manifest", "promotion_registry"):
            self._verify_declared(self.section("parent")[name], f"parent.{name}")
        if (
            report.get("schema") != "halfkp_resumable_paired_match_v1"
            or report.get("candidate_sha256") != candidate_sha
            or report.get("parent_sha256") != self.section("parent")["model"]["sha256"]
            or report.get("openings_sha256") != paired["openings"]["sha256"]
            or report.get("pairs") != paired["pairs"]
            or report.get("games") != paired["pairs"] * 2
            or completed.get("report_sha256") != sha256_file(report_path)
        ):
            raise TerminalBlocked("paired match report verification failed")
        require_finite(report, "match_report")
        for name in ("score", "paired_score_ci95_lower", "paired_score_ci95_upper"):
            value = report.get(name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TerminalBlocked(f"paired match lacks finite {name}")
        if not 0.0 <= report["paired_score_ci95_lower"] <= report["paired_score_ci95_upper"] <= 1.0:
            raise TerminalBlocked("paired match confidence interval is invalid")
        if not isinstance(report.get("passes_safety"), bool):
            raise TerminalBlocked("paired match safety flag is missing")
        self.record_artifact(f"{label}_match", report_path)
        return report

    @staticmethod
    def match_regressed(report: Mapping[str, Any], gate: Mapping[str, Any]) -> str | None:
        if report.get("passes_safety") is not True:
            return "paired_match_safety_failure"
        upper = float(report["paired_score_ci95_upper"])
        catastrophic = gate.get("catastrophic_ci_upper_below")
        if catastrophic is not None and upper < float(catastrophic):
            return "paired_ci_upper_below_catastrophic_threshold"
        score_below = gate.get("score_below")
        and_upper = gate.get("and_ci_upper_below")
        if (
            score_below is not None
            and and_upper is not None
            and float(report["score"]) < float(score_below)
            and upper < float(and_upper)
        ):
            return "point_score_and_paired_ci_show_regression"
        return None

    def evaluate_outer(
        self,
        identity: Mapping[str, Any],
        archive: Path,
        phase: str,
        outer: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        audit = self.run_audit(identity, archive, phase, outer)
        match = self.run_match(
            identity,
            archive / "model.bin",
            archive / "model.bin.manifest.json",
            f"phase-{phase.lower()}-outer-{outer:04d}",
            phase,
            outer,
        )
        return audit, match

    def phase_a(self, identity: Mapping[str, Any]) -> dict[str, Any]:
        phase = self.section("phases")["A"]
        candidate = self.resolve(phase["candidate_model"]["path"])
        if sha256_file(candidate) != phase["candidate_model"]["sha256"]:
            raise TerminalBlocked("phase A candidate changed")
        manifest_path = self.run_root / "phase-a-candidate.manifest.json"
        self.write_candidate_manifest(manifest_path, candidate, "A", 4)
        report = self.run_match(
            identity, candidate, manifest_path, "phase-a-direct", "A", None
        )
        reason = self.match_regressed(report, phase["gate"])
        if reason:
            raise GateStop(f"phase_A:{reason}")
        self.completed_step("phase_A_gate", {"score": report["score"]})
        return report

    def phase_b(
        self, identity: Mapping[str, Any]
    ) -> tuple[float, dict[str, Any], dict[str, Any]]:
        phase = self.section("phases")["B"]
        primary_rate = float(phase["learning_rate"])
        fallback_rate = float(phase["fallback_learning_rate"])
        decision_path = self.run_root / "phase-b-learning-rate-decision.json"
        decision_preexisting = decision_path.exists()
        if decision_path.exists():
            decision = load_json(decision_path)
            expected_rates = {"primary": primary_rate, "fallback": fallback_rate}
            expected_reasons = {
                "primary": "primary_completed_safely",
                "fallback": "primary_parameter_or_outer_delta_safety_limit",
            }
            if (
                decision.get("schema") != "halfkp_mmto_phase_b_rate_decision_v1"
                or decision.get("config_sha256") != self.config_sha
                or decision.get("variant") not in ("primary", "fallback")
                or float(decision.get("learning_rate", -1.0))
                != expected_rates.get(decision.get("variant"), -2.0)
                or decision.get("reason")
                != expected_reasons.get(decision.get("variant"))
            ):
                raise TerminalBlocked("phase B learning-rate decision is invalid")
            variant = decision["variant"]
        else:
            variant = "primary"

        if variant == "primary":
            selected_rate = primary_rate
            output_dir = self.resolve(phase["output_dir"])
            try:
                training = self.run_training_outer(
                    identity, "B", phase, output_dir, 1, selected_rate, "primary"
                )
            except CommandFailure as error:
                if not error.deterministic or not self._log_tail_contains(
                    error.log_path,
                    ("outer delta limit exceeded", "parameter bound touched"),
                ):
                    raise TerminalBlocked(str(error)) from error
                if decision_preexisting:
                    raise TerminalBlocked(
                        "immutable primary learning-rate decision cannot change"
                    ) from error
                variant = "fallback"
                selected_rate = fallback_rate
                output_dir = self.resolve(phase["fallback_output_dir"])
                atomic_json(
                    decision_path,
                    {
                        "schema": "halfkp_mmto_phase_b_rate_decision_v1",
                        "decided_at": utc_now(),
                        "config_sha256": self.config_sha,
                        "variant": variant,
                        "learning_rate": selected_rate,
                        "reason": "primary_parameter_or_outer_delta_safety_limit",
                    },
                )
                training = self.run_training_outer(
                    identity, "B", phase, output_dir, 1, selected_rate, "fallback"
                )
            else:
                if not decision_preexisting:
                    atomic_json(
                        decision_path,
                        {
                            "schema": "halfkp_mmto_phase_b_rate_decision_v1",
                            "decided_at": utc_now(),
                            "config_sha256": self.config_sha,
                            "variant": "primary",
                            "learning_rate": selected_rate,
                            "reason": "primary_completed_safely",
                        },
                    )
        else:
            selected_rate = fallback_rate
            output_dir = self.resolve(phase["fallback_output_dir"])
            if float(decision.get("learning_rate", -1.0)) != selected_rate:
                raise TerminalBlocked("phase B fallback rate decision changed")
            training = self.run_training_outer(
                identity, "B", phase, output_dir, 1, selected_rate, "fallback"
            )
        archive = self.archive_outer("B", 1, training)
        audit, match = self.evaluate_outer(identity, archive, "B", 1)
        if not audit.get("offline_improves"):
            raise GateStop("phase_B:fixed_audit_loss_did_not_improve")
        reason = self.match_regressed(match, phase["gate"])
        if reason:
            raise GateStop(f"phase_B:{reason}")
        self.completed_step(
            "phase_B_gate",
            {"learning_rate": selected_rate, "score": match["score"]},
        )
        return selected_rate, audit, match

    def phase_c(self, identity: Mapping[str, Any], learning_rate: float) -> None:
        phase = self.section("phases")["C"]
        output_dir = self.resolve(phase["output_dir"])
        results: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
        max_outer = int(phase["max_outer"])
        checkpoint_path = output_dir / "checkpoint.json"
        live_completed = 0
        if checkpoint_path.exists():
            value = load_json(checkpoint_path).get("completed_outer")
            if not isinstance(value, int) or value < 0 or value > max_outer:
                raise TerminalBlocked("phase C live checkpoint outer is invalid")
            live_completed = value
        for outer in range(1, max_outer + 1):
            if outer >= 3:
                previous_audit, previous_match = results[outer - 1]
                extension = phase["extension_gate"]
                prior_score = results[outer - 2][1]["score"] if outer > 2 else 0.0
                if not (
                    previous_match.get("passes_safety") is True
                    and float(previous_match["score"]) >= float(extension["min_score"])
                    and float(previous_match["score"]) >= float(prior_score)
                    and previous_audit.get("offline_strong_improvement") is True
                ):
                    self.save_status(
                        status="completed",
                        terminal_reason=f"phase_C_outer_{outer - 1}_no_extension",
                        current_phase=None,
                        current_step=None,
                        current_outer=None,
                    )
                    return
            if live_completed > outer:
                archive = (
                    self.run_root
                    / "archives"
                    / "phase-c"
                    / f"outer_{outer:04d}"
                )
                if not archive.exists():
                    raise TerminalBlocked(
                        f"phase C outer {outer} archive missing behind live checkpoint"
                    )
                self.verify_archive(archive, "C", outer)
            else:
                training = self.run_training_outer(
                    identity, "C", phase, output_dir, outer, learning_rate, "full"
                )
                archive = self.archive_outer("C", outer, training)
            audit, match = self.evaluate_outer(identity, archive, "C", outer)
            results[outer] = (audit, match)
            if not audit.get("offline_improves"):
                raise GateStop(f"phase_C_outer_{outer}:fixed_audit_loss_did_not_improve")
            reason = self.match_regressed(match, phase["gate"])
            if reason:
                raise GateStop(f"phase_C_outer_{outer}:{reason}")
            self.completed_step(
                f"phase_C_outer_{outer:04d}_gate", {"score": match["score"]}
            )
        self.save_status(
            status="completed",
            terminal_reason=f"phase_C_outer_{max_outer}_complete",
            current_phase=None,
            current_step=None,
            current_outer=None,
        )

    def run_workflow(self) -> None:
        self.acquire_lock()
        try:
            if self.status.get("status") in {
                "completed",
                "stopped_by_gate",
                "blocked",
            }:
                return
            self.check_stop()
            preflight = self.validate_preflight(require_clean=True)
            self.save_status(
                status="running",
                started_at=self.status.get("started_at") or utc_now(),
                preflight=preflight,
                terminal_reason=None,
            )
            identity = self.ensure_identity()
            self.phase_a(identity)
            learning_rate, _, _ = self.phase_b(identity)
            self.phase_c(identity, learning_rate)
        except StopRequested as error:
            self.save_status(
                status="stopped_by_user",
                terminal_reason=str(error),
                current_step=None,
                child_pid=None,
            )
        except GateStop as error:
            self.save_status(
                status="stopped_by_gate",
                terminal_reason=str(error),
                current_step=None,
                child_pid=None,
            )
        except (TerminalBlocked, CommandFailure) as error:
            self.save_status(
                status="blocked",
                terminal_reason=str(error),
                last_error=str(error),
                current_step=None,
                child_pid=None,
            )
        finally:
            self.close()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "command", choices=("validate", "run", "status", "request-stop", "clear-stop")
    )
    result.add_argument(
        "--config",
        type=Path,
        default=Path("training/halfkp64-scratch-v1/unattended_mmto_20260806.json"),
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        supervisor = Supervisor(args.config)
        if args.command == "validate":
            print(json.dumps(supervisor.validate_preflight(), ensure_ascii=False, indent=2))
            return 0
        if args.command == "status":
            print(json.dumps(supervisor.status, ensure_ascii=False, indent=2))
            return 0
        if args.command == "request-stop":
            supervisor.run_root.mkdir(parents=True, exist_ok=True)
            atomic_json(
                supervisor.stop_path,
                {"schema": "halfkp_mmto_stop_request_v1", "requested_at": utc_now()},
            )
            print(f"stop requested: {supervisor.stop_path}")
            return 0
        if args.command == "clear-stop":
            if supervisor.status.get("status") in {"completed", "stopped_by_gate"}:
                raise TerminalBlocked(
                    "completed or Gate-stopped workflow requires a new reviewed config"
                )
            if supervisor.stop_path.exists():
                supervisor.stop_path.unlink()
                fsync_directory(supervisor.stop_path.parent)
            if supervisor.status.get("status") in {
                "stopped_by_user",
                "blocked",
                "crashed",
            }:
                supervisor.save_status(
                    status="resume_requested",
                    terminal_reason=None,
                    last_error=None,
                    attempts={},
                    supervisor_crash_count=0,
                )
            print("stop request cleared")
            return 0
        signal.signal(signal.SIGTERM, supervisor.handle_signal)
        signal.signal(signal.SIGINT, supervisor.handle_signal)
        supervisor.run_workflow()
        return 0
    except TerminalBlocked as error:
        print(f"unattended supervisor blocked: {error}", file=sys.stderr)
        return 2
    except BaseException as error:
        stop_restarting = False
        try:
            if "supervisor" in locals():
                crash_count = int(supervisor.status.get("supervisor_crash_count", 0)) + 1
                stop_restarting = crash_count >= supervisor.max_attempts
                supervisor.save_status(
                    status="blocked" if stop_restarting else "crashed",
                    last_error=repr(error),
                    terminal_reason=(
                        "repeated_unexpected_supervisor_exception"
                        if stop_restarting
                        else "unexpected_supervisor_exception"
                    ),
                    child_pid=None,
                    supervisor_crash_count=crash_count,
                )
                supervisor.close()
        except BaseException:
            pass
        print(f"unattended supervisor crashed: {error!r}", file=sys.stderr)
        return 0 if stop_restarting else 1


if __name__ == "__main__":
    raise SystemExit(main())

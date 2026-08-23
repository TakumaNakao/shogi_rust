#!/usr/bin/env python3
"""Run halfkp_promotion_match one opening pair at a time.

The Rust match runner intentionally treats a match as one process.  This wrapper
keeps its game semantics, but publishes each two-game color-swapped opening pair
as an immutable JSON artifact.  Re-running the same command verifies provenance
and existing artifacts, then executes only missing pairs.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence


LINEAGE_ID = "halfkp64-scratch-v1"
RUN_SCHEMA = "halfkp_resumable_paired_match_run_v1"
PAIR_SCHEMA = "halfkp_resumable_paired_match_pair_v1"
REPORT_SCHEMA = "halfkp_resumable_paired_match_v1"
COMPLETED_SCHEMA = "halfkp_resumable_paired_match_completed_v1"
FINGERPRINT_SCHEMA = "halfkp_resumable_paired_match_fingerprint_v1"
DEFAULT_BOOTSTRAP_RESAMPLES = 100_000
DEFAULT_BOOTSTRAP_SEED = 20260806
# A promotion is a superiority claim, not merely non-inferiority.  Pair-level
# child reports retain the historical diagnostics; only the aggregate report
# may promote a candidate.
PROMOTION_MIN_SCORE = 0.525
PROMOTION_MIN_PAIRED_LOWER = 0.5
COUNT_KEYS = (
    "candidate_wins",
    "parent_wins",
    "draws",
    "candidate_black_wins",
    "candidate_white_wins",
    "search_failures",
    "missing_root_results",
    "illegal_moves",
    "non_finite_scores",
    "max_ply_draws",
    "repetition_draws",
    "perpetual_check_games",
)
SAFETY_COUNT_KEYS = (
    "search_failures",
    "missing_root_results",
    "illegal_moves",
    "non_finite_scores",
)


class MatchRunnerError(RuntimeError):
    """A fail-closed validation or child-process error."""


@dataclasses.dataclass(frozen=True)
class SelectedOpening:
    pair_index: int
    source_line: int
    raw_json: bytes
    opening_family: str
    sfen: str
    ply: int

    @property
    def input_bytes(self) -> bytes:
        return self.raw_json + b"\n"

    @property
    def record_sha256(self) -> str:
        return sha256_bytes(self.raw_json)

    @property
    def input_sha256(self) -> str:
        return sha256_bytes(self.input_bytes)


@dataclasses.dataclass(frozen=True)
class RunInputs:
    binary: Path
    candidate: Path
    candidate_manifest: Path
    parent: Path
    parent_manifest: Path
    registry: Path | None
    openings: Path
    output_dir: Path
    pairs: int
    nodes: int
    max_depth: int
    max_plies: int
    bootstrap_resamples: int
    bootstrap_seed: int
    parent_authorization: Path | None = None
    openings_format: str = "dataset-jsonl"
    opening_selection: str = "first-unique"
    opening_seed: int = 0
    time_limit_ms: int | None = None


_active_child: subprocess.Popen[str] | None = None
_received_signal: int | None = None
_kill_timer: threading.Timer | None = None


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


def fingerprint_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_replace_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_replace_json(path: Path, value: Any) -> None:
    atomic_replace_bytes(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n")


def atomic_create_json(path: Path, value: Any) -> None:
    """Atomically create an immutable JSON file, refusing replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise MatchRunnerError(f"refusing to overwrite immutable artifact: {path}") from error
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MatchRunnerError(f"cannot read valid JSON from {path}: {error}") from error
    if not isinstance(value, dict):
        raise MatchRunnerError(f"expected a JSON object in {path}")
    return value


def select_openings(
    path: Path,
    count: int,
    openings_format: str = "dataset-jsonl",
    opening_selection: str = "first-unique",
    opening_seed: int = 0,
) -> list[SelectedOpening]:
    """Mirror the Rust match loader's first-N valid unique-family selection."""

    if count <= 0:
        raise MatchRunnerError("pairs must be positive")
    selected: list[SelectedOpening] = []
    if openings_format == "usi-lines":
        try:
            lines = [
                (source_line, line.rstrip(b"\r\n"))
                for source_line, line in enumerate(path.read_bytes().splitlines(), 1)
                if line.strip()
            ]
        except OSError as error:
            raise MatchRunnerError(f"cannot read opening suite {path}: {error}") from error
        if opening_selection == "sha256-seed":
            lines.sort(key=lambda item: hashlib.sha256(opening_seed.to_bytes(8, "little") + item[1]).hexdigest())
        elif opening_selection != "first-unique":
            raise MatchRunnerError(f"unknown opening selection: {opening_selection}")
        for index, (source_line, raw_line) in enumerate(lines[:count]):
            text = raw_line.decode("utf-8", errors="strict").strip()
            if not text.startswith("startpos") and not text.startswith("sfen "):
                raise MatchRunnerError(f"opening line {index + 1} is not a USI position")
            moves = text.split()[2:] if "moves" in text.split() else []
            # Each child receives a one-line temporary suite; Rust therefore reports
            # the local family as usi-line-0. Keep the original source line separately.
            selected.append(SelectedOpening(index, source_line, raw_line, "usi-line-0", text, len(moves)))
        if len(selected) != count:
            raise MatchRunnerError(f"requested {count} USI openings, found {len(selected)}")
        return selected
    if openings_format != "dataset-jsonl" or opening_selection != "first-unique":
        raise MatchRunnerError("dataset-jsonl supports only first-unique selection")
    seen_families: set[str] = set()
    try:
        source = path.open("rb")
    except OSError as error:
        raise MatchRunnerError(f"cannot open opening suite {path}: {error}") from error
    with source:
        for line_number, raw_line in enumerate(source, 1):
            raw_json = raw_line.rstrip(b"\r\n")
            try:
                record = json.loads(raw_json)
            except (UnicodeError, json.JSONDecodeError) as error:
                raise MatchRunnerError(
                    f"cannot parse opening line {line_number} in {path}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise MatchRunnerError(f"opening line {line_number} is not an object")
            try:
                split = record["split"]
                family = record["opening_family"]
                sfen = record["sfen"]
                ply = record["ply"]
            except KeyError as error:
                raise MatchRunnerError(
                    f"opening line {line_number} lacks {error.args[0]}"
                ) from error
            if not isinstance(family, str) or not isinstance(sfen, str) or not isinstance(ply, int):
                raise MatchRunnerError(f"opening line {line_number} has invalid field types")
            if split != "valid" or not 8 <= ply <= 40 or family in seen_families:
                continue
            seen_families.add(family)
            selected.append(
                SelectedOpening(
                    pair_index=len(selected),
                    source_line=line_number,
                    raw_json=raw_json,
                    opening_family=family,
                    sfen=sfen,
                    ply=ply,
                )
            )
            if len(selected) == count:
                break
    if len(selected) != count:
        raise MatchRunnerError(
            f"requested {count} unique valid opening families, found {len(selected)}"
        )
    return selected


def _require_regular_file(path: Path, label: str, executable: bool = False) -> None:
    if not path.is_file():
        raise MatchRunnerError(f"{label} is not a file: {path}")
    if executable and not os.access(path, os.X_OK):
        raise MatchRunnerError(f"{label} is not executable: {path}")


def _manifest_model_hash(
    path: Path, expected_model: Path, allowed_lineages: tuple[str, ...] = (LINEAGE_ID,)
) -> str:
    manifest = load_json(path)
    if manifest.get("lineage_id") not in allowed_lineages:
        raise MatchRunnerError(f"model manifest has wrong lineage: {path}")
    model = manifest.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("sha256"), str):
        raise MatchRunnerError(f"model manifest lacks model.sha256: {path}")
    actual = sha256_file(expected_model)
    if model["sha256"] != actual:
        raise MatchRunnerError(f"model manifest SHA does not match {expected_model}")
    return actual


def build_provenance(inputs: RunInputs, openings: Sequence[SelectedOpening]) -> tuple[dict[str, Any], str]:
    for path, label, executable in (
        (inputs.binary, "promotion match binary", True),
        (inputs.candidate, "candidate", False),
        (inputs.candidate_manifest, "candidate manifest", False),
        (inputs.parent, "parent", False),
        (inputs.parent_manifest, "parent manifest", False),
        *(([(inputs.registry, "promotion registry", False)] if inputs.registry is not None else [])),
        *(([(inputs.parent_authorization, "parent authorization", False)] if inputs.parent_authorization is not None else [])),
        (inputs.openings, "opening suite", False),
    ):
        _require_regular_file(path, label, executable)

    openings_sha_before = sha256_file(inputs.openings)
    if (inputs.registry is None) == (inputs.parent_authorization is None):
        raise MatchRunnerError("exactly one of registry and parent_authorization is required")
    if (inputs.nodes <= 0) or (inputs.time_limit_ms is not None and inputs.time_limit_ms <= 0):
        raise MatchRunnerError("node/time budget must be positive")
    if inputs.time_limit_ms is not None and inputs.nodes <= 0:
        raise MatchRunnerError("time budget requires a positive compatibility node value")
    current_openings = select_openings(inputs.openings, inputs.pairs, inputs.openings_format, inputs.opening_selection, inputs.opening_seed)
    openings_sha_after = sha256_file(inputs.openings)
    if openings_sha_before != openings_sha_after or list(openings) != current_openings:
        raise MatchRunnerError("opening suite changed while provenance was captured")

    parent_lineages = (LINEAGE_ID,)
    authorization: dict[str, Any] | None = None
    if inputs.parent_authorization is not None:
        authorization = load_json(inputs.parent_authorization)
        if authorization.get("schema") != "halfkp_workflow_parent_authorization_v1":
            raise MatchRunnerError("parent authorization has wrong schema")
        if authorization.get("authorization_kind") == "evaluation_only_external_baseline":
            parent_lineages = (LINEAGE_ID, "v2.5.0-kpp-derived-evaluation-only")
    candidate_hash = _manifest_model_hash(inputs.candidate_manifest, inputs.candidate)
    parent_hash = _manifest_model_hash(inputs.parent_manifest, inputs.parent, parent_lineages)
    if candidate_hash == parent_hash:
        raise MatchRunnerError("candidate and parent are identical")
    if inputs.registry is not None:
        registry = load_json(inputs.registry)
        accepted = registry.get("accepted")
        if registry.get("lineage_id") != LINEAGE_ID or not isinstance(accepted, list):
            raise MatchRunnerError("promotion registry has wrong lineage or shape")
        if not any(isinstance(entry, dict) and entry.get("model_sha256") == parent_hash for entry in accepted):
            raise MatchRunnerError("parent is not accepted in the scratch promotion registry")
    else:
        assert authorization is not None
        model = authorization.get("model")
        if not isinstance(model, dict) or model.get("sha256") != parent_hash:
            raise MatchRunnerError("parent authorization does not match parent model")

    selected = [
        {
            "pair_index": opening.pair_index,
            "source_line": opening.source_line,
            "opening_family": opening.opening_family,
            "ply": opening.ply,
            "record_sha256": opening.record_sha256,
            "single_opening_file_sha256": opening.input_sha256,
        }
        for opening in openings
    ]
    fingerprint_inputs: dict[str, Any] = {
        "schema": FINGERPRINT_SCHEMA,
        "lineage_id": LINEAGE_ID,
        "promotion_match_binary_sha256": sha256_file(inputs.binary),
        "candidate_sha256": candidate_hash,
        "candidate_manifest_sha256": sha256_file(inputs.candidate_manifest),
        "parent_sha256": parent_hash,
        "parent_manifest_sha256": sha256_file(inputs.parent_manifest),
        "registry_sha256": sha256_file(inputs.registry) if inputs.registry is not None else None,
        "parent_authorization_sha256": sha256_file(inputs.parent_authorization) if inputs.parent_authorization is not None else None,
        "openings_sha256": openings_sha_after,
        "pairs": inputs.pairs,
        "nodes": None if inputs.time_limit_ms is not None else inputs.nodes,
        "time_limit_ms": inputs.time_limit_ms,
        "openings_format": inputs.openings_format,
        "opening_selection": inputs.opening_selection,
        "opening_seed": inputs.opening_seed,
        "max_depth": inputs.max_depth,
        "max_plies": inputs.max_plies,
        "bootstrap": {
            "method": "opening_pair_percentile",
            "resamples": inputs.bootstrap_resamples,
            "seed": inputs.bootstrap_seed,
        },
        "selected_openings": selected,
    }
    return fingerprint_inputs, fingerprint_json(fingerprint_inputs)


def verify_runtime_inputs(
    inputs: RunInputs, fingerprint_inputs: Mapping[str, Any]
) -> None:
    """Fail closed if any frozen match input changed after fingerprinting."""
    checks = (
        (
            inputs.binary,
            "promotion match binary",
            "promotion_match_binary_sha256",
            True,
        ),
        (inputs.candidate, "candidate", "candidate_sha256", False),
        (
            inputs.candidate_manifest,
            "candidate manifest",
            "candidate_manifest_sha256",
            False,
        ),
        (inputs.parent, "parent", "parent_sha256", False),
        (
            inputs.parent_manifest,
            "parent manifest",
            "parent_manifest_sha256",
            False,
        ),
        *(([(inputs.registry, "promotion registry", "registry_sha256", False)] if inputs.registry is not None else [])),
        *(([(inputs.parent_authorization, "parent authorization", "parent_authorization_sha256", False)] if inputs.parent_authorization is not None else [])),
        (inputs.openings, "opening suite", "openings_sha256", False),
    )
    for path, label, field, executable in checks:
        _require_regular_file(path, label, executable)
        expected = fingerprint_inputs.get(field)
        if not isinstance(expected, str) or sha256_file(path) != expected:
            raise MatchRunnerError(f"{label} changed after provenance fingerprinting")


def initialize_or_validate_run(
    path: Path,
    fingerprint_inputs: Mapping[str, Any],
    fingerprint: str,
    display_paths: Mapping[str, str],
) -> dict[str, Any]:
    if path.exists():
        existing = load_json(path)
        if existing.get("schema") != RUN_SCHEMA or existing.get("provenance_fingerprint") != fingerprint:
            raise MatchRunnerError("existing run provenance fingerprint does not match")
        if existing.get("fingerprint_inputs") != fingerprint_inputs:
            raise MatchRunnerError("existing run provenance payload does not match")
        return existing
    run = {
        "schema": RUN_SCHEMA,
        "created_at": utc_now(),
        "provenance_fingerprint": fingerprint,
        "fingerprint_inputs": dict(fingerprint_inputs),
        "paths": dict(display_paths),
    }
    atomic_create_json(path, run)
    return run


def wilson_interval(successes: float, trials: int) -> tuple[float, float]:
    if trials <= 0 or not math.isfinite(successes) or successes < 0.0 or successes > trials:
        raise MatchRunnerError("invalid Wilson interval input")
    n = float(trials)
    p = successes / n
    z = 1.959_963_984_540_054
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise MatchRunnerError("cannot take a percentile of an empty sequence")
    if not 0.0 <= probability <= 1.0:
        raise MatchRunnerError("percentile probability must be within [0, 1]")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def paired_percentile_bootstrap(
    pair_scores: Sequence[float], resamples: int, seed: int
) -> tuple[float, float]:
    if not pair_scores or resamples <= 0:
        raise MatchRunnerError("bootstrap needs pair scores and positive resamples")
    population = sorted(float(score) for score in pair_scores)
    if any(not math.isfinite(score) or score < 0.0 or score > 1.0 for score in population):
        raise MatchRunnerError("pair score is outside [0, 1]")
    if population[0] == population[-1]:
        return population[0], population[0]
    generator = random.Random(seed)
    size = len(population)
    means = [
        math.fsum(population[generator.randrange(size)] for _ in range(size)) / size
        for _ in range(resamples)
    ]
    means.sort()
    return percentile(means, 0.025), percentile(means, 0.975)


def _integer_count(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise MatchRunnerError(f"invalid non-negative integer {label}")
    return value


def validate_child_report(
    report: Mapping[str, Any],
    opening: SelectedOpening,
    fingerprint_inputs: Mapping[str, Any],
) -> None:
    expected_scalars = {
        "schema": "halfkp_promotion_match_v1",
        "lineage_id": LINEAGE_ID,
        "candidate_sha256": fingerprint_inputs["candidate_sha256"],
        "parent_sha256": fingerprint_inputs["parent_sha256"],
        "openings_sha256": opening.input_sha256,
        "node_budget_per_move": fingerprint_inputs["nodes"],
        "max_depth": fingerprint_inputs["max_depth"],
        "max_plies": fingerprint_inputs["max_plies"],
        "pairs": 1,
        "games": 2,
    }
    if "time_limit_ms" in fingerprint_inputs:
        expected_scalars["time_limit_ms"] = fingerprint_inputs["time_limit_ms"]
    for key, expected in expected_scalars.items():
        if report.get(key) != expected:
            raise MatchRunnerError(
                f"pair {opening.pair_index}: child report {key} mismatch"
            )
    counts = report.get("counts")
    games = report.get("game_reports")
    if not isinstance(counts, dict) or not isinstance(games, list) or len(games) != 2:
        raise MatchRunnerError(f"pair {opening.pair_index}: malformed child result")
    normalized_counts = {key: _integer_count(counts.get(key), key) for key in COUNT_KEYS}
    if normalized_counts["candidate_wins"] + normalized_counts["parent_wins"] + normalized_counts["draws"] != 2:
        raise MatchRunnerError(f"pair {opening.pair_index}: WDL count is not two games")
    colors: set[str] = set()
    result_counts = {"candidate_win": 0, "parent_win": 0, "draw": 0}
    candidate_color_wins = {"black": 0, "white": 0}
    termination_counts = {
        "search_failures": 0,
        "missing_root_results": 0,
        "illegal_moves": 0,
        "non_finite_scores": 0,
        "max_ply_draws": 0,
        "repetition_draws": 0,
        "perpetual_check_games": 0,
    }
    termination_to_count = {
        "search_failure": "search_failures",
        "missing_root": "missing_root_results",
        "illegal_move": "illegal_moves",
        "non_finite_score": "non_finite_scores",
        "max_plies": "max_ply_draws",
        "repetition": "repetition_draws",
        "perpetual_check": "perpetual_check_games",
    }
    for game in games:
        if not isinstance(game, dict):
            raise MatchRunnerError(f"pair {opening.pair_index}: malformed game report")
        if game.get("pair") != 0 or game.get("opening_family") != opening.opening_family or game.get("opening_ply") != opening.ply:
            raise MatchRunnerError(f"pair {opening.pair_index}: game opening identity mismatch")
        color = game.get("candidate_color")
        result = game.get("result")
        if color not in ("black", "white") or color in colors or result not in result_counts:
            raise MatchRunnerError(f"pair {opening.pair_index}: invalid color-swapped games")
        colors.add(color)
        result_counts[result] += 1
        termination = game.get("termination")
        if not isinstance(termination, str):
            raise MatchRunnerError(f"pair {opening.pair_index}: invalid termination")
        if termination in termination_to_count:
            termination_counts[termination_to_count[termination]] += 1
        if result == "candidate_win":
            candidate_color_wins[color] += 1
    if colors != {"black", "white"}:
        raise MatchRunnerError(f"pair {opening.pair_index}: both candidate colors are required")
    if (
        result_counts["candidate_win"] != normalized_counts["candidate_wins"]
        or result_counts["parent_win"] != normalized_counts["parent_wins"]
        or result_counts["draw"] != normalized_counts["draws"]
        or candidate_color_wins["black"] != normalized_counts["candidate_black_wins"]
        or candidate_color_wins["white"] != normalized_counts["candidate_white_wins"]
    ):
        raise MatchRunnerError(f"pair {opening.pair_index}: game reports and counts disagree")
    for key, observed in termination_counts.items():
        if normalized_counts[key] != observed:
            raise MatchRunnerError(
                f"pair {opening.pair_index}: termination and {key} count disagree"
            )
    expected_score = (normalized_counts["candidate_wins"] + 0.5 * normalized_counts["draws"]) / 2.0
    score = report.get("score")
    if not isinstance(score, (int, float)) or not math.isclose(float(score), expected_score, abs_tol=1e-15):
        raise MatchRunnerError(f"pair {opening.pair_index}: child score mismatch")
    expected_lower, _ = wilson_interval(
        normalized_counts["candidate_wins"] + 0.5 * normalized_counts["draws"], 2
    )
    reported_lower = report.get("score_wilson_95_lower")
    safety = all(normalized_counts[key] == 0 for key in SAFETY_COUNT_KEYS)
    expected_flags = {
        "passes_point_score": expected_score >= 0.5,
        "passes_noninferiority": expected_lower >= 0.475,
        "passes_safety": safety,
        "promotion_gate_passed": expected_score >= 0.5
        and expected_lower >= 0.475
        and safety,
    }
    if not isinstance(reported_lower, (int, float)) or not math.isclose(
        float(reported_lower), expected_lower, abs_tol=1e-15
    ):
        raise MatchRunnerError(f"pair {opening.pair_index}: Wilson lower mismatch")
    for key, expected in expected_flags.items():
        if report.get(key) is not expected:
            raise MatchRunnerError(f"pair {opening.pair_index}: child {key} mismatch")


def make_pair_artifact(
    opening: SelectedOpening,
    provenance_fingerprint: str,
    fingerprint_inputs: Mapping[str, Any],
    child_report: Mapping[str, Any],
) -> dict[str, Any]:
    validate_child_report(child_report, opening, fingerprint_inputs)
    return {
        "schema": PAIR_SCHEMA,
        "provenance_fingerprint": provenance_fingerprint,
        "pair_index": opening.pair_index,
        "source_line": opening.source_line,
        "opening": {
            "opening_family": opening.opening_family,
            "sfen": opening.sfen,
            "ply": opening.ply,
            "record_sha256": opening.record_sha256,
            "single_opening_file_sha256": opening.input_sha256,
        },
        "child_report": dict(child_report),
    }


def validate_pair_artifact(
    artifact: Mapping[str, Any],
    opening: SelectedOpening,
    provenance_fingerprint: str,
    fingerprint_inputs: Mapping[str, Any],
) -> None:
    if (
        artifact.get("schema") != PAIR_SCHEMA
        or artifact.get("provenance_fingerprint") != provenance_fingerprint
        or artifact.get("pair_index") != opening.pair_index
        or artifact.get("source_line") != opening.source_line
    ):
        raise MatchRunnerError(f"pair {opening.pair_index}: artifact provenance mismatch")
    identity = artifact.get("opening")
    if not isinstance(identity, dict) or identity != {
        "opening_family": opening.opening_family,
        "sfen": opening.sfen,
        "ply": opening.ply,
        "record_sha256": opening.record_sha256,
        "single_opening_file_sha256": opening.input_sha256,
    }:
        raise MatchRunnerError(f"pair {opening.pair_index}: artifact opening mismatch")
    child = artifact.get("child_report")
    if not isinstance(child, dict):
        raise MatchRunnerError(f"pair {opening.pair_index}: child report is missing")
    validate_child_report(child, opening, fingerprint_inputs)


def pair_path(pair_dir: Path, index: int) -> Path:
    return pair_dir / f"pair_{index:04d}.json"


def scan_pair_artifacts(
    pair_dir: Path,
    openings: Sequence[SelectedOpening],
    provenance_fingerprint: str,
    fingerprint_inputs: Mapping[str, Any],
) -> tuple[dict[int, dict[str, Any]], list[int]]:
    expected_names = {pair_path(pair_dir, opening.pair_index).name for opening in openings}
    if pair_dir.exists():
        unexpected = sorted(
            path.name
            for path in pair_dir.glob("pair_*.json")
            if path.name not in expected_names
        )
        if unexpected:
            raise MatchRunnerError(f"unexpected pair artifacts: {', '.join(unexpected)}")
    artifacts: dict[int, dict[str, Any]] = {}
    missing: list[int] = []
    for opening in openings:
        path = pair_path(pair_dir, opening.pair_index)
        if not path.exists():
            missing.append(opening.pair_index)
            continue
        artifact = load_json(path)
        validate_pair_artifact(
            artifact, opening, provenance_fingerprint, fingerprint_inputs
        )
        artifacts[opening.pair_index] = artifact
    return artifacts, missing


def aggregate_pair_artifacts(
    artifacts: Sequence[Mapping[str, Any]],
    provenance_fingerprint: str,
    fingerprint_inputs: Mapping[str, Any],
    artifact_metadata: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_pairs = int(fingerprint_inputs["pairs"])
    if len(artifacts) != expected_pairs or len(artifact_metadata) != expected_pairs:
        raise MatchRunnerError("cannot aggregate an incomplete match")
    ordered = sorted(artifacts, key=lambda artifact: int(artifact["pair_index"]))
    if [int(artifact["pair_index"]) for artifact in ordered] != list(range(expected_pairs)):
        raise MatchRunnerError("pair indices are incomplete or duplicated")
    counts = {key: 0 for key in COUNT_KEYS}
    pair_scores: list[float] = []
    game_reports: list[dict[str, Any]] = []
    for artifact in ordered:
        child = artifact["child_report"]
        for key in COUNT_KEYS:
            counts[key] += _integer_count(child["counts"].get(key), key)
        pair_scores.append(float(child["score"]))
        pair_index = int(artifact["pair_index"])
        for child_game in child["game_reports"]:
            game = dict(child_game)
            game["pair"] = pair_index
            game_reports.append(game)
    games = expected_pairs * 2
    if counts["candidate_wins"] + counts["parent_wins"] + counts["draws"] != games:
        raise MatchRunnerError("aggregate WDL count does not equal expected games")
    score = (counts["candidate_wins"] + 0.5 * counts["draws"]) / games
    wilson_lower, wilson_upper = wilson_interval(
        counts["candidate_wins"] + 0.5 * counts["draws"], games
    )
    bootstrap = fingerprint_inputs["bootstrap"]
    paired_lower, paired_upper = paired_percentile_bootstrap(
        pair_scores, int(bootstrap["resamples"]), int(bootstrap["seed"])
    )
    safety = all(counts[key] == 0 for key in SAFETY_COUNT_KEYS)
    distribution = {
        f"{value:.2f}": sum(math.isclose(score_value, value, abs_tol=1e-15) for score_value in pair_scores)
        for value in (0.0, 0.25, 0.5, 0.75, 1.0)
    }
    return {
        "schema": REPORT_SCHEMA,
        "lineage_id": LINEAGE_ID,
        "created_at": utc_now(),
        "provenance_fingerprint": provenance_fingerprint,
        "candidate_sha256": fingerprint_inputs["candidate_sha256"],
        "parent_sha256": fingerprint_inputs["parent_sha256"],
        "openings_sha256": fingerprint_inputs["openings_sha256"],
        "promotion_match_binary_sha256": fingerprint_inputs[
            "promotion_match_binary_sha256"
        ],
        "node_budget_per_move": fingerprint_inputs["nodes"],
        "time_limit_ms": fingerprint_inputs.get("time_limit_ms"),
        "max_depth": fingerprint_inputs["max_depth"],
        "max_plies": fingerprint_inputs["max_plies"],
        "pairs": expected_pairs,
        "games": games,
        "score": score,
        "score_wilson_95_lower": wilson_lower,
        "score_wilson_95_upper": wilson_upper,
        "paired_score_ci95_lower": paired_lower,
        "paired_score_ci95_upper": paired_upper,
        "opening_pair_bootstrap": {
            "method": "percentile",
            "unit": "two_color_swapped_games_per_opening",
            "resamples": bootstrap["resamples"],
            "seed": bootstrap["seed"],
            "ci95_lower": paired_lower,
            "ci95_upper": paired_upper,
        },
        "pair_score_distribution": distribution,
        "passes_point_score": score >= 0.5,
        "passes_noninferiority": wilson_lower >= 0.475,
        "passes_paired_noninferiority": paired_lower >= 0.475,
        "passes_superiority_point": score >= PROMOTION_MIN_SCORE,
        "passes_paired_superiority": paired_lower > PROMOTION_MIN_PAIRED_LOWER,
        "passes_safety": safety,
        "promotion_gate_passed": (
            score >= PROMOTION_MIN_SCORE
            and paired_lower > PROMOTION_MIN_PAIRED_LOWER
            and safety
        ),
        "counts": counts,
        "pair_artifacts": list(artifact_metadata),
        "game_reports": game_reports,
    }


def _signal_handler(signum: int, _frame: Any) -> None:
    global _kill_timer, _received_signal
    _received_signal = signum
    child = _active_child
    if child is not None and child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if _kill_timer is None:
            _kill_timer = threading.Timer(10.0, _force_kill_child, args=(child,))
            _kill_timer.daemon = True
            _kill_timer.start()


def _force_kill_child(child: subprocess.Popen[str]) -> None:
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_child_pair(
    inputs: RunInputs,
    opening: SelectedOpening,
    fingerprint_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    global _active_child
    verify_runtime_inputs(inputs, fingerprint_inputs)
    attempts_dir = inputs.output_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"pair_{opening.pair_index:04d}.", dir=attempts_dir
    ) as temporary_name:
        temporary = Path(temporary_name)
        opening_path = temporary / "opening.jsonl"
        child_json = temporary / "child.json"
        opening_path.write_bytes(opening.input_bytes)
        command = [
            str(inputs.binary),
            "--candidate",
            str(inputs.candidate),
            "--candidate-manifest",
            str(inputs.candidate_manifest),
            "--parent",
            str(inputs.parent),
            "--parent-manifest",
            str(inputs.parent_manifest),
            *( ["--registry", str(inputs.registry)] if inputs.registry is not None else ["--parent-authorization", str(inputs.parent_authorization)] ),
            "--openings",
            str(opening_path),
            "--openings-format",
            inputs.openings_format,
            "--opening-selection",
            inputs.opening_selection,
            "--opening-seed",
            str(inputs.opening_seed),
            "--pairs",
            "1",
            *( ["--time-limit-ms", str(inputs.time_limit_ms)] if inputs.time_limit_ms is not None else ["--nodes", str(inputs.nodes)] ),
            "--max-depth",
            str(inputs.max_depth),
            "--max-plies",
            str(inputs.max_plies),
            "--json",
            str(child_json),
        ]
        try:
            try:
                _active_child = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
            except OSError as error:
                raise MatchRunnerError(
                    f"cannot start promotion match for pair {opening.pair_index}: {error}"
                ) from error
            output, _ = _active_child.communicate()
            return_code = _active_child.returncode
        finally:
            _active_child = None
        if _received_signal is not None:
            raise KeyboardInterrupt
        if return_code != 0:
            if output:
                print(output, file=sys.stderr, end="" if output.endswith("\n") else "\n")
            raise MatchRunnerError(
                f"promotion match failed for pair {opening.pair_index} with exit {return_code}"
            )
        verify_runtime_inputs(inputs, fingerprint_inputs)
        child_report = load_json(child_json)
        validate_child_report(child_report, opening, fingerprint_inputs)
        return child_report


def verify_completed_run(
    completed_path: Path,
    report_path: Path,
    pair_dir: Path,
    fingerprint: str,
    openings: Sequence[SelectedOpening],
    fingerprint_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    completed = load_json(completed_path)
    if completed.get("schema") != COMPLETED_SCHEMA or completed.get("provenance_fingerprint") != fingerprint:
        raise MatchRunnerError("completed marker provenance mismatch")
    if completed.get("report_sha256") != sha256_file(report_path):
        raise MatchRunnerError("completed report SHA mismatch")
    report = load_json(report_path)
    if report.get("schema") != REPORT_SCHEMA or report.get("provenance_fingerprint") != fingerprint:
        raise MatchRunnerError("completed report provenance mismatch")
    metadata = report.get("pair_artifacts")
    if not isinstance(metadata, list) or len(metadata) != report.get("pairs"):
        raise MatchRunnerError("completed report has incomplete pair metadata")
    seen_indices: set[int] = set()
    for entry in metadata:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not isinstance(entry.get("sha256"), str):
            raise MatchRunnerError("invalid completed pair metadata")
        pair_index = entry.get("pair_index")
        if (
            not isinstance(pair_index, int)
            or isinstance(pair_index, bool)
            or not 0 <= pair_index < len(openings)
            or pair_index in seen_indices
        ):
            raise MatchRunnerError("invalid or duplicate completed pair index")
        seen_indices.add(pair_index)
        expected_path = pair_path(pair_dir, pair_index)
        if entry["path"] != str(expected_path.relative_to(pair_dir.parent)):
            raise MatchRunnerError("completed pair path does not match its index")
        artifact_path = pair_dir.parent / entry["path"]
        if (
            not artifact_path.is_file()
            or sha256_file(artifact_path) != entry["sha256"]
            or entry.get("bytes") != artifact_path.stat().st_size
        ):
            raise MatchRunnerError(f"completed pair SHA mismatch: {artifact_path}")
        artifact = load_json(artifact_path)
        validate_pair_artifact(
            artifact,
            openings[pair_index],
            fingerprint,
            fingerprint_inputs,
        )
    if seen_indices != set(range(len(openings))):
        raise MatchRunnerError("completed pair metadata does not cover every pair")
    return report


def run(inputs: RunInputs) -> dict[str, Any]:
    if (
        inputs.pairs <= 0
        or inputs.nodes <= 0
        or inputs.max_depth <= 0
        or inputs.max_depth > 255
        or inputs.max_plies <= 0
        or inputs.bootstrap_resamples <= 0
    ):
        raise MatchRunnerError("invalid positive match or bootstrap parameter")
    inputs.output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = inputs.output_dir / ".lock"
    lock_descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise MatchRunnerError(f"another paired match runner holds {lock_path}") from error

        openings = select_openings(
            inputs.openings,
            inputs.pairs,
            inputs.openings_format,
            inputs.opening_selection,
            inputs.opening_seed,
        )
        fingerprint_inputs, fingerprint = build_provenance(inputs, openings)
        verify_runtime_inputs(inputs, fingerprint_inputs)
        pair_dir = inputs.output_dir / "pairs"
        run_path = inputs.output_dir / "run.json"
        report_path = inputs.output_dir / "report.json"
        completed_path = inputs.output_dir / "completed.json"
        display_paths = {
            "binary": str(inputs.binary),
            "candidate": str(inputs.candidate),
            "candidate_manifest": str(inputs.candidate_manifest),
            "parent": str(inputs.parent),
            "parent_manifest": str(inputs.parent_manifest),
            "registry": str(inputs.registry) if inputs.registry is not None else None,
            "parent_authorization": str(inputs.parent_authorization) if inputs.parent_authorization is not None else None,
            "openings": str(inputs.openings),
        }
        if not run_path.exists() and (
            report_path.exists()
            or completed_path.exists()
            or any(pair_dir.glob("pair_*.json"))
        ):
            raise MatchRunnerError(
                "output artifacts exist without the required run provenance"
            )
        initialize_or_validate_run(
            run_path, fingerprint_inputs, fingerprint, display_paths
        )
        if completed_path.exists():
            report = verify_completed_run(
                completed_path,
                report_path,
                pair_dir,
                fingerprint,
                openings,
                fingerprint_inputs,
            )
            print(
                f"match already complete pairs={report['pairs']} score={report['score']:.6f}"
            )
            return report

        artifacts, missing = scan_pair_artifacts(
            pair_dir, openings, fingerprint, fingerprint_inputs
        )
        if missing:
            print(
                f"resuming paired match completed={len(artifacts)} missing={len(missing)} "
                f"fingerprint={fingerprint}"
            )
        for index in missing:
            if _received_signal is not None:
                raise KeyboardInterrupt
            opening = openings[index]
            child_report = run_child_pair(inputs, opening, fingerprint_inputs)
            artifact = make_pair_artifact(
                opening, fingerprint, fingerprint_inputs, child_report
            )
            destination = pair_path(pair_dir, index)
            atomic_create_json(destination, artifact)
            artifacts[index] = artifact
            print(
                f"pair={index + 1}/{inputs.pairs} score={child_report['score']:.3f} "
                f"saved={destination}"
            )

        verify_runtime_inputs(inputs, fingerprint_inputs)
        ordered_artifacts = [artifacts[index] for index in range(inputs.pairs)]
        artifact_metadata = [
            {
                "pair_index": index,
                "path": str(pair_path(pair_dir, index).relative_to(inputs.output_dir)),
                "sha256": sha256_file(pair_path(pair_dir, index)),
                "bytes": pair_path(pair_dir, index).stat().st_size,
            }
            for index in range(inputs.pairs)
        ]
        report = aggregate_pair_artifacts(
            ordered_artifacts,
            fingerprint,
            fingerprint_inputs,
            artifact_metadata,
        )
        atomic_replace_json(report_path, report)
        completed = {
            "schema": COMPLETED_SCHEMA,
            "completed_at": utc_now(),
            "provenance_fingerprint": fingerprint,
            "pairs": inputs.pairs,
            "games": inputs.pairs * 2,
            "report": report_path.name,
            "report_sha256": sha256_file(report_path),
        }
        atomic_create_json(completed_path, completed)
        print(
            f"match complete pairs={inputs.pairs} score={report['score']:.6f} "
            f"paired_ci95=[{report['opening_pair_bootstrap']['ci95_lower']:.6f},"
            f"{report['opening_pair_bootstrap']['ci95_upper']:.6f}]"
        )
        return report
    finally:
        os.close(lock_descriptor)


def parse_args(argv: Sequence[str] | None = None) -> RunInputs:
    parser = argparse.ArgumentParser(
        description="Run resumable, atomic two-game HalfKP opening pairs"
    )
    parser.add_argument(
        "--binary", type=Path, default=Path("target/release/halfkp_promotion_match")
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    auth = parser.add_mutually_exclusive_group(required=True)
    auth.add_argument("--registry", type=Path)
    auth.add_argument("--parent-authorization", type=Path)
    parser.add_argument("--openings", type=Path, required=True)
    parser.add_argument("--openings-format", choices=("dataset-jsonl", "usi-lines"), default="dataset-jsonl")
    parser.add_argument("--opening-selection", choices=("first-unique", "sha256-seed"), default="first-unique")
    parser.add_argument("--opening-seed", type=int, default=0)
    parser.add_argument("--pairs", type=int, default=250)
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--time-limit-ms", type=int)
    parser.add_argument("--max-depth", type=int, default=64)
    parser.add_argument("--max-plies", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    namespace = parser.parse_args(argv)
    if namespace.nodes is not None and namespace.time_limit_ms is not None:
        parser.error("--nodes and --time-limit-ms are mutually exclusive")
    nodes = namespace.nodes if namespace.nodes is not None else (1 if namespace.time_limit_ms is not None else 20_000)
    return RunInputs(
        binary=namespace.binary.resolve(),
        candidate=namespace.candidate.resolve(),
        candidate_manifest=namespace.candidate_manifest.resolve(),
        parent=namespace.parent.resolve(),
        parent_manifest=namespace.parent_manifest.resolve(),
        registry=namespace.registry.resolve() if namespace.registry is not None else None,
        parent_authorization=namespace.parent_authorization.resolve() if namespace.parent_authorization is not None else None,
        openings=namespace.openings.resolve(),
        output_dir=namespace.output_dir.resolve(),
        pairs=namespace.pairs,
        nodes=nodes,
        max_depth=namespace.max_depth,
        max_plies=namespace.max_plies,
        bootstrap_resamples=namespace.bootstrap_resamples,
        bootstrap_seed=namespace.bootstrap_seed,
        openings_format=namespace.openings_format,
        opening_selection=namespace.opening_selection,
        opening_seed=namespace.opening_seed,
        time_limit_ms=namespace.time_limit_ms,
    )


def main(argv: Sequence[str] | None = None) -> int:
    global _kill_timer, _received_signal
    _received_signal = None
    _kill_timer = None
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    try:
        run(parse_args(argv))
        return 0
    except KeyboardInterrupt:
        received = _received_signal or signal.SIGINT
        print(f"paired match interrupted by signal {received}", file=sys.stderr)
        return 128 + received
    except MatchRunnerError as error:
        print(f"paired match error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

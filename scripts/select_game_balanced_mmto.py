#!/usr/bin/env python3
"""Build deterministic game-balanced MMTO train and validation subsets.

The emitted JSONL records are copied byte-for-byte from the frozen source
dataset.  Selection and output order are based only on SHA-256 keys, so they
do not depend on Python hash iteration order.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence


DERIVATION_SCHEMA = "halfkp_mmto_game_balanced_derivation_v1"
MANIFEST_SCHEMA = "rank_value_dataset_v4_manifest"
RECORD_SCHEMA = "rank_value_dataset_v3_position"
PHASES = ("opening", "middlegame", "endgame")


class SelectionError(RuntimeError):
    """Raised when source provenance or an existing artifact is unsafe."""


@dataclass(frozen=True)
class SourceArtifact:
    name: str
    path: Path
    sha256: str
    records: int


@dataclass(frozen=True)
class FrozenSource:
    manifest_path: Path
    manifest_sha256: str
    min_player_rate: int
    min_opponent_rate: int
    outputs: dict[str, SourceArtifact]


@dataclass(frozen=True)
class Candidate:
    line_no: int
    raw: bytes
    game_key: str
    opening_family: str
    phase: str
    sfen: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _rank(seed: int, domain: str, *values: str) -> bytes:
    digest = hashlib.sha256()
    for value in (domain, str(seed), *values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.digest()


def load_frozen_source(
    manifest_path: Path, required_outputs: Sequence[str]
) -> FrozenSource:
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise SelectionError(f"source manifest does not exist: {manifest_path}")
    manifest_sha = sha256_file(manifest_path)
    with manifest_path.open(encoding="utf-8") as source:
        manifest = json.load(source)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise SelectionError("source is not a rank_value_dataset_v4 manifest")
    min_player_rate = manifest.get("min_player_rate")
    min_opponent_rate = manifest.get("min_opponent_rate")
    if min_player_rate != 4000 or min_opponent_rate != 4000:
        raise SelectionError("source is not the frozen WDoor rate-4000 dataset")

    declared = {entry.get("path"): entry for entry in manifest.get("output_files", [])}
    outputs: dict[str, SourceArtifact] = {}
    for name in required_outputs:
        entry = declared.get(name)
        if not entry or not isinstance(entry.get("records"), int):
            raise SelectionError(f"source manifest lacks {name}")
        path = (manifest_path.parent / name).resolve()
        if not path.is_file():
            raise SelectionError(f"source artifact does not exist: {path}")
        actual_sha = sha256_file(path)
        if actual_sha != entry.get("sha256"):
            raise SelectionError(f"source artifact SHA-256 mismatch: {path}")
        outputs[name] = SourceArtifact(
            name=name,
            path=path,
            sha256=actual_sha,
            records=entry["records"],
        )
    return FrozenSource(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        min_player_rate=min_player_rate,
        min_opponent_rate=min_opponent_rate,
        outputs=outputs,
    )


def _candidate(raw: bytes, line_no: int, expected_split: str) -> Candidate:
    if not raw.endswith(b"\n"):
        raise SelectionError(f"unterminated source JSONL line {line_no}")
    try:
        record = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SelectionError(f"invalid JSON at source line {line_no}: {error}") from error
    required = ("game_key", "opening_family", "phase", "sfen")
    if (
        record.get("schema") != RECORD_SCHEMA
        or record.get("split") != expected_split
        or any(not isinstance(record.get(field), str) or not record[field] for field in required)
        or record["phase"] not in PHASES
    ):
        raise SelectionError(f"invalid {expected_split} provenance at source line {line_no}")
    return Candidate(
        line_no=line_no,
        raw=raw,
        game_key=record["game_key"],
        opening_family=record["opening_family"],
        phase=record["phase"],
        sfen=record["sfen"],
    )


def iter_candidates(path: Path, expected_split: str) -> Iterator[Candidate]:
    with path.open("rb") as source:
        for line_no, raw in enumerate(source, 1):
            if not raw.strip():
                raise SelectionError(f"blank source JSONL line {line_no}")
            yield _candidate(raw, line_no, expected_split)


def _better(
    new: Candidate, old: Candidate, seed: int, domain: str, group: tuple[str, ...]
) -> bool:
    new_key = (
        _rank(seed, domain, *group, new.sfen),
        hashlib.sha256(new.raw).digest(),
        new.line_no,
    )
    old_key = (
        _rank(seed, domain, *group, old.sfen),
        hashlib.sha256(old.raw).digest(),
        old.line_no,
    )
    return new_key < old_key


def _first_pass(
    artifact: SourceArtifact, split: str, seed: int
) -> tuple[dict[str, Candidate], dict[tuple[str, str], Candidate], dict[str, object]]:
    per_game: dict[str, Candidate] = {}
    per_game_phase: dict[tuple[str, str], Candidate] = {}
    games: set[str] = set()
    families: set[str] = set()
    phase_counts = {phase: 0 for phase in PHASES}
    records = 0
    for item in iter_candidates(artifact.path, split):
        records += 1
        games.add(item.game_key)
        families.add(item.opening_family)
        phase_counts[item.phase] += 1
        old_game = per_game.get(item.game_key)
        if old_game is None or _better(
            item, old_game, seed, f"{split}-one-per-game", (item.game_key,)
        ):
            per_game[item.game_key] = item
        group = (item.game_key, item.phase)
        old_phase = per_game_phase.get(group)
        if old_phase is None or _better(
            item, old_phase, seed, f"{split}-one-per-game-phase", group
        ):
            per_game_phase[group] = item
    if records != artifact.records:
        raise SelectionError(
            f"{split} record count mismatch: manifest={artifact.records}, actual={records}"
        )
    stats: dict[str, object] = {
        "records": records,
        "games": len(games),
        "opening_families": len(families),
        "game_phase_groups": len(per_game_phase),
        "phase": phase_counts,
    }
    return per_game, per_game_phase, stats


def _sort_for_stage(items: Sequence[Candidate], seed: int, domain: str) -> list[Candidate]:
    return sorted(
        items,
        key=lambda item: (
            _rank(seed, domain, item.game_key, item.phase, item.sfen),
            hashlib.sha256(item.raw).digest(),
            item.line_no,
        ),
    )


def _global_fill(
    artifact: SourceArtifact,
    split: str,
    seed: int,
    selected_line_nos: set[int],
    count: int,
) -> list[Candidate]:
    if count == 0:
        return []
    heap: list[tuple[int, int, int, Candidate]] = []
    for item in iter_candidates(artifact.path, split):
        if item.line_no in selected_line_nos:
            continue
        priority = int.from_bytes(
            _rank(seed, f"{split}-global-fill", item.game_key, item.phase, item.sfen),
            "big",
        )
        entry = (-priority, -item.line_no, item.line_no, item)
        if len(heap) < count:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    if len(heap) != count:
        raise SelectionError(f"{split} has only {len(heap)} records available for {count} fill")
    return [entry[3] for entry in heap]


def select_split(
    artifact: SourceArtifact, split: str, target: int, seed: int
) -> tuple[list[Candidate], dict[str, object]]:
    per_game, per_game_phase, source_stats = _first_pass(artifact, split, seed)
    if split == "train":
        selected = list(per_game_phase.values())
        coverage = "all_game_phase_groups"
    elif split == "valid":
        selected = list(per_game.values())
        coverage = "all_games_then_game_phase_fill"
        covered_groups = {(item.game_key, item.phase) for item in selected}
        secondary = [
            item
            for group, item in per_game_phase.items()
            if group not in covered_groups
        ]
        room = max(0, target - len(selected))
        selected.extend(
            _sort_for_stage(secondary, seed, "valid-secondary-game-phase")[:room]
        )
    else:
        raise SelectionError(f"unsupported split: {split}")

    if len(selected) > target:
        raise SelectionError(
            f"{split} target {target} cannot retain required coverage of {len(selected)} records"
        )
    selected_lines = {item.line_no for item in selected}
    selected.extend(
        _global_fill(
            artifact,
            split,
            seed,
            selected_lines,
            target - len(selected),
        )
    )
    if len({item.line_no for item in selected}) != target:
        raise SelectionError(f"{split} selection contains duplicate source lines")
    selected = _sort_for_stage(selected, seed, f"{split}-output-order")
    selected_games = {item.game_key for item in selected}
    selected_groups = {(item.game_key, item.phase) for item in selected}
    selected_sfens = {item.sfen for item in selected}
    if len(selected_sfens) != target:
        raise SelectionError(f"{split} selection contains duplicate exact SFEN")
    stats = {
        "source": source_stats,
        "selected": {
            "records": target,
            "games": len(selected_games),
            "opening_families": len({item.opening_family for item in selected}),
            "game_phase_groups": len(selected_groups),
            "unique_sfen": len(selected_sfens),
            "phase": {
                phase: sum(item.phase == phase for item in selected) for phase in PHASES
            },
        },
        "coverage": coverage,
    }
    if split == "train" and len(selected_groups) != source_stats["game_phase_groups"]:
        raise SelectionError("train selection did not retain every game-phase group")
    if split == "valid" and len(selected_games) != source_stats["games"]:
        raise SelectionError("valid selection did not retain every game")
    return selected, stats


def _write_records(path: Path, records: Sequence[Candidate]) -> tuple[str, int]:
    digest = hashlib.sha256()
    with path.open("xb") as output:
        for item in records:
            output.write(item.raw)
            digest.update(item.raw)
        output.flush()
        os.fsync(output.fileno())
    return digest.hexdigest(), len(records)


def _write_json(path: Path, value: object) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    with path.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def _count_lines(path: Path) -> int:
    with path.open("rb") as source:
        return sum(1 for line in source if line)


def validate_reusable_output(
    output_dir: Path,
    expected_schema: str,
    expected_request: dict[str, object],
    expected_outputs: dict[str, int],
) -> dict[str, object] | None:
    if not output_dir.exists():
        return None
    manifest_path = output_dir / "manifest.json"
    try:
        with manifest_path.open(encoding="utf-8") as source:
            manifest = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise SelectionError(f"existing output has no valid manifest: {output_dir}") from error
    if (
        manifest.get("schema") != expected_schema
        or manifest.get("derivation", {}).get("request") != expected_request
        or manifest.get("stage_fingerprint")
        != hashlib.sha256(_canonical_json(expected_request)).hexdigest()
    ):
        raise SelectionError(f"existing output provenance does not match request: {output_dir}")
    artifacts = manifest.get("output_files", [])
    if [artifact.get("path") for artifact in artifacts] != list(expected_outputs):
        raise SelectionError("existing output manifest has unexpected artifacts")
    for artifact in artifacts:
        name = artifact.get("path")
        if not isinstance(name, str) or Path(name).name != name:
            raise SelectionError("existing output manifest has an unsafe artifact path")
        path = output_dir / name
        if (
            not path.is_file()
            or artifact.get("records") != expected_outputs[name]
            or sha256_file(path) != artifact.get("sha256")
            or _count_lines(path) != artifact.get("records")
        ):
            raise SelectionError(f"existing output artifact does not match manifest: {path}")
    return manifest


@contextmanager
def staged_output_directory(output_dir: Path) -> Iterator[Path]:
    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        yield staging
        staging_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(staging_fd)
        finally:
            os.close(staging_fd)
        os.replace(staging, output_dir)
        directory_fd = os.open(output_dir.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def build_dataset(
    source_manifest: Path,
    output_dir: Path,
    seed: int,
    train_records: int = 100_000,
    valid_records: int = 10_000,
) -> tuple[dict[str, object], bool]:
    if seed < 0 or train_records <= 0 or valid_records <= 0:
        raise SelectionError("seed must be non-negative and record targets must be positive")
    frozen = load_frozen_source(source_manifest, ("train.jsonl", "valid.jsonl"))
    request: dict[str, object] = {
        "schema": DERIVATION_SCHEMA,
        "seed": seed,
        "train_records": train_records,
        "valid_records": valid_records,
        "source_manifest_sha256": frozen.manifest_sha256,
        "source_train_sha256": frozen.outputs["train.jsonl"].sha256,
        "source_valid_sha256": frozen.outputs["valid.jsonl"].sha256,
    }
    output_dir = output_dir.resolve()
    reusable = validate_reusable_output(
        output_dir,
        MANIFEST_SCHEMA,
        request,
        {"train.jsonl": train_records, "valid.jsonl": valid_records},
    )
    if reusable is not None:
        return reusable, True

    train, train_stats = select_split(
        frozen.outputs["train.jsonl"], "train", train_records, seed
    )
    valid, valid_stats = select_split(
        frozen.outputs["valid.jsonl"], "valid", valid_records, seed
    )
    train_games = {item.game_key for item in train}
    valid_games = {item.game_key for item in valid}
    train_sfens = {item.sfen for item in train}
    valid_sfens = {item.sfen for item in valid}
    overlap = {
        "game_keys": len(train_games & valid_games),
        "exact_sfen": len(train_sfens & valid_sfens),
    }
    if overlap != {"game_keys": 0, "exact_sfen": 0}:
        raise SelectionError(f"selected train/valid overlap is non-zero: {overlap}")

    with staged_output_directory(output_dir) as staging:
        train_sha, train_count = _write_records(staging / "train.jsonl", train)
        valid_sha, valid_count = _write_records(staging / "valid.jsonl", valid)
        manifest: dict[str, object] = {
            "schema_version": 4,
            "schema": MANIFEST_SCHEMA,
            "stage_fingerprint": hashlib.sha256(_canonical_json(request)).hexdigest(),
            "min_player_rate": frozen.min_player_rate,
            "min_opponent_rate": frozen.min_opponent_rate,
            "derivation": {
                "request": request,
                "selection": {
                    "hash": "sha256(domain\\0seed\\0game_key\\0phase\\0sfen\\0)",
                    "train": "one_per_game_phase_then_global_hash_fill",
                    "valid": "one_per_game_then_game_phase_hash_fill_then_global_hash_fill",
                    "output_order": "independent_split_output_order_hash",
                    "source_lines": "copied_byte_for_byte",
                },
                "source_manifest": {
                    "path": str(frozen.manifest_path),
                    "sha256": frozen.manifest_sha256,
                },
            },
            "output_files": [
                {"path": "train.jsonl", "sha256": train_sha, "records": train_count},
                {"path": "valid.jsonl", "sha256": valid_sha, "records": valid_count},
            ],
            "stats": {
                "train": train_stats,
                "valid": valid_stats,
                "overlap": overlap,
            },
        }
        _write_json(staging / "manifest.json", manifest)
    return manifest, False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-records", type=int, default=100_000)
    parser.add_argument("--valid-records", type=int, default=10_000)
    args = parser.parse_args()
    try:
        manifest, reused = build_dataset(
            args.source_manifest,
            args.output_dir,
            args.seed,
            args.train_records,
            args.valid_records,
        )
    except SelectionError as error:
        raise SystemExit(f"error: {error}") from error
    print(
        f"output={args.output_dir} reused={str(reused).lower()} "
        f"train={manifest['output_files'][0]['records']} "
        f"valid={manifest['output_files'][1]['records']}"
    )


if __name__ == "__main__":
    main()

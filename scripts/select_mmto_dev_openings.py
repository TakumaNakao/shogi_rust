#!/usr/bin/env python3
"""Freeze a deterministic 250-pair intermediate MMTO opening suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from select_game_balanced_mmto import (
    SelectionError,
    display_path,
    iter_candidates,
    load_frozen_source,
    sha256_file,
    staged_output_directory,
    validate_reusable_output,
    verify_frozen_source_unchanged,
)


MANIFEST_SCHEMA = "halfkp_mmto_dev_openings_v1_manifest"
DERIVATION_SCHEMA = "halfkp_mmto_dev_opening_selection_v1"
FINAL_RESERVE_SCHEMA = "halfkp_final_opening_reserve_v1"
FINAL_RESERVE_MANIFEST_SCHEMA = "halfkp_final_opening_reserve_v1_manifest"


@dataclass(frozen=True)
class OpeningCandidate:
    raw: bytes
    line_no: int
    game_key: str
    opening_family: str
    sfen: str
    ply: int
    phase: str


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _selection_rank(seed: int, item: OpeningCandidate) -> tuple[bytes, bytes, int]:
    # This exact key is frozen in the training plan.
    rank = hashlib.sha256(
        f"{seed}\0{item.game_key}\0{item.sfen}".encode("utf-8")
    ).digest()
    return rank, hashlib.sha256(item.raw).digest(), item.line_no


def _load_final_reserve(
    reserve_path: Path, reserve_manifest_path: Path
) -> tuple[set[str], dict[str, object]]:
    reserve_path = reserve_path.resolve()
    reserve_manifest_path = reserve_manifest_path.resolve()
    if not reserve_path.is_file() or not reserve_manifest_path.is_file():
        raise SelectionError("final opening reserve or its manifest is missing")
    reserve_sha = sha256_file(reserve_path)
    reserve_manifest_sha = sha256_file(reserve_manifest_path)
    try:
        with reserve_manifest_path.open(encoding="utf-8") as source:
            manifest = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise SelectionError("invalid final opening reserve manifest") from error
    artifact = manifest.get("artifact", {})
    if (
        manifest.get("schema") != FINAL_RESERVE_MANIFEST_SCHEMA
        or artifact.get("sha256") != reserve_sha
        or not isinstance(artifact.get("records"), int)
    ):
        raise SelectionError("final opening reserve provenance mismatch")

    sfens: set[str] = set()
    records = 0
    with reserve_path.open("rb") as source:
        for line_no, raw in enumerate(source, 1):
            if not raw.endswith(b"\n"):
                raise SelectionError(f"unterminated final reserve line {line_no}")
            try:
                record = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise SelectionError(f"invalid final reserve line {line_no}") from error
            sfen = record.get("sfen")
            if record.get("schema") != FINAL_RESERVE_SCHEMA or not isinstance(sfen, str) or not sfen:
                raise SelectionError(f"invalid final reserve record at line {line_no}")
            records += 1
            sfens.add(sfen)
    if records != artifact["records"] or len(sfens) != records:
        raise SelectionError("final opening reserve count or SFEN uniqueness mismatch")
    provenance: dict[str, object] = {
        "path": display_path(reserve_path),
        "sha256": reserve_sha,
        "records": records,
        "manifest_path": display_path(reserve_manifest_path),
        "manifest_sha256": reserve_manifest_sha,
    }
    return sfens, provenance


def _load_exclusion_sfens(path: Path, split: str) -> set[str]:
    return {item.sfen for item in iter_candidates(path, split)}


def _valid_candidates(path: Path) -> list[OpeningCandidate]:
    output: list[OpeningCandidate] = []
    for item in iter_candidates(path, "valid"):
        record = json.loads(item.raw)
        ply = record.get("ply")
        if not isinstance(ply, int):
            raise SelectionError(f"valid opening has no integer ply at line {item.line_no}")
        output.append(
            OpeningCandidate(
                raw=item.raw,
                line_no=item.line_no,
                game_key=item.game_key,
                opening_family=item.opening_family,
                sfen=item.sfen,
                ply=ply,
                phase=item.phase,
            )
        )
    return output


def _write_openings(path: Path, records: Sequence[OpeningCandidate]) -> str:
    digest = hashlib.sha256()
    with path.open("xb") as output:
        for item in records:
            output.write(item.raw)
            digest.update(item.raw)
        output.flush()
        os.fsync(output.fileno())
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    with path.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def build_openings(
    source_manifest: Path,
    final_reserve: Path,
    output_dir: Path,
    seed: int,
    pairs: int = 250,
    final_reserve_manifest: Path | None = None,
) -> tuple[dict[str, object], bool]:
    if seed < 0 or pairs <= 0:
        raise SelectionError("seed must be non-negative and pairs must be positive")
    if final_reserve_manifest is None:
        final_reserve_manifest = final_reserve.with_suffix(".manifest.json")
    frozen = load_frozen_source(
        source_manifest, ("train.jsonl", "valid.jsonl", "test.jsonl")
    )
    final_sfens, final_provenance = _load_final_reserve(
        final_reserve, final_reserve_manifest
    )
    request: dict[str, object] = {
        "schema": DERIVATION_SCHEMA,
        "seed": seed,
        "pairs": pairs,
        "source_manifest_sha256": frozen.manifest_sha256,
        "source_train_sha256": frozen.outputs["train.jsonl"].sha256,
        "source_valid_sha256": frozen.outputs["valid.jsonl"].sha256,
        "source_test_sha256": frozen.outputs["test.jsonl"].sha256,
        "final_reserve_sha256": final_provenance["sha256"],
        "final_reserve_manifest_sha256": final_provenance["manifest_sha256"],
    }
    output_dir = output_dir.resolve()
    reusable = validate_reusable_output(
        output_dir, MANIFEST_SCHEMA, request, {"openings.jsonl": pairs}
    )
    if reusable is not None:
        return reusable, True

    train_sfens = _load_exclusion_sfens(frozen.outputs["train.jsonl"].path, "train")
    test_sfens = _load_exclusion_sfens(frozen.outputs["test.jsonl"].path, "test")
    candidates = _valid_candidates(frozen.outputs["valid.jsonl"].path)
    excluded = {"train": 0, "test": 0, "final_reserve": 0}
    eligible: list[OpeningCandidate] = []
    in_ply_range = 0
    for item in candidates:
        if not 8 <= item.ply <= 40:
            continue
        in_ply_range += 1
        blocked = False
        for name, sfens in (
            ("train", train_sfens),
            ("test", test_sfens),
            ("final_reserve", final_sfens),
        ):
            if item.sfen in sfens:
                excluded[name] += 1
                blocked = True
        if not blocked:
            eligible.append(item)
    eligible.sort(key=lambda item: _selection_rank(seed, item))

    selected: list[OpeningCandidate] = []
    seen_families: set[str] = set()
    seen_sfens: set[str] = set()
    duplicate_family_skipped = 0
    duplicate_sfen_skipped = 0
    for item in eligible:
        if item.opening_family in seen_families:
            duplicate_family_skipped += 1
            continue
        if item.sfen in seen_sfens:
            duplicate_sfen_skipped += 1
            continue
        selected.append(item)
        seen_families.add(item.opening_family)
        seen_sfens.add(item.sfen)
        if len(selected) == pairs:
            break
    if len(selected) != pairs:
        raise SelectionError(
            f"requested {pairs} unique opening families/SFENs, found {len(selected)}"
        )

    output_overlap = {
        "train_exact_sfen": len(seen_sfens & train_sfens),
        "test_exact_sfen": len(seen_sfens & test_sfens),
        "final_reserve_exact_sfen": len(seen_sfens & final_sfens),
    }
    if any(output_overlap.values()):
        raise SelectionError(f"opening exclusion verification failed: {output_overlap}")
    ply_values = [item.ply for item in selected]

    with staged_output_directory(output_dir) as staging:
        output_sha = _write_openings(staging / "openings.jsonl", selected)
        manifest: dict[str, object] = {
            "schema_version": 1,
            "schema": MANIFEST_SCHEMA,
            "stage_fingerprint": hashlib.sha256(_canonical_json(request)).hexdigest(),
            "derivation": {
                "request": request,
                "selection": {
                    "source_filter": "split valid and 8 <= ply <= 40",
                    "exclusions": "exact SFEN in frozen train, frozen test, or final reserve",
                    "rank": "ascending sha256(seed\\0game_key\\0sfen)",
                    "uniqueness": "opening_family and exact SFEN",
                    "source_lines": "copied_byte_for_byte",
                },
                "source_manifest": {
                    "path": display_path(frozen.manifest_path),
                    "sha256": frozen.manifest_sha256,
                },
                "final_reserve": final_provenance,
            },
            "output_files": [
                {"path": "openings.jsonl", "sha256": output_sha, "records": pairs}
            ],
            "stats": {
                "source_valid_records": len(candidates),
                "in_ply_range": in_ply_range,
                "eligible_after_exclusions": len(eligible),
                "excluded_records": excluded,
                "duplicate_family_skipped_before_completion": duplicate_family_skipped,
                "duplicate_sfen_skipped_before_completion": duplicate_sfen_skipped,
                "selected": {
                    "pairs": pairs,
                    "unique_opening_families": len(seen_families),
                    "unique_exact_sfen": len(seen_sfens),
                    "unique_games": len({item.game_key for item in selected}),
                    "ply_min": min(ply_values),
                    "ply_max": max(ply_values),
                    "phase": {
                        phase: sum(item.phase == phase for item in selected)
                        for phase in ("opening", "middlegame", "endgame")
                    },
                },
                "output_overlap": output_overlap,
            },
        }
        _write_json(staging / "manifest.json", manifest)
        verify_frozen_source_unchanged(frozen)
        if (
            sha256_file(final_reserve.resolve()) != final_provenance["sha256"]
            or sha256_file(final_reserve_manifest.resolve())
            != final_provenance["manifest_sha256"]
        ):
            raise SelectionError("final opening reserve changed during selection")
    return manifest, False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--final-reserve", type=Path, required=True)
    parser.add_argument("--final-reserve-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pairs", type=int, default=250)
    args = parser.parse_args()
    try:
        manifest, reused = build_openings(
            args.source_manifest,
            args.final_reserve,
            args.output_dir,
            args.seed,
            args.pairs,
            args.final_reserve_manifest,
        )
    except SelectionError as error:
        raise SystemExit(f"error: {error}") from error
    print(
        f"output={args.output_dir} reused={str(reused).lower()} "
        f"pairs={manifest['output_files'][0]['records']}"
    )


if __name__ == "__main__":
    main()

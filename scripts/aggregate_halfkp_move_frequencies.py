#!/usr/bin/env python3
"""Aggregate observed WDoor moves for identical positions.

This is the S01 data-audit stage for the opening/middlegame move-ranking
plan.  It deliberately groups only by the exact SFEN board, side to move,
and hand (the first three SFEN fields).  Move counters are ignored.  The
input rows are not treated as value labels; only the observed move and its
metadata are retained.

The script accepts the v3/v4 JSONL position format produced by
``dataset_build``.  It writes one aggregate row per position and split.  A
position occurring in more than one split is excluded from every output
split, preventing accidental train/eval leakage when the source was built
without a position-preserving split audit.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys
from typing import Any


PHASES = ("opening", "middlegame", "endgame")


def normalized_position(sfen: str) -> str | None:
    fields = sfen.split()
    if len(fields) < 3:
        return None
    # Exact board + side + hand identity.  The move counter is intentionally
    # ignored because it is not part of the HalfKP position features.
    return " ".join(fields[:3])


def source_sha256(paths: list[pathlib.Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def split_name(path: pathlib.Path, value: dict[str, Any]) -> str:
    value_split = value.get("split")
    if value_split in {"train", "valid", "test"}:
        return value_split
    stem = path.stem
    if stem in {"train", "valid", "test"}:
        return stem
    raise ValueError(f"cannot infer split for {path}")


def phase_name(value: dict[str, Any]) -> str:
    phase = value.get("phase")
    if phase == "late":
        return "endgame"
    if phase in PHASES:
        return phase
    try:
        ply = int(value.get("ply", 0))
    except (TypeError, ValueError):
        ply = 0
    if ply <= 40:
        return "opening"
    if ply <= 90:
        return "middlegame"
    return "endgame"


def read_groups(paths: list[pathlib.Path]) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    groups: dict[str, dict[str, Any]] = {}
    stats = collections.Counter()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                stats["raw_records"] += 1
                try:
                    value = json.loads(line)
                    split = split_name(path, value)
                    sfen = str(value["sfen"])
                    key = normalized_position(sfen)
                    move = str(value["teacher_move"])
                    if key is None or not move:
                        raise ValueError("missing normalized position or teacher_move")
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                    stats["malformed_records"] += 1
                    if stats["malformed_records"] <= 5:
                        print(f"warning: {path}:{line_number}: {error}", file=sys.stderr)
                    continue

                group_key = f"{split}\0{key}"
                group = groups.setdefault(
                    group_key,
                    {
                        "split": split,
                        "position_key": key,
                        "sfen": sfen,
                        "moves": collections.Counter(),
                        "phase_counts": collections.Counter(),
                        "plies": [],
                        "legal_moves": [],
                        "game_keys": set(),
                        "sources": set(),
                        "opening_families": set(),
                        "license_ids": set(),
                        "source_uris": set(),
                        "side_to_move": str(value.get("side_to_move", "")),
                    },
                )
                group["moves"][move] += 1
                group["phase_counts"][phase_name(value)] += 1
                try:
                    group["plies"].append(int(value.get("ply", 0)))
                except (TypeError, ValueError):
                    pass
                try:
                    group["legal_moves"].append(int(value.get("legal_moves", 0)))
                except (TypeError, ValueError):
                    pass
                if value.get("game_key") is not None:
                    group["game_keys"].add(str(value["game_key"]))
                if value.get("source") is not None:
                    group["sources"].add(str(value["source"]))
                if value.get("opening_family") is not None:
                    group["opening_families"].add(str(value["opening_family"]))
                if value.get("license_id") is not None:
                    group["license_ids"].add(str(value["license_id"]))
                if value.get("source_uri") is not None:
                    group["source_uris"].add(str(value["source_uri"]))
    return groups, dict(stats)


def median(values: list[int]) -> int | None:
    if not values:
        return None
    values = sorted(values)
    return values[(len(values) - 1) // 2]


def write_outputs(
    groups: dict[str, dict[str, Any]],
    output_dir: pathlib.Path,
    stats: dict[str, Any],
    source_files: list[pathlib.Path],
) -> dict[str, Any]:
    by_position: dict[str, set[str]] = collections.defaultdict(set)
    for group in groups.values():
        by_position[group["position_key"]].add(group["split"])
    collision_positions = {
        key for key, splits in by_position.items() if len(splits) > 1
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        split: (output_dir / f"{split}.jsonl").open("w", encoding="utf-8")
        for split in ("train", "valid", "test")
    }
    try:
        counters = collections.Counter()
        for group in sorted(groups.values(), key=lambda item: (item["split"], item["position_key"])):
            split = group["split"]
            key = group["position_key"]
            if key in collision_positions:
                counters[f"cross_split_collision_{split}"] += 1
                continue
            moves = [
                {"move": move, "count": count}
                for move, count in sorted(group["moves"].items())
            ]
            phases = group["phase_counts"]
            phase = max(PHASES, key=lambda item: (phases[item], -PHASES.index(item)))
            row = {
                "schema": "halfkp_move_frequency_v1",
                "split": split,
                "game_key": sorted(group["game_keys"])[0] if group["game_keys"] else "frequency-group",
                "opening_family": sorted(group["opening_families"])[0]
                if group["opening_families"]
                else "frequency-group",
                "license_id": sorted(group["license_ids"])[0]
                if group["license_ids"]
                else "user-authorized:wdoor-public-archive",
                "source_uri": sorted(group["source_uris"])[0]
                if group["source_uris"]
                else "https://wdoor.c.u-tokyo.ac.jp/shogi/",
                "position_key": key,
                "sfen": group["sfen"],
                "side_to_move": group["side_to_move"],
                "phase": phase,
                "phase_counts": dict(sorted(phases.items())),
                "ply_median": median(group["plies"]),
                "legal_moves_median": median(group["legal_moves"]),
                "teacher_moves": moves,
                "total_observations": sum(item["count"] for item in moves),
                "unique_moves": len(moves),
                "unique_games": len(group["game_keys"]),
                "source_count": len(group["sources"]),
            }
            handles[split].write(json.dumps(row, separators=(",", ":")) + "\n")
            counters[f"positions_{split}"] += 1
            counters[f"observations_{split}"] += row["total_observations"]
            if row["unique_moves"] > 1:
                counters[f"multi_move_positions_{split}"] += 1
                counters[f"multi_move_observations_{split}"] += row["total_observations"]
            counters[f"phase_{split}_{phase}"] += 1
    finally:
        for handle in handles.values():
            handle.close()

    result = {
        "schema": "halfkp_move_frequency_manifest_v1",
        "source_files": [str(path) for path in source_files],
        "source_sha256": source_sha256(source_files),
        "raw_records": stats.get("raw_records", 0),
        "malformed_records": stats.get("malformed_records", 0),
        "input_groups": len(groups),
        "cross_split_collision_positions": len(collision_positions),
        "output_files": [
            {
                "path": f"{split}.jsonl",
                "sha256": hashlib.sha256(
                    (output_dir / f"{split}.jsonl").read_bytes()
                ).hexdigest(),
                "records": counters.get(f"positions_{split}", 0),
            }
            for split in ("train", "valid", "test")
        ],
        **dict(sorted(counters.items())),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--min-player-rate", type=int)
    parser.add_argument("--min-opponent-rate", type=int)
    args = parser.parse_args()
    groups, stats = read_groups(args.input)
    result = write_outputs(groups, args.output_dir, stats, args.input)
    result["min_player_rate"] = args.min_player_rate
    result["min_opponent_rate"] = args.min_opponent_rate
    (args.output_dir / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

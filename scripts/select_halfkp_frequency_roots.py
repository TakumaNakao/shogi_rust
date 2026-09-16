#!/usr/bin/env python3
"""Select a phase-balanced, game-capped subset of frequency roots.

The input is the output of ``aggregate_halfkp_move_frequencies.py``.  A
deterministic eight-root reservoir is kept per representative game, then the
lowest hash priorities are selected to hit the requested opening/middlegame/
endgame counts.  Valid/test files are copied unchanged; only train is
subsampled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
from collections import defaultdict
from typing import Any


PHASES = ("opening", "middlegame", "endgame")


def priority(seed: int, row: dict[str, Any]) -> bytes:
    return hashlib.sha256(
        f"halfkp-frequency-select-v1\0{seed}\0{row.get('split')}\0{row.get('position_key')}".encode()
    ).digest()


def select_train(
    path: pathlib.Path, targets: dict[str, int], seed: int, max_per_game: int
) -> tuple[dict[str, list[tuple[bytes, str]]], dict[str, int]]:
    by_game: dict[str, list[tuple[bytes, str, str]]] = defaultdict(list)
    seen = defaultdict(int)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            phase = row.get("phase")
            if phase not in PHASES or targets[phase] <= 0:
                continue
            game = str(row.get("game_key", "frequency-group"))
            key = game
            item = (priority(seed, row), line, str(row.get("position_key", "")))
            seen[phase] += 1
            bucket = by_game[key]
            bucket.append(item)
            bucket.sort(key=lambda value: value[0])
            if len(bucket) > max_per_game:
                bucket.pop()
    candidates: dict[str, list[tuple[bytes, str]]] = defaultdict(list)
    for rows in by_game.values():
        for item in rows:
            row = json.loads(item[1])
            candidates[row["phase"]].append((item[0], item[1]))
    for phase in PHASES:
        candidates[phase].sort(key=lambda value: value[0])
        candidates[phase] = candidates[phase][: targets[phase]]
    return candidates, dict(seen)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--opening", type=int, required=True)
    parser.add_argument("--middlegame", type=int, required=True)
    parser.add_argument("--endgame", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--max-roots-per-game", type=int, default=8)
    args = parser.parse_args()
    if args.max_roots_per_game <= 0 or any(
        value < 0 for value in (args.opening, args.middlegame, args.endgame)
    ):
        parser.error("targets must be non-negative and max-roots-per-game positive")
    targets = {
        "opening": args.opening,
        "middlegame": args.middlegame,
        "endgame": args.endgame,
    }
    input_train = args.input_dir / "train.jsonl"
    candidates, seen = select_train(
        input_train, targets, args.seed, args.max_roots_per_game
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_train = args.output_dir / "train.jsonl"
    with output_train.open("w", encoding="utf-8") as handle:
        selected_rows = [item for phase in PHASES for item in candidates[phase]]
        selected_rows.sort(key=lambda item: item[0])
        for _key, line in selected_rows:
            handle.write(line)
    for split in ("valid", "test"):
        source = args.input_dir / f"{split}.jsonl"
        destination = args.output_dir / f"{split}.jsonl"
        shutil.copyfile(source, destination)
    output_files = []
    for split in ("train", "valid", "test"):
        path = args.output_dir / f"{split}.jsonl"
        output_files.append(
            {
                "path": path.name,
                "sha256": sha256(path),
                "records": sum(
                    1
                    for _ in path.open(encoding="utf-8")
                ),
            }
        )
    selected = {phase: len(candidates[phase]) for phase in PHASES}
    input_manifest = json.loads((args.input_dir / "manifest.json").read_text())
    manifest = {
        "schema": "halfkp_move_frequency_manifest_v1",
        "selection_schema": "halfkp_frequency_selection_manifest_v1",
        "input_dir": str(args.input_dir),
        "input_manifest_sha256": sha256(args.input_dir / "manifest.json"),
        "min_player_rate": input_manifest.get("min_player_rate"),
        "min_opponent_rate": input_manifest.get("min_opponent_rate"),
        "seed": args.seed,
        "max_roots_per_game": args.max_roots_per_game,
        "targets": targets,
        "seen_train_by_phase": seen,
        "selected_train_by_phase": selected,
        "selected_train_records": sum(selected.values()),
        "output_files": output_files,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

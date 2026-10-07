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
import sqlite3
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


def read_groups(
    paths: list[pathlib.Path], db_path: pathlib.Path
) -> tuple[sqlite3.Connection, dict[str, int]]:
    """Ingest records into a disk-backed aggregate.

    The earlier in-memory implementation used several gigabytes for a few
    million unique roots.  SQLite keeps the same exact grouping semantics but
    bounds RAM usage, which is important on the CPU-only training machine.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.executescript(
        """
        DROP TABLE IF EXISTS groups;
        DROP TABLE IF EXISTS moves;
        DROP TABLE IF EXISTS phases;
        DROP TABLE IF EXISTS game_keys;
        DROP TABLE IF EXISTS sources;
        CREATE TABLE groups (
            split TEXT NOT NULL,
            position_key TEXT NOT NULL,
            sfen TEXT NOT NULL,
            side_to_move TEXT NOT NULL,
            game_key TEXT NOT NULL,
            opening_family TEXT NOT NULL,
            license_id TEXT NOT NULL,
            source_uri TEXT NOT NULL,
            total_observations INTEGER NOT NULL DEFAULT 0,
            ply_sum INTEGER NOT NULL DEFAULT 0,
            ply_count INTEGER NOT NULL DEFAULT 0,
            legal_sum INTEGER NOT NULL DEFAULT 0,
            legal_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (split, position_key)
        );
        CREATE TABLE moves (
            split TEXT NOT NULL,
            position_key TEXT NOT NULL,
            move TEXT NOT NULL,
            count INTEGER NOT NULL,
            PRIMARY KEY (split, position_key, move)
        );
        CREATE TABLE phases (
            split TEXT NOT NULL,
            position_key TEXT NOT NULL,
            phase TEXT NOT NULL,
            count INTEGER NOT NULL,
            PRIMARY KEY (split, position_key, phase)
        );
        CREATE TABLE game_keys (
            split TEXT NOT NULL,
            position_key TEXT NOT NULL,
            game_key TEXT NOT NULL,
            PRIMARY KEY (split, position_key, game_key)
        );
        CREATE TABLE sources (
            split TEXT NOT NULL,
            position_key TEXT NOT NULL,
            source TEXT NOT NULL,
            PRIMARY KEY (split, position_key, source)
        );
        CREATE INDEX groups_position ON groups(position_key);
        """
    )
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

                phase = phase_name(value)
                side_to_move = str(value.get("side_to_move", ""))
                game_key = str(value.get("game_key", "frequency-group"))
                opening_family = str(value.get("opening_family", "frequency-group"))
                license_id = str(
                    value.get("license_id", "user-authorized:wdoor-public-archive")
                )
                source_uri = str(
                    value.get("source_uri", "https://wdoor.c.u-tokyo.ac.jp/shogi/")
                )
                connection.execute(
                    """
                    INSERT INTO groups(
                        split,position_key,sfen,side_to_move,game_key,opening_family,
                        license_id,source_uri,total_observations
                    ) VALUES(?,?,?,?,?,?,?,?,1)
                    ON CONFLICT(split,position_key) DO UPDATE SET
                        total_observations=total_observations+1
                    """,
                    (
                        split,
                        key,
                        sfen,
                        side_to_move,
                        game_key,
                        opening_family,
                        license_id,
                        source_uri,
                    ),
                )
                try:
                    ply = int(value.get("ply", 0))
                except (TypeError, ValueError):
                    ply = 0
                try:
                    legal_moves = int(value.get("legal_moves", 0))
                except (TypeError, ValueError):
                    legal_moves = 0
                connection.execute(
                    "UPDATE groups SET ply_sum=ply_sum+?, ply_count=ply_count+1, legal_sum=legal_sum+?, legal_count=legal_count+1 WHERE split=? AND position_key=?",
                    (ply, legal_moves, split, key),
                )
                connection.execute(
                    "INSERT INTO moves(split,position_key,move,count) VALUES(?,?,?,1) ON CONFLICT(split,position_key,move) DO UPDATE SET count=count+1",
                    (split, key, move),
                )
                connection.execute(
                    "INSERT INTO phases(split,position_key,phase,count) VALUES(?,?,?,1) ON CONFLICT(split,position_key,phase) DO UPDATE SET count=count+1",
                    (split, key, phase),
                )
                connection.execute(
                    "INSERT OR IGNORE INTO game_keys(split,position_key,game_key) VALUES(?,?,?)",
                    (split, key, game_key),
                )
                source = str(value.get("source", ""))
                if source:
                    connection.execute(
                        "INSERT OR IGNORE INTO sources(split,position_key,source) VALUES(?,?,?)",
                        (split, key, source),
                    )
        connection.commit()
    connection.commit()
    return connection, dict(stats)


def write_outputs(
    connection: sqlite3.Connection,
    output_dir: pathlib.Path,
    stats: dict[str, Any],
    source_files: list[pathlib.Path],
    shuffle_seed: int | None,
) -> dict[str, Any]:
    connection.executescript(
        """
        DROP TABLE IF EXISTS collision_positions;
        CREATE TEMP TABLE collision_positions AS
          SELECT position_key FROM groups GROUP BY position_key HAVING COUNT(DISTINCT split)>1;
        """
    )
    collision_positions = connection.execute(
        "SELECT COUNT(*) FROM collision_positions"
    ).fetchone()[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        split: (output_dir / f"{split}.jsonl").open("w", encoding="utf-8")
        for split in ("train", "valid", "test")
    }
    try:
        counters = collections.Counter()
        if shuffle_seed is None:
            order_sql = "g.split, g.position_key"
            order_params: tuple[Any, ...] = ()
        else:
            connection.create_function(
                "frequency_order",
                3,
                lambda seed, split, key: hashlib.sha256(
                    f"halfkp-frequency-order-v1\0{seed}\0{split}\0{key}".encode()
                ).digest(),
            )
            order_sql = "g.split, frequency_order(?, g.split, g.position_key)"
            order_params = (shuffle_seed,)
        groups_cursor = connection.execute(
            f"""
            SELECT g.split,g.position_key,g.sfen,g.side_to_move,g.game_key,
                   g.opening_family,g.license_id,g.source_uri,g.total_observations,
                   g.ply_sum,g.ply_count,g.legal_sum,g.legal_count
            FROM groups AS g LEFT JOIN collision_positions AS c
              ON c.position_key=g.position_key
            WHERE c.position_key IS NULL
            ORDER BY {order_sql}
            """,
            order_params,
        )
        for group in groups_cursor:
            (
                split,
                key,
                sfen,
                side_to_move,
                game_key,
                opening_family,
                license_id,
                source_uri,
                total_observations,
                ply_sum,
                ply_count,
                legal_sum,
                legal_count,
            ) = group
            moves = [
                {"move": move, "count": count}
                for move, count in connection.execute(
                    "SELECT move,count FROM moves WHERE split=? AND position_key=? ORDER BY move",
                    (split, key),
                )
            ]
            phases = dict(
                connection.execute(
                    "SELECT phase,count FROM phases WHERE split=? AND position_key=?",
                    (split, key),
                ).fetchall()
            )
            phase = max(PHASES, key=lambda item: (phases.get(item, 0), -PHASES.index(item)))
            unique_games = connection.execute(
                "SELECT COUNT(*) FROM game_keys WHERE split=? AND position_key=?",
                (split, key),
            ).fetchone()[0]
            source_count = connection.execute(
                "SELECT COUNT(*) FROM sources WHERE split=? AND position_key=?",
                (split, key),
            ).fetchone()[0]
            row = {
                "schema": "halfkp_move_frequency_v1",
                "split": split,
                "game_key": game_key or "frequency-group",
                "opening_family": opening_family or "frequency-group",
                "license_id": license_id or "user-authorized:wdoor-public-archive",
                "source_uri": source_uri or "https://wdoor.c.u-tokyo.ac.jp/shogi/",
                "position_key": key,
                "sfen": sfen,
                "side_to_move": side_to_move,
                "phase": phase,
                "phase_counts": dict(sorted(phases.items())),
                "ply_mean": (ply_sum / ply_count) if ply_count else None,
                "legal_moves_mean": (legal_sum / legal_count) if legal_count else None,
                "teacher_moves": moves,
                "total_observations": total_observations,
                "unique_moves": len(moves),
                "unique_games": unique_games,
                "source_count": source_count,
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
        "input_groups": connection.execute("SELECT COUNT(*) FROM groups").fetchone()[0],
        "cross_split_collision_positions": collision_positions,
        "shuffle_seed": shuffle_seed,
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
    parser.add_argument("--shuffle-seed", type=int, default=20260916)
    args = parser.parse_args()
    db_path = args.output_dir / ".move_frequency.sqlite3"
    connection, stats = read_groups(args.input, db_path)
    try:
        result = write_outputs(connection, args.output_dir, stats, args.input, args.shuffle_seed)
    finally:
        connection.close()
        try:
            db_path.unlink()
        except FileNotFoundError:
            pass
    result["min_player_rate"] = args.min_player_rate
    result["min_opponent_rate"] = args.min_opponent_rate
    (args.output_dir / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

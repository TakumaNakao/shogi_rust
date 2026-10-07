#!/usr/bin/env python3
"""Build S09's 2M frequency dataset from frozen 1M + a new WDoor cohort.

Selection is disk-backed so the 1.8M-root source does not require a large
in-memory Python object graph.  It excludes all old split positions and keeps
opening families in the same split across the old and new cohorts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sqlite3
import sys
from collections import Counter
from typing import Any


PHASES = ("opening", "middlegame", "endgame")
TOTAL_TARGETS = {"opening": 416_000, "middlegame": 1_104_000, "endgame": 480_000}
ADD_TARGETS = {"opening": 208_000, "middlegame": 552_000, "endgame": 240_000}
GAME_CAPS = {"opening": 25, "middlegame": 25, "endgame": 14}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_outputs(directory: pathlib.Path, manifest: dict[str, Any]) -> dict[str, pathlib.Path]:
    outputs: dict[str, pathlib.Path] = {}
    for item in manifest.get("output_files", []):
        name = pathlib.Path(item["path"]).name
        path = directory / name
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise RuntimeError(f"manifest output mismatch: {path}")
        outputs[name.removesuffix(".jsonl")] = path
    for split in ("train", "valid", "test"):
        if split not in outputs:
            raise RuntimeError(f"manifest lacks {split}.jsonl")
    return outputs


def position_key(row: dict[str, Any]) -> str:
    key = row.get("position_key")
    if isinstance(key, str) and key:
        return key
    fields = str(row.get("sfen", "")).split()
    if len(fields) < 3:
        raise RuntimeError("record has no usable position identity")
    return " ".join(fields[:3])


def priority(seed: int, phase: str, key: str) -> bytes:
    return hashlib.sha256(f"halfkp-s09-frequency-v1\0{seed}\0{phase}\0{key}".encode()).digest()


def ingest_base(connection: sqlite3.Connection, paths: dict[str, pathlib.Path]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in ("train", "valid", "test"):
        count = 0
        with paths[split].open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                family = str(row.get("opening_family", ""))
                key = position_key(row)
                connection.execute("INSERT OR IGNORE INTO base_positions VALUES(?,?,?)", (key, split, family))
                if family:
                    previous = connection.execute(
                        "SELECT split FROM base_families WHERE family=?", (family,)
                    ).fetchone()
                    if previous is not None and previous[0] != split:
                        raise RuntimeError(f"base opening family crosses splits: {family}")
                    connection.execute("INSERT OR IGNORE INTO base_families VALUES(?,?)", (family, split))
                count += 1
        counts[split] = count
        connection.commit()
    return counts


def ingest_addition(connection: sqlite3.Connection, path: pathlib.Path, seed: int) -> Counter[str]:
    counts: Counter[str] = Counter()
    batch: list[tuple[str, str, str, bytes, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            phase = row.get("phase")
            if phase not in PHASES:
                raise RuntimeError(f"unsupported phase: {phase}")
            key = position_key(row)
            family = str(row.get("opening_family", ""))
            base_split = connection.execute(
                "SELECT split FROM base_families WHERE family=?", (family,)
            ).fetchone() if family else None
            if base_split is not None and base_split[0] != "train":
                counts["excluded_base_eval_family"] += 1
                continue
            if connection.execute("SELECT 1 FROM base_positions WHERE key=?", (key,)).fetchone():
                counts["excluded_base_position"] += 1
                continue
            encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            batch.append((key, phase, str(row.get("game_key", "")), priority(seed, phase, key), encoded))
            counts[f"eligible_{phase}"] += 1
            if len(batch) >= 10_000:
                connection.executemany("INSERT OR IGNORE INTO candidates VALUES(?,?,?,?,?)", batch)
                connection.commit()
                batch.clear()
    if batch:
        connection.executemany("INSERT OR IGNORE INTO candidates VALUES(?,?,?,?,?)", batch)
        connection.commit()
    return counts


def choose_additions(connection: sqlite3.Connection) -> dict[str, int]:
    connection.executescript(
        """
        DROP TABLE IF EXISTS selected;
        CREATE TABLE selected (position_key TEXT PRIMARY KEY, phase TEXT NOT NULL, payload TEXT NOT NULL);
        """
    )
    selected_counts: dict[str, int] = {}
    for phase in PHASES:
        cap = GAME_CAPS[phase]
        target = ADD_TARGETS[phase]
        connection.execute(
            """
            INSERT INTO selected(position_key,phase,payload)
            SELECT position_key,phase,payload FROM (
                SELECT position_key,phase,payload,priority,
                       ROW_NUMBER() OVER(PARTITION BY game_key ORDER BY priority) AS game_rank
                FROM candidates WHERE phase=?
            ) WHERE game_rank<=?
            ORDER BY priority LIMIT ?
            """,
            (phase, cap, target),
        )
        selected_counts[phase] = connection.execute(
            "SELECT COUNT(*) FROM selected WHERE phase=?", (phase,)
        ).fetchone()[0]
        if selected_counts[phase] != target:
            raise RuntimeError(
                f"insufficient collision-free {phase} roots: "
                f"{selected_counts[phase]} < {target}"
            )
        connection.commit()
    return selected_counts


def write_manifest(
    path: pathlib.Path,
    train: pathlib.Path,
    valid: pathlib.Path,
    test: pathlib.Path,
    train_records: int,
    valid_records: int,
    test_records: int,
    base_manifest_sha: str,
    addition_manifest_sha: str,
    base_counts: dict[str, int],
    addition_counts: dict[str, int],
    selected_counts: dict[str, int],
    seed: int,
) -> None:
    files = [file_record(file, count_lines(file)) for file in (train, valid, test)]
    manifest = {
        "schema": "halfkp_move_frequency_manifest_v1",
        "schema_version": 1,
        "min_player_rate": None,
        "min_opponent_rate": None,
        "selected_train_records": train_records,
        "selected_train_by_phase": TOTAL_TARGETS,
        "source_policy": "S07 2015-2017 rated-frequency corpus plus collision-free 2012-2014 WDoor observed-move frequency corpus; historical rating availability is recorded per cohort; no engine evaluation labels",
        "selection_policy": {
            "seed": seed,
            "addition_by_phase": ADD_TARGETS,
            "addition_game_caps_by_phase": GAME_CAPS,
            "exclude_prior_positions": True,
            "opening_family_split_consistent_with_base": True,
        },
        "source_cohorts": [
            {"name": "s07-rated-frequency", "years": [2015, 2016, 2017], "rate_floor": 2500, "dataset_manifest_sha256": base_manifest_sha},
            {"name": "historical-wdoor-observed-moves", "years": [2012, 2013, 2014], "rate_floor": None, "dataset_manifest_sha256": addition_manifest_sha},
        ],
        "base_split_records": base_counts,
        "addition_eligible_records": dict(addition_counts),
        "addition_selected_by_phase": selected_counts,
        "output_files": files,
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_record(path: pathlib.Path, records: int) -> dict[str, Any]:
    return {"path": path.name, "records": records, "sha256": sha256(path)}


def count_lines(path: pathlib.Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def make_control_manifest(
    control_dir: pathlib.Path,
    base_doc: dict[str, Any],
    base_paths: dict[str, pathlib.Path],
    valid_path: pathlib.Path,
    test_path: pathlib.Path,
) -> None:
    control_dir.mkdir(parents=True, exist_ok=True)
    destinations = {name: control_dir / f"{name}.jsonl" for name in ("train", "valid", "test")}
    sources = {"train": base_paths["train"], "valid": valid_path, "test": test_path}
    for name in ("train", "valid", "test"):
        try:
            destinations[name].hardlink_to(sources[name])
        except OSError:
            shutil.copyfile(sources[name], destinations[name])
    output_files = [
        file_record(destinations[name], count_lines(destinations[name]))
        for name in ("train", "valid", "test")
    ]
    control_manifest = {
        "schema": "halfkp_move_frequency_manifest_v1",
        "schema_version": 1,
        # The training split has a 2500 rate floor, but the shared validation
        # and test splits are historical and un-rated.  Avoid misrepresenting
        # a mixed-split manifest with a single corpus-wide floor.
        "min_player_rate": None,
        "min_opponent_rate": None,
        "selected_train_records": base_doc["selected_train_records"],
        "selected_train_by_phase": base_doc.get("selected_train_by_phase", {}),
        "source_policy": "S07 2015-2017 WDoor move-frequency train subset; validation is the same frozen collision-free 2012-2014 holdout used by S09 2M",
        "source_cohorts": [
            {"name": "s07-rated-frequency", "years": [2015, 2016, 2017], "rate_floor": 2500},
            {"name": "shared-holdout-only", "years": [2012, 2013, 2014], "rate_floor": None},
        ],
        "output_files": output_files,
    }
    (control_dir / "manifest.json").write_text(
        json.dumps(control_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_balanced_valid(connection: sqlite3.Connection, path: pathlib.Path, seed: int) -> int:
    targets = {"opening": 2_496, "middlegame": 6_624, "endgame": 2_880}
    # The pool was split by game family before this sampling.  Keep the
    # validation set small and phase-balanced for repeated pass-level checks.
    for phase in PHASES:
        available = connection.execute(
            "SELECT COUNT(*) FROM validation_rows WHERE phase=?", (phase,)
        ).fetchone()[0]
        if available < targets[phase]:
            raise RuntimeError(f"not enough collision-free {phase} validation roots: {available}")
    with path.open("w", encoding="utf-8") as output:
        for phase in PHASES:
            cursor = connection.execute(
                "SELECT payload FROM validation_rows WHERE phase=? ORDER BY priority LIMIT ?",
                (phase, targets[phase]),
            )
            for (payload,) in cursor:
                output.write(payload + "\n")
    return sum(targets.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=pathlib.Path, required=True)
    parser.add_argument("--base-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--addition-dir", type=pathlib.Path, required=True)
    parser.add_argument("--addition-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_doc = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    addition_doc = json.loads(args.addition_manifest.read_text(encoding="utf-8"))
    if base_doc.get("selected_train_records") != 1_000_000:
        raise RuntimeError("base dataset must be the frozen S07 1M dataset")
    base_paths = manifest_outputs(args.base_dir, base_doc)
    addition_paths = manifest_outputs(args.addition_dir, addition_doc)
    base_manifest_sha = sha256(args.base_manifest)
    addition_manifest_sha = sha256(args.addition_manifest)

    db_path = args.output_dir / ".selection.sqlite"
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-32768")
    connection.executescript(
        """
        CREATE TABLE base_positions(key TEXT PRIMARY KEY,split TEXT NOT NULL,family TEXT NOT NULL);
        CREATE TABLE base_families(family TEXT PRIMARY KEY,split TEXT NOT NULL);
        CREATE TABLE candidates(position_key TEXT PRIMARY KEY,phase TEXT NOT NULL,game_key TEXT NOT NULL,priority BLOB NOT NULL,payload TEXT NOT NULL);
        CREATE TABLE final_train(priority BLOB PRIMARY KEY,payload TEXT NOT NULL);
        CREATE TABLE validation_rows(position_key TEXT PRIMARY KEY,phase TEXT NOT NULL,priority BLOB NOT NULL,payload TEXT NOT NULL);
        CREATE INDEX candidate_phase_priority ON candidates(phase,priority);
        CREATE INDEX candidate_game_phase_priority ON candidates(phase,game_key,priority);
        CREATE INDEX validation_phase_priority ON validation_rows(phase,priority);
        """
    )
    base_counts = ingest_base(connection, base_paths)
    addition_counts = ingest_addition(connection, addition_paths["train"], args.seed)
    selected_counts = choose_additions(connection)

    # Deterministically shuffle the full combined stream.  The trainer keeps
    # input order, so concatenating old then new cohorts would bias AdaGrad.
    final_batch: list[tuple[bytes, str]] = []
    for line in base_paths["train"].open(encoding="utf-8"):
        row = json.loads(line)
        key = position_key(row)
        final_batch.append((priority(args.seed, "combined", key), line.rstrip("\n")))
        if len(final_batch) >= 10_000:
            connection.executemany("INSERT OR IGNORE INTO final_train VALUES(?,?)", final_batch)
            connection.commit()
            final_batch.clear()
    if final_batch:
        connection.executemany("INSERT OR IGNORE INTO final_train VALUES(?,?)", final_batch)
        connection.commit()
    for (payload,) in connection.execute("SELECT payload FROM selected"):
        row = json.loads(payload)
        final_batch.append((priority(args.seed, "combined", position_key(row)), payload))
        if len(final_batch) >= 10_000:
            connection.executemany("INSERT OR IGNORE INTO final_train VALUES(?,?)", final_batch)
            connection.commit()
            final_batch.clear()
    if final_batch:
        connection.executemany("INSERT OR IGNORE INTO final_train VALUES(?,?)", final_batch)
        connection.commit()

    train_path = args.output_dir / "train.jsonl"
    valid_path = args.output_dir / "valid.jsonl"
    test_path = args.output_dir / "test.jsonl"
    with train_path.open("w", encoding="utf-8") as output:
        for (payload,) in connection.execute("SELECT payload FROM final_train ORDER BY priority"):
            output.write(payload + "\n")
    base_train_count = base_counts["train"]
    total_train = connection.execute("SELECT COUNT(*) FROM final_train").fetchone()[0]
    if total_train != 2_000_000:
        raise RuntimeError(f"expected 2M train roots, got {total_train}")

    # Keep the new cohort's own held-out games. Exclude exact roots already
    # present in the S07 split and retain the split by opening family.
    eval_root_records = Counter()
    with test_path.open("w", encoding="utf-8") as test_output:
        for name in ("valid", "test"):
            for line in addition_paths[name].open(encoding="utf-8"):
                row = json.loads(line)
                key = position_key(row)
                if connection.execute("SELECT 1 FROM base_positions WHERE key=?", (key,)).fetchone():
                    eval_root_records[f"excluded_{name}_base_position"] += 1
                    continue
                family = str(row.get("opening_family", ""))
                base_split = connection.execute(
                    "SELECT split FROM base_families WHERE family=?", (family,)
                ).fetchone() if family else None
                if base_split is not None and base_split[0] != name:
                    eval_root_records[f"excluded_{name}_base_family"] += 1
                    continue
                if name == "valid":
                    phase = row.get("phase")
                    if phase not in PHASES:
                        raise RuntimeError(f"unsupported validation phase: {phase}")
                    connection.execute(
                        "INSERT OR IGNORE INTO validation_rows VALUES(?,?,?,?)",
                        (key, phase, priority(args.seed, phase, key), json.dumps(row, ensure_ascii=False, separators=(",", ":"))),
                    )
                else:
                    eval_root_records["kept_test"] += 1
                    test_output.write(line if line.endswith("\n") else line + "\n")
            connection.commit()
    eval_root_records["kept_valid"] = write_balanced_valid(connection, valid_path, args.seed)

    write_manifest(
        args.output_dir / "manifest.json", train_path, valid_path, test_path,
        total_train, eval_root_records["kept_valid"], eval_root_records["kept_test"],
        base_manifest_sha, addition_manifest_sha, base_counts, addition_counts,
        selected_counts, args.seed,
    )
    make_control_manifest(args.output_dir / "control", base_doc, base_paths, valid_path, test_path)
    connection.close()
    db_path.unlink(missing_ok=True)
    print(json.dumps({
        "train_records": total_train,
        "base_records": base_counts,
        "addition_selected_by_phase": selected_counts,
        "addition_eligible": dict(addition_counts),
        "valid_records": eval_root_records["kept_valid"],
        "test_records": eval_root_records["kept_test"],
        "manifest_sha256": sha256(args.output_dir / "manifest.json"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)

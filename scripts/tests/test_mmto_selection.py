import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import select_game_balanced_mmto as balanced  # noqa: E402
import select_mmto_dev_openings as openings  # noqa: E402


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(split, game, family, phase, sfen, ply, marker):
    return {
        "schema": "rank_value_dataset_v3_position",
        "split": split,
        "game_key": game,
        "opening_family": family,
        "phase": phase,
        "sfen": sfen,
        "ply": ply,
        "marker": marker,
    }


def write_jsonl(path: Path, rows):
    lines = []
    with path.open("wb") as output:
        for index, row in enumerate(rows):
            # Vary harmless whitespace so byte preservation is observable.
            payload = json.dumps(row, ensure_ascii=False, separators=(",", ": "))
            line = (payload + (" " if index % 2 else "") + "\n").encode()
            output.write(line)
            lines.append(line)
    return lines


def write_source(root: Path, train_rows, valid_rows, test_rows):
    declared = []
    source_lines = {}
    for name, rows in (
        ("train.jsonl", train_rows),
        ("valid.jsonl", valid_rows),
        ("test.jsonl", test_rows),
    ):
        path = root / name
        source_lines[name] = write_jsonl(path, rows)
        declared.append(
            {"path": name, "sha256": file_sha256(path), "records": len(rows)}
        )
    manifest = {
        "schema_version": 4,
        "schema": "rank_value_dataset_v4_manifest",
        "min_player_rate": 4000,
        "min_opponent_rate": 4000,
        "output_files": declared,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    return manifest_path, source_lines


class GameBalancedSelectionTests(unittest.TestCase):
    def test_display_path_is_repository_relative_independent_of_cwd(self):
        target = balanced.REPOSITORY_ROOT / "docs"
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            try:
                os.chdir(temporary)
                self.assertEqual(balanced.display_path(target), "docs")
            finally:
                os.chdir(original)

    def test_deterministic_coverage_byte_preservation_and_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = [
                record("train", "t1", "tf1", "opening", "t1-o1", 10, "a"),
                record("train", "t1", "tf1", "opening", "t1-o2", 20, "b"),
                record("train", "t1", "tf2", "middlegame", "t1-m", 50, "c"),
                record("train", "t2", "tf3", "opening", "t2-o", 30, "d"),
                record("train", "t2", "tf4", "endgame", "t2-e", 100, "e"),
                record("train", "t3", "tf5", "middlegame", "t3-m", 60, "f"),
                record("train", "t3", "tf6", "endgame", "t3-e1", 110, "g"),
                record("train", "t3", "tf6", "endgame", "t3-e2", 120, "h"),
            ]
            valid = [
                record("valid", "v1", "vf1", "opening", "v1-o1", 10, "i"),
                record("valid", "v1", "vf2", "opening", "v1-o2", 20, "j"),
                record("valid", "v1", "vf3", "middlegame", "v1-m", 50, "k"),
                record("valid", "v2", "vf4", "opening", "v2-o", 30, "l"),
                record("valid", "v2", "vf5", "endgame", "v2-e", 100, "m"),
                record("valid", "v3", "vf6", "opening", "v3-o", 15, "n"),
                record("valid", "v3", "vf7", "middlegame", "v3-m", 60, "o"),
                record("valid", "v3", "vf8", "endgame", "v3-e", 110, "p"),
            ]
            manifest_path, source_lines = write_source(root, train, valid, [])

            first = root / "derived-a"
            second = root / "derived-b"
            manifest, reused = balanced.build_dataset(
                manifest_path, first, seed=77, train_records=7, valid_records=5
            )
            self.assertFalse(reused)
            self.assertEqual(manifest["stats"]["train"]["selected"]["records"], 7)
            self.assertEqual(
                manifest["stats"]["train"]["selected"]["game_phase_groups"],
                manifest["stats"]["train"]["source"]["game_phase_groups"],
            )
            self.assertEqual(
                manifest["stats"]["valid"]["selected"]["games"],
                manifest["stats"]["valid"]["source"]["games"],
            )
            self.assertEqual(
                manifest["stats"]["overlap"], {"game_keys": 0, "exact_sfen": 0}
            )

            train_output = (first / "train.jsonl").read_bytes().splitlines(keepends=True)
            valid_output = (first / "valid.jsonl").read_bytes().splitlines(keepends=True)
            self.assertTrue(all(line in source_lines["train.jsonl"] for line in train_output))
            self.assertTrue(all(line in source_lines["valid.jsonl"] for line in valid_output))

            balanced.build_dataset(
                manifest_path, second, seed=77, train_records=7, valid_records=5
            )
            self.assertEqual((first / "train.jsonl").read_bytes(), (second / "train.jsonl").read_bytes())
            self.assertEqual((first / "valid.jsonl").read_bytes(), (second / "valid.jsonl").read_bytes())
            self.assertEqual((first / "manifest.json").read_bytes(), (second / "manifest.json").read_bytes())

            before = (first / "manifest.json").stat().st_mtime_ns
            _, reused = balanced.build_dataset(
                manifest_path, first, seed=77, train_records=7, valid_records=5
            )
            self.assertTrue(reused)
            self.assertEqual(before, (first / "manifest.json").stat().st_mtime_ns)
            with self.assertRaises(balanced.SelectionError):
                balanced.build_dataset(
                    manifest_path, first, seed=78, train_records=7, valid_records=5
                )


class OpeningSelectionTests(unittest.TestCase):
    def test_exclusions_uniqueness_determinism_and_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = [
                record("train", "t1", "tf", "opening", "blocked-train", 10, "t")
            ]
            test = [
                record("test", "x1", "xf", "opening", "blocked-test", 10, "x")
            ]
            valid = [
                record("valid", "v1", "a", "opening", "blocked-train", 10, "a"),
                record("valid", "v2", "b", "opening", "blocked-test", 12, "b"),
                record("valid", "v3", "c", "opening", "blocked-final", 14, "c"),
                record("valid", "v4", "d", "opening", "candidate-1", 16, "d"),
                record("valid", "v5", "e", "opening", "candidate-2", 18, "e"),
                record("valid", "v6", "f", "opening", "candidate-3", 20, "f"),
                record("valid", "v7", "f", "opening", "candidate-4", 22, "g"),
                record("valid", "v8", "h", "opening", "candidate-5", 24, "h"),
                record("valid", "v9", "i", "middlegame", "too-late", 41, "i"),
            ]
            manifest_path, source_lines = write_source(root, train, valid, test)
            reserve_path = root / "final.jsonl"
            reserve_rows = [
                {
                    "schema": "halfkp_final_opening_reserve_v1",
                    "split": "valid",
                    "opening_family": "reserve",
                    "sfen": "blocked-final",
                    "ply": 20,
                }
            ]
            write_jsonl(reserve_path, reserve_rows)
            reserve_manifest = {
                "schema_version": 1,
                "schema": "halfkp_final_opening_reserve_v1_manifest",
                "artifact": {
                    "path": reserve_path.name,
                    "sha256": file_sha256(reserve_path),
                    "records": 1,
                },
            }
            reserve_path.with_suffix(".manifest.json").write_text(
                json.dumps(reserve_manifest) + "\n"
            )

            first = root / "openings-a"
            second = root / "openings-b"
            manifest, reused = openings.build_openings(
                manifest_path, reserve_path, first, seed=20260731, pairs=3
            )
            self.assertFalse(reused)
            selected_lines = (first / "openings.jsonl").read_bytes().splitlines(keepends=True)
            self.assertEqual(len(selected_lines), 3)
            self.assertTrue(all(line in source_lines["valid.jsonl"] for line in selected_lines))
            selected = [json.loads(line) for line in selected_lines]
            self.assertEqual(len({row["opening_family"] for row in selected}), 3)
            self.assertEqual(len({row["sfen"] for row in selected}), 3)
            self.assertFalse(
                {row["sfen"] for row in selected}
                & {"blocked-train", "blocked-test", "blocked-final"}
            )
            self.assertEqual(
                manifest["stats"]["output_overlap"],
                {
                    "train_exact_sfen": 0,
                    "test_exact_sfen": 0,
                    "final_reserve_exact_sfen": 0,
                },
            )
            self.assertEqual(
                manifest["stats"]["excluded_records"],
                {"train": 1, "test": 1, "final_reserve": 1},
            )

            openings.build_openings(
                manifest_path, reserve_path, second, seed=20260731, pairs=3
            )
            self.assertEqual((first / "openings.jsonl").read_bytes(), (second / "openings.jsonl").read_bytes())
            self.assertEqual((first / "manifest.json").read_bytes(), (second / "manifest.json").read_bytes())
            _, reused = openings.build_openings(
                manifest_path, reserve_path, first, seed=20260731, pairs=3
            )
            self.assertTrue(reused)


if __name__ == "__main__":
    unittest.main()

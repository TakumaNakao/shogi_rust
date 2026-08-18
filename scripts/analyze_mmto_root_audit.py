#!/usr/bin/env python3
"""Compare per-root MMTO audit JSONL files with game-cluster bootstrap."""

import argparse
import json
import random
from pathlib import Path


def load(path: Path):
    rows = [json.loads(line) for line in path.open()]
    by_root = {}
    for row in rows:
        root_key = row.get("root_key") or row.get("root_hash")
        if not root_key or root_key in by_root:
            raise SystemExit(f"audit has missing/duplicate root_key: {path}")
        by_root[root_key] = row
    return rows, by_root


def bootstrap(values, seed, iterations=100000):
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iterations):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return [means[int(iterations * 0.025)], means[int(iterations * 0.975) - 1]]


def game_means(rows, field):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["game_key"], []).append(float(row[field]))
    return {
        game: sum(values) / len(values)
        for game, values in grouped.items()
    }


def summary(rows):
    count = len(rows)
    pair_count = sum(row["pairs"] for row in rows)
    loss_games = game_means(rows, "mean_pair_loss")
    regret_games = game_means(rows, "regret_cp")
    return {
        "roots": count,
        "games": len(loss_games),
        "agreement": sum(row["agreement"] for row in rows) / count,
        "mean_pair_loss_game_equal": sum(loss_games.values()) / len(loss_games),
        "mean_pair_loss_root_equal": sum(row["mean_pair_loss"] for row in rows) / count,
        "mean_pair_loss_pair_weighted": sum(
            row["mean_pair_loss"] * row["pairs"] for row in rows
        ) / pair_count,
        "mean_regret_cp_game_equal": sum(regret_games.values()) / len(regret_games),
        "mean_regret_cp_root_equal": sum(row["regret_cp"] for row in rows) / count,
        "qnodes_per_root": sum(row["qnodes"] for row in rows) / count,
        "rejected_bounds": sum(row["rejected_bounds"] for row in rows),
    }


def compare(parent, candidate, seed, require_exact_root_set=True, bootstrap_resamples=100000):
    if require_exact_root_set and set(parent) != set(candidate):
        missing = sorted(set(parent) - set(candidate))[:3]
        extra = sorted(set(candidate) - set(parent))[:3]
        raise SystemExit(f"root set mismatch: missing={missing} extra={extra}")
    keys = sorted(set(parent) & set(candidate))
    loss_by_game = {}
    regret_by_game = {}
    transitions = {
        "unchanged": 0,
        "wrong_to_teacher": 0,
        "teacher_to_wrong": 0,
        "non_teacher_to_non_teacher": 0,
    }
    for key in keys:
        before = parent[key]
        after = candidate[key]
        if before["game_key"] != after["game_key"]:
            raise SystemExit(f"game key mismatch for root {key}")
        loss_by_game.setdefault(before["game_key"], []).append(
            after["mean_pair_loss"] - before["mean_pair_loss"]
        )
        regret_by_game.setdefault(before["game_key"], []).append(
            after["regret_cp"] - before["regret_cp"]
        )
        before_teacher = before["best_move"] == before["teacher_move"]
        after_teacher = after["best_move"] == after["teacher_move"]
        if before["best_move"] == after["best_move"]:
            transitions["unchanged"] += 1
        elif not before_teacher and after_teacher:
            transitions["wrong_to_teacher"] += 1
        elif before_teacher and not after_teacher:
            transitions["teacher_to_wrong"] += 1
        else:
            transitions["non_teacher_to_non_teacher"] += 1
    game_loss_delta = [sum(values) / len(values) for values in loss_by_game.values()]
    game_regret_delta = [sum(values) / len(values) for values in regret_by_game.values()]
    return {
        "roots": len(keys),
        "games": len(game_loss_delta),
        "loss_delta_mean_game_equal": sum(game_loss_delta) / len(game_loss_delta),
        "loss_delta_bootstrap_95_game_equal": bootstrap(
            game_loss_delta, seed, bootstrap_resamples
        ),
        "regret_delta_mean_cp_game_equal": sum(game_regret_delta) / len(game_regret_delta),
        "regret_delta_bootstrap_95_game_equal": bootstrap(
            game_regret_delta, seed + 1, bootstrap_resamples
        ),
        "best_move_transitions": transitions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--candidate", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-exact-root-set", action="store_true", default=True)
    parser.add_argument("--cluster", choices=("game_key",), default="game_key")
    parser.add_argument("--bootstrap-resamples", type=int, default=100000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260823)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    parent_rows, parent = load(args.parent)
    report = {
        "schema": "halfkp_mmto_root_audit_comparison_v1",
        "parent": {"path": str(args.parent), "summary": summary(parent_rows)},
        "comparisons": {},
    }
    for index, (label, path_text) in enumerate(args.candidate):
        rows, candidate = load(Path(path_text))
        report["comparisons"][label] = {
            "path": path_text,
            "summary": summary(rows),
            "versus_parent": compare(
                parent,
                candidate,
                args.bootstrap_seed if args.seed is None else args.seed + index * 10,
                args.require_exact_root_set,
                args.bootstrap_resamples,
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare per-root MMTO audit JSONL files with game-cluster bootstrap."""

import argparse
import json
import random
from pathlib import Path


def load(path: Path):
    rows = [json.loads(line) for line in path.open()]
    by_game = {row["game_key"]: row for row in rows}
    if len(by_game) != len(rows):
        raise SystemExit(f"audit is not one-root-per-game: {path}")
    return rows, by_game


def bootstrap(values, seed, iterations=10000):
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iterations):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return [means[int(iterations * 0.025)], means[int(iterations * 0.975) - 1]]


def summary(rows):
    count = len(rows)
    pair_count = sum(row["pairs"] for row in rows)
    return {
        "roots": count,
        "games": len({row["game_key"] for row in rows}),
        "agreement": sum(row["agreement"] for row in rows) / count,
        "mean_pair_loss_root_equal": sum(row["mean_pair_loss"] for row in rows) / count,
        "mean_pair_loss_pair_weighted": sum(
            row["mean_pair_loss"] * row["pairs"] for row in rows
        ) / pair_count,
        "mean_regret_cp": sum(row["regret_cp"] for row in rows) / count,
        "qnodes_per_root": sum(row["qnodes"] for row in rows) / count,
        "rejected_bounds": sum(row["rejected_bounds"] for row in rows),
    }


def compare(parent, candidate, seed):
    keys = sorted(set(parent) & set(candidate))
    loss_delta = [candidate[key]["mean_pair_loss"] - parent[key]["mean_pair_loss"] for key in keys]
    regret_delta = [candidate[key]["regret_cp"] - parent[key]["regret_cp"] for key in keys]
    transitions = {
        "unchanged": 0,
        "wrong_to_teacher": 0,
        "teacher_to_wrong": 0,
        "non_teacher_to_non_teacher": 0,
    }
    for key in keys:
        before = parent[key]
        after = candidate[key]
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
    return {
        "roots": len(keys),
        "loss_delta_mean": sum(loss_delta) / len(loss_delta),
        "loss_delta_bootstrap_95": bootstrap(loss_delta, seed),
        "regret_delta_mean_cp": sum(regret_delta) / len(regret_delta),
        "regret_delta_bootstrap_95": bootstrap(regret_delta, seed + 1),
        "best_move_transitions": transitions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--candidate", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260802)
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
            "versus_parent": compare(parent, candidate, args.seed + index * 10),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

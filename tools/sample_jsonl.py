#!/usr/bin/env python3
"""Create deterministic JSONL subsets without loading the whole file."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


def reject_reason(record: dict[str, Any], args: argparse.Namespace) -> str | None:
    if args.exclude_in_check and record.get("in_check") is True:
        return "in_check"
    legal_moves = record.get("legal_moves")
    if (
        args.min_legal_moves > 0
        and isinstance(legal_moves, int)
        and legal_moves < args.min_legal_moves
    ):
        return "few_legal_moves"
    if args.max_abs_root_score_cp > 0.0:
        root_score = record.get("teacher_root_score")
        if isinstance(root_score, (int, float)) and abs(float(root_score)) > args.max_abs_root_score_cp:
            return "root_score_out_of_range"
    if args.max_selected_regret_cp > 0.0:
        selected_regrets = [
            float(candidate.get("regret", 0.0))
            for candidate in record.get("candidates", [])
            if isinstance(candidate, dict) and candidate.get("selected_by_student") is True
        ]
        if selected_regrets and min(selected_regrets) > args.max_selected_regret_cp:
            return "selected_regret_out_of_range"
    if args.max_candidate_regret_cp > 0.0:
        candidate_regrets = [
            float(candidate.get("regret", 0.0))
            for candidate in record.get("candidates", [])
            if isinstance(candidate, dict)
        ]
        if candidate_regrets and max(candidate_regrets) > args.max_candidate_regret_cp:
            return "candidate_regret_out_of_range"
    return None


def iter_eligible_lines(input_path: Path, args: argparse.Namespace):
    stats = {
        "seen": 0,
        "eligible": 0,
        "invalid_json": 0,
        "in_check": 0,
        "few_legal_moves": 0,
        "root_score_out_of_range": 0,
        "selected_regret_out_of_range": 0,
        "candidate_regret_out_of_range": 0,
    }
    with input_path.open("r", encoding="utf-8") as src:
        for line_number, line in enumerate(src, start=1):
            stats["seen"] += 1
            if not line.strip():
                continue
            if (
                args.exclude_in_check
                or args.min_legal_moves > 0
                or args.max_abs_root_score_cp > 0.0
                or args.max_selected_regret_cp > 0.0
                or args.max_candidate_regret_cp > 0.0
            ):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    stats["invalid_json"] += 1
                    continue
                reason = reject_reason(record, args)
                if reason is not None:
                    stats[reason] += 1
                    continue
            stats["eligible"] += 1
            yield line_number, line, stats


def write_head(input_path: Path, output_path: Path, limit: int, args: argparse.Namespace) -> tuple[int, dict[str, int]]:
    written = 0
    last_stats: dict[str, int] = {}
    with output_path.open("w", encoding="utf-8") as dst:
        for _, line, stats in iter_eligible_lines(input_path, args):
            last_stats = stats
            if limit > 0 and written >= limit:
                break
            dst.write(line)
            written += 1
    return written, last_stats


def write_reservoir(
    input_path: Path, output_path: Path, limit: int, seed: int, args: argparse.Namespace
) -> tuple[int, dict[str, int]]:
    if limit <= 0:
        return write_head(input_path, output_path, limit, args)

    rng = random.Random(seed)
    reservoir: list[tuple[int, str]] = []
    seen = 0
    last_stats: dict[str, int] = {}
    for line_number, line, stats in iter_eligible_lines(input_path, args):
        last_stats = stats
        seen += 1
        if len(reservoir) < limit:
            reservoir.append((line_number, line))
            continue
        replace_idx = rng.randrange(seen)
        if replace_idx < limit:
            reservoir[replace_idx] = (line_number, line)

    reservoir.sort(key=lambda item: item[0])
    with output_path.open("w", encoding="utf-8") as dst:
        for _, line in reservoir:
            dst.write(line)
    return len(reservoir), last_stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample lines from a JSONL file.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lines", required=True, type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=("head", "shuffle"), default="shuffle")
    parser.add_argument("--exclude-in-check", action="store_true")
    parser.add_argument("--min-legal-moves", type=int, default=0)
    parser.add_argument("--max-abs-root-score-cp", type=float, default=0.0)
    parser.add_argument("--max-selected-regret-cp", type=float, default=0.0)
    parser.add_argument("--max-candidate-regret-cp", type=float, default=0.0)
    args = parser.parse_args()

    if args.lines < 0:
        raise SystemExit("--lines must be non-negative")
    if not args.input.is_file():
        raise SystemExit(f"missing input: {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "head":
        written, stats = write_head(args.input, args.output, args.lines, args)
    else:
        written, stats = write_reservoir(args.input, args.output, args.lines, args.seed, args)
    if args.lines > 0 and written < args.lines:
        print(
            f"warning: requested {args.lines} lines but only wrote {written}",
            file=sys.stderr,
        )
    stats_text = " ".join(f"{key}={value}" for key, value in sorted(stats.items()))
    print(
        f"sample_jsonl mode={args.mode} seed={args.seed} lines={written} output={args.output} {stats_text}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

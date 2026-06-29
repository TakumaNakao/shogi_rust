#!/usr/bin/env python3
"""Convert bench_failure_miner JSONL into mmto_tree_dump JSONL input."""

import argparse
import json
from pathlib import Path


WEIGHT_MODES = ("none", "regret", "combined")
BAD_MOVE_SOURCES = ("timed", "actual", "timed-or-actual", "actual-or-timed")


def finite_number(value: object) -> float | None:
    if isinstance(value, (int, float)):
        rendered = float(value)
        if rendered == rendered and rendered not in (float("inf"), float("-inf")):
            return rendered
    return None


def clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def select_bad_move(item: dict, source: str) -> tuple[str, float | None]:
    timed_move = clean_text(item.get("timed_move"))
    actual_move = clean_text(item.get("actual_move"))
    timed_regret = finite_number(item.get("timed_regret_cp"))
    actual_regret = finite_number(item.get("actual_regret_cp"))

    if source == "timed":
        return timed_move, timed_regret
    if source == "actual":
        return actual_move, actual_regret
    if source == "timed-or-actual":
        if timed_move:
            return timed_move, timed_regret
        return actual_move, actual_regret
    if actual_move:
        return actual_move, actual_regret
    return timed_move, timed_regret


def compute_sample_weight(
    mode: str,
    regret: float,
    actual_minus_timed_regret: float,
    scale_cp: float,
    max_sample_weight: float,
) -> float:
    if mode == "none":
        return 1.0
    if mode == "regret":
        signal = max(0.0, regret)
    else:
        signal = max(max(0.0, regret), max(0.0, actual_minus_timed_regret))
    return min(max_sample_weight, 1.0 + signal / scale_cp)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract explicit teacher/student bad move pairs from bench failure JSONL."
    )
    parser.add_argument("--input", required=True, type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bad-move-source", choices=BAD_MOVE_SOURCES, default="timed-or-actual")
    parser.add_argument("--min-regret-cp", type=float, default=150.0)
    parser.add_argument("--max-regret-cp", type=float, default=100000.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dedupe-sfen", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclude-in-check", action="store_true")
    parser.add_argument("--require-new-loss", action="store_true")
    parser.add_argument("--require-bad-flag", action="store_true")
    parser.add_argument("--weight-mode", choices=WEIGHT_MODES, default="combined")
    parser.add_argument("--weight-scale-cp", type=float, default=100.0)
    parser.add_argument("--max-sample-weight", type=float, default=5.0)
    args = parser.parse_args()

    if args.min_regret_cp < 0.0:
        raise SystemExit("--min-regret-cp must be non-negative")
    if args.max_regret_cp < 0.0:
        raise SystemExit("--max-regret-cp must be non-negative")
    if args.max_regret_cp > 0.0 and args.max_regret_cp < args.min_regret_cp:
        raise SystemExit("--max-regret-cp must be >= --min-regret-cp, or 0 to disable")
    if args.weight_scale_cp <= 0.0:
        raise SystemExit("--weight-scale-cp must be positive")
    if args.max_sample_weight < 1.0:
        raise SystemExit("--max-sample-weight must be >= 1")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen_sfens: set[str] = set()
    counts = {
        "read": 0,
        "written": 0,
        "skipped_json": 0,
        "skipped_result": 0,
        "skipped_in_check": 0,
        "skipped_moves": 0,
        "skipped_regret": 0,
        "skipped_bad_flag": 0,
        "skipped_dedupe": 0,
    }

    with args.output.open("w", encoding="utf-8") as out:
        for path in args.input:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    counts["read"] += 1
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        counts["skipped_json"] += 1
                        continue
                    if not isinstance(item, dict):
                        counts["skipped_json"] += 1
                        continue
                    if args.require_new_loss and clean_text(item.get("result")) != "BaselineWin":
                        counts["skipped_result"] += 1
                        continue
                    if args.exclude_in_check and bool(item.get("in_check")):
                        counts["skipped_in_check"] += 1
                        continue

                    sfen = clean_text(item.get("sfen"))
                    teacher_move = clean_text(item.get("teacher_move"))
                    bad_move, regret = select_bad_move(item, args.bad_move_source)
                    if not sfen or not teacher_move or not bad_move or teacher_move == bad_move:
                        counts["skipped_moves"] += 1
                        continue
                    if regret is None or regret < args.min_regret_cp:
                        counts["skipped_regret"] += 1
                        continue
                    if args.max_regret_cp > 0.0 and regret > args.max_regret_cp:
                        counts["skipped_regret"] += 1
                        continue
                    if args.require_bad_flag:
                        flag_name = (
                            "timed_bad"
                            if args.bad_move_source.startswith("timed")
                            else "actual_bad"
                        )
                        if not bool(item.get(flag_name)):
                            counts["skipped_bad_flag"] += 1
                            continue
                    if args.dedupe_sfen and sfen in seen_sfens:
                        counts["skipped_dedupe"] += 1
                        continue
                    seen_sfens.add(sfen)

                    actual_minus_timed = (
                        finite_number(item.get("actual_minus_timed_regret_cp")) or 0.0
                    )
                    sample_weight = compute_sample_weight(
                        args.weight_mode,
                        regret,
                        actual_minus_timed,
                        args.weight_scale_cp,
                        args.max_sample_weight,
                    )
                    out.write(
                        json.dumps(
                            {
                                "sfen": sfen,
                                "sample_weight": sample_weight,
                                "teacher_move": teacher_move,
                                "student_move": bad_move,
                                "source": "bench_failure",
                                "record": clean_text(item.get("record")),
                                "ply": item.get("ply"),
                                "bad_move_source": args.bad_move_source,
                                "bad_regret": regret,
                                "input_file": str(path),
                                "input_line": line_number,
                            },
                            ensure_ascii=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    counts["written"] += 1
                    if args.limit > 0 and counts["written"] >= args.limit:
                        break
            if args.limit > 0 and counts["written"] >= args.limit:
                break

    for key, value in counts.items():
        print(f"{key}={value}")
    print(f"output={args.output}")
    return 0 if counts["written"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

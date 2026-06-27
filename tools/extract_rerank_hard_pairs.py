#!/usr/bin/env python3
"""Extract explicit teacher/bad move pairs from mmto_rerank_gate JSON."""

import argparse
import json
from pathlib import Path


def finite_number(value: object) -> float | None:
    if isinstance(value, (int, float)):
        rendered = float(value)
        if rendered == rendered and rendered not in (float("inf"), float("-inf")):
            return rendered
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert mmto_rerank_gate hard_positions into mmto_tree_dump JSONL input."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-regret-delta-cp", type=float, default=0.0)
    parser.add_argument("--min-candidate-regret-cp", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    hard_positions = payload.get("hard_positions", [])
    if not isinstance(hard_positions, list):
        raise SystemExit("input JSON does not contain a hard_positions array")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with args.output.open("w", encoding="utf-8") as out:
        for item in hard_positions:
            if not isinstance(item, dict):
                skipped += 1
                continue
            sfen = str(item.get("sfen", "")).strip()
            teacher_move = str(item.get("teacher_best_move", "")).strip()
            candidate_move = str(item.get("candidate_move", "")).strip()
            regret_delta = finite_number(item.get("regret_delta"))
            candidate_regret = finite_number(item.get("candidate_regret"))
            if not sfen or not teacher_move or not candidate_move:
                skipped += 1
                continue
            if regret_delta is None or regret_delta < args.min_regret_delta_cp:
                skipped += 1
                continue
            if candidate_regret is None or candidate_regret < args.min_candidate_regret_cp:
                skipped += 1
                continue
            out.write(
                json.dumps(
                    {
                        "sfen": sfen,
                        "teacher_move": teacher_move,
                        "student_move": candidate_move,
                        "regret_delta": regret_delta,
                        "candidate_regret": candidate_regret,
                    },
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            written += 1
            if args.limit > 0 and written >= args.limit:
                break

    print(f"written={written} skipped={skipped} output={args.output}")
    if written == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

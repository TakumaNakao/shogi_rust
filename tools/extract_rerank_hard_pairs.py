#!/usr/bin/env python3
"""Extract explicit teacher/bad move pairs from mmto_rerank_gate JSON."""

import argparse
import json
from pathlib import Path


WEIGHT_MODES = ("none", "regret-delta", "candidate-regret", "combined")


def finite_number(value: object) -> float | None:
    if isinstance(value, (int, float)):
        rendered = float(value)
        if rendered == rendered and rendered not in (float("inf"), float("-inf")):
            return rendered
    return None


def compute_sample_weight(
    mode: str,
    regret_delta: float,
    candidate_regret: float,
    scale_cp: float,
    max_sample_weight: float,
) -> float:
    if mode == "none":
        return 1.0
    if mode == "regret-delta":
        signal = max(0.0, regret_delta)
    elif mode == "candidate-regret":
        signal = max(0.0, candidate_regret)
    elif mode == "combined":
        signal = max(max(0.0, regret_delta), max(0.0, candidate_regret))
    else:
        raise ValueError(f"unsupported weight mode: {mode}")
    return min(max_sample_weight, 1.0 + signal / scale_cp)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert mmto_rerank_gate hard_positions into mmto_tree_dump JSONL input."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-regret-delta-cp", type=float, default=0.0)
    parser.add_argument("--min-candidate-regret-cp", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--weight-mode", choices=WEIGHT_MODES, default="combined")
    parser.add_argument("--weight-scale-cp", type=float, default=100.0)
    parser.add_argument("--max-sample-weight", type=float, default=5.0)
    args = parser.parse_args()
    if args.weight_scale_cp <= 0.0:
        raise SystemExit("--weight-scale-cp must be positive")
    if args.max_sample_weight < 1.0:
        raise SystemExit("--max-sample-weight must be >= 1")

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
            sample_weight = compute_sample_weight(
                args.weight_mode,
                regret_delta,
                candidate_regret,
                args.weight_scale_cp,
                args.max_sample_weight,
            )
            out.write(
                json.dumps(
                    {
                        "sfen": sfen,
                        "sample_weight": sample_weight,
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

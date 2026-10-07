#!/usr/bin/env python3
"""Summarize two HKST teacher runs without treating result Brier as teacher proof."""
import argparse
import collections
import json
import math
import struct
from pathlib import Path

RECORD = struct.Struct("<IQHBBBBff")


def skip_position(data: bytes, pos: int) -> int:
    side, black, white, _ = struct.unpack_from("<4B", data, pos)
    if side > 1:
        raise ValueError("invalid side")
    return pos + 12 + 4 * (black + white)


def read(path: Path):
    data = path.read_bytes()
    if data[:8] != b"HKST0002":
        raise ValueError(f"invalid header: {path}")
    pos, records = 20, {}
    depths, phases = collections.Counter(), collections.Counter()
    while pos < len(data):
        start = pos
        magic, position_hash, ply, phase, result, candidates, depth, score, weight = RECORD.unpack_from(data, pos)
        if magic != 0x31524B48 or candidates == 0:
            raise ValueError(f"invalid record at {start}")
        pos += RECORD.size
        pos = skip_position(data, pos)
        for _ in range(candidates):
            pos += 8
            pos = skip_position(data, pos)
        records[position_hash] = {"score": score, "depth": depth, "phase": phase, "ply": ply}
        depths[str(depth)] += 1
        phases[str(phase)] += 1
    if pos != len(data):
        raise ValueError(f"trailing bytes in {path}")
    return records, {"records": len(records), "depth_counts": depths, "phase_counts": phases}


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    return values[min(len(values) - 1, int((len(values) - 1) * fraction))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--low", type=Path, required=True)
    parser.add_argument("--high", type=Path, required=True)
    parser.add_argument("--roots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    low, low_summary = read(args.low)
    high, high_summary = read(args.high)
    common = sorted(set(low) & set(high))
    deltas = [high[key]["score"] - low[key]["score"] for key in common]
    by_phase = {}
    for phase in (0, 1, 2):
        phase_deltas = [deltas[index] for index, key in enumerate(common) if low[key]["phase"] == phase]
        by_phase[str(phase)] = {
            "common": len(phase_deltas),
            "mean_delta_cp": sum(phase_deltas) / len(phase_deltas) if phase_deltas else None,
            "mean_abs_delta_cp": sum(abs(value) for value in phase_deltas) / len(phase_deltas) if phase_deltas else None,
            "p50_abs_delta_cp": percentile([abs(value) for value in phase_deltas], 0.50),
            "p95_abs_delta_cp": percentile([abs(value) for value in phase_deltas], 0.95),
            "changed_over_100cp": sum(abs(value) > 100.0 for value in phase_deltas),
        }
    result = {
        "schema": "halfkp_teacher_depth_audit_summary_v1",
        "low": {"path": str(args.low), **low_summary},
        "high": {"path": str(args.high), **high_summary},
        "roots": {"path": str(args.roots), "records": sum(1 for _ in args.roots.open())},
        "common_records": len(common),
        "score_delta_high_minus_low_cp": {
            "mean": sum(deltas) / len(deltas) if deltas else None,
            "mean_abs": sum(abs(value) for value in deltas) / len(deltas) if deltas else None,
            "p50_abs": percentile([abs(value) for value in deltas], 0.50),
            "p95_abs": percentile([abs(value) for value in deltas], 0.95),
            "max_abs": max((abs(value) for value in deltas), default=None),
            "changed_over_100cp": sum(abs(value) > 100.0 for value in deltas),
        },
        "by_phase": by_phase,
        "completion_loss_vs_roots": {
            "low": 5001 - len(low),
            "high": 5001 - len(high),
        },
        "interpretation": "Descriptive teacher-deepening audit; score change is not an external evaluation target.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Select a deterministic, phase-balanced root set for teacher audits."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--per-phase", type=int, default=1667)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    if args.per_phase <= 0:
        raise SystemExit("--per-phase must be positive")
    inputs = sorted(args.input_dir.glob("root-*.jsonl"))
    if not inputs:
        raise SystemExit("no root JSONL files found")
    buckets = {"opening": {}, "middlegame": {}, "endgame": {}}
    for path in inputs:
        with path.open() as stream:
            for line_number, line in enumerate(stream):
                record = json.loads(line)
                phase = record.get("phase")
                if phase not in buckets:
                    continue
                sfen = record.get("sfen")
                if not sfen:
                    continue
                key = hashlib.sha256(
                    f"{args.seed}:{phase}:{sfen}".encode("utf-8")
                ).hexdigest()
                buckets[phase].setdefault(key, (path, line_number, line))
    selected = {}
    for phase, records in buckets.items():
        if len(records) < args.per_phase:
            raise SystemExit(f"{phase} has only {len(records)} unique SFENs")
        selected[phase] = [records[key] for key in sorted(records)[: args.per_phase]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for phase in ("opening", "middlegame", "endgame"):
            for _, _, line in selected[phase]:
                stream.write(line)
    manifest = {
        "schema": "halfkp_teacher_audit_roots_v1",
        "seed": args.seed,
        "per_phase": args.per_phase,
        "records": sum(len(value) for value in selected.values()),
        "phase_counts": {phase: len(value) for phase, value in selected.items()},
        "selection_key": "sha256(seed:phase:sfen)",
        "input_files": [
            {"path": str(path), "sha256": sha256(path)} for path in inputs
        ],
        "output": {"path": str(args.output), "sha256": sha256(args.output)},
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

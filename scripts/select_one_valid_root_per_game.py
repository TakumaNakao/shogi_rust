#!/usr/bin/env python3
"""Create a deterministic, one-root-per-game valid audit set."""

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()

    selected = {}
    with args.input.open() as source:
        for line_no, line in enumerate(source, 1):
            record = json.loads(line)
            if record.get("schema") != "rank_value_dataset_v3_position":
                raise SystemExit(f"invalid schema at line {line_no}")
            if record.get("split") != "valid":
                raise SystemExit(f"non-valid record at line {line_no}")
            if not record.get("game_key") or not record.get("sfen"):
                raise SystemExit(f"missing key at line {line_no}")
            key = record["game_key"]
            rank_key = hashlib.sha256(
                f"{args.seed}\0{key}\0{record['sfen']}".encode()
            ).hexdigest()
            candidate = (rank_key, line_no, record)
            if key not in selected or candidate[:2] < selected[key][:2]:
                selected[key] = candidate

    records = [item[2] for item in sorted(selected.values(), key=lambda item: item[:2])]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "schema": "halfkp_mmto_one_root_per_game_audit_v1",
        "input": str(args.input),
        "output": str(args.output),
        "seed": args.seed,
        "records": len(records),
        "unique_game_keys": len({record["game_key"] for record in records}),
        "unique_opening_families": len({record["opening_family"] for record in records}),
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

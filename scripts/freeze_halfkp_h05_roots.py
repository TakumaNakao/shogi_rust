#!/usr/bin/env python3
"""Freeze a deterministic 500k-root source for the H05 teacher A/B."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("shards", type=Path, nargs=10)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = 0
    with args.output.open("wb") as destination:
        for shard in args.shards:
            with shard.open("rb") as source:
                for line in source:
                    if line.strip():
                        destination.write(line)
                        records += 1
    if records != 500_000:
        raise SystemExit(f"fixed root count is {records}, expected 500000")
    manifest = {
        "schema": "halfkp_h05_fixed_roots_v1",
        "records": records,
        "source_shards": [{"path": str(path), "sha256": sha256(path)} for path in args.shards],
        "output": {"path": str(args.output), "sha256": sha256(args.output)},
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

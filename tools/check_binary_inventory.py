#!/usr/bin/env python3
"""Verify that Cargo, src/bin, and docs/binaries.md describe the same targets."""

import json
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    metadata = json.loads(
        subprocess.check_output(
            ["cargo", "metadata", "--no-deps", "--format-version", "1"],
            cwd=ROOT,
            text=True,
        )
    )
    package = next(package for package in metadata["packages"] if package["name"] == "shogi_ai")
    targets = {target["name"] for target in package["targets"] if "bin" in target["kind"]}

    source_targets = {path.stem for path in (ROOT / "src" / "bin").glob("*.rs")}
    source_targets.add("kpp_learn")
    docs = (ROOT / "docs" / "binaries.md").read_text(encoding="utf-8")
    undocumented = sorted(name for name in targets if f"`{name}`" not in docs)

    errors = []
    if targets != source_targets:
        errors.append(f"Cargo-only targets: {sorted(targets - source_targets)}")
        errors.append(f"source-only targets: {sorted(source_targets - targets)}")
    if undocumented:
        errors.append(f"targets missing from docs/binaries.md: {undocumented}")

    if errors:
        print("binary inventory mismatch", file=sys.stderr)
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"binary inventory OK: {len(targets)} explicit targets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

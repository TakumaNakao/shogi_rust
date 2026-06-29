#!/usr/bin/env python3
"""Write a reproducibility manifest for long training runs."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shlex
import subprocess
from pathlib import Path
from typing import Any


def run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def parse_key_path(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"expected LABEL=PATH, got: {spec}")
    label, path = spec.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError(f"expected LABEL=PATH, got: {spec}")
    return label, Path(path)


def parse_key_value(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got: {spec}")
    key, value = spec.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got: {spec}")
    return key, value


def file_info(path: Path, *, line_count: bool) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return info

    stat = path.stat()
    info["bytes"] = stat.st_size
    info["mtime_utc"] = dt.datetime.fromtimestamp(
        stat.st_mtime, tz=dt.timezone.utc
    ).isoformat()

    sha = hashlib.sha256()
    lines = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            sha.update(chunk)
            if line_count:
                lines += chunk.count(b"\n")
    info["sha256"] = sha.hexdigest()
    if line_count:
        info["lines"] = lines
    return info


def collect_env(keys: list[str]) -> dict[str, str]:
    return {key: os.environ[key] for key in keys if key in os.environ}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write a JSON manifest for a training or dump run."
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--kind", required=True)
    parser.add_argument("--command", nargs=argparse.REMAINDER)
    parser.add_argument("--arg", action="append", default=[], type=parse_key_value)
    parser.add_argument("--input", action="append", default=[], type=parse_key_path)
    parser.add_argument("--weight", action="append", default=[], type=parse_key_path)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--line-count", action="append", default=[])
    args = parser.parse_args()

    line_count_labels = set(args.line_count)
    status_short = run_git(["status", "--short"])
    manifest: dict[str, Any] = {
        "schema": "shogi_ai_training_manifest_v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "kind": args.kind,
        "run_dir": str(args.run_dir) if args.run_dir else None,
        "command": " ".join(shlex.quote(part) for part in args.command or []),
        "parameters": dict(args.arg),
        "inputs": {
            label: file_info(path, line_count=label in line_count_labels)
            for label, path in args.input
        },
        "weights": {
            label: file_info(path, line_count=False) for label, path in args.weight
        },
        "environment": {
            "cwd": os.getcwd(),
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "vars": collect_env(args.env),
        },
        "git": {
            "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "commit": run_git(["rev-parse", "HEAD"]),
            "status_short": status_short,
            "dirty": bool(status_short),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"manifest: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

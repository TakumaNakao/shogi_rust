#!/usr/bin/env python3
"""Summarize MMTO training run directories."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


METRIC_RE = re.compile(r"([A-Za-z0-9_]+)=(-?(?:inf|nan|[0-9]+(?:\.[0-9]+)?))%?")


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def parse_metrics(line: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, raw_value in METRIC_RE.findall(line):
        if raw_value == "inf":
            value = float("inf")
        elif raw_value == "nan":
            value = float("nan")
        else:
            value = float(raw_value)
        if key == "selected_regret":
            key = "selected_regret_mean"
        if key == "teacher_match":
            value /= 100.0
        metrics[key] = value
    return metrics


def best_epoch_from_train(text: str) -> int | None:
    matches = re.findall(r"best_epoch=([0-9]+)", text)
    if not matches:
        return None
    return int(matches[-1])


def metric_line(text: str, prefix: str) -> dict[str, float] | None:
    for line in text.splitlines():
        if line.startswith(prefix):
            return parse_metrics(line)
    return None


def last_epoch_valid(text: str, epoch: int | None) -> dict[str, float] | None:
    if epoch == 0:
        return metric_line(text, "baseline valid:")
    lines = text.splitlines()
    if epoch is not None:
        prefix = f"epoch {epoch} valid:"
        for line in lines:
            if line.startswith(prefix):
                return parse_metrics(line)
    for line in reversed(lines):
        if re.match(r"epoch [0-9]+ valid:", line):
            return parse_metrics(line)
    return None


def gate_status(text: str, name: str) -> str:
    if not text:
        return "missing"
    upper_name = name.upper()
    if f"{upper_name} GATE PASSED" in text:
        return "pass"
    if f"{upper_name} GATE FAILED" in text:
        return "fail"
    if "PASSED" in text:
        return "pass"
    if "FAILED" in text:
        return "fail"
    return "unknown"


def gate_reason(text: str) -> str:
    for line in reversed(text.splitlines()):
        if "FAILED:" in line:
            return line.strip()
    return ""


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float | None, *, pct: bool = False) -> str:
    if value is None:
        return "-"
    if pct:
        return f"{value * 100:.2f}%"
    return f"{value:.2f}"


def delta(best: dict[str, float] | None, base: dict[str, float] | None, key: str) -> float | None:
    if not best or not base or key not in best or key not in base:
        return None
    return best[key] - base[key]


def summarize_run(run_dir: Path) -> dict[str, Any]:
    train_text = read_text(run_dir / "train_stdout.log")
    score_text = read_text(run_dir / "score_gate_stdout.log")
    rerank_text = read_text(run_dir / "rerank_gate_stdout.log")
    manifest = load_manifest(run_dir)

    best_epoch = best_epoch_from_train(train_text)
    baseline_valid = metric_line(train_text, "baseline valid:")
    best_valid = last_epoch_valid(train_text, best_epoch)
    parameters = manifest.get("parameters", {})
    inputs = manifest.get("inputs", {})
    run_seed = (
        parameters.get("SUBSET_SEED")
        or parameters.get("DUMP_SEED")
        or parameters.get("PROTECTION_SEED")
        or ""
    )
    train_input = (
        inputs.get("train")
        or inputs.get("train_protection")
        or inputs.get("train_strong")
        or {}
    )
    valid_input = (
        inputs.get("valid")
        or inputs.get("valid_top")
        or inputs.get("valid_strong")
        or {}
    )

    return {
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "score_gate": gate_status(score_text, "score"),
        "rerank_gate": gate_status(rerank_text, "rerank"),
        "rerank_reason": gate_reason(rerank_text),
        "subset_mode": parameters.get("SUBSET_MODE", ""),
        "subset_seed": run_seed,
        "train_lines": train_input.get("lines"),
        "valid_lines": valid_input.get("lines"),
        "valid_selected_delta": delta(best_valid, baseline_valid, "selected_regret_mean"),
        "valid_p90_delta": delta(best_valid, baseline_valid, "p90"),
        "valid_p95_delta": delta(best_valid, baseline_valid, "p95"),
        "valid_match_delta": delta(best_valid, baseline_valid, "teacher_match"),
        "valid_bad50_delta": delta(best_valid, baseline_valid, "bad50"),
        "valid_bad100_delta": delta(best_valid, baseline_valid, "bad100"),
    }


def print_markdown(summaries: list[dict[str, Any]]) -> None:
    print(
        "| run | seed | best | score | rerank | d_selected | d_p90 | d_p95 | d_match | d_bad50 | d_bad100 |"
    )
    print("|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|")
    for item in summaries:
        run_name = Path(item["run_dir"]).name
        print(
            "| {run} | {seed} | {best} | {score} | {rerank} | {sel} | {p90} | {p95} | {match} | {bad50} | {bad100} |".format(
                run=run_name,
                seed=item["subset_seed"] or "-",
                best=item["best_epoch"] if item["best_epoch"] is not None else "-",
                score=item["score_gate"],
                rerank=item["rerank_gate"],
                sel=fmt(item["valid_selected_delta"]),
                p90=fmt(item["valid_p90_delta"]),
                p95=fmt(item["valid_p95_delta"]),
                match=fmt(item["valid_match_delta"], pct=True),
                bad50=fmt(item["valid_bad50_delta"], pct=True),
                bad100=fmt(item["valid_bad100_delta"], pct=True),
            )
        )

    failed = [item for item in summaries if item["rerank_reason"]]
    if failed:
        print()
        print("Rerank failures:")
        for item in failed:
            print(f"- {Path(item['run_dir']).name}: {item['rerank_reason']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize MMTO run directories.")
    parser.add_argument("run_dir", nargs="+", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    summaries = [summarize_run(path) for path in args.run_dir]
    print_markdown(summaries)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(summaries, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

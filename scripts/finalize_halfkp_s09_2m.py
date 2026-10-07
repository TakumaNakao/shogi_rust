#!/usr/bin/env python3
"""Freeze S09's final candidate metadata and reserved paired-match inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib


PARENT_SHA = "149b79c5d29457f88b7f6355c5ccf1f106ae25eddc647717debbe4a34fc1e82c"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=pathlib.Path, required=True)
    parser.add_argument("--dataset-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--parent", type=pathlib.Path, required=True)
    parser.add_argument("--parent-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--v250", type=pathlib.Path, required=True)
    parser.add_argument("--taya36", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    if sha256(args.parent) != PARENT_SHA:
        raise SystemExit("v2.7.1 parent SHA mismatch")
    parent_manifest = json.loads(args.parent_manifest.read_text())
    if parent_manifest.get("model", {}).get("sha256") != PARENT_SHA:
        raise SystemExit("v2.7.1 parent manifest does not match the frozen parent")
    checkpoint = json.loads((args.run_dir / "checkpoint.json").read_text())
    if checkpoint.get("completed_outer") != 8:
        raise SystemExit(f"2M training is incomplete: outer{checkpoint.get('completed_outer')}")
    data_manifest = json.loads(args.dataset_manifest.read_text())
    if data_manifest.get("selected_train_records") != 2_000_000:
        raise SystemExit("expected exactly 2,000,000 frozen train roots")

    metrics = []
    for outer in (2, 4, 8):
        report = json.loads((args.run_dir / "metrics" / f"outer_{outer:04d}.json").read_text())
        valid = report["valid"]
        metrics.append({
            "outer": outer,
            "model_sha256": report["model_sha256"],
            "mean_loss_per_pair": valid["mean_loss_per_pair"],
            "agreement": valid["agreement"],
            "phase_loss": [item["mean_loss"] for item in valid["phase"]],
            "aborted_roots": valid["aborted_roots"],
            "terminal_roots": valid["terminal_roots"],
        })

    model = args.run_dir / "model.bin"
    model_hash = sha256(model)
    if model_hash != metrics[-1]["model_sha256"]:
        raise SystemExit("outer8 metric model hash does not match final candidate")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_manifest = {
        "schema_version": 1,
        "lineage_id": "halfkp64-scratch-v1",
        "stage": "halfkp_mmto_train_s09_frequency_2m",
        "selected_outer": 8,
        "parent_model_sha256": PARENT_SHA,
        "dataset_manifest_sha256": sha256(args.dataset_manifest),
        "model": {"path": str(model.resolve()), "sha256": model_hash, "bytes": model.stat().st_size},
        "selection_reason": "precommitted 2M-root outer8 endpoint; outer2/4/8 snapshots retained",
        "teacher_policy": "observed WDoor move-frequency soft targets only; no external evaluation labels",
        "metrics": metrics,
    }
    (args.output_dir / "candidate.manifest.json").write_text(
        json.dumps(candidate_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "training_summary.json").write_text(
        json.dumps({"dataset_manifest_sha256": sha256(args.dataset_manifest), "metrics": metrics}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [line for line in args.taya36.read_bytes().splitlines() if line.strip()]
    if len(lines) < 500:
        raise SystemExit(f"taya36 only has {len(lines)} lines; reserved S09 range needs 500")
    (args.output_dir / "v271_openings.sfen").write_bytes(b"\n".join(lines[450:475]) + b"\n")
    (args.output_dir / "v250_openings.sfen").write_bytes(b"\n".join(lines[475:500]) + b"\n")

    v250_hash = sha256(args.v250)
    (args.output_dir / "v250_parent.manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "lineage_id": "v2.5.0-kpp-derived-evaluation-only",
        "model": {"path": str(args.v250.resolve()), "sha256": v250_hash, "bytes": args.v250.stat().st_size},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "v250_parent_authorization.json").write_text(json.dumps({
        "schema": "halfkp_workflow_parent_authorization_v1",
        "authorization_kind": "evaluation_only_external_baseline",
        "model": {"path": str(args.v250.resolve()), "sha256": v250_hash},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_sha256": model_hash, "dataset_manifest_sha256": sha256(args.dataset_manifest), "metrics": metrics}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

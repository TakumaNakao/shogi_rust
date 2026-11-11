#!/usr/bin/env python3
"""Write run-local lineage records for continuing the released v2.7.1 model."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import shutil


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--optimizer", type=pathlib.Path, required=True)
    parser.add_argument("--source-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--gate-summary", type=pathlib.Path, required=True)
    parser.add_argument("--canonical-registry", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    expected_model = "149b79c5d29457f88b7f6355c5ccf1f106ae25eddc647717debbe4a34fc1e82c"
    expected_optimizer = "57d6e916ded2255e80785354a619a6ddad374104b5de765b63473a534a3564f8"
    if sha256(args.model) != expected_model or sha256(args.optimizer) != expected_optimizer:
        raise SystemExit("v2.7.1 model/optimizer artifact SHA mismatch")
    run = args.output_dir
    run.mkdir(parents=True, exist_ok=True)
    model_manifest = {
        "schema_version": 1,
        "lineage_id": "halfkp64-scratch-v1",
        "generation": 7,
        "parent_model_sha256": "162e17bacea3de80ecdbbdae765ca1157246c4febb6fa6a15f8d36b900d011dc",
        "model": {"path": str(args.model.resolve()), "sha256": expected_model, "bytes": args.model.stat().st_size},
        "released_as": "v2.7.1",
        "release_source_manifest_sha256": sha256(args.source_manifest),
        "optimizer": {"path": str(args.optimizer.resolve()), "sha256": expected_optimizer, "bytes": args.optimizer.stat().st_size},
    }
    model_manifest_path = run / "v2.7.1.model.manifest.json"
    model_manifest_path.write_text(json.dumps(model_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    registry = json.loads(args.canonical_registry.read_text(encoding="utf-8"))
    registry["accepted"] = [entry for entry in registry["accepted"] if entry["model_sha256"] != expected_model]
    registry["accepted"].append({
        "model_sha256": expected_model,
        "generation": 7,
        "gate_report_sha256": sha256(args.gate_summary),
        "decision": "accepted_released_research_parent",
        "accepted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "decision_commit": "v2.7.1",
    })
    registry_path = run / "research_parent_registry.json"
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.copyfile(args.gate_summary, run / "v2.7.1_gate_summary.json")
    print(json.dumps({
        "generation": 7,
        "model_sha256": expected_model,
        "optimizer_sha256": expected_optimizer,
        "gate_summary_sha256": sha256(args.gate_summary),
        "model_manifest": str(model_manifest_path),
        "research_registry": str(registry_path),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

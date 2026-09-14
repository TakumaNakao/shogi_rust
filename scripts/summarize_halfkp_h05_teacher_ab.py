#!/usr/bin/env python3
"""Write a compact manifest comparison for the H05 teacher A/B."""
import argparse
import json
from pathlib import Path


def model_summary(path: Path) -> dict:
    value = json.loads(path.read_text())
    return {
        "path": str(path),
        "model_sha256": value.get("model", {}).get("sha256"),
        "completed_epoch": value.get("completed_epoch"),
        "selected_epoch": value.get("selected_epoch"),
        "selected_validation_score": value.get("selected_validation_score"),
        "final_validation": value.get("final_validation"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--low-manifest", type=Path, required=True)
    parser.add_argument("--high-manifest", type=Path, required=True)
    args = parser.parse_args()
    value = {
        "schema": "halfkp_h05_teacher_ab_summary_v1",
        "parent_model_sha256": "162e17bacea3de80ecdbbdae765ca1157246c4febb6fa6a15f8d36b900d011dc",
        "fixed_root_manifest": "roots-500k.manifest.json",
        "epochs": 16,
        "teacher_100k": model_summary(args.low_manifest),
        "teacher_300k": model_summary(args.high_manifest),
    }
    args.output.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()

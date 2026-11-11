#!/usr/bin/env python3
"""Compare uninterrupted and interrupted/resumed training for both objectives."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from run_halfkp_listwise_experiment import ROOT, S07, S09, PARENT, OPTIMIZER, sha, write, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, default=ROOT / "target/release/halfkp_mmto_train")
    args = parser.parse_args()
    base = args.output_dir.resolve()
    base.mkdir(parents=True, exist_ok=False)
    data = base / "dataset"
    data.mkdir()
    outputs = []
    for split, count in [("train", 32), ("valid", 8), ("test", 0)]:
        rows = []
        if count:
            for line in (S07 / f"{split}.jsonl").open():
                if json.loads(line)["phase"] == "opening":
                    rows.append(line)
                    if len(rows) == count:
                        break
        if len(rows) != count:
            raise RuntimeError("insufficient smoke input")
        path = data / f"{split}.jsonl"
        path.write_text("".join(rows))
        outputs.append({"path": path.name, "records": count, "sha256": sha(path)})
    write(data / "manifest.json", {"schema": "halfkp_move_frequency_manifest_v1", "min_player_rate": 2500,
                                  "min_opponent_rate": 2500, "output_files": outputs})
    lineage = S09 / "research-parent"

    def command(run, loss, resume=False):
        result = [str(args.trainer.resolve()), "--train", str(data / "train.jsonl"),
            "--valid", str(data / "valid.jsonl"), "--dataset-manifest", str(data / "manifest.json"),
            "--expected-dataset-manifest-sha256", sha(data / "manifest.json"), "--init", str(PARENT),
            "--init-optimizer", str(OPTIMIZER), "--model-manifest", str(lineage / "v2.7.1.model.manifest.json"),
            "--promotion-registry", str(lineage / "research_parent_registry.json"),
            "--gate-report", str(lineage / "v2.7.1_gate_summary.json"), "--parent-generation", "7",
            "--output-dir", str(run), "--outer-iterations", "2", "--frequency-loss", loss,
            "--root-batch-size", "4", "--search-block-roots", "4", "--checkpoint-roots", "4",
            "--jobs", "2", "--max-valid-roots", "8", "--pv-probe-roots", "4",
            "--learning-rate", "0.0001", "--tau-cp", "300"]
        return result + (["--resume"] if resume else [])

    results = {}
    for loss in ("ranknet", "listwise"):
        reference = base / f"{loss}-reference"
        resumed = base / f"{loss}-resumed"
        with (base / f"{loss}-reference.log").open("w") as log:
            subprocess.run(command(reference, loss), stdout=log, stderr=subprocess.STDOUT, check=True)
        with (base / f"{loss}-interrupted.log").open("w") as log:
            process = subprocess.Popen(command(resumed, loss), stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 120
                interrupted_root = None
                while time.monotonic() < deadline and process.poll() is None:
                    checkpoint = resumed / "checkpoint.json"
                    if checkpoint.exists():
                        state = read(checkpoint)
                        if state.get("active_outer") == 1 and state["roots_completed"] >= 4:
                            interrupted_root = state["roots_completed"]
                            process.kill()
                            break
                    time.sleep(0.01)
                process.wait(timeout=10)
                if interrupted_root is None:
                    raise RuntimeError("did not exercise an active-outer checkpoint")
            finally:
                if process.poll() is None:
                    process.kill(); process.wait()
        checkpoint_before = (resumed / "checkpoint.json").read_bytes()
        model_before = sha(resumed / "model.bin")
        with (base / f"{loss}-mismatch.log").open("w") as log:
            bad = subprocess.run(command(resumed, "listwise" if loss == "ranknet" else "ranknet", True), stdout=log, stderr=subprocess.STDOUT)
        if bad.returncode == 0 or checkpoint_before != (resumed / "checkpoint.json").read_bytes() or model_before != sha(resumed / "model.bin"):
            raise RuntimeError("mismatched objective resumed or changed saved artifacts")
        with (base / f"{loss}-resumed.log").open("w") as log:
            subprocess.run(command(resumed, loss, True), stdout=log, stderr=subprocess.STDOUT, check=True)
        for artifact in ("model.bin", "optimizer.bin"):
            if sha(reference / artifact) != sha(resumed / artifact):
                raise RuntimeError(f"{loss} resume changed {artifact}")
        results[loss] = {"interrupted_after_roots": interrupted_root, "model_sha256": sha(resumed / "model.bin"),
                         "optimizer_sha256": sha(resumed / "optimizer.bin"), "bitwise_resume_equal": True,
                         "wrong_objective_rejected_without_writes": True}
        print(loss, results[loss], flush=True)
    write(base / "report.json", {"passed": True, "results": results})


if __name__ == "__main__":
    main()

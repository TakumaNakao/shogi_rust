#!/usr/bin/env python3
"""S11: fixed-quality 1M Listwise, eight passes, then three 50-game matches."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

import run_halfkp_listwise_experiment as previous
from run_halfkp_listwise_experiment import (
    ROOT, RUNS, S07, S09, PARENT, OPTIMIZER, EXACT,
    read, write, sha, freeze_json, snapshot, model_manifest,
)

OLD = RUNS / "s10-listwise-20261026-restart"
PILOT = OLD / "pilot-listwise-outer8/model.bin"


def compare_positions(parent, candidate):
    if len(parent) != 1000 or len(candidate) != 1000:
        raise RuntimeError("incomplete fixed suite")
    counts = {"lost_successes": 0, "gained_successes": 0,
              "parent_successful": 0, "candidate_successful": 0,
              "common_conclusive": 0, "common_parent_successful": 0,
              "common_candidate_successful": 0}
    for a, b in zip(parent, candidate):
        if (a["id"], a["sfen"]) != (b["id"], b["sfen"]):
            raise RuntimeError("fixed-suite position mismatch")
        if a["search_failed"] or b["search_failed"]:
            raise RuntimeError("fixed-suite internal search failure")
        counts["parent_successful"] += a["successful"]
        counts["candidate_successful"] += b["successful"]
        counts["lost_successes"] += a["successful"] and not b["successful"]
        counts["gained_successes"] += b["successful"] and not a["successful"]
        if a["depth"] and b["depth"]:
            counts["common_conclusive"] += 1
            counts["common_parent_successful"] += a["successful"]
            counts["common_candidate_successful"] += b["successful"]
    counts["within_budget_tolerance"] = counts["parent_successful"] - counts["candidate_successful"] <= 10
    return counts


class Experiment(previous.Experiment):
    def prepare_openings(self):
        path = self.base / "openings.json"
        if path.exists():
            return
        old = read(OLD / "openings.json")
        used = set()
        for record in RUNS.rglob("game_*.usi"):
            with record.open(errors="replace") as stream:
                for _ in range(4):
                    line = stream.readline().strip()
                    if line.startswith("start_sfen "):
                        used.add(line.removeprefix("start_sfen "))
        taya_parent = old["sets"]["confirm_parent"][:25]
        taya_v250 = old["sets"]["confirm_v250"]
        if any(x in used for x in taya_parent + taya_v250):
            raise RuntimeError("reserved confirmation openings were consumed")
        families, positions, shallow = set(), set(), []
        reserved = {x for values in old["sets"].values() for x in values}
        for line in (S09 / "dataset-2m/test.jsonl").open():
            row = json.loads(line)
            command = "sfen " + row["sfen"]
            if (not 8 <= row["ply_mean"] <= 20 or command in used or command in reserved
                    or row["opening_family"] in families or row["position_key"] in positions):
                continue
            families.add(row["opening_family"])
            positions.add(row["position_key"])
            shallow.append(command)
            if len(shallow) == 25:
                break
        if len(shallow) != 25 or len(taya_parent) != 25 or len(taya_v250) != 25:
            raise RuntimeError("insufficient independent openings")
        for train in (S07 / "train.jsonl", S09 / "dataset-2m/train.jsonl"):
            for line in train.open():
                if json.loads(line)["position_key"] in positions:
                    raise RuntimeError("shallow opening overlaps frozen train")
        freeze_json(path, {"sets": {"parent": taya_parent, "shallow": shallow, "v250": taya_v250},
                           "source_sha256": sha(OLD / "openings.json"),
                           "test_sha256": sha(S09 / "dataset-2m/test.jsonl"),
                           "novelty": "unused recorded starts, not lifetime model-input novelty"})

    def detailed(self, label, model, nodes):
        output = self.base / "evaluation" / f"{label}-{nodes}-positions.json"
        summary = output.with_name(f"{label}-{nodes}-summary.json")
        freeze_json(output.with_suffix(".provenance.json"),
                    {"model": sha(model), "dataset": sha(EXACT), "nodes": nodes})
        if not output.exists() or not summary.exists():
            # A killed write is never considered a completed evaluation.
            output.unlink(missing_ok=True)
            summary.unlink(missing_ok=True)
            manifest = model_manifest(model, self.base / "evaluation" / f"{label}-model.json")
            self.command(f"{label}-{nodes}", [self.tools / "halfkp_search_gate", "--model", model,
                "--model-manifest", manifest, "--dataset", EXACT, "--nodes", nodes,
                "--json", summary, "--details-json", output])
        details = read(output)
        if read(summary)["model_sha256"] != sha(model) or len(details) != 1000:
            raise RuntimeError("invalid detailed evaluation")
        return details

    def check_candidate(self, label, model):
        self.tactics(label, model)
        for nodes in (20000, 200000):
            comparison = compare_positions(self.detailed("parent", PARENT, nodes),
                                           self.detailed(label, model, nodes))
            write(self.base / "evaluation" / f"{label}-{nodes}-comparison.json", comparison)
            if not comparison["within_budget_tolerance"]:
                raise RuntimeError(f"review required: {label} tactical loss exceeds 10/1000 at {nodes} nodes")
        a = read(self.base / "evaluation/parent-captures.json")["cases"]
        b = read(self.base / "evaluation" / f"{label}-captures.json")["cases"]
        if len(a) != len(b) or any(x["id"] != y["id"] or (x["passed"] and not y["passed"])
                                   for x, y in zip(a, b)):
            raise RuntimeError("review required: new basic capture failure")

    def run(self):
        self.prepare()
        freeze_json(self.base / "s11-config.json", {
            "runner_sha256": sha(Path(__file__)), "pilot_sha256": sha(PILOT),
            "train_sha256": sha(S07 / "train.jsonl"), "valid_sha256": sha(S07 / "valid.jsonl"),
            "snapshots": [1, 2, 4, 8], "lr": 1e-4, "clip": 1, "root_batch": 128,
            "jobs": 12, "release": False, "max_tactical_net_loss": 10})
        self.tactics("parent", PARENT)
        self.check_candidate("pilot", PILOT)
        run = self.base / "listwise-1m"
        lineage = S09 / "research-parent"
        for outer in (1, 2, 4, 8):
            checkpoint = run / "checkpoint.json"
            current = read(checkpoint)["completed_outer"] if checkpoint.exists() else 0
            if current < outer:
                command = [self.tools / "halfkp_mmto_train", "--train", S07 / "train.jsonl",
                    "--valid", S07 / "valid.jsonl", "--dataset-manifest", S07 / "manifest.json",
                    "--expected-dataset-manifest-sha256", sha(S07 / "manifest.json"), "--init", PARENT,
                    "--init-optimizer", OPTIMIZER, "--model-manifest", lineage / "v2.7.1.model.manifest.json",
                    "--promotion-registry", lineage / "research_parent_registry.json",
                    "--gate-report", lineage / "v2.7.1_gate_summary.json", "--parent-generation", 7,
                    "--output-dir", run, "--outer-iterations", outer, "--frequency-loss", "listwise",
                    "--root-batch-size", 128, "--search-block-roots", 1024, "--jobs", 12,
                    "--learning-rate", .0001, "--output-learning-rate-ratio", .1, "--tau-cp", 300,
                    "--gradient-clip-norm", 1, "--max-valid-roots", 12000, "--pv-probe-roots", 256,
                    "--checkpoint-roots", 25000, "--qsearch-node-limit", 2000000]
                if checkpoint.exists():
                    command.append("--resume")
                self.command(f"listwise-1m-outer{outer}", command)
            destination = self.base / f"listwise-1m-outer{outer}"
            snapshot(run, destination, outer)
            self.check_candidate(f"outer{outer}", destination / "model.bin")
        model = self.base / "listwise-1m-outer8/model.bin"
        self.evaluate("final-old", model, S07 / "valid.jsonl")
        self.evaluate("final-new", model, S09 / "dataset-2m/valid.jsonl")
        parent = self.match("final-vs-parent", model, PARENT, "parent")
        shallow = self.match("final-vs-parent-shallow", model, PARENT, "shallow")
        reference = self.match("final-vs-v250", model,
            ROOT / "policy_weights_halfkp64_kpp_distilled_v2.5.0.binary", "v250")
        write(self.base / "decision.json", {"research_candidate": parent["score"] >= .54 and shallow["score"] >= .5,
              "parent_score": parent["score"], "shallow_score": shallow["score"], "v250_score": reference["score"],
              "release_promoted": False, "next_action": "review independent intervals and regressions"})
        self.state("COMPLETE", state="completed", release_promoted=False)
        write(self.base / "COMPLETE.json", read(self.base / "status.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUNS / "s11-listwise-1m-20261031")
    args = parser.parse_args()
    experiment = Experiment(args.run_dir)
    with (experiment.base / ".lock").open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            experiment.run()
        except BaseException as error:
            experiment.state(experiment.stage, state="failed", error=str(error))
            raise


if __name__ == "__main__":
    main()

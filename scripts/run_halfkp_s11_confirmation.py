#!/usr/bin/env python3
"""Fixed S11 candidate: two independent 50-game confirmation matches, no training."""
import argparse
import fcntl
from pathlib import Path
import shutil

import run_halfkp_listwise_experiment as shared
from run_halfkp_listwise_experiment import ROOT, RUNS, PARENT, PARENT_SHA, read, write, sha, freeze_json

SOURCE = RUNS / "s11-listwise-1m-20261031"
CANDIDATE = SOURCE / "listwise-1m-outer8/model.bin"
V250 = ROOT / "policy_weights_halfkp64_kpp_distilled_v2.5.0.binary"
CANDIDATE_SHA = "109d7aaa6fbfb27418d358d76045f30b71e10f8b0a7191df5a184632fc2c2a08"
V250_SHA = "91784d6f03f70580468a1509f968cbaaf598fbce084c22a739cb870344fb7c00"


def select_openings(lines, used):
    selected = list(dict.fromkeys(line.strip() for line in lines if line.strip() and line.strip() not in used))[:25]
    if len(selected) != 25:
        raise RuntimeError("fewer than 25 unused taya36 starts")
    return selected


def prepare(experiment):
    for model, expected in [(CANDIDATE, CANDIDATE_SHA), (PARENT, PARENT_SHA), (V250, V250_SHA)]:
        if sha(model) != expected:
            raise RuntimeError(f"weight SHA mismatch: {model}")
    binaries = read(SOURCE / "binaries.json")
    for name in ["usi_engine", "usi_benchmark"]:
        source = SOURCE / "tools" / name
        if sha(source) != binaries[name]:
            raise RuntimeError(f"source binary mismatch: {name}")
        target = experiment.tools / name
        if not target.exists():
            shutil.copy2(source, target)
        if sha(target) != binaries[name]:
            raise RuntimeError(f"pinned binary mismatch: {name}")
    freeze_json(experiment.base / "provenance.json", {
        "script_sha256": sha(Path(__file__)), "shared_runner_sha256": sha(Path(shared.__file__)),
        "candidate": CANDIDATE_SHA, "parent": PARENT_SHA, "v250": V250_SHA,
        "binaries": {k: binaries[k] for k in ["usi_engine", "usi_benchmark"]},
        "taya_sha256": sha(ROOT / "taya36.sfen"),
        "games_per_opponent": 50, "time_ms": 5000, "threads": 1,
        "depth": 32, "max_plies": 256, "automatic_training_or_release": False})
    path = experiment.base / "openings.json"
    if not path.exists():
        used = set()
        for record in RUNS.rglob("game_*.usi"):
            with record.open(errors="replace") as stream:
                for _ in range(4):
                    line = stream.readline().strip()
                    if line.startswith("start_sfen "):
                        used.add(line.removeprefix("start_sfen "))
        for record in RUNS.rglob("pair_*.json"):
            opening = read(record).get("opening", {}).get("sfen")
            if opening:
                used.add(opening)
        starts = select_openings((ROOT / "taya36.sfen").read_text().splitlines(), used)
        freeze_json(path, {"sets": {"confirm": starts}, "historical_used_commands": len(used),
                          "novelty": "unused recorded start commands, not lifetime training-input novelty"})
    starts = read(path)["sets"]["confirm"]
    if len(starts) != 25 or len(set(starts)) != 25:
        raise RuntimeError("invalid frozen opening set")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUNS / "s11-confirmation-20261110")
    args = parser.parse_args()
    experiment = shared.Experiment(args.run_dir)
    with (experiment.base / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            experiment.state("preflight", state="running")
            prepare(experiment)
            parent = experiment.match("confirm-vs-parent", CANDIDATE, PARENT, "confirm")
            v250 = experiment.match("confirm-vs-v250", CANDIDATE, V250, "confirm")
            write(experiment.base / "results.json", {"parent": parent, "v250": v250,
                  "meaning": "new independent confirmation results; historical matches are not pooled here",
                  "next_action": "review results; no automatic training or release"})
            experiment.state("COMPLETE", state="completed", games=100, release_promoted=False)
            write(experiment.base / "COMPLETE.json", read(experiment.base / "status.json"))
        except BaseException as error:
            experiment.state(experiment.stage, state="failed", error=str(error))
            raise


if __name__ == "__main__":
    main()

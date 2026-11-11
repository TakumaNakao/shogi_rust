#!/usr/bin/env python3
"""Run the 2026-10-25 R2--R5 plan, retaining immutable stage/pair artifacts.

This is a single experiment runner, not a general job scheduler. Re-run the
same command to validate saved inputs and resume the first unfinished stage.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import selectors
import struct
import subprocess
import sys
import time

from run_halfkp_paired_match import paired_percentile_bootstrap

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/halfkp64-scratch-v1"
S07 = RUNS / "s07-retry-rate2500-v3-20260922-selected-1m"
S09 = RUNS / "s09-frequency-2m-20261007"
PARENT = RUNS / "s08-retry-rate2500-v3-20260929/outer16/model.bin"
OPTIMIZER = PARENT.with_name("optimizer.bin")
EXACT = RUNS / "exact-rank-v4-dev1k/valid.hkcl"
HEALTH = RUNS / "legal-health-v5-stratified/valid.hkcl"
PARENT_SHA = "149b79c5d29457f88b7f6355c5ccf1f106ae25eddc647717debbe4a34fc1e82c"
OPTIMIZER_SHA = "57d6e916ded2255e80785354a619a6ddad374104b5de765b63473a534a3564f8"
TOOLS = ("halfkp_mmto_train", "halfkp_frequency_evaluate", "usi_engine",
         "usi_benchmark", "halfkp_search_gate", "halfkp_scratch_eval", "halfkp_curriculum_audit")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def validate_hkcl_header(path: Path, *, exact: bool = False) -> dict:
    """Fail before expensive validation if an old or incompatible corpus is selected."""
    with path.open("rb") as stream:
        prefix = stream.read(32)
        if len(prefix) != 32:
            raise RuntimeError(f"truncated HKCL header: {path}")
        magic, version, hidden, inputs, records, metadata_length = struct.unpack("<8sIIIQI", prefix)
        if (magic, version, hidden, inputs) != (b"HKCL0004", 4, 64, 105480):
            raise RuntimeError(f"incompatible HKCL header: {path}: magic={magic!r}, version={version}, hidden={hidden}, inputs={inputs}; expected HKCL0004/4/64/105480")
        if records == 0 or metadata_length > 16 * 1024 * 1024:
            raise RuntimeError(f"empty or oversized HKCL header: {path}")
        payload = stream.read(metadata_length)
        if len(payload) != metadata_length:
            raise RuntimeError(f"truncated HKCL metadata: {path}")
        metadata = json.loads(payload)
    if metadata.get("lineage_id") != "halfkp64-scratch-v1":
        raise RuntimeError(f"wrong HKCL lineage: {path}")
    if exact:
        config = metadata.get("generator_config", {})
        targets = config.get("targets", {})
        if config.get("rule_version") != 3 or targets.get("defense_pairs", 0) <= 0 or targets.get("mate3", 0) <= 0:
            raise RuntimeError(f"exact suite requires rule-v3 attacking and defending proofs: {path}")
    return {"records": records, "metadata": metadata}


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def freeze_json(path: Path, value) -> None:
    if path.exists():
        if read(path) != value:
            raise RuntimeError(f"frozen configuration changed: {path}")
    else:
        write(path, value)


def model_manifest(model: Path, destination: Path) -> Path:
    freeze_json(destination, {"schema_version": 1, "lineage_id": "halfkp64-scratch-v1",
                             "model": {"path": str(model), "sha256": sha(model)}})
    return destination


def verify_snapshot(destination: Path, outer: int) -> None:
    metric = read(destination / "metric.json")
    if metric["outer"] != outer or sha(destination / "model.bin") != metric["model_sha256"] or sha(destination / "optimizer.bin") != metric["optimizer_sha256"]:
        raise RuntimeError(f"snapshot mismatch: {destination}")


def snapshot(run: Path, destination: Path, outer: int) -> None:
    if destination.exists():
        verify_snapshot(destination, outer)
        return
    if read(run / "checkpoint.json")["completed_outer"] != outer:
        raise RuntimeError("cannot reconstruct an earlier unsaved snapshot from final weights")
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.mkdir(exist_ok=True)
    for name in ("model.bin", "optimizer.bin"):
        shutil.copyfile(run / name, temporary / name)
    shutil.copyfile(run / f"metrics/outer_{outer:04d}.json", temporary / "metric.json")
    verify_snapshot(temporary, outer)
    temporary.rename(destination)


def parse_pair(directory: Path, opening: str) -> dict:
    records = sorted((directory / "games").glob("game_*.usi"))
    if len(records) != 2:
        raise RuntimeError(f"expected two saved USI games: {directory}")
    results = []
    colors = []
    for path in records:
        fields = dict(line.split(" ", 1) for line in path.read_text().splitlines()[:4])
        if fields["start_sfen"] != opening:
            raise RuntimeError("saved USI opening differs from frozen opening")
        if fields["reason"] not in ("Resign", "RepetitionDraw", "PerpetualCheckLoss", "MaxPliesDraw"):
            raise RuntimeError(f"unsafe game termination: {fields['reason']}")
        results.append(fields["result"])
        colors.append(fields["new_as"])
    if set(colors) != {"black", "white"}:
        raise RuntimeError(f"pair did not swap colors: {colors}")
    values = {"NewWin": 1.0, "BaselineWin": 0.0, "Draw": 0.5}
    return {"score": sum(values[value] for value in results) / 2,
            "candidate_wins": results.count("NewWin"), "baseline_wins": results.count("BaselineWin"),
            "draws": results.count("Draw"),
            "records": {path.name: sha(path) for path in records}}


def usi_probe(engine: Path, model: Path, position: str) -> tuple[str, list[str]]:
    process = subprocess.Popen([str(engine)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    pending = b""
    transcript = []

    def send(command):
        process.stdin.write((command + "\n").encode())
        process.stdin.flush()

    def until(prefix):
        nonlocal pending
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            while b"\n" in pending:
                raw, pending = pending.split(b"\n", 1)
                line = raw.decode(errors="replace").strip()
                transcript.append(line)
                if line.startswith(prefix):
                    return line
            if not selector.select(max(0, deadline - time.monotonic())):
                break
            chunk = os.read(process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError("USI probe engine exited before bestmove")
            pending += chunk
        raise RuntimeError("USI probe timed out")

    try:
        send("usi"); until("usiok")
        send(f"setoption name EvalFile value {model}")
        send("setoption name Threads value 1")
        send("isready"); until("readyok")
        send("usinewgame")
        send("position " + position)
        send("go depth 32 movetime 5000")
        best = until("bestmove ").split()[1]
        if any("Internal search error" in line for line in transcript):
            raise RuntimeError("USI probe search failure")
        return best, transcript
    finally:
        selector.close()
        if process.poll() is None:
            send("quit")
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        process.stdin.close()
        process.stdout.close()


class Experiment:
    def __init__(self, base: Path):
        self.base = base.resolve()
        self.base.mkdir(parents=True, exist_ok=True)
        self.tools = self.base / "tools"
        self.tools.mkdir(exist_ok=True)
        self.stage = "initializing"

    def state(self, stage: str, **details):
        self.stage = stage
        write(self.base / "status.json", {"stage": stage, "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(), **details})
        print(stage, details, flush=True)

    def command(self, name: str, command: list):
        self.state(name, state="running", command=[str(item) for item in command])
        log = self.base / "logs" / f"{name}.log"
        log.parent.mkdir(exist_ok=True)
        with log.open("a") as stream:
            subprocess.run([str(item) for item in command], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)

    def prepare(self):
        validate_hkcl_header(EXACT, exact=True)
        validate_hkcl_header(HEALTH)
        if sha(PARENT) != PARENT_SHA or sha(OPTIMIZER) != OPTIMIZER_SHA:
            raise RuntimeError("released parent bundle mismatch")
        binaries = {}
        for tool in TOOLS:
            target = self.tools / tool
            if not target.exists():
                shutil.copy2(ROOT / "target/release" / tool, target)
            binaries[tool] = sha(target)
        freeze_json(self.base / "binaries.json", binaries)
        freeze_json(self.base / "runner.json", {"script_sha256": sha(Path(__file__)),
                    "selector_sha256": sha(ROOT / "scripts/select_halfkp_frequency_roots.py")})
        inputs = [S07 / "manifest.json", S09 / "dataset-2m/manifest.json", EXACT, HEALTH,
                  S09 / "research-parent/v2.7.1.model.manifest.json",
                  S09 / "research-parent/research_parent_registry.json",
                  S09 / "research-parent/v2.7.1_gate_summary.json"]
        freeze_json(self.base / "inputs.json", {str(path): sha(path) for path in inputs})
        audit = self.base / "preflight/hkcl-audit.json"
        audit_inputs = {str(path): sha(path) for path in (EXACT, HEALTH)}
        audit_provenance = audit.with_name("hkcl-audit-inputs.json")
        freeze_json(audit_provenance, audit_inputs)
        if not audit.exists():
            self.command("hkcl-preflight", [self.tools / "halfkp_curriculum_audit", EXACT, HEALTH,
                                          "--revalidate-proofs", "--report", audit])
        if not read(audit)["passed"] or read(audit)["proofs"] != 1000:
            raise RuntimeError("HKCL readability/proof audit failed or exact suite was incomplete")
        for name, outer in [("frozen-control-outer4", 4), ("frozen-2m-outer2", 2), ("frozen-2m-outer8", 8)]:
            verify_snapshot(S09 / name, outer)
        dataset = self.base / "dataset-100k"
        if not (dataset / "manifest.json").exists():
            if dataset.exists():
                shutil.rmtree(dataset)  # Only an uncommitted generated subset in this experiment.
            self.command("prepare-100k", [sys.executable, ROOT / "scripts/select_halfkp_frequency_roots.py",
                "--input-dir", S07, "--output-dir", dataset, "--opening", 20800,
                "--middlegame", 55200, "--endgame", 24000, "--seed", 20261025, "--max-roots-per-game", 64])
        manifest = read(dataset / "manifest.json")
        if manifest["selected_train_records"] != 100000:
            raise RuntimeError("pilot did not reach 100k roots")
        for output in manifest["output_files"]:
            if sha(dataset / output["path"]) != output["sha256"]:
                raise RuntimeError("pilot dataset hash mismatch")
        self.prepare_openings()

    def prepare_openings(self):
        path = self.base / "openings.json"
        if path.exists():
            return
        # The reserved historical test is consumed as a match-opening source,
        # never as training labels or a validation-based model selector.
        families = set()
        positions = set()
        shallow = []
        for line in (S09 / "dataset-2m/test.jsonl").open():
            row = json.loads(line)
            if not 8 <= row["ply_mean"] <= 20 or row["opening_family"] in families or row["position_key"] in positions:
                continue
            families.add(row["opening_family"])
            positions.add(row["position_key"])
            shallow.append("sfen " + row["sfen"])
            if len(shallow) == 50:
                break
        if len(shallow) != 50:
            raise RuntimeError("not enough independent shallow opening families")
        taya = [line.strip() for line in (ROOT / "taya36.sfen").read_text().splitlines() if line.strip()]
        used = set()
        for record in RUNS.rglob("game_*.usi"):
            with record.open(errors="replace") as stream:
                for _ in range(4):
                    line = stream.readline().strip()
                    if line.startswith("start_sfen "):
                        used.add(line.removeprefix("start_sfen "))
        for record in RUNS.rglob("pair_*.json"):
            value = read(record)
            opening = value.get("opening", {}).get("sfen")
            if opening:
                used.add(opening)
        unused_taya = list(dict.fromkeys(line for line in taya if line not in used))
        if len(unused_taya) < 100:
            raise RuntimeError(f"only {len(unused_taya)} unused recorded taya commands; reserve a new confirmation set before training")
        # Reserve all sets before training and retain the exact historical-use audit.
        selected = {"r2_shallow": shallow[:25], "pilot_shallow": shallow[25:],
                    "pilot_taya": unused_taya[:25], "confirm_parent": unused_taya[25:75], "confirm_v250": unused_taya[75:100]}
        if any(len(set(value)) != len(value) for value in selected.values()):
            raise RuntimeError("duplicate match openings")
        # Verify the shallow root identities against the two frozen train sets.
        keys = {" ".join(line.split()[1:4]) for line in shallow}
        for train in (S07 / "train.jsonl", S09 / "dataset-2m/train.jsonl"):
            for line in train.open():
                if json.loads(line)["position_key"] in keys:
                    raise RuntimeError("shallow match opening overlaps the current frozen train")
        freeze_json(path, {"source_test_sha256": sha(S09 / "dataset-2m/test.jsonl"),
                           "source_taya_sha256": sha(ROOT / "taya36.sfen"), "sets": selected,
                           "unused_taya_commands": len(unused_taya), "historical_commands": len(used),
                           "used_commands_sha256": hashlib.sha256("\n".join(sorted(used)).encode()).hexdigest(),
                           "novelty": "not present as exact recorded opening commands; not a proof of lifetime network-input novelty"})

    def evaluate(self, name: str, model: Path, valid: Path):
        output = self.base / "evaluation" / f"{name}.json"
        if output.exists():
            report = read(output)
            if report["model_sha256"] != sha(model) or report["input_sha256"] != [sha(valid)]:
                raise RuntimeError("evaluation provenance mismatch")
            return
        self.command(name, [self.tools / "halfkp_frequency_evaluate", "--model", model, "--input", valid,
                           "--output", output, "--jobs", 12, "--max-roots", 12000, "--tau-cp", 300])

    def tactics(self, name: str, model: Path):
        manifest = model_manifest(model, self.base / "evaluation" / f"{name}-model.json")
        for mode, tool, dataset, extra in [
            ("exact", "halfkp_search_gate", EXACT, ["--nodes", 20000]),
            ("health", "halfkp_scratch_eval", HEALTH, ["--bootstrap-iterations", 1000]),
        ]:
            output = self.base / "evaluation" / f"{name}-{mode}.json"
            if output.exists():
                if read(output)["model_sha256"] != sha(model):
                    raise RuntimeError("tactical report model mismatch")
                continue
            self.command(f"{name}-{mode}", [self.tools / tool, "--model", model, "--model-manifest", manifest,
                                           "--dataset", dataset, "--json", output, *extra])
            if mode == "exact" and read(output)["search_failed"]:
                raise RuntimeError("fixed tactical suite had a search failure")
        output = self.base / "evaluation" / f"{name}-captures.json"
        fixtures = ROOT / "tests/fixtures/search/material_capture_cases.json"
        if output.exists():
            if read(output)["model_sha256"] != sha(model) or read(output)["fixture_sha256"] != sha(fixtures):
                raise RuntimeError("capture probe provenance mismatch")
            return
        self.state(f"{name}-captures", state="running")
        cases = []
        for case in read(fixtures)["cases"]:
            best, transcript = usi_probe(self.tools / "usi_engine", model, case["position"])
            cases.append({**case, "best_move": best, "passed": best in case["capture_moves"], "transcript": transcript})
        write(output, {"model_sha256": sha(model), "fixture_sha256": sha(fixtures),
                       "successful": sum(case["passed"] for case in cases), "cases": cases,
                       "meaning": "immediate undefended-piece capture probes; not unique optimal-move proofs"})

    def match(self, name: str, candidate: Path, baseline: Path, opening_key: str) -> dict:
        directory = self.base / "matches" / name
        directory.mkdir(parents=True, exist_ok=True)
        openings = read(self.base / "openings.json")["sets"][opening_key]
        configuration = {"candidate_sha256": sha(candidate), "baseline_sha256": sha(baseline),
                         "engine_sha256": sha(self.tools / "usi_engine"), "harness_sha256": sha(self.tools / "usi_benchmark"),
                         "openings": openings, "time_limit_ms": 5000, "depth": 32, "threads": 1,
                         "max_plies": 256, "seed": 20261025}
        freeze_json(directory / "config.json", configuration)
        pairs = []
        for index, opening in enumerate(openings):
            pair_dir = directory / f"pair_{index:04d}"
            marker = pair_dir / "complete.json"
            if marker.exists():
                pair = parse_pair(pair_dir, opening)
                if pair != read(marker):
                    raise RuntimeError("completed USI pair changed")
            else:
                if pair_dir.exists():
                    shutil.rmtree(pair_dir)  # Partial pair only; completed pairs are immutable.
                pair_dir.mkdir()
                opening_file = pair_dir / "opening.sfen"
                opening_file.write_text(opening + "\n")
                self.command(f"{name}-pair-{index:04d}", [self.tools / "usi_benchmark",
                    "--new-engine", self.tools / "usi_engine", "--baseline-engine", self.tools / "usi_engine",
                    "--new-weights", candidate, "--baseline-weights", baseline, "--positions", opening_file,
                    "--games", 2, "--depth", 32, "--time-limit-ms", 5000, "--new-threads", 1,
                    "--baseline-threads", 1, "--max-plies", 256, "--jobs", 1, "--seed", 20261025,
                    "--record-dir", pair_dir / "games"])
                pair = parse_pair(pair_dir, opening)
                write(marker, pair)
            pairs.append(pair)
            self.state(name, state="running", completed_pairs=len(pairs), total_pairs=len(openings))
        scores = [pair["score"] for pair in pairs]
        ci = paired_percentile_bootstrap(scores, 100000, 20261025)
        report = {"config": configuration, "games": len(pairs) * 2, "pairs": len(pairs),
                  "score": sum(scores) / len(scores), "paired_ci95": ci,
                  **{key: sum(pair[key] for pair in pairs) for key in ("candidate_wins", "baseline_wins", "draws")}}
        write(directory / "report.json", report)
        return report

    def train(self, label: str, dataset: Path, loss: str) -> Path:
        run = self.base / label
        lineage = S09 / "research-parent"
        for outer in (2, 4, 8):
            checkpoint = run / "checkpoint.json"
            current = read(checkpoint)["completed_outer"] if checkpoint.exists() else 0
            if current < outer:
                command = [self.tools / "halfkp_mmto_train", "--train", dataset / "train.jsonl",
                    "--valid", dataset / "valid.jsonl", "--dataset-manifest", dataset / "manifest.json",
                    "--expected-dataset-manifest-sha256", sha(dataset / "manifest.json"), "--init", PARENT,
                    "--init-optimizer", OPTIMIZER, "--model-manifest", lineage / "v2.7.1.model.manifest.json",
                    "--promotion-registry", lineage / "research_parent_registry.json",
                    "--gate-report", lineage / "v2.7.1_gate_summary.json", "--parent-generation", 7,
                    "--output-dir", run, "--outer-iterations", outer, "--frequency-loss", loss,
                    "--root-batch-size", 128, "--search-block-roots", 1024, "--jobs", 12,
                    "--learning-rate", 0.0001, "--output-learning-rate-ratio", 0.1, "--tau-cp", 300,
                    "--gradient-clip-norm", 1, "--max-valid-roots", 12000, "--pv-probe-roots", 256,
                    "--checkpoint-roots", 25000, "--qsearch-node-limit", 2000000]
                if checkpoint.exists():
                    command.append("--resume")
                self.command(f"{label}-outer{outer}", command)
            snapshot(run, self.base / f"{label}-outer{outer}", outer)
        return self.base / f"{label}-outer8/model.bin"

    def run(self):
        self.prepare()
        old_valid = S07 / "valid.jsonl"
        new_valid = S09 / "dataset-2m/valid.jsonl"
        models = {"parent": PARENT, "control-a4": S09 / "frozen-control-outer4/model.bin",
                  "s09-b2": S09 / "frozen-2m-outer2/model.bin", "s09-b8": S09 / "frozen-2m-outer8/model.bin"}
        for label, model in models.items():
            for cohort, valid in [("old", old_valid), ("new", new_valid)]:
                self.evaluate(f"r2-{label}-{cohort}", model, valid)
        for label in ("parent", "s09-b8"):
            self.tactics(f"r2-{label}", models[label])
        self.match("r2-s09-vs-parent-shallow", models["s09-b8"], PARENT, "r2_shallow")
        self.state("R2_COMPLETE", state="completed")
        a = self.train("pilot-ranknet", self.base / "dataset-100k", "ranknet")
        b = self.train("pilot-listwise", self.base / "dataset-100k", "listwise")
        for label, model in [("ranknet", a), ("listwise", b)]:
            for cohort, valid in [("old", old_valid), ("new", new_valid)]:
                self.evaluate(f"r3-{label}-{cohort}", model, valid)
            self.tactics(f"r3-{label}", model)
        ba = self.match("r4-listwise-vs-ranknet", b, a, "pilot_shallow")
        bp = self.match("r4-listwise-vs-parent", b, PARENT, "pilot_taya")
        exact_parent = read(self.base / "evaluation/r2-parent-exact.json")
        exact_b = read(self.base / "evaluation/r3-listwise-exact.json")
        # A conservative aggregate regression screen, not a proof that no
        # individual tactical case regressed. Keep the reports for inspection.
        tactics_ok = (exact_b["conclusive"] >= exact_parent["conclusive"] and
                      exact_b["successful"] >= exact_parent["successful"])
        parent_captures = read(self.base / "evaluation/r2-parent-captures.json")["cases"]
        b_captures = read(self.base / "evaluation/r3-listwise-captures.json")["cases"]
        tactics_ok = tactics_ok and all(not parent_case["passed"] or candidate_case["passed"]
                                       for parent_case, candidate_case in zip(parent_captures, b_captures))
        expand = ba["score"] >= 0.54 and bp["score"] >= 0.54 and tactics_ok
        write(self.base / "pilot-decision.json", {"expand_to_1m": expand, "vs_ranknet": ba["score"],
              "vs_parent": bp["score"], "aggregate_tactics_no_drop": tactics_ok,
              "reason": "predeclared operational screen; CI crossing 50% alone does not stop expansion"})
        if expand:
            large = self.train("listwise-1m", S07, "listwise")
            self.tactics("r5-listwise-1m", large)
            self.match("r5-confirm-vs-parent", large, PARENT, "confirm_parent")
            self.match("r5-reference-vs-v250", large, ROOT / "policy_weights_halfkp64_kpp_distilled_v2.5.0.binary", "confirm_v250")
        self.state("COMPLETE", state="completed", expanded_to_1m=expand,
                   release_promoted=False, next_action="review saved reports; no automatic release or new algorithm")
        write(self.base / "COMPLETE.json", read(self.base / "status.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUNS / "s10-listwise-20261026-restart")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    experiment = Experiment(args.run_dir)
    with (experiment.base / ".lock").open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.prepare_only:
                experiment.prepare()
                experiment.state("PREPARED", state="prepared")
            else:
                experiment.run()
        except BaseException as error:
            experiment.state(experiment.stage, state="failed", error=str(error))
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

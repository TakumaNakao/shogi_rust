#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SCREEN_DIR="$ROOT_DIR/runs/halfkp64-scratch-v1/s05-outer8-vs-gen6-taya36-5s-25pair"
PIPELINE_DIR="$ROOT_DIR/runs/halfkp64-scratch-v1/s05-outer8-pipeline-20260918"
CANDIDATE="$ROOT_DIR/runs/halfkp64-scratch-v1/s04-soft-target-250k-8pass-20260916/model.bin"
CANDIDATE_MANIFEST="$ROOT_DIR/training/halfkp64-scratch-v1/s05_candidate_outer8_manifest_20260917.json"
PARENT="$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen5-20260910/candidate/model-epoch24.bin"
PARENT_MANIFEST="$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen5-20260910/candidate/model-epoch24.bin.manifest.json"
REGISTRY="$ROOT_DIR/training/halfkp64-scratch-v1/promotion_registry.json"
BINARY="$ROOT_DIR/target/release/halfkp_promotion_match"
MATCH="$ROOT_DIR/scripts/run_halfkp_paired_match.py"
TAYA36="$ROOT_DIR/taya36.sfen"
TEST_ROOTS="$ROOT_DIR/runs/halfkp64-scratch-v1/s04-selected-250k-v3-20260916/test.jsonl"
DECISION="$PIPELINE_DIR/decision.json"

mkdir -p "$PIPELINE_DIR"
exec 9>"$PIPELINE_DIR/.lock"
flock -n 9 || exit 0
exec > >(tee -a "$PIPELINE_DIR/pipeline.log") 2>&1

if [[ ! -s "$SCREEN_DIR/completed.json" || ! -s "$SCREEN_DIR/report.json" ]]; then
  echo "S05 screen is not complete; refusing to continue"
  exit 2
fi

python3 - "$SCREEN_DIR/report.json" "$DECISION" <<'PY'
import json
import pathlib
import sys

report_path, decision_path = map(pathlib.Path, sys.argv[1:])
report = json.loads(report_path.read_text())
counts = report.get("counts", {})
safety = all(int(counts.get(key, 0)) == 0 for key in ("search_failures", "missing_root_results", "illegal_moves", "non_finite_scores"))
score = float(report.get("score", 0.0))
decision = {
    "schema": "halfkp_s05_decision_v1",
    "screen": {
        "report": str(report_path),
        "candidate_wins": counts.get("candidate_wins"),
        "parent_wins": counts.get("parent_wins"),
        "draws": counts.get("draws"),
        "score": score,
        "paired_ci95": [report.get("opening_pair_bootstrap", {}).get("ci95_lower"), report.get("opening_pair_bootstrap", {}).get("ci95_upper")],
        "safety": safety,
    },
    "screen_pass": score > 0.50 and safety,
    "endgame_non_regression": None,
    "confirmation_pass": None,
    "v250_comparison": None,
}
decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
print("screen_pass", decision["screen_pass"], "score", score, "safety", safety)
if not decision["screen_pass"]:
    raise SystemExit(10)
PY

python3 - "$TEST_ROOTS" "$PIPELINE_DIR/endgame_openings.sfen" "$PIPELINE_DIR/endgame_openings_manifest.json" <<'PY'
import hashlib
import json
import pathlib
import sys

source, output, manifest = map(pathlib.Path, sys.argv[1:])
selected = []
seen_games = set()
for line_number, raw in enumerate(source.read_bytes().splitlines(), 1):
    record = json.loads(raw)
    if record.get("split") != "test" or record.get("phase") != "endgame":
        continue
    game_key = record.get("game_key")
    sfen = record.get("sfen")
    if not isinstance(game_key, str) or game_key in seen_games or not isinstance(sfen, str):
        continue
    seen_games.add(game_key)
    selected.append({"source_line": line_number, "game_key": game_key, "sfen": sfen})
    if len(selected) == 25:
        break
if len(selected) != 25:
    raise SystemExit(f"expected 25 independent endgame roots, found {len(selected)}")
output.write_text("".join(f"sfen {entry['sfen']}\n" for entry in selected))
manifest.write_text(json.dumps({
    "schema": "halfkp_s05_endgame_openings_v1",
    "source": str(source),
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "selected_records": selected,
    "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
}, indent=2, sort_keys=True) + "\n")
PY

run_match() {
  local openings="$1"
  local pairs="$2"
  local seed="$3"
  local output="$4"
  python3 "$MATCH" \
    --binary "$BINARY" \
    --candidate "$CANDIDATE" --candidate-manifest "$CANDIDATE_MANIFEST" \
    --parent "$PARENT" --parent-manifest "$PARENT_MANIFEST" \
    --registry "$REGISTRY" --openings "$openings" \
    --openings-format usi-lines --opening-selection first-unique \
    --pairs "$pairs" --time-limit-ms 5000 --max-depth 64 --max-plies 256 \
    --bootstrap-resamples 100000 --bootstrap-seed "$seed" --output-dir "$output"
}

run_match "$PIPELINE_DIR/endgame_openings.sfen" 25 20260918 "$PIPELINE_DIR/endgame-match"
python3 - "$PIPELINE_DIR/endgame-match/report.json" "$DECISION" <<'PY'
import json
import pathlib
import sys

report_path, decision_path = map(pathlib.Path, sys.argv[1:])
report = json.loads(report_path.read_text())
counts = report.get("counts", {})
safety = all(int(counts.get(key, 0)) == 0 for key in ("search_failures", "missing_root_results", "illegal_moves", "non_finite_scores"))
score = float(report.get("score", 0.0))
ci = report.get("opening_pair_bootstrap", {})
decision = json.loads(decision_path.read_text())
decision["endgame_non_regression"] = {
    "candidate_wins": counts.get("candidate_wins"),
    "parent_wins": counts.get("parent_wins"),
    "draws": counts.get("draws"),
    "score": score,
    "paired_ci95": [ci.get("ci95_lower"), ci.get("ci95_upper")],
    "safety": safety,
    "pass": score >= 0.45 and safety,
}
decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
print("endgame_non_regression", decision["endgame_non_regression"])
if not decision["endgame_non_regression"]["pass"]:
    raise SystemExit(11)
PY

python3 - "$TAYA36" "$PIPELINE_DIR/confirmation_openings.sfen" <<'PY'
import pathlib
import sys

source, output = map(pathlib.Path, sys.argv[1:])
lines = [line for line in source.read_bytes().splitlines() if line.strip()]
selected = lines[25:125]
if len(selected) != 100:
    raise SystemExit(f"expected 100 unused taya36 openings, found {len(selected)}")
output.write_bytes(b"\n".join(selected) + b"\n")
PY

run_match "$PIPELINE_DIR/confirmation_openings.sfen" 100 20260919 "$PIPELINE_DIR/confirmation-match"
python3 - "$PIPELINE_DIR/confirmation-match/report.json" "$DECISION" <<'PY'
import json
import pathlib
import sys

report_path, decision_path = map(pathlib.Path, sys.argv[1:])
report = json.loads(report_path.read_text())
counts = report.get("counts", {})
safety = all(int(counts.get(key, 0)) == 0 for key in ("search_failures", "missing_root_results", "illegal_moves", "non_finite_scores"))
score = float(report.get("score", 0.0))
ci = report.get("opening_pair_bootstrap", {})
lower = float(ci.get("ci95_lower", 0.0))
decision = json.loads(decision_path.read_text())
decision["confirmation_pass"] = {
    "candidate_wins": counts.get("candidate_wins"),
    "parent_wins": counts.get("parent_wins"),
    "draws": counts.get("draws"),
    "score": score,
    "paired_ci95": [lower, ci.get("ci95_upper")],
    "safety": safety,
    "pass": score >= 0.525 and lower > 0.50 and safety,
}
decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
print("confirmation_pass", decision["confirmation_pass"])
if not decision["confirmation_pass"]["pass"]:
    raise SystemExit(12)
PY

V250_DIR="$PIPELINE_DIR/v250-taya36-5s-50g"
RUST_FONTCONFIG_DLOPEN=1 \
NEW_WEIGHTS="$CANDIDATE" \
BASELINE_WEIGHTS="$ROOT_DIR/policy_weights_halfkp64_kpp_distilled_v2.5.0.binary" \
POSITIONS="$TAYA36" GAMES=50 TIME_LIMIT_MS=5000 DEPTH=32 MAX_PLIES=256 \
SEED=20260920 THREADS=1 JOBS=1 OUT_DIR="$V250_DIR" \
bash scripts/bench_halfkp64_learned_vs_v250.sh

python3 - "$V250_DIR/match.log" "$DECISION" <<'PY'
import json
import pathlib
import re
import sys

log_path, decision_path = map(pathlib.Path, sys.argv[1:])
log = log_path.read_text()

def find_int(pattern):
    match = re.search(pattern, log)
    return int(match.group(1)) if match else None

rate = re.search(r"new total score rate: ([0-9.]+)%", log)
decision = json.loads(decision_path.read_text())
decision["v250_comparison"] = {
    "candidate_wins": find_int(r"new wins: (\d+)"),
    "v250_wins": find_int(r"baseline wins: (\d+)"),
    "draws": find_int(r"draws: (\d+)"),
    "candidate_total_score_rate": float(rate.group(1)) / 100 if rate else None,
    "status": "completed",
}
decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
print("v250_comparison", decision["v250_comparison"])
PY

echo "S05 pipeline complete: $DECISION"

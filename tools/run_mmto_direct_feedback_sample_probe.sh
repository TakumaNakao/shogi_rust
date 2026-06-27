#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script. Run it as: bash tools/run_mmto_direct_feedback_sample_probe.sh"
  return 2
fi

cd "$(dirname "$0")/.."

RUN_DIR="${RUN_DIR:-data/mmto/runs/direct_feedback_sample_probe_$(date -u +%Y%m%d_%H%M%S)}"
BASELINE_WEIGHTS="${BASELINE_WEIGHTS:-policy_weights_v2.1.0.binary}"
CANDIDATE_WEIGHTS="${CANDIDATE_WEIGHTS:-data/mmto/runs/mmto_softguard_feedback_probe_20260627_141905/initial_benchgate/iter_1/best.raw.binary}"
TEACHER_WEIGHTS="${TEACHER_WEIGHTS:-$BASELINE_WEIGHTS}"
POSITIONS="${POSITIONS:-data/mmto/runs/mmto_direct_feedback_probe_20260627_151908/direct_feedback_input.sfen}"
MAX_POSITIONS_LIST="${MAX_POSITIONS_LIST:-40 80}"
BASELINE_DEPTH="${BASELINE_DEPTH:-3}"
CANDIDATE_DEPTH="${CANDIDATE_DEPTH:-3}"
TEACHER_DEPTH="${TEACHER_DEPTH:-5}"
JOBS="${JOBS:-0}"
SEED="${SEED:-18001}"
HARD_POSITION_LIMIT="${HARD_POSITION_LIMIT:-2000}"
MIN_FREE_GB="${MIN_FREE_GB:-5}"

for path in "$BASELINE_WEIGHTS" "$CANDIDATE_WEIGHTS" "$TEACHER_WEIGHTS" "$POSITIONS"; do
  if [[ ! -e "$path" ]]; then
    echo "missing required input: $path" >&2
    exit 1
  fi
done

FREE_KB="$(df -Pk . | awk 'NR==2 {print $4}')"
MIN_FREE_KB=$((MIN_FREE_GB * 1024 * 1024))
if (( FREE_KB < MIN_FREE_KB )); then
  echo "free disk is too low: $((FREE_KB / 1024 / 1024))GB < ${MIN_FREE_GB}GB" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"

{
  echo "RUN_DIR=$RUN_DIR"
  echo "BASELINE_WEIGHTS=$BASELINE_WEIGHTS"
  echo "CANDIDATE_WEIGHTS=$CANDIDATE_WEIGHTS"
  echo "TEACHER_WEIGHTS=$TEACHER_WEIGHTS"
  echo "POSITIONS=$POSITIONS"
  echo "MAX_POSITIONS_LIST=$MAX_POSITIONS_LIST"
  echo "BASELINE_DEPTH=$BASELINE_DEPTH CANDIDATE_DEPTH=$CANDIDATE_DEPTH TEACHER_DEPTH=$TEACHER_DEPTH"
  echo "JOBS=$JOBS SEED=$SEED"
  df -h .
  wc -l "$POSITIONS"
} | tee "$RUN_DIR/config.txt"

env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin mmto_rerank_gate

summary_path="$RUN_DIR/summary.tsv"
printf "max_positions\tstatus\tseconds\tsamples\thard_positions\tteacher_candidate_diff\tcandidate_mean\tcandidate_p95\tbaseline_mean\tbaseline_p95\tjson_bytes\n" \
  > "$summary_path"

for max_positions in $MAX_POSITIONS_LIST; do
  label="max${max_positions}_d${BASELINE_DEPTH}_${CANDIDATE_DEPTH}_${TEACHER_DEPTH}"
  json_path="$RUN_DIR/${label}.json"
  stdout_path="$RUN_DIR/${label}.stdout.log"
  status_path="$RUN_DIR/${label}.status"
  start_seconds="$(date +%s)"
  set +e
  env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_rerank_gate \
    --baseline-weights "$BASELINE_WEIGHTS" \
    --candidate-weights "$CANDIDATE_WEIGHTS" \
    --teacher-weights "$TEACHER_WEIGHTS" \
    --input "$POSITIONS" \
    --dedupe-sfen \
    --max-positions "$max_positions" \
    --baseline-depth "$BASELINE_DEPTH" \
    --candidate-depth "$CANDIDATE_DEPTH" \
    --teacher-depth "$TEACHER_DEPTH" \
    --jobs "$JOBS" \
    --seed "$SEED" \
    --hard-position-limit "$HARD_POSITION_LIMIT" \
    --json-output "$json_path" \
    2>&1 | tee "$stdout_path"
  status="${PIPESTATUS[0]}"
  set -e
  end_seconds="$(date +%s)"
  seconds=$((end_seconds - start_seconds))
  echo "$status" > "$status_path"

  python3 - "$json_path" "$max_positions" "$status" "$seconds" >> "$summary_path" <<'PY'
import json
import os
import sys

path, max_positions, status, seconds = sys.argv[1:5]
if not os.path.exists(path):
    print(f"{max_positions}\t{status}\t{seconds}\t0\t0\t0\t0\t0\t0\t0\t0")
    raise SystemExit

with open(path, "r", encoding="utf-8") as handle:
    payload = json.load(handle)

hard = payload.get("hard_positions", [])
diff = 0
for item in hard:
    if item.get("teacher_best_move") != item.get("candidate_move"):
        diff += 1

baseline = payload.get("baseline", {})
candidate = payload.get("candidate", {})
size = os.path.getsize(path)
print(
    "\t".join(
        [
            str(max_positions),
            str(status),
            str(seconds),
            str(candidate.get("samples", 0)),
            str(len(hard)),
            str(diff),
            f"{float(candidate.get('mean_regret_cp', 0.0)):.2f}",
            f"{float(candidate.get('p95_regret_cp', 0.0)):.2f}",
            f"{float(baseline.get('mean_regret_cp', 0.0)):.2f}",
            f"{float(baseline.get('p95_regret_cp', 0.0)):.2f}",
            str(size),
        ]
    )
)
PY
done

if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "$summary_path" | tee "$RUN_DIR/summary.txt"
else
  cp "$summary_path" "$RUN_DIR/summary.txt"
  cat "$RUN_DIR/summary.txt"
fi
du -sh "$RUN_DIR" | tee "$RUN_DIR/du_after.txt"
df -h . | tee "$RUN_DIR/df_after.txt"

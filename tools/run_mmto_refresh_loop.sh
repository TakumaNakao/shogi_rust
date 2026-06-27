#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script. Run it as: SOURCE_RUN_DIR=... bash tools/run_mmto_refresh_loop.sh"
  return 2
fi

cd "$(dirname "$0")/.."

SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-}"
if [[ -z "$SOURCE_RUN_DIR" ]]; then
  echo "SOURCE_RUN_DIR is required." >&2
  exit 1
fi

WEIGHTS="${WEIGHTS:-policy_weights_v2.1.0.binary}"
INITIAL_CANDIDATE_WEIGHTS="${INITIAL_CANDIDATE_WEIGHTS:-$WEIGHTS}"
LOOP_DIR="${LOOP_DIR:-data/mmto/runs/mmto_refresh_loop_$(date -u +%Y%m%d_%H%M%S)}"
ITERATIONS="${ITERATIONS:-3}"
KEEP_PASSED_WEIGHTS="${KEEP_PASSED_WEIGHTS:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-6}"
BEST_METRIC="${BEST_METRIC:-p95-regret}"
BEST_GUARD_MAX_REGRET_INCREASE_CP="${BEST_GUARD_MAX_REGRET_INCREASE_CP:-0}"
BEST_GUARD_BAD100_INCREASE="${BEST_GUARD_BAD100_INCREASE:-0}"

if [[ ! -f "$INITIAL_CANDIDATE_WEIGHTS" ]]; then
  echo "missing initial candidate weights: $INITIAL_CANDIDATE_WEIGHTS" >&2
  exit 1
fi
if (( ITERATIONS <= 0 )); then
  echo "ITERATIONS must be greater than zero" >&2
  exit 1
fi

FREE_KB="$(df -Pk . | awk 'NR==2 {print $4}')"
MIN_FREE_KB=$((MIN_FREE_GB * 1024 * 1024))
if (( FREE_KB < MIN_FREE_KB )); then
  echo "free disk is too low: $((FREE_KB / 1024 / 1024))GB < ${MIN_FREE_GB}GB" >&2
  exit 1
fi

mkdir -p "$LOOP_DIR"

current_candidate="$INITIAL_CANDIDATE_WEIGHTS"
last_generated_candidate=""

echo "Starting MMTO refresh loop."
echo "LOOP_DIR=$LOOP_DIR"
echo "SOURCE_RUN_DIR=$SOURCE_RUN_DIR"
echo "INITIAL_CANDIDATE_WEIGHTS=$INITIAL_CANDIDATE_WEIGHTS"
echo "ITERATIONS=$ITERATIONS"
echo "BEST_METRIC=$BEST_METRIC"
echo "BEST_GUARD_MAX_REGRET_INCREASE_CP=$BEST_GUARD_MAX_REGRET_INCREASE_CP"
echo "BEST_GUARD_BAD100_INCREASE=$BEST_GUARD_BAD100_INCREASE"

for iteration in $(seq 1 "$ITERATIONS"); do
  iter_dir="$LOOP_DIR/iter_${iteration}"
  mkdir -p "$iter_dir"
  echo "iteration=$iteration candidate=$current_candidate"

  set +e
  SOURCE_RUN_DIR="$SOURCE_RUN_DIR" \
    CANDIDATE_WEIGHTS="$current_candidate" \
    RUN_DIR="$iter_dir" \
    KEEP_CANDIDATE_RAW=1 \
    BEST_METRIC="$BEST_METRIC" \
    BEST_GUARD_MAX_REGRET_INCREASE_CP="$BEST_GUARD_MAX_REGRET_INCREASE_CP" \
    BEST_GUARD_BAD100_INCREASE="$BEST_GUARD_BAD100_INCREASE" \
    bash tools/run_mmto_refresh_from_candidate.sh \
    > "$iter_dir/pipeline_stdout.log" 2>&1
  status=$?
  set -e

  if [[ -f "$iter_dir/candidate.raw.binary" ]]; then
    rm -f "$iter_dir/candidate.raw.binary"
  fi

  if (( status != 0 )); then
    echo "iteration=$iteration failed with status=$status. Rejecting new candidate."
    rm -f "$iter_dir/best.raw.binary"
    echo "LOOP_DIR=$LOOP_DIR"
    echo "FINAL_CANDIDATE=$current_candidate"
    exit "$status"
  fi

  if ! grep -q "refresh offline gates passed" "$iter_dir/pipeline_stdout.log"; then
    echo "iteration=$iteration did not pass offline gates. Stopping loop."
    rm -f "$iter_dir/best.raw.binary"
    echo "LOOP_DIR=$LOOP_DIR"
    echo "FINAL_CANDIDATE=$current_candidate"
    exit 0
  fi

  if [[ ! -f "$iter_dir/best.raw.binary" ]]; then
    echo "iteration=$iteration passed but best.raw.binary is missing" >&2
    echo "LOOP_DIR=$LOOP_DIR"
    exit 1
  fi

  previous_generated="$last_generated_candidate"
  current_candidate="$iter_dir/best.raw.binary"
  last_generated_candidate="$current_candidate"
  echo "$current_candidate" > "$LOOP_DIR/current_candidate_path.txt"
  echo "iteration=$iteration accepted candidate=$current_candidate"

  if [[ "$KEEP_PASSED_WEIGHTS" != "1" && -n "$previous_generated" && "$previous_generated" != "$current_candidate" ]]; then
    case "$previous_generated" in
      "$LOOP_DIR"/*) rm -f "$previous_generated" ;;
    esac
  fi
done

echo "refresh loop completed all iterations."
echo "LOOP_DIR=$LOOP_DIR"
echo "FINAL_CANDIDATE=$current_candidate"

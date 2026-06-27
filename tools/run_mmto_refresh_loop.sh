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

BASE_WEIGHTS="${WEIGHTS:-policy_weights_v2.1.0.binary}"
TEACHER_WEIGHTS="${TEACHER_WEIGHTS:-$BASE_WEIGHTS}"
INITIAL_CANDIDATE_WEIGHTS="${INITIAL_CANDIDATE_WEIGHTS:-$BASE_WEIGHTS}"
LOOP_DIR="${LOOP_DIR:-data/mmto/runs/mmto_refresh_loop_$(date -u +%Y%m%d_%H%M%S)}"
ITERATIONS="${ITERATIONS:-3}"
KEEP_PASSED_WEIGHTS="${KEEP_PASSED_WEIGHTS:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-6}"
BEST_METRIC="${BEST_METRIC:-p95-regret}"
BEST_GUARD_MAX_REGRET_INCREASE_CP="${BEST_GUARD_MAX_REGRET_INCREASE_CP:-0}"
BEST_GUARD_BAD100_INCREASE="${BEST_GUARD_BAD100_INCREASE:-0}"
HARD_FEEDBACK_ON_FAILURE="${HARD_FEEDBACK_ON_FAILURE:-1}"

if [[ ! -f "$INITIAL_CANDIDATE_WEIGHTS" ]]; then
  echo "missing initial candidate weights: $INITIAL_CANDIDATE_WEIGHTS" >&2
  exit 1
fi
if [[ ! -f "$BASE_WEIGHTS" ]]; then
  echo "missing base weights: $BASE_WEIGHTS" >&2
  exit 1
fi
if [[ ! -f "$TEACHER_WEIGHTS" ]]; then
  echo "missing teacher weights: $TEACHER_WEIGHTS" >&2
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
echo "BASE_WEIGHTS=$BASE_WEIGHTS"
echo "TEACHER_WEIGHTS=$TEACHER_WEIGHTS"
echo "INITIAL_CANDIDATE_WEIGHTS=$INITIAL_CANDIDATE_WEIGHTS"
echo "ITERATIONS=$ITERATIONS"
echo "BEST_METRIC=$BEST_METRIC"
echo "BEST_GUARD_MAX_REGRET_INCREASE_CP=$BEST_GUARD_MAX_REGRET_INCREASE_CP"
echo "BEST_GUARD_BAD100_INCREASE=$BEST_GUARD_BAD100_INCREASE"
echo "HARD_FEEDBACK_ON_FAILURE=$HARD_FEEDBACK_ON_FAILURE"

accept_candidate() {
  local accepted_path="$1"
  local label="$2"
  local previous_generated="$last_generated_candidate"
  current_candidate="$accepted_path"
  last_generated_candidate="$current_candidate"
  echo "$current_candidate" > "$LOOP_DIR/current_candidate_path.txt"
  echo "$label accepted candidate=$current_candidate"

  if [[ "$KEEP_PASSED_WEIGHTS" != "1" && -n "$previous_generated" && "$previous_generated" != "$current_candidate" ]]; then
    case "$previous_generated" in
      "$LOOP_DIR"/*) rm -f "$previous_generated" ;;
    esac
  fi
}

try_hard_feedback() {
  local failed_dir="$1"
  local iteration="$2"

  if [[ "$HARD_FEEDBACK_ON_FAILURE" != "1" ]]; then
    return 1
  fi
  if [[ ! -s "$failed_dir/rerank_gate.json" ]]; then
    return 1
  fi
  if [[ ! -s "$failed_dir/train.tree.jsonl" || ! -s "$failed_dir/valid.tree.jsonl" ]]; then
    return 1
  fi

  local feedback_dir="$failed_dir/hard_feedback"
  mkdir -p "$feedback_dir"
  echo "iteration=$iteration starting weighted hard feedback from $failed_dir/rerank_gate.json"

  set +e
  SOURCE_RUN_DIR="$failed_dir" \
    WEIGHTS="$current_candidate" \
    TEACHER_WEIGHTS="$TEACHER_WEIGHTS" \
    BASE_TRAIN="$failed_dir/train.tree.jsonl" \
    BASE_VALID="$failed_dir/valid.tree.jsonl" \
    SCORE_POSITIONS="$failed_dir/score_positions.sfen" \
    RUN_DIR="$feedback_dir" \
    KEEP_CANDIDATE_RAW=1 \
    BEST_METRIC="$BEST_METRIC" \
    BEST_GUARD_MAX_REGRET_INCREASE_CP="$BEST_GUARD_MAX_REGRET_INCREASE_CP" \
    BEST_GUARD_BAD100_INCREASE="$BEST_GUARD_BAD100_INCREASE" \
    bash tools/run_mmto_dagger_from_run.sh \
    > "$feedback_dir/pipeline_stdout.log" 2>&1
  local feedback_status=$?
  set -e

  rm -f "$feedback_dir/candidate.raw.binary"

  if (( feedback_status != 0 )); then
    echo "iteration=$iteration hard feedback failed with status=$feedback_status."
    rm -f "$feedback_dir/best.raw.binary"
    return 1
  fi
  if ! grep -q "dagger offline gates passed" "$feedback_dir/pipeline_stdout.log"; then
    echo "iteration=$iteration hard feedback did not pass offline gates."
    rm -f "$feedback_dir/best.raw.binary"
    return 1
  fi
  if [[ ! -f "$feedback_dir/best.raw.binary" ]]; then
    echo "iteration=$iteration hard feedback passed but best.raw.binary is missing" >&2
    return 1
  fi

  accept_candidate "$feedback_dir/best.raw.binary" "iteration=$iteration hard_feedback"
  return 0
}

for iteration in $(seq 1 "$ITERATIONS"); do
  iter_dir="$LOOP_DIR/iter_${iteration}"
  mkdir -p "$iter_dir"
  echo "iteration=$iteration candidate=$current_candidate"

  set +e
  SOURCE_RUN_DIR="$SOURCE_RUN_DIR" \
    WEIGHTS="$current_candidate" \
    TEACHER_WEIGHTS="$TEACHER_WEIGHTS" \
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
    if try_hard_feedback "$iter_dir" "$iteration"; then
      continue
    fi
    echo "LOOP_DIR=$LOOP_DIR"
    echo "FINAL_CANDIDATE=$current_candidate"
    if [[ -s "$iter_dir/rerank_gate.json" ]]; then
      exit 0
    fi
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

  accept_candidate "$iter_dir/best.raw.binary" "iteration=$iteration refresh"
done

echo "refresh loop completed all iterations."
echo "LOOP_DIR=$LOOP_DIR"
echo "FINAL_CANDIDATE=$current_candidate"

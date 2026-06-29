#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script. Run it as: bash tools/run_strong_teacher_feedback_screen.sh"
  return 2
fi

cd "$(dirname "$0")/.."

SCREEN_NAME="${SCREEN_NAME:-strong_teacher_feedback}"
SCREEN_DIR="${SCREEN_DIR:-data/mmto/runs/${SCREEN_NAME}_$(date -u +%Y%m%d_%H%M%S)}"
SEEDS="${SEEDS:-7501 7601}"

SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-data/mmto/runs/pv_sibling_feedback_20260628_190857}"
HARD_LIMIT="${HARD_LIMIT:-300}"
FEEDBACK_LIMIT="${FEEDBACK_LIMIT:-900}"
PROTECTION_LINES="${PROTECTION_LINES:-1200}"
VALID_LINES="${VALID_LINES:-400}"
EPOCHS="${EPOCHS:-8}"
RERANK_MAX_POSITIONS="${RERANK_MAX_POSITIONS:-400}"
MIN_FREE_GB="${MIN_FREE_GB:-1}"
KEEP_SCREEN_BINARY="${KEEP_SCREEN_BINARY:-0}"

mkdir -p "$SCREEN_DIR"

echo "Starting strong-teacher feedback screen."
echo "SCREEN_DIR=$SCREEN_DIR"
echo "SCREEN_NAME=$SCREEN_NAME"
echo "SEEDS=$SEEDS"
echo "SOURCE_RUN_DIR=$SOURCE_RUN_DIR"
echo "HARD_LIMIT=$HARD_LIMIT FEEDBACK_LIMIT=$FEEDBACK_LIMIT PROTECTION_LINES=$PROTECTION_LINES"
echo "EPOCHS=$EPOCHS RERANK_MAX_POSITIONS=$RERANK_MAX_POSITIONS KEEP_SCREEN_BINARY=$KEEP_SCREEN_BINARY"

run_dirs=()
for seed in $SEEDS; do
  run_dir="$SCREEN_DIR/seed_${seed}"
  run_dirs+=("$run_dir")
  mkdir -p "$run_dir"
  echo "screen seed=$seed run_dir=$run_dir"

  set +e
  SOURCE_RUN_DIR="$SOURCE_RUN_DIR" \
    RUN_DIR="$run_dir" \
    MIN_FREE_GB="$MIN_FREE_GB" \
    HARD_LIMIT="$HARD_LIMIT" \
    FEEDBACK_LIMIT="$FEEDBACK_LIMIT" \
    PROTECTION_LINES="$PROTECTION_LINES" \
    VALID_LINES="$VALID_LINES" \
    EPOCHS="$EPOCHS" \
    RERANK_MAX_POSITIONS="$RERANK_MAX_POSITIONS" \
    DUMP_SEED="$seed" \
    FEEDBACK_SEED="$((seed + 101))" \
    PROTECTION_SEED="$((seed + 202))" \
    RUN_BENCH=0 \
    LOSS_MODE="${LOSS_MODE:-listwise-leaf}" \
    STREAM_TRAIN="${STREAM_TRAIN:-1}" \
    BEST_METRIC="${BEST_METRIC:-feedback-violation}" \
    BEST_GUARD_MAX_REGRET_INCREASE_CP="${BEST_GUARD_MAX_REGRET_INCREASE_CP:-0}" \
    BEST_GUARD_BAD100_INCREASE="${BEST_GUARD_BAD100_INCREASE:-0}" \
    BEST_GUARD_TEACHER_MATCH_DROP_PCT="${BEST_GUARD_TEACHER_MATCH_DROP_PCT:-0}" \
    BEST_GUARD_FEEDBACK_VIOLATION_INCREASE="${BEST_GUARD_FEEDBACK_VIOLATION_INCREASE:-0}" \
    BEST_GUARD_FEEDBACK_LOSS_INCREASE="${BEST_GUARD_FEEDBACK_LOSS_INCREASE:--1}" \
    FEEDBACK_GOOD_MOVE="${FEEDBACK_GOOD_MOVE:-teacher}" \
    bash tools/run_pv_sibling_strong_teacher_feedback.sh \
    > "$run_dir/pipeline_stdout.log" 2>&1
  status=$?
  set -e
  echo "seed=$seed exit_code=$status" | tee "$run_dir/exit"

  if [[ "$KEEP_SCREEN_BINARY" != "1" ]]; then
    rm -f "$run_dir/best.raw.binary" "$run_dir/candidate.raw.binary"
    if [[ -f "$run_dir/final_binary_status.txt" ]]; then
      printf 'kept_best_raw=0\nscreen_binary_deleted=1\n' > "$run_dir/final_binary_status.txt"
    fi
  fi
done

python3 tools/summarize_mmto_runs.py "${run_dirs[@]}" \
  --json-output "$SCREEN_DIR/summary.json" \
  | tee "$SCREEN_DIR/summary.md"

echo "SCREEN_DIR=$SCREEN_DIR"

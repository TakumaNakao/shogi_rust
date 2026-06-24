#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script. Run it as: BASE_RUN_DIR=... bash tools/run_mmto_hard_stage.sh"
  return 2
fi

cd "$(dirname "$0")/.."

BASE_RUN_DIR="${BASE_RUN_DIR:-}"
if [[ -z "$BASE_RUN_DIR" ]]; then
  echo "BASE_RUN_DIR is required. Example: BASE_RUN_DIR=data/mmto/runs/mmto_rerank_long_... bash tools/run_mmto_hard_stage.sh" >&2
  exit 1
fi

BASELINE_WEIGHTS="${BASELINE_WEIGHTS:-policy_weights_v2.1.0.binary}"
STUDENT_WEIGHTS="${STUDENT_WEIGHTS:-$BASE_RUN_DIR/best.raw.binary}"
HARD_POSITIONS="${HARD_POSITIONS:-$BASE_RUN_DIR/hard_positions.sfen}"

if [[ ! -f "$STUDENT_WEIGHTS" ]]; then
  echo "missing student weights: $STUDENT_WEIGHTS" >&2
  exit 1
fi

if [[ ! -s "$HARD_POSITIONS" ]]; then
  echo "missing or empty hard positions: $HARD_POSITIONS" >&2
  exit 1
fi

export WEIGHTS="$STUDENT_WEIGHTS"
export TEACHER_WEIGHTS="$BASELINE_WEIGHTS"
export POSITIONS="$HARD_POSITIONS"
export MAX_POSITIONS="${MAX_POSITIONS:-1000}"
export VALID_PERCENT="${VALID_PERCENT:-20}"
export JOBS="${JOBS:-$(nproc)}"

export TEACHER_DEPTH="${TEACHER_DEPTH:-6}"
export STUDENT_DEPTH="${STUDENT_DEPTH:-4}"
export RERANK_TEACHER_DEPTH="${RERANK_TEACHER_DEPTH:-6}"
export RERANK_MAX_POSITIONS="${RERANK_MAX_POSITIONS:-1000}"

export TEACHER_SCORE_TOP="${TEACHER_SCORE_TOP:-48}"
export CANDIDATE_TOP="${CANDIDATE_TOP:-48}"
export SCORE_ALL_LEGAL_FOR_VALID="${SCORE_ALL_LEGAL_FOR_VALID:-1}"

export EPOCHS="${EPOCHS:-3}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export LEARNING_RATE="${LEARNING_RATE:-0.00006}"
export TEACHER_TOP_K="${TEACHER_TOP_K:-3}"
export STUDENT_BAD_TOP_K="${STUDENT_BAD_TOP_K:-16}"
export BAD_CANDIDATE_SCOPE="${BAD_CANDIDATE_SCOPE:-model-top}"
export MIN_REGRET_CP="${MIN_REGRET_CP:-20}"
export MAX_PAIRS_PER_SAMPLE="${MAX_PAIRS_PER_SAMPLE:-64}"
export MAX_WEIGHT_DELTA="${MAX_WEIGHT_DELTA:-0.02}"
export ANCHOR_L2="${ANCHOR_L2:-0.0003}"

export BLEND_RATIOS="${BLEND_RATIOS:-0.01 0.02 0.05}"
export KEEP_CANDIDATE_RAW="${KEEP_CANDIDATE_RAW:-0}"
export MIN_FREE_GB="${MIN_FREE_GB:-10}"
export RUN_DIR="${RUN_DIR:-data/mmto/runs/mmto_hard_stage_$(date -u +%Y%m%d_%H%M%S)}"

echo "Starting MMTO hard-position DAgger stage."
echo "BASE_RUN_DIR=$BASE_RUN_DIR"
echo "STUDENT_WEIGHTS=$STUDENT_WEIGHTS"
echo "TEACHER_WEIGHTS=$TEACHER_WEIGHTS"
echo "HARD_POSITIONS=$HARD_POSITIONS"
echo "RUN_DIR=$RUN_DIR"

bash tools/run_mmto_rerank_pipeline.sh

#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script. Run it as: bash tools/run_mmto_rerank_long.sh"
  return 2
fi

cd "$(dirname "$0")/.."

export POSITIONS="${POSITIONS:-converted_records2016_10818.sfen}"
export MAX_POSITIONS="${MAX_POSITIONS:-10000}"
export VALID_PERCENT="${VALID_PERCENT:-10}"
export JOBS="${JOBS:-$(nproc)}"

export TEACHER_DEPTH="${TEACHER_DEPTH:-5}"
export STUDENT_DEPTH="${STUDENT_DEPTH:-4}"
export RERANK_TEACHER_DEPTH="${RERANK_TEACHER_DEPTH:-5}"
export RERANK_MAX_POSITIONS="${RERANK_MAX_POSITIONS:-1000}"

export TEACHER_SCORE_TOP="${TEACHER_SCORE_TOP:-24}"
export CANDIDATE_TOP="${CANDIDATE_TOP:-24}"
export SCORE_ALL_LEGAL_FOR_VALID="${SCORE_ALL_LEGAL_FOR_VALID:-1}"

export EPOCHS="${EPOCHS:-10}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export LEARNING_RATE="${LEARNING_RATE:-0.00012}"
export TEACHER_TOP_K="${TEACHER_TOP_K:-2}"
export STUDENT_BAD_TOP_K="${STUDENT_BAD_TOP_K:-12}"
export BAD_CANDIDATE_SCOPE="${BAD_CANDIDATE_SCOPE:-model-top}"
export MIN_REGRET_CP="${MIN_REGRET_CP:-15}"
export MAX_PAIRS_PER_SAMPLE="${MAX_PAIRS_PER_SAMPLE:-32}"
export MAX_WEIGHT_DELTA="${MAX_WEIGHT_DELTA:-0.03}"
export ANCHOR_L2="${ANCHOR_L2:-0.0002}"

export BLEND_RATIOS="${BLEND_RATIOS:-0.02 0.05}"
export KEEP_CANDIDATE_RAW="${KEEP_CANDIDATE_RAW:-0}"
export MIN_FREE_GB="${MIN_FREE_GB:-10}"
export RUN_DIR="${RUN_DIR:-data/mmto/runs/mmto_rerank_long_$(date -u +%Y%m%d_%H%M%S)}"

echo "Starting long MMTO rerank run."
echo "This is intended for unattended multi-hour learning."
echo "RUN_DIR=$RUN_DIR"

bash tools/run_mmto_rerank_pipeline.sh

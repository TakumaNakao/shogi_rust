#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script. Run it as: bash tools/run_mmto_large_margin_teacher_pairs.sh"
  return 2
fi

cd "$(dirname "$0")/.."

if [[ -z "${SOURCE_RUN_DIR:-}" ]]; then
  for candidate in \
    data/mmto/runs/mmto_rerank_long_20260624_140151 \
    data/mmto/runs/mmto_loss_top_20260625_142715; do
    if [[ -f "$candidate/train.tree.jsonl" && -f "$candidate/valid.tree.jsonl" ]]; then
      SOURCE_RUN_DIR="$candidate"
      break
    fi
  done
fi

if [[ -z "${SOURCE_RUN_DIR:-}" ]]; then
  echo "SOURCE_RUN_DIR is required and no reusable local MMTO dump was found." >&2
  exit 1
fi

export SOURCE_RUN_DIR
export RUN_DIR="${RUN_DIR:-data/mmto/runs/mmto_large_margin_teacher_pairs_$(date -u +%Y%m%d_%H%M%S)}"
export WEIGHTS="${WEIGHTS:-policy_weights_v2.1.0.binary}"
export TEACHER_WEIGHTS="${TEACHER_WEIGHTS:-$WEIGHTS}"

export TRAIN_LINES="${TRAIN_LINES:-9000}"
export VALID_LINES="${VALID_LINES:-1000}"
export EPOCHS="${EPOCHS:-6}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export LEARNING_RATE="${LEARNING_RATE:-0.00008}"
export OPTIMIZER="${OPTIMIZER:-adagrad}"
export ADAGRAD_EPSILON="${ADAGRAD_EPSILON:-1e-6}"

export TEACHER_TOP_K="${TEACHER_TOP_K:-1}"
export STUDENT_BAD_TOP_K="${STUDENT_BAD_TOP_K:-999999}"
export BAD_CANDIDATE_SCOPE="${BAD_CANDIDATE_SCOPE:-all-candidates}"
export MIN_REGRET_CP="${MIN_REGRET_CP:-100}"
export MAX_PAIRS_PER_SAMPLE="${MAX_PAIRS_PER_SAMPLE:-16}"
export PAIR_MINING="${PAIR_MINING:-loss-top}"
export PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-score-gap}"
export PAIR_WEIGHT_SCALE_CP="${PAIR_WEIGHT_SCALE_CP:-150}"
export MAX_PAIR_WEIGHT="${MAX_PAIR_WEIGHT:-3}"

export LOSS_MODE="${LOSS_MODE:-pairwise}"
export BEST_METRIC="${BEST_METRIC:-teacher-mismatch}"
export BEST_GUARD_MAX_REGRET_INCREASE_CP="${BEST_GUARD_MAX_REGRET_INCREASE_CP:-80}"
export BEST_GUARD_BAD100_INCREASE="${BEST_GUARD_BAD100_INCREASE:-0.02}"
export BEST_GUARD_TEACHER_MATCH_DROP_PCT="${BEST_GUARD_TEACHER_MATCH_DROP_PCT:-0.0}"
export ANCHOR_L2="${ANCHOR_L2:-0.0002}"
export MAX_WEIGHT_DELTA="${MAX_WEIGHT_DELTA:-0.003}"

export RERANK_MAX_POSITIONS="${RERANK_MAX_POSITIONS:-1000}"
export RERANK_REQUIRE_MEAN_REGRET_IMPROVEMENT_CP="${RERANK_REQUIRE_MEAN_REGRET_IMPROVEMENT_CP:-0.0}"
export RERANK_REQUIRE_P90_REGRET_IMPROVEMENT_CP="${RERANK_REQUIRE_P90_REGRET_IMPROVEMENT_CP:-0.0}"
export RERANK_REQUIRE_P95_REGRET_IMPROVEMENT_CP="${RERANK_REQUIRE_P95_REGRET_IMPROVEMENT_CP:-0.0}"
export RERANK_REQUIRE_MATCH_RATE_IMPROVEMENT_PCT="${RERANK_REQUIRE_MATCH_RATE_IMPROVEMENT_PCT:-0.0}"
export RERANK_DEDUPE_SFEN="${RERANK_DEDUPE_SFEN:-1}"

export BLEND_RATIOS="${BLEND_RATIOS:-}"
export KEEP_CANDIDATE_RAW="${KEEP_CANDIDATE_RAW:-0}"
export MIN_FREE_GB="${MIN_FREE_GB:-6}"

echo "Starting large-margin teacher-pair MMTO experiment."
echo "SOURCE_RUN_DIR=$SOURCE_RUN_DIR"
echo "RUN_DIR=$RUN_DIR"
echo "TRAIN_LINES=$TRAIN_LINES VALID_LINES=$VALID_LINES EPOCHS=$EPOCHS"
echo "MIN_REGRET_CP=$MIN_REGRET_CP MAX_PAIRS_PER_SAMPLE=$MAX_PAIRS_PER_SAMPLE PAIR_WEIGHT_MODE=$PAIR_WEIGHT_MODE"
echo "BEST_METRIC=$BEST_METRIC OPTIMIZER=$OPTIMIZER"

exec bash tools/run_mmto_from_dump.sh

#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script. Run it as: bash tools/run_bonanza_root_pipeline.sh"
  return 2
fi

cd "$(dirname "$0")/.."

WEIGHTS="${WEIGHTS:-policy_weights_v2.1.0.binary}"
TEACHER_WEIGHTS="${TEACHER_WEIGHTS:-$WEIGHTS}"
INPUTS="${INPUTS:-data/wdoor/extract/2026}"
RUN_DIR="${RUN_DIR:-data/mmto/runs/bonanza_root_$(date -u +%Y%m%d_%H%M%S)}"
SEED="${SEED:-9601}"
MIN_FREE_GB="${MIN_FREE_GB:-6}"

MAX_RECORDS="${MAX_RECORDS:-2000}"
MIN_PLY="${MIN_PLY:-16}"
MAX_PLY="${MAX_PLY:-120}"
MIN_PLAYER_RATE="${MIN_PLAYER_RATE:-4000}"
VALID_PERCENT="${VALID_PERCENT:-10}"
TEST_PERCENT="${TEST_PERCENT:-0}"
DECISIVE_ONLY="${DECISIVE_ONLY:-1}"
WINNER_ONLY="${WINNER_ONLY:-0}"
EXCLUDE_LOSER_AFTER_PLY="${EXCLUDE_LOSER_AFTER_PLY:-100}"
DATASET_MIN_LEGAL_MOVES="${DATASET_MIN_LEGAL_MOVES:-2}"
DATASET_EXCLUDE_IN_CHECK="${DATASET_EXCLUDE_IN_CHECK:-1}"

TREE_MAX_POSITIONS="${TREE_MAX_POSITIONS:-$MAX_RECORDS}"
TREE_VALID_PERCENT="${TREE_VALID_PERCENT:-10}"
POSITION_CHUNK_SIZE="${POSITION_CHUNK_SIZE:-32}"
JOBS="${JOBS:-4}"
TEACHER_DEPTH="${TEACHER_DEPTH:-4}"
STUDENT_DEPTH="${STUDENT_DEPTH:-3}"
TEACHER_SCORE_TOP="${TEACHER_SCORE_TOP:-24}"
CANDIDATE_TOP="${CANDIDATE_TOP:-24}"
SCORE_ALL_LEGAL_FOR_VALID="${SCORE_ALL_LEGAL_FOR_VALID:-1}"
MAX_ABS_ROOT_SCORE="${MAX_ABS_ROOT_SCORE:-3000}"

EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-0.00025}"
TEACHER_TOP_K="${TEACHER_TOP_K:-2}"
STUDENT_BAD_TOP_K="${STUDENT_BAD_TOP_K:-12}"
BAD_CANDIDATE_SCOPE="${BAD_CANDIDATE_SCOPE:-model-top}"
MIN_REGRET_CP="${MIN_REGRET_CP:-15}"
MAX_PAIRS_PER_SAMPLE="${MAX_PAIRS_PER_SAMPLE:-32}"
PAIR_MINING="${PAIR_MINING:-loss-top}"
PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-bad-regret}"
PAIR_WEIGHT_SCALE_CP="${PAIR_WEIGHT_SCALE_CP:-100}"
MAX_PAIR_WEIGHT="${MAX_PAIR_WEIGHT:-3}"
LOSS_MODE="${LOSS_MODE:-listwise-leaf}"
LISTWISE_FEATURE_SOURCE="${LISTWISE_FEATURE_SOURCE:-teacher-leaf}"
LISTWISE_HARD_NEGATIVE_WEIGHT="${LISTWISE_HARD_NEGATIVE_WEIGHT:-0.02}"
LISTWISE_HARD_NEGATIVE_MIN_REGRET_CP="${LISTWISE_HARD_NEGATIVE_MIN_REGRET_CP:-15}"
GAME_TEACHER_MARGIN_WEIGHT="${GAME_TEACHER_MARGIN_WEIGHT:-0.05}"
GAME_TEACHER_MAX_REGRET_CP="${GAME_TEACHER_MAX_REGRET_CP:-150}"
GAME_TEACHER_MIN_BAD_REGRET_CP="${GAME_TEACHER_MIN_BAD_REGRET_CP:-15}"
BEST_METRIC="${BEST_METRIC:-p95-regret}"
ANCHOR_L2="${ANCHOR_L2:-0.0002}"
MAX_WEIGHT_DELTA="${MAX_WEIGHT_DELTA:-0.001}"

RERANK_TEACHER_DEPTH="${RERANK_TEACHER_DEPTH:-5}"
RERANK_MAX_POSITIONS="${RERANK_MAX_POSITIONS:-400}"
RERANK_REQUIRE_MEAN_REGRET_IMPROVEMENT_CP="${RERANK_REQUIRE_MEAN_REGRET_IMPROVEMENT_CP:-0.0}"
RERANK_REQUIRE_P90_REGRET_IMPROVEMENT_CP="${RERANK_REQUIRE_P90_REGRET_IMPROVEMENT_CP:-0.0}"
RERANK_REQUIRE_P95_REGRET_IMPROVEMENT_CP="${RERANK_REQUIRE_P95_REGRET_IMPROVEMENT_CP:-0.0}"
RERANK_REQUIRE_MATCH_RATE_IMPROVEMENT_PCT="${RERANK_REQUIRE_MATCH_RATE_IMPROVEMENT_PCT:-0.0}"
KEEP_CANDIDATE_RAW="${KEEP_CANDIDATE_RAW:-0}"
BLEND_RATIOS="${BLEND_RATIOS:-}"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "missing weights: $WEIGHTS" >&2
  exit 1
fi
if [[ ! -f "$TEACHER_WEIGHTS" ]]; then
  echo "missing teacher weights: $TEACHER_WEIGHTS" >&2
  exit 1
fi

FREE_KB="$(df -Pk . | awk 'NR==2 {print $4}')"
MIN_FREE_KB=$((MIN_FREE_GB * 1024 * 1024))
if (( FREE_KB < MIN_FREE_KB )); then
  echo "free disk is too low: $((FREE_KB / 1024 / 1024))GB < ${MIN_FREE_GB}GB" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"

echo "Starting Bonanza-style root selection pipeline."
echo "RUN_DIR=$RUN_DIR"
echo "INPUTS=$INPUTS"
echo "WEIGHTS=$WEIGHTS"
echo "TEACHER_WEIGHTS=$TEACHER_WEIGHTS"
echo "MAX_RECORDS=$MAX_RECORDS MIN_PLAYER_RATE=$MIN_PLAYER_RATE MIN_PLY=$MIN_PLY MAX_PLY=$MAX_PLY"
echo "TEACHER_DEPTH=$TEACHER_DEPTH STUDENT_DEPTH=$STUDENT_DEPTH TOP=$TEACHER_SCORE_TOP/$CANDIDATE_TOP"
echo "GAME_TEACHER_MARGIN_WEIGHT=$GAME_TEACHER_MARGIN_WEIGHT GAME_TEACHER_MAX_REGRET_CP=$GAME_TEACHER_MAX_REGRET_CP"

env RUST_FONTCONFIG_DLOPEN=1 cargo build --release \
  --bin dataset_build \
  --bin mmto_tree_dump \
  --bin rank_stats \
  --bin mmto_tree_train \
  --bin mmto_score_gate \
  --bin mmto_rerank_gate \
  --bin adjust_weights

dataset_args=(
  --output-dir "$RUN_DIR/dataset"
  --seed "$SEED"
  --valid-percent "$VALID_PERCENT"
  --test-percent "$TEST_PERCENT"
  --max-records "$MAX_RECORDS"
  --min-ply "$MIN_PLY"
  --min-legal-moves "$DATASET_MIN_LEGAL_MOVES"
)

for input in $INPUTS; do
  dataset_args+=(--input "$input")
done
if [[ -n "$MAX_PLY" ]]; then
  dataset_args+=(--max-ply "$MAX_PLY")
fi
if [[ -n "$MIN_PLAYER_RATE" ]]; then
  dataset_args+=(--min-player-rate "$MIN_PLAYER_RATE")
fi
if [[ "$DECISIVE_ONLY" == "1" ]]; then
  dataset_args+=(--decisive-only)
fi
if [[ "$WINNER_ONLY" == "1" ]]; then
  dataset_args+=(--winner-only)
fi
if [[ -n "$EXCLUDE_LOSER_AFTER_PLY" ]]; then
  dataset_args+=(--exclude-loser-after-ply "$EXCLUDE_LOSER_AFTER_PLY")
fi
if [[ "$DATASET_EXCLUDE_IN_CHECK" == "1" ]]; then
  dataset_args+=(--exclude-in-check)
fi

env RUST_FONTCONFIG_DLOPEN=1 target/release/dataset_build "${dataset_args[@]}" \
  | tee "$RUN_DIR/dataset_build_stdout.log"

cat "$RUN_DIR/dataset/train.jsonl" "$RUN_DIR/dataset/valid.jsonl" "$RUN_DIR/dataset/test.jsonl" \
  > "$RUN_DIR/dataset/all.jsonl"
wc -l "$RUN_DIR/dataset/"*.jsonl | tee "$RUN_DIR/dataset_counts.txt"

dump_args=(
  --student-weights "$WEIGHTS"
  --teacher-weights "$TEACHER_WEIGHTS"
  --input "$RUN_DIR/dataset/all.jsonl"
  --train-output "$RUN_DIR/train.tree.jsonl"
  --valid-output "$RUN_DIR/valid.tree.jsonl"
  --teacher-depth "$TEACHER_DEPTH"
  --student-depth "$STUDENT_DEPTH"
  --teacher-score-top "$TEACHER_SCORE_TOP"
  --candidate-top "$CANDIDATE_TOP"
  --position-chunk-size "$POSITION_CHUNK_SIZE"
  --valid-percent "$TREE_VALID_PERCENT"
  --seed "$SEED"
  --jobs "$JOBS"
  --min-legal-moves 2
  --exclude-in-check
  --max-abs-root-score "$MAX_ABS_ROOT_SCORE"
)
if [[ -n "$TREE_MAX_POSITIONS" ]]; then
  dump_args+=(--max-positions "$TREE_MAX_POSITIONS")
fi
if [[ "$SCORE_ALL_LEGAL_FOR_VALID" == "1" ]]; then
  dump_args+=(--score-all-legal-for-valid)
fi

env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_tree_dump "${dump_args[@]}" \
  | tee "$RUN_DIR/dump_stdout.log"

env RUST_FONTCONFIG_DLOPEN=1 target/release/rank_stats \
  --input "$RUN_DIR/train.tree.jsonl" \
  --input "$RUN_DIR/valid.tree.jsonl" \
  --json-output "$RUN_DIR/rank_stats.json" \
  | tee "$RUN_DIR/rank_stats_stdout.log"

env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_tree_train \
  --weights "$WEIGHTS" \
  --train "$RUN_DIR/train.tree.jsonl" \
  --valid "$RUN_DIR/valid.tree.jsonl" \
  --output "$RUN_DIR/candidate.raw.binary" \
  --best-checkpoint-path "$RUN_DIR/best.raw.binary" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --teacher-top-k "$TEACHER_TOP_K" \
  --student-bad-top-k "$STUDENT_BAD_TOP_K" \
  --bad-candidate-scope "$BAD_CANDIDATE_SCOPE" \
  --min-regret-cp "$MIN_REGRET_CP" \
  --max-pairs-per-sample "$MAX_PAIRS_PER_SAMPLE" \
  --pair-mining "$PAIR_MINING" \
  --pair-weight-mode "$PAIR_WEIGHT_MODE" \
  --pair-weight-scale-cp "$PAIR_WEIGHT_SCALE_CP" \
  --max-pair-weight "$MAX_PAIR_WEIGHT" \
  --optimizer adagrad \
  --loss-mode "$LOSS_MODE" \
  --listwise-feature-source "$LISTWISE_FEATURE_SOURCE" \
  --listwise-hard-negative-weight "$LISTWISE_HARD_NEGATIVE_WEIGHT" \
  --listwise-hard-negative-min-regret-cp "$LISTWISE_HARD_NEGATIVE_MIN_REGRET_CP" \
  --game-teacher-margin-weight "$GAME_TEACHER_MARGIN_WEIGHT" \
  --game-teacher-max-regret-cp "$GAME_TEACHER_MAX_REGRET_CP" \
  --game-teacher-min-bad-regret-cp "$GAME_TEACHER_MIN_BAD_REGRET_CP" \
  --margin-cp 50 \
  --softplus-temp-cp 100 \
  --bad-regret-cp 300 \
  --bad-regret-thresholds-cp 50,100,200,300 \
  --best-metric "$BEST_METRIC" \
  --selected-regret-cap-cp 300 \
  --freeze-material \
  --anchor-l2 "$ANCHOR_L2" \
  --max-weight-delta "$MAX_WEIGHT_DELTA" \
  --log-path "$RUN_DIR/train.csv" \
  | tee "$RUN_DIR/train_stdout.log"

BEST_EPOCH="$(
  grep -o 'best_epoch=[0-9]*' "$RUN_DIR/train_stdout.log" \
    | tail -1 \
    | cut -d= -f2
)"

if [[ -z "$BEST_EPOCH" || "$BEST_EPOCH" == "0" ]]; then
  echo "best_epoch=${BEST_EPOCH:-none}: baseline is still best. Rejecting this run."
  rm -f "$RUN_DIR/candidate.raw.binary" "$RUN_DIR/best.raw.binary"
  echo "RUN_DIR=$RUN_DIR"
  exit 0
fi

set +e
env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_score_gate \
  --baseline-weights "$WEIGHTS" \
  --candidate-weights "$RUN_DIR/best.raw.binary" \
  --input "$RUN_DIR/dataset/all.jsonl" \
  --max-positions "$TREE_MAX_POSITIONS" \
  --seed 7201 \
  --p95-limit-cp 50 \
  --max-limit-cp 200 \
  --mean-limit-cp 10 \
  --fail-on-material-drift-cp 5 \
  --json-output "$RUN_DIR/score_gate.json" \
  | tee "$RUN_DIR/score_gate_stdout.log"
score_status="${PIPESTATUS[0]}"
set -e

if [[ "$score_status" != "0" ]]; then
  echo "score gate failed. Rejecting this run."
  if [[ "$KEEP_CANDIDATE_RAW" != "1" ]]; then
    rm -f "$RUN_DIR/candidate.raw.binary" "$RUN_DIR/best.raw.binary"
  fi
  echo "RUN_DIR=$RUN_DIR"
  exit "$score_status"
fi

set +e
env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_rerank_gate \
  --baseline-weights "$WEIGHTS" \
  --candidate-weights "$RUN_DIR/best.raw.binary" \
  --teacher-weights "$TEACHER_WEIGHTS" \
  --input "$RUN_DIR/valid.tree.jsonl" \
  --max-positions "$RERANK_MAX_POSITIONS" \
  --seed 7202 \
  --baseline-depth "$STUDENT_DEPTH" \
  --candidate-depth "$STUDENT_DEPTH" \
  --teacher-depth "$RERANK_TEACHER_DEPTH" \
  --bad-regret-thresholds-cp 50,100,200,300 \
  --require-mean-regret-improvement-cp "$RERANK_REQUIRE_MEAN_REGRET_IMPROVEMENT_CP" \
  --require-p90-regret-improvement-cp "$RERANK_REQUIRE_P90_REGRET_IMPROVEMENT_CP" \
  --require-p95-regret-improvement-cp "$RERANK_REQUIRE_P95_REGRET_IMPROVEMENT_CP" \
  --require-match-rate-improvement-pct "$RERANK_REQUIRE_MATCH_RATE_IMPROVEMENT_PCT" \
  --hard-position-limit 1000 \
  --json-output "$RUN_DIR/rerank_gate.json" \
  | tee "$RUN_DIR/rerank_gate_stdout.log"
rerank_status="${PIPESTATUS[0]}"
set -e

if [[ -f "$RUN_DIR/rerank_gate.json" ]]; then
  python3 - "$RUN_DIR/rerank_gate.json" > "$RUN_DIR/hard_positions.sfen" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
for pos in payload.get("hard_positions", []):
    print(pos["sfen"])
PY
fi

if [[ "$rerank_status" != "0" ]]; then
  echo "rerank gate failed. Hard positions were saved to $RUN_DIR/hard_positions.sfen when available."
  if [[ "$KEEP_CANDIDATE_RAW" != "1" ]]; then
    rm -f "$RUN_DIR/candidate.raw.binary" "$RUN_DIR/best.raw.binary"
  fi
  echo "RUN_DIR=$RUN_DIR"
  exit "$rerank_status"
fi

for R in $BLEND_RATIOS; do
  env RUST_FONTCONFIG_DLOPEN=1 target/release/adjust_weights \
    --input "$WEIGHTS" \
    --blend-target "$RUN_DIR/best.raw.binary" \
    --blend-ratio "$R" \
    --output "$RUN_DIR/blend_${R}.binary"
done

if [[ "$KEEP_CANDIDATE_RAW" != "1" ]]; then
  rm -f "$RUN_DIR/candidate.raw.binary"
fi

echo "offline gates passed"
echo "RUN_DIR=$RUN_DIR"
ls -lh "$RUN_DIR"

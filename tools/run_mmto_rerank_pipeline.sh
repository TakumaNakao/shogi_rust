#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script. Run it as: bash tools/run_mmto_rerank_pipeline.sh"
  return 2
fi

cd "$(dirname "$0")/.."

WEIGHTS="${WEIGHTS:-policy_weights_v2.1.0.binary}"
TEACHER_WEIGHTS="${TEACHER_WEIGHTS:-$WEIGHTS}"
POSITIONS="${POSITIONS:-taya36.sfen}"
MAX_POSITIONS="${MAX_POSITIONS:-3000}"
VALID_PERCENT="${VALID_PERCENT:-10}"
JOBS="${JOBS:-$(nproc)}"
TEACHER_SCORE_TOP="${TEACHER_SCORE_TOP:-16}"
CANDIDATE_TOP="${CANDIDATE_TOP:-16}"
SCORE_ALL_LEGAL_FOR_VALID="${SCORE_ALL_LEGAL_FOR_VALID:-0}"

TEACHER_DEPTH="${TEACHER_DEPTH:-4}"
STUDENT_DEPTH="${STUDENT_DEPTH:-3}"
RERANK_TEACHER_DEPTH="${RERANK_TEACHER_DEPTH:-5}"
RERANK_MAX_POSITIONS="${RERANK_MAX_POSITIONS:-300}"
RERANK_REQUIRE_MEAN_REGRET_IMPROVEMENT_CP="${RERANK_REQUIRE_MEAN_REGRET_IMPROVEMENT_CP:-0.5}"
RERANK_REQUIRE_P90_REGRET_IMPROVEMENT_CP="${RERANK_REQUIRE_P90_REGRET_IMPROVEMENT_CP:-0.0}"
RERANK_REQUIRE_P95_REGRET_IMPROVEMENT_CP="${RERANK_REQUIRE_P95_REGRET_IMPROVEMENT_CP:-0.0}"
RERANK_REQUIRE_MATCH_RATE_IMPROVEMENT_PCT="${RERANK_REQUIRE_MATCH_RATE_IMPROVEMENT_PCT:-0.0}"

EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"
TEACHER_TOP_K="${TEACHER_TOP_K:-2}"
STUDENT_BAD_TOP_K="${STUDENT_BAD_TOP_K:-6}"
BAD_CANDIDATE_SCOPE="${BAD_CANDIDATE_SCOPE:-student-top}"
MIN_REGRET_CP="${MIN_REGRET_CP:-50}"
MAX_PAIRS_PER_SAMPLE="${MAX_PAIRS_PER_SAMPLE:-16}"
PAIR_MINING="${PAIR_MINING:-loss-top}"
PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-bad-regret}"
PAIR_WEIGHT_SCALE_CP="${PAIR_WEIGHT_SCALE_CP:-100}"
MAX_PAIR_WEIGHT="${MAX_PAIR_WEIGHT:-3}"
MAX_WEIGHT_DELTA="${MAX_WEIGHT_DELTA:-0.05}"
ANCHOR_L2="${ANCHOR_L2:-0.0002}"
BLEND_RATIOS="${BLEND_RATIOS:-0.01 0.02 0.05 0.10}"
KEEP_CANDIDATE_RAW="${KEEP_CANDIDATE_RAW:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-6}"

RUN_DIR="${RUN_DIR:-data/mmto/runs/mmto_rerank_$(date -u +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "missing weights: $WEIGHTS" >&2
  exit 1
fi

if [[ ! -f "$TEACHER_WEIGHTS" ]]; then
  echo "missing teacher weights: $TEACHER_WEIGHTS" >&2
  exit 1
fi

if [[ ! -f "$POSITIONS" ]]; then
  echo "missing positions: $POSITIONS" >&2
  exit 1
fi

echo "RUN_DIR=$RUN_DIR"
echo "WEIGHTS=$WEIGHTS"
echo "TEACHER_WEIGHTS=$TEACHER_WEIGHTS"
echo "POSITIONS=$POSITIONS"
echo "MAX_POSITIONS=$MAX_POSITIONS"
echo "TEACHER_DEPTH=$TEACHER_DEPTH STUDENT_DEPTH=$STUDENT_DEPTH"
echo "TEACHER_SCORE_TOP=$TEACHER_SCORE_TOP CANDIDATE_TOP=$CANDIDATE_TOP SCORE_ALL_LEGAL_FOR_VALID=$SCORE_ALL_LEGAL_FOR_VALID"
echo "BAD_CANDIDATE_SCOPE=$BAD_CANDIDATE_SCOPE MIN_REGRET_CP=$MIN_REGRET_CP MAX_PAIRS_PER_SAMPLE=$MAX_PAIRS_PER_SAMPLE"
echo "PAIR_MINING=$PAIR_MINING PAIR_WEIGHT_MODE=$PAIR_WEIGHT_MODE PAIR_WEIGHT_SCALE_CP=$PAIR_WEIGHT_SCALE_CP MAX_PAIR_WEIGHT=$MAX_PAIR_WEIGHT"

FREE_KB="$(df -Pk . | awk 'NR==2 {print $4}')"
MIN_FREE_KB=$((MIN_FREE_GB * 1024 * 1024))
if (( FREE_KB < MIN_FREE_KB )); then
  echo "free disk is too low: $((FREE_KB / 1024 / 1024))GB < ${MIN_FREE_GB}GB" >&2
  echo "Clean old generated runs first: bash tools/clean_mmto_runs.sh" >&2
  exit 1
fi

env RUST_FONTCONFIG_DLOPEN=1 cargo build --release \
  --bin mmto_tree_dump \
  --bin mmto_tree_train \
  --bin mmto_score_gate \
  --bin mmto_rerank_gate \
  --bin adjust_weights

dump_args=(
  --student-weights "$WEIGHTS"
  --teacher-weights "$TEACHER_WEIGHTS"
  --input "$POSITIONS"
  --train-output "$RUN_DIR/train.tree.jsonl"
  --valid-output "$RUN_DIR/valid.tree.jsonl"
  --teacher-depth "$TEACHER_DEPTH"
  --student-depth "$STUDENT_DEPTH"
  --teacher-score-top "$TEACHER_SCORE_TOP"
  --candidate-top "$CANDIDATE_TOP"
  --max-positions "$MAX_POSITIONS"
  --valid-percent "$VALID_PERCENT"
  --seed 7101
  --jobs "$JOBS"
  --min-legal-moves 2
  --exclude-in-check
  --max-abs-root-score 3000
)

if [[ "$SCORE_ALL_LEGAL_FOR_VALID" == "1" ]]; then
  dump_args+=(--score-all-legal-for-valid)
fi

env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_tree_dump \
  "${dump_args[@]}" \
  | tee "$RUN_DIR/dump_stdout.log"

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
  --margin-cp 50 \
  --softplus-temp-cp 100 \
  --bad-regret-cp 300 \
  --bad-regret-thresholds-cp 50,100,200,300 \
  --best-metric p95-regret \
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
  echo "Deleting large candidate weight copies."
  rm -f "$RUN_DIR/candidate.raw.binary" "$RUN_DIR/best.raw.binary"
  echo "RUN_DIR=$RUN_DIR"
  echo "Stop here. Do not run offline gates or benchmarks for this candidate."
  exit 0
fi

env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_score_gate \
  --baseline-weights "$WEIGHTS" \
  --candidate-weights "$RUN_DIR/best.raw.binary" \
  --input "$POSITIONS" \
  --max-positions "$MAX_POSITIONS" \
  --seed 7201 \
  --p95-limit-cp 50 \
  --max-limit-cp 200 \
  --mean-limit-cp 10 \
  --fail-on-material-drift-cp 5 \
  --json-output "$RUN_DIR/score_gate.json" \
  | tee "$RUN_DIR/score_gate_stdout.log"

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
    rm -f "$RUN_DIR/candidate.raw.binary"
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

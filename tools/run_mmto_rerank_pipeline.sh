#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script. Run it as: bash tools/run_mmto_rerank_pipeline.sh"
  return 2
fi

cd "$(dirname "$0")/.."

WEIGHTS="${WEIGHTS:-policy_weights_v2.1.0.binary}"
POSITIONS="${POSITIONS:-taya36.sfen}"
MAX_POSITIONS="${MAX_POSITIONS:-3000}"
VALID_PERCENT="${VALID_PERCENT:-10}"
JOBS="${JOBS:-$(nproc)}"

TEACHER_DEPTH="${TEACHER_DEPTH:-4}"
STUDENT_DEPTH="${STUDENT_DEPTH:-3}"
RERANK_TEACHER_DEPTH="${RERANK_TEACHER_DEPTH:-5}"
RERANK_MAX_POSITIONS="${RERANK_MAX_POSITIONS:-300}"

EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"
MIN_REGRET_CP="${MIN_REGRET_CP:-50}"
MAX_WEIGHT_DELTA="${MAX_WEIGHT_DELTA:-0.05}"
ANCHOR_L2="${ANCHOR_L2:-0.0002}"

RUN_DIR="${RUN_DIR:-data/mmto/runs/mmto_rerank_$(date -u +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "missing weights: $WEIGHTS" >&2
  exit 1
fi

if [[ ! -f "$POSITIONS" ]]; then
  echo "missing positions: $POSITIONS" >&2
  exit 1
fi

echo "RUN_DIR=$RUN_DIR"
echo "WEIGHTS=$WEIGHTS"
echo "POSITIONS=$POSITIONS"

env RUST_FONTCONFIG_DLOPEN=1 cargo build --release \
  --bin mmto_tree_dump \
  --bin mmto_tree_train \
  --bin mmto_score_gate \
  --bin mmto_rerank_gate \
  --bin adjust_weights

env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_tree_dump \
  --student-weights "$WEIGHTS" \
  --teacher-weights "$WEIGHTS" \
  --input "$POSITIONS" \
  --train-output "$RUN_DIR/train.tree.jsonl" \
  --valid-output "$RUN_DIR/valid.tree.jsonl" \
  --teacher-depth "$TEACHER_DEPTH" \
  --student-depth "$STUDENT_DEPTH" \
  --teacher-score-top 16 \
  --candidate-top 16 \
  --max-positions "$MAX_POSITIONS" \
  --valid-percent "$VALID_PERCENT" \
  --seed 7101 \
  --jobs "$JOBS" \
  --min-legal-moves 2 \
  --exclude-in-check \
  --max-abs-root-score 3000 \
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
  --teacher-top-k 2 \
  --student-bad-top-k 6 \
  --min-regret-cp "$MIN_REGRET_CP" \
  --max-pairs-per-sample 16 \
  --optimizer adagrad \
  --margin-cp 50 \
  --softplus-temp-cp 100 \
  --bad-regret-cp 300 \
  --bad-regret-thresholds-cp 50,100,200,300 \
  --best-metric selected-regret \
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

env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_rerank_gate \
  --baseline-weights "$WEIGHTS" \
  --candidate-weights "$RUN_DIR/best.raw.binary" \
  --teacher-weights "$WEIGHTS" \
  --input "$RUN_DIR/valid.tree.jsonl" \
  --max-positions "$RERANK_MAX_POSITIONS" \
  --seed 7202 \
  --baseline-depth "$STUDENT_DEPTH" \
  --candidate-depth "$STUDENT_DEPTH" \
  --teacher-depth "$RERANK_TEACHER_DEPTH" \
  --bad-regret-thresholds-cp 50,100,200,300 \
  --allow-mean-regret-increase-cp 0 \
  --allow-p90-regret-increase-cp 0 \
  --allow-p95-regret-increase-cp 0 \
  --allow-bad-ratio-increase 0 \
  --hard-position-limit 1000 \
  --json-output "$RUN_DIR/rerank_gate.json" \
  | tee "$RUN_DIR/rerank_gate_stdout.log"

python3 - "$RUN_DIR/rerank_gate.json" > "$RUN_DIR/hard_positions.sfen" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
for pos in payload.get("hard_positions", []):
    print(pos["sfen"])
PY

for R in 0.01 0.02 0.05 0.10; do
  env RUST_FONTCONFIG_DLOPEN=1 target/release/adjust_weights \
    --input "$WEIGHTS" \
    --blend-target "$RUN_DIR/best.raw.binary" \
    --blend-ratio "$R" \
    --output "$RUN_DIR/blend_${R}.binary"
done

echo "offline gates passed"
echo "RUN_DIR=$RUN_DIR"
ls -lh "$RUN_DIR"

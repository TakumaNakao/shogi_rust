#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script. Run it as: bash tools/run_pv_sibling_listwise_from_dump.sh"
  return 2
fi

cd "$(dirname "$0")/.."

SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-data/mmto/runs/pv_sibling_feedback_20260628_190857}"
SOURCE_TRAIN="${SOURCE_TRAIN:-$SOURCE_RUN_DIR/train.pv.tree.jsonl}"
SOURCE_VALID="${SOURCE_VALID:-$SOURCE_RUN_DIR/valid.pv.tree.jsonl}"

WEIGHTS="${WEIGHTS:-policy_weights_v2.1.0.binary}"
TEACHER_WEIGHTS="${TEACHER_WEIGHTS:-$WEIGHTS}"
RUN_DIR="${RUN_DIR:-data/mmto/runs/pv_sibling_listwise_$(date -u +%Y%m%d_%H%M%S)}"
MIN_FREE_GB="${MIN_FREE_GB:-4}"

VALID_LINES="${VALID_LINES:-1000}"
EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-0.00008}"
MAX_WEIGHT_DELTA="${MAX_WEIGHT_DELTA:-0.02}"
ANCHOR_L2="${ANCHOR_L2:-0.0003}"

LISTWISE_TEACHER_TOP_K="${LISTWISE_TEACHER_TOP_K:-16}"
LISTWISE_CANDIDATE_TOP_K="${LISTWISE_CANDIDATE_TOP_K:-16}"
LISTWISE_MIN_SELECTED_REGRET_CP="${LISTWISE_MIN_SELECTED_REGRET_CP:-30}"
LISTWISE_MAX_TEACHER_SCORE_ABS_CP="${LISTWISE_MAX_TEACHER_SCORE_ABS_CP:-10000}"
LISTWISE_REGRET_CAP_CP="${LISTWISE_REGRET_CAP_CP:-600}"
LISTWISE_WEIGHT_MODE="${LISTWISE_WEIGHT_MODE:-model-regret}"
LISTWISE_WEIGHT_SCALE_CP="${LISTWISE_WEIGHT_SCALE_CP:-100}"
LISTWISE_MAX_SAMPLE_WEIGHT="${LISTWISE_MAX_SAMPLE_WEIGHT:-3}"
TEACHER_TEMPERATURE_CP="${TEACHER_TEMPERATURE_CP:-80}"
MODEL_TEMPERATURE_CP="${MODEL_TEMPERATURE_CP:-80}"
TEACHER_TOP_CE_WEIGHT="${TEACHER_TOP_CE_WEIGHT:-0.2}"
CURRENT_TOP_MARGIN_WEIGHT="${CURRENT_TOP_MARGIN_WEIGHT:-0.1}"
CURRENT_TOP_MIN_BAD_REGRET_CP="${CURRENT_TOP_MIN_BAD_REGRET_CP:-30}"
BEST_METRIC="${BEST_METRIC:-valid-loss}"

STUDENT_DEPTH="${STUDENT_DEPTH:-2}"
TEACHER_DEPTH="${TEACHER_DEPTH:-3}"
RERANK_MAX_POSITIONS="${RERANK_MAX_POSITIONS:-1000}"
JOBS="${JOBS:-4}"

RUN_BENCH="${RUN_BENCH:-1}"
BENCH_POSITIONS="${BENCH_POSITIONS:-taya36.sfen}"
BENCH_GAMES="${BENCH_GAMES:-20}"
BENCH_DEPTH="${BENCH_DEPTH:-5}"
BENCH_TIME_LIMIT_MS="${BENCH_TIME_LIMIT_MS:-100}"
BENCH_MAX_PLIES="${BENCH_MAX_PLIES:-200}"
BENCH_JOBS="${BENCH_JOBS:-2}"
BENCH_SEED="${BENCH_SEED:-36001}"
KEEP_MIN_NEW_WINS="${KEEP_MIN_NEW_WINS:-12}"

for path in "$SOURCE_TRAIN" "$SOURCE_VALID" "$WEIGHTS" "$TEACHER_WEIGHTS"; do
  if [[ ! -f "$path" ]]; then
    echo "missing required file: $path" >&2
    exit 1
  fi
done

FREE_KB="$(df -Pk . | awk 'NR==2 {print $4}')"
MIN_FREE_KB=$((MIN_FREE_GB * 1024 * 1024))
if (( FREE_KB < MIN_FREE_KB )); then
  echo "free disk is too low: $((FREE_KB / 1024 / 1024))GB < ${MIN_FREE_GB}GB" >&2
  echo "Clean old generated runs first: bash tools/clean_mmto_runs.sh" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"

echo "Starting PV sibling hard-node listwise run."
echo "RUN_DIR=$RUN_DIR"
echo "SOURCE_TRAIN=$SOURCE_TRAIN"
echo "SOURCE_VALID=$SOURCE_VALID"
echo "WEIGHTS=$WEIGHTS"
echo "LISTWISE_TEACHER_TOP_K=$LISTWISE_TEACHER_TOP_K"
echo "LISTWISE_CANDIDATE_TOP_K=$LISTWISE_CANDIDATE_TOP_K"
echo "LISTWISE_MIN_SELECTED_REGRET_CP=$LISTWISE_MIN_SELECTED_REGRET_CP"
echo "LISTWISE_WEIGHT_MODE=$LISTWISE_WEIGHT_MODE"
echo "BEST_METRIC=$BEST_METRIC"
echo "LEARNING_RATE=$LEARNING_RATE MAX_WEIGHT_DELTA=$MAX_WEIGHT_DELTA"

env RUST_FONTCONFIG_DLOPEN=1 cargo build --release \
  --bin mmto_tree_train \
  --bin mmto_score_gate \
  --bin mmto_rerank_gate \
  --bin usi_engine \
  --bin usi_benchmark \
  --bin record_analyze

head -n "$VALID_LINES" "$SOURCE_VALID" > "$RUN_DIR/valid.top.tree.jsonl"
wc -l "$SOURCE_TRAIN" "$RUN_DIR/valid.top.tree.jsonl" | tee "$RUN_DIR/subset_counts.txt"

env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_tree_train \
  --weights "$WEIGHTS" \
  --train "$SOURCE_TRAIN" \
  --valid "$RUN_DIR/valid.top.tree.jsonl" \
  --output "$RUN_DIR/candidate.raw.binary" \
  --best-checkpoint-path "$RUN_DIR/best.raw.binary" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --max-weight-delta "$MAX_WEIGHT_DELTA" \
  --anchor-l2 "$ANCHOR_L2" \
  --loss-mode listwise-leaf \
  --listwise-feature-source teacher-leaf \
  --listwise-teacher-top-k "$LISTWISE_TEACHER_TOP_K" \
  --listwise-candidate-top-k "$LISTWISE_CANDIDATE_TOP_K" \
  --listwise-min-selected-regret-cp "$LISTWISE_MIN_SELECTED_REGRET_CP" \
  --listwise-max-teacher-score-abs-cp "$LISTWISE_MAX_TEACHER_SCORE_ABS_CP" \
  --listwise-regret-cap-cp "$LISTWISE_REGRET_CAP_CP" \
  --listwise-weight-mode "$LISTWISE_WEIGHT_MODE" \
  --listwise-weight-scale-cp "$LISTWISE_WEIGHT_SCALE_CP" \
  --listwise-max-sample-weight "$LISTWISE_MAX_SAMPLE_WEIGHT" \
  --teacher-temperature-cp "$TEACHER_TEMPERATURE_CP" \
  --model-temperature-cp "$MODEL_TEMPERATURE_CP" \
  --teacher-top-ce-weight "$TEACHER_TOP_CE_WEIGHT" \
  --current-top-margin-weight "$CURRENT_TOP_MARGIN_WEIGHT" \
  --current-top-min-bad-regret-cp "$CURRENT_TOP_MIN_BAD_REGRET_CP" \
  --best-metric "$BEST_METRIC" \
  --optimizer adagrad \
  --separate-aux-adagrad \
  --freeze-material \
  --selected-regret-cap-cp 300 \
  --bad-regret-cp 300 \
  --bad-regret-thresholds-cp 50,100,200,300 \
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
  echo "kept_best_raw=0" > "$RUN_DIR/final_binary_status.txt"
  echo "RUN_DIR=$RUN_DIR"
  exit 0
fi

rm -f "$RUN_DIR/candidate.raw.binary"

python3 - "$RUN_DIR/valid.top.tree.jsonl" "$RUN_DIR/score_positions.sfen" <<'PY'
import json
import sys

input_path, output_path = sys.argv[1:3]
seen = set()
with open(input_path, encoding="utf-8") as handle, open(output_path, "w", encoding="utf-8") as out:
    for line in handle:
        if not line.strip():
            continue
        sfen = json.loads(line)["sfen"].strip()
        if sfen and sfen not in seen:
            seen.add(sfen)
            out.write(sfen + "\n")
print(f"score_positions={len(seen)}")
PY

set +e
env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_score_gate \
  --baseline-weights "$WEIGHTS" \
  --candidate-weights "$RUN_DIR/best.raw.binary" \
  --input "$RUN_DIR/score_positions.sfen" \
  --max-positions 1000 \
  --seed 7301 \
  --p95-limit-cp 50 \
  --max-limit-cp 200 \
  --mean-limit-cp 10 \
  --json-output "$RUN_DIR/score_gate.json" \
  | tee "$RUN_DIR/score_gate_stdout.log"
score_status="${PIPESTATUS[0]}"
set -e

if [[ "$score_status" != "0" ]]; then
  echo "score gate failed. Rejecting this run."
  rm -f "$RUN_DIR/best.raw.binary"
  echo "kept_best_raw=0" > "$RUN_DIR/final_binary_status.txt"
  echo "RUN_DIR=$RUN_DIR"
  exit "$score_status"
fi

set +e
env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_rerank_gate \
  --baseline-weights "$WEIGHTS" \
  --candidate-weights "$RUN_DIR/best.raw.binary" \
  --teacher-weights "$TEACHER_WEIGHTS" \
  --input "$SOURCE_VALID" \
  --max-positions "$RERANK_MAX_POSITIONS" \
  --baseline-depth "$STUDENT_DEPTH" \
  --candidate-depth "$STUDENT_DEPTH" \
  --teacher-depth "$TEACHER_DEPTH" \
  --seed 7301 \
  --jobs "$JOBS" \
  --bad-regret-thresholds-cp 50,100,200,300 \
  --allow-mean-regret-increase-cp 0 \
  --allow-p90-regret-increase-cp 0 \
  --allow-p95-regret-increase-cp 0 \
  --allow-bad-ratio-increase 0 \
  --allow-phase-mean-regret-increase-cp=-1 \
  --allow-phase-p90-regret-increase-cp=-1 \
  --allow-phase-p95-regret-increase-cp=-1 \
  --allow-phase-bad-ratio-increase=-1 \
  --hard-position-limit 1000 \
  --print-worst 20 \
  --json-output "$RUN_DIR/rerank_gate.json" \
  | tee "$RUN_DIR/rerank_gate_stdout.log"
rerank_status="${PIPESTATUS[0]}"
set -e

if [[ -f "$RUN_DIR/rerank_gate.json" ]]; then
  python3 - "$RUN_DIR/rerank_gate.json" > "$RUN_DIR/hard_positions.sfen" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
for pos in payload.get("hard_positions", []):
    print(pos["sfen"])
PY
fi

if [[ "$rerank_status" != "0" ]]; then
  echo "rerank gate failed. Rejecting this run."
  rm -f "$RUN_DIR/best.raw.binary"
  echo "kept_best_raw=0" > "$RUN_DIR/final_binary_status.txt"
  echo "RUN_DIR=$RUN_DIR"
  exit "$rerank_status"
fi

if [[ "$RUN_BENCH" == "1" ]]; then
  BENCH_DIR="$RUN_DIR/bench${BENCH_GAMES}"
  env RUST_FONTCONFIG_DLOPEN=1 target/release/usi_benchmark \
    --new-engine target/release/usi_engine \
    --baseline-engine target/release/usi_engine \
    --new-weights "$RUN_DIR/best.raw.binary" \
    --baseline-weights "$WEIGHTS" \
    --positions "$BENCH_POSITIONS" \
    --games "$BENCH_GAMES" \
    --depth "$BENCH_DEPTH" \
    --time-limit-ms "$BENCH_TIME_LIMIT_MS" \
    --max-plies "$BENCH_MAX_PLIES" \
    --adjudicate-at-max-plies \
    --jobs "$BENCH_JOBS" \
    --seed "$BENCH_SEED" \
    --record-dir "$BENCH_DIR" \
    | tee "$RUN_DIR/bench_stdout.log"

  env RUST_FONTCONFIG_DLOPEN=1 target/release/record_analyze \
    --record-dir "$BENCH_DIR" \
    --weights "$WEIGHTS" \
    > "$RUN_DIR/record_analyze.log"

  NEW_WINS="$(awk '/^new wins:/ {print $3}' "$RUN_DIR/bench_stdout.log" | tail -1)"
  NEW_WINS="${NEW_WINS:-0}"
  if (( NEW_WINS >= KEEP_MIN_NEW_WINS )); then
    echo "kept_best_raw=1" > "$RUN_DIR/final_binary_status.txt"
  else
    rm -f "$RUN_DIR/best.raw.binary"
    echo "kept_best_raw=0" > "$RUN_DIR/final_binary_status.txt"
  fi
else
  echo "kept_best_raw=1" > "$RUN_DIR/final_binary_status.txt"
fi

echo "PV sibling hard-node listwise run finished."
echo "RUN_DIR=$RUN_DIR"
cat "$RUN_DIR/final_binary_status.txt"
find "$RUN_DIR" -maxdepth 1 -type f -name '*.binary' -printf '%s %p\n'

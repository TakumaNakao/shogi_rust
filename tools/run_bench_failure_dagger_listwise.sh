#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script. Run it as: bash tools/run_bench_failure_dagger_listwise.sh"
  return 2
fi

cd "$(dirname "$0")/.."

WEIGHTS="${WEIGHTS:-policy_weights_v2.1.0.binary}"
TEACHER_WEIGHTS="${TEACHER_WEIGHTS:-$WEIGHTS}"
STUDENT_WEIGHTS_FOR_DUMP="${STUDENT_WEIGHTS_FOR_DUMP:-$WEIGHTS}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-data/mmto/runs/pv_sibling_feedback_20260628_190857}"
SOURCE_TRAIN="${SOURCE_TRAIN:-$SOURCE_RUN_DIR/train.pv.tree.jsonl}"
SOURCE_VALID="${SOURCE_VALID:-$SOURCE_RUN_DIR/valid.pv.tree.jsonl}"
BENCH_FAILURE_JSONL="${BENCH_FAILURE_JSONL:-data/mmto/runs/protected_feedback_phasefix_40gate_fb050_i05_20260628_233749/bench_failure.jsonl}"
RUN_DIR="${RUN_DIR:-data/mmto/runs/bench_failure_dagger_listwise_$(date -u +%Y%m%d_%H%M%S)}"
MIN_FREE_GB="${MIN_FREE_GB:-6}"

PAIR_BAD_MOVE_SOURCE="${PAIR_BAD_MOVE_SOURCE:-timed-or-actual}"
PAIR_MIN_REGRET_CP="${PAIR_MIN_REGRET_CP:-150}"
PAIR_MAX_REGRET_CP="${PAIR_MAX_REGRET_CP:-100000}"
PAIR_LIMIT="${PAIR_LIMIT:-300}"
PAIR_EXCLUDE_IN_CHECK="${PAIR_EXCLUDE_IN_CHECK:-0}"
PAIR_REQUIRE_NEW_LOSS="${PAIR_REQUIRE_NEW_LOSS:-0}"
PAIR_REQUIRE_BAD_FLAG="${PAIR_REQUIRE_BAD_FLAG:-0}"
PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-combined}"
PAIR_WEIGHT_SCALE_CP="${PAIR_WEIGHT_SCALE_CP:-100}"
PAIR_MAX_SAMPLE_WEIGHT="${PAIR_MAX_SAMPLE_WEIGHT:-5}"

DUMP_TEACHER_DEPTH="${DUMP_TEACHER_DEPTH:-5}"
DUMP_STUDENT_DEPTH="${DUMP_STUDENT_DEPTH:-4}"
DUMP_TEACHER_SCORE_TOP="${DUMP_TEACHER_SCORE_TOP:-24}"
DUMP_CANDIDATE_TOP="${DUMP_CANDIDATE_TOP:-24}"
DUMP_POSITION_CHUNK_SIZE="${DUMP_POSITION_CHUNK_SIZE:-16}"
DUMP_JOBS="${DUMP_JOBS:-2}"
DUMP_MAX_POSITIONS="${DUMP_MAX_POSITIONS:-300}"
DUMP_MAX_ABS_ROOT_SCORE="${DUMP_MAX_ABS_ROOT_SCORE:-30000}"
DUMP_EXCLUDE_IN_CHECK="${DUMP_EXCLUDE_IN_CHECK:-0}"

VALID_LINES="${VALID_LINES:-1000}"
EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-0.0008}"
MAX_WEIGHT_DELTA="${MAX_WEIGHT_DELTA:-0.005}"
ANCHOR_L2="${ANCHOR_L2:-0.0002}"
STREAM_TRAIN="${STREAM_TRAIN:-1}"
STREAM_TRAIN_EVAL_MAX_SAMPLES="${STREAM_TRAIN_EVAL_MAX_SAMPLES:-3000}"
REPLAY_WEIGHT="${REPLAY_WEIGHT:-0.25}"
EXTRA_VALID_BEST_WEIGHT="${EXTRA_VALID_BEST_WEIGHT:-1.0}"

LISTWISE_FEATURE_SOURCE="${LISTWISE_FEATURE_SOURCE:-student-leaf}"
LISTWISE_TEACHER_TOP_K="${LISTWISE_TEACHER_TOP_K:-16}"
LISTWISE_CANDIDATE_TOP_K="${LISTWISE_CANDIDATE_TOP_K:-16}"
LISTWISE_MIN_SELECTED_REGRET_CP="${LISTWISE_MIN_SELECTED_REGRET_CP:-30}"
LISTWISE_MAX_TEACHER_SCORE_ABS_CP="${LISTWISE_MAX_TEACHER_SCORE_ABS_CP:-30000}"
LISTWISE_REGRET_CAP_CP="${LISTWISE_REGRET_CAP_CP:-600}"
LISTWISE_WEIGHT_MODE="${LISTWISE_WEIGHT_MODE:-model-regret}"
LISTWISE_WEIGHT_SCALE_CP="${LISTWISE_WEIGHT_SCALE_CP:-100}"
LISTWISE_MAX_SAMPLE_WEIGHT="${LISTWISE_MAX_SAMPLE_WEIGHT:-3}"
LISTWISE_HARD_NEGATIVE_WEIGHT="${LISTWISE_HARD_NEGATIVE_WEIGHT:-0.02}"
LISTWISE_HARD_NEGATIVE_MIN_REGRET_CP="${LISTWISE_HARD_NEGATIVE_MIN_REGRET_CP:-50}"
TEACHER_TOP_CE_WEIGHT="${TEACHER_TOP_CE_WEIGHT:-0.05}"
CURRENT_TOP_MARGIN_WEIGHT="${CURRENT_TOP_MARGIN_WEIGHT:-0.05}"
CURRENT_TOP_MIN_BAD_REGRET_CP="${CURRENT_TOP_MIN_BAD_REGRET_CP:-30}"
GAME_TEACHER_MARGIN_WEIGHT="${GAME_TEACHER_MARGIN_WEIGHT:-0.05}"
GAME_TEACHER_MAX_REGRET_CP="${GAME_TEACHER_MAX_REGRET_CP:-300}"
GAME_TEACHER_MIN_BAD_REGRET_CP="${GAME_TEACHER_MIN_BAD_REGRET_CP:-15}"
POLICY_ANCHOR_WEIGHTS="${POLICY_ANCHOR_WEIGHTS:-}"
POLICY_ANCHOR_WEIGHT="${POLICY_ANCHOR_WEIGHT:-0}"
POLICY_ANCHOR_TEMPERATURE_CP="${POLICY_ANCHOR_TEMPERATURE_CP:-100}"
POLICY_ANCHOR_FEATURE_SOURCE="${POLICY_ANCHOR_FEATURE_SOURCE:-student-leaf}"
POLICY_ANCHOR_MARGIN_WEIGHT="${POLICY_ANCHOR_MARGIN_WEIGHT:-0}"
POLICY_ANCHOR_MARGIN_CP="${POLICY_ANCHOR_MARGIN_CP:-50}"
POLICY_ANCHOR_MARGIN_SOFTPLUS_TEMP_CP="${POLICY_ANCHOR_MARGIN_SOFTPLUS_TEMP_CP:-100}"

BEST_METRIC="${BEST_METRIC:-bad100-regret}"
BEST_GUARD_MAX_REGRET_INCREASE_CP="${BEST_GUARD_MAX_REGRET_INCREASE_CP:--1}"
BEST_GUARD_BAD100_INCREASE="${BEST_GUARD_BAD100_INCREASE:-0}"
BEST_GUARD_TEACHER_MATCH_DROP_PCT="${BEST_GUARD_TEACHER_MATCH_DROP_PCT:-0}"

RERANK_MAX_POSITIONS="${RERANK_MAX_POSITIONS:-800}"
RERANK_STUDENT_DEPTH="${RERANK_STUDENT_DEPTH:-3}"
RERANK_TEACHER_DEPTH="${RERANK_TEACHER_DEPTH:-4}"
RERANK_JOBS="${RERANK_JOBS:-4}"

RUN_BENCH="${RUN_BENCH:-1}"
BENCH_POSITIONS="${BENCH_POSITIONS:-taya36.sfen}"
BENCH_GAMES="${BENCH_GAMES:-20}"
BENCH_DEPTH="${BENCH_DEPTH:-5}"
BENCH_TIME_LIMIT_MS="${BENCH_TIME_LIMIT_MS:-100}"
BENCH_MAX_PLIES="${BENCH_MAX_PLIES:-200}"
BENCH_JOBS="${BENCH_JOBS:-2}"
BENCH_SEED="${BENCH_SEED:-37801}"
KEEP_MIN_NEW_WINS="${KEEP_MIN_NEW_WINS:-12}"

for path in "$WEIGHTS" "$TEACHER_WEIGHTS" "$STUDENT_WEIGHTS_FOR_DUMP" "$SOURCE_TRAIN" "$SOURCE_VALID"; do
  if [[ ! -f "$path" ]]; then
    echo "missing required file: $path" >&2
    exit 1
  fi
done
for path in $BENCH_FAILURE_JSONL; do
  if [[ ! -f "$path" ]]; then
    echo "missing bench failure jsonl: $path" >&2
    exit 1
  fi
done

FREE_KB="$(df -Pk . | awk 'NR==2 {print $4}')"
MIN_FREE_KB=$((MIN_FREE_GB * 1024 * 1024))
if (( FREE_KB < MIN_FREE_KB )); then
  echo "free disk is too low: $((FREE_KB / 1024 / 1024))GB < ${MIN_FREE_GB}GB" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"

echo "Starting bench-failure DAgger listwise run."
echo "RUN_DIR=$RUN_DIR"
echo "SOURCE_TRAIN=$SOURCE_TRAIN"
echo "SOURCE_VALID=$SOURCE_VALID"
echo "BENCH_FAILURE_JSONL=$BENCH_FAILURE_JSONL"
echo "PAIR_REGRET_RANGE=$PAIR_MIN_REGRET_CP..$PAIR_MAX_REGRET_CP LIMIT=$PAIR_LIMIT BAD_MOVE_SOURCE=$PAIR_BAD_MOVE_SOURCE"
echo "DUMP_DEPTHS teacher=$DUMP_TEACHER_DEPTH student=$DUMP_STUDENT_DEPTH max_positions=$DUMP_MAX_POSITIONS"
echo "LOSS=listwise-leaf feature=$LISTWISE_FEATURE_SOURCE replay_weight=$REPLAY_WEIGHT best_metric=$BEST_METRIC"
echo "POLICY_ANCHOR_WEIGHTS=$POLICY_ANCHOR_WEIGHTS POLICY_ANCHOR_WEIGHT=$POLICY_ANCHOR_WEIGHT POLICY_ANCHOR_MARGIN_WEIGHT=$POLICY_ANCHOR_MARGIN_WEIGHT"
echo "STREAM_TRAIN=$STREAM_TRAIN eval_max=$STREAM_TRAIN_EVAL_MAX_SAMPLES"

env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --features training-tools,benchmark-tools,research-tools \
  --bin mmto_tree_dump \
  --bin mmto_tree_train \
  --bin mmto_score_gate \
  --bin mmto_rerank_gate \
  --bin usi_engine \
  --bin usi_benchmark \
  --bin record_analyze

pair_args=(--input)
for path in $BENCH_FAILURE_JSONL; do
  pair_args+=("$path")
done
pair_args+=(
  --output "$RUN_DIR/bench_failure_pairs.jsonl"
  --bad-move-source "$PAIR_BAD_MOVE_SOURCE"
  --min-regret-cp "$PAIR_MIN_REGRET_CP"
  --max-regret-cp "$PAIR_MAX_REGRET_CP"
  --limit "$PAIR_LIMIT"
  --weight-mode "$PAIR_WEIGHT_MODE"
  --weight-scale-cp "$PAIR_WEIGHT_SCALE_CP"
  --max-sample-weight "$PAIR_MAX_SAMPLE_WEIGHT"
)
if [[ "$PAIR_EXCLUDE_IN_CHECK" == "1" ]]; then
  pair_args+=(--exclude-in-check)
fi
if [[ "$PAIR_REQUIRE_NEW_LOSS" == "1" ]]; then
  pair_args+=(--require-new-loss)
fi
if [[ "$PAIR_REQUIRE_BAD_FLAG" == "1" ]]; then
  pair_args+=(--require-bad-flag)
fi
tools/extract_bench_failure_pairs.py "${pair_args[@]}" \
  | tee "$RUN_DIR/extract_pairs_stdout.log"

dump_args=(
  --student-weights "$STUDENT_WEIGHTS_FOR_DUMP"
  --teacher-weights "$TEACHER_WEIGHTS"
  --input "$RUN_DIR/bench_failure_pairs.jsonl"
  --train-output "$RUN_DIR/bench_failure.train.tree.jsonl"
  --valid-output "$RUN_DIR/bench_failure.valid.tree.jsonl"
  --teacher-depth "$DUMP_TEACHER_DEPTH"
  --student-depth "$DUMP_STUDENT_DEPTH"
  --teacher-score-top "$DUMP_TEACHER_SCORE_TOP"
  --candidate-top "$DUMP_CANDIDATE_TOP"
  --position-chunk-size "$DUMP_POSITION_CHUNK_SIZE"
  --valid-percent 0
  --seed 9841
  --jobs "$DUMP_JOBS"
  --min-legal-moves 2
  --max-abs-root-score "$DUMP_MAX_ABS_ROOT_SCORE"
  --max-positions "$DUMP_MAX_POSITIONS"
)
if [[ "$DUMP_EXCLUDE_IN_CHECK" == "1" ]]; then
  dump_args+=(--exclude-in-check)
fi
env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_tree_dump "${dump_args[@]}" \
  | tee "$RUN_DIR/bench_failure_dump_stdout.log"

if [[ ! -s "$RUN_DIR/bench_failure.train.tree.jsonl" ]]; then
  echo "bench failure dump produced no train samples. Rejecting this run."
  echo "kept_best_raw=0" > "$RUN_DIR/final_binary_status.txt"
  echo "RUN_DIR=$RUN_DIR"
  exit 1
fi

head -n "$VALID_LINES" "$SOURCE_VALID" > "$RUN_DIR/valid.top.tree.jsonl"
wc -l "$SOURCE_TRAIN" "$RUN_DIR/valid.top.tree.jsonl" "$RUN_DIR/bench_failure.train.tree.jsonl" \
  | tee "$RUN_DIR/subset_counts.txt"

policy_anchor_args=()
if [[ -n "$POLICY_ANCHOR_WEIGHTS" ]]; then
  if [[ ! -f "$POLICY_ANCHOR_WEIGHTS" ]]; then
    echo "missing policy anchor weights: $POLICY_ANCHOR_WEIGHTS" >&2
    exit 1
  fi
  policy_anchor_args+=(
    --policy-anchor-weights "$POLICY_ANCHOR_WEIGHTS"
    --policy-anchor-weight "$POLICY_ANCHOR_WEIGHT"
    --policy-anchor-temperature-cp "$POLICY_ANCHOR_TEMPERATURE_CP"
    --policy-anchor-feature-source "$POLICY_ANCHOR_FEATURE_SOURCE"
    --policy-anchor-margin-weight "$POLICY_ANCHOR_MARGIN_WEIGHT"
    --policy-anchor-margin-cp "$POLICY_ANCHOR_MARGIN_CP"
    --policy-anchor-margin-softplus-temp-cp "$POLICY_ANCHOR_MARGIN_SOFTPLUS_TEMP_CP"
  )
fi

train_args=(
  --weights "$WEIGHTS"
  --train "$SOURCE_TRAIN"
  --valid "$RUN_DIR/valid.top.tree.jsonl"
  --extra-valid "bench_failure=$RUN_DIR/bench_failure.train.tree.jsonl"
  --extra-valid-best-weight "$EXTRA_VALID_BEST_WEIGHT"
  --output "$RUN_DIR/candidate.raw.binary"
  --best-checkpoint-path "$RUN_DIR/best.raw.binary"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --learning-rate "$LEARNING_RATE"
  --optimizer adagrad
  --loss-mode listwise-leaf
  --listwise-feature-source "$LISTWISE_FEATURE_SOURCE"
  --listwise-teacher-top-k "$LISTWISE_TEACHER_TOP_K"
  --listwise-candidate-top-k "$LISTWISE_CANDIDATE_TOP_K"
  --listwise-min-selected-regret-cp "$LISTWISE_MIN_SELECTED_REGRET_CP"
  --listwise-max-teacher-score-abs-cp "$LISTWISE_MAX_TEACHER_SCORE_ABS_CP"
  --listwise-regret-cap-cp "$LISTWISE_REGRET_CAP_CP"
  --listwise-weight-mode "$LISTWISE_WEIGHT_MODE"
  --listwise-weight-scale-cp "$LISTWISE_WEIGHT_SCALE_CP"
  --listwise-max-sample-weight "$LISTWISE_MAX_SAMPLE_WEIGHT"
  --listwise-hard-negative-weight "$LISTWISE_HARD_NEGATIVE_WEIGHT"
  --listwise-hard-negative-min-regret-cp "$LISTWISE_HARD_NEGATIVE_MIN_REGRET_CP"
  --teacher-top-ce-weight "$TEACHER_TOP_CE_WEIGHT"
  --current-top-margin-weight "$CURRENT_TOP_MARGIN_WEIGHT"
  --current-top-min-bad-regret-cp "$CURRENT_TOP_MIN_BAD_REGRET_CP"
  --game-teacher-margin-weight "$GAME_TEACHER_MARGIN_WEIGHT"
  --game-teacher-max-regret-cp "$GAME_TEACHER_MAX_REGRET_CP"
  --game-teacher-min-bad-regret-cp "$GAME_TEACHER_MIN_BAD_REGRET_CP"
  --bad-regret-cp 300
  --bad-regret-thresholds-cp 50,100,200,300
  --best-metric "$BEST_METRIC"
  --best-guard-max-regret-increase-cp="$BEST_GUARD_MAX_REGRET_INCREASE_CP"
  --best-guard-bad100-increase="$BEST_GUARD_BAD100_INCREASE"
  --best-guard-teacher-match-drop-pct="$BEST_GUARD_TEACHER_MATCH_DROP_PCT"
  --selected-regret-cap-cp 300
  --freeze-material
  --anchor-l2 "$ANCHOR_L2"
  --max-weight-delta "$MAX_WEIGHT_DELTA"
  --replay-train "$RUN_DIR/bench_failure.train.tree.jsonl"
  --replay-weight "$REPLAY_WEIGHT"
  "${policy_anchor_args[@]}"
  --log-path "$RUN_DIR/train.csv"
)
if [[ "$STREAM_TRAIN" == "1" ]]; then
  train_args+=(--stream-train --stream-train-eval-max-samples "$STREAM_TRAIN_EVAL_MAX_SAMPLES")
fi

env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_tree_train "${train_args[@]}" \
  | tee "$RUN_DIR/train_stdout.log"

BEST_EPOCH="$(
  sed -n 's/.*best_epoch=\([0-9][0-9]*\).*/\1/p' "$RUN_DIR/train_stdout.log" \
    | tail -1
)"

rm -f "$RUN_DIR/candidate.raw.binary"
if [[ -z "$BEST_EPOCH" || "$BEST_EPOCH" == "0" ]]; then
  echo "best_epoch=${BEST_EPOCH:-none}: baseline is still best. Rejecting this run."
  rm -f "$RUN_DIR/best.raw.binary"
  echo "kept_best_raw=0" > "$RUN_DIR/final_binary_status.txt"
  echo "RUN_DIR=$RUN_DIR"
  exit 0
fi

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
  --seed 9842 \
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
  --baseline-depth "$RERANK_STUDENT_DEPTH" \
  --candidate-depth "$RERANK_STUDENT_DEPTH" \
  --teacher-depth "$RERANK_TEACHER_DEPTH" \
  --seed 9843 \
  --jobs "$RERANK_JOBS" \
  --bad-regret-thresholds-cp 50,100,200,300 \
  --allow-mean-regret-increase-cp 0 \
  --allow-p90-regret-increase-cp 0 \
  --allow-p95-regret-increase-cp 0 \
  --allow-bad-ratio-increase 0 \
  --hard-position-limit 1000 \
  --print-worst 20 \
  --json-output "$RUN_DIR/rerank_gate.json" \
  | tee "$RUN_DIR/rerank_gate_stdout.log"
rerank_status="${PIPESTATUS[0]}"
set -e
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

echo "bench-failure DAgger listwise run finished."
echo "RUN_DIR=$RUN_DIR"
cat "$RUN_DIR/final_binary_status.txt"
find "$RUN_DIR" -maxdepth 1 -type f -name '*.binary' -printf '%s %p\n'

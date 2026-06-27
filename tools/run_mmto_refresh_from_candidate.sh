#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script. Run it as: SOURCE_RUN_DIR=... CANDIDATE_WEIGHTS=... bash tools/run_mmto_refresh_from_candidate.sh"
  return 2
fi

cd "$(dirname "$0")/.."

SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-}"
if [[ -z "$SOURCE_RUN_DIR" ]]; then
  echo "SOURCE_RUN_DIR is required." >&2
  exit 1
fi

WEIGHTS="${WEIGHTS:-policy_weights_v2.1.0.binary}"
TEACHER_WEIGHTS="${TEACHER_WEIGHTS:-$WEIGHTS}"
CANDIDATE_WEIGHTS="${CANDIDATE_WEIGHTS:-$SOURCE_RUN_DIR/best.raw.binary}"
SOURCE_TRAIN="${SOURCE_TRAIN:-$SOURCE_RUN_DIR/train.tree.jsonl}"
SOURCE_VALID="${SOURCE_VALID:-$SOURCE_RUN_DIR/valid.tree.jsonl}"
SOURCE_SCORE_POSITIONS="${SOURCE_SCORE_POSITIONS:-$SOURCE_RUN_DIR/score_positions.sfen}"
REFRESH_INPUTS="${REFRESH_INPUTS:-$SOURCE_TRAIN}"
RUN_DIR="${RUN_DIR:-data/mmto/runs/mmto_refresh_$(date -u +%Y%m%d_%H%M%S)}"
MIN_FREE_GB="${MIN_FREE_GB:-6}"

TRAIN_MODE="${TRAIN_MODE:-mixed}"
BASE_TRAIN_LINES="${BASE_TRAIN_LINES:-0}"
BASE_VALID_LINES="${BASE_VALID_LINES:-0}"
REFRESH_MAX_POSITIONS="${REFRESH_MAX_POSITIONS:-600}"
REFRESH_VALID_PERCENT="${REFRESH_VALID_PERCENT:-10}"
REFRESH_SCORE_ALL_LEGAL_FOR_VALID="${REFRESH_SCORE_ALL_LEGAL_FOR_VALID:-0}"

TEACHER_DEPTH="${TEACHER_DEPTH:-4}"
STUDENT_DEPTH="${STUDENT_DEPTH:-3}"
RERANK_TEACHER_DEPTH="${RERANK_TEACHER_DEPTH:-5}"
TEACHER_SCORE_TOP="${TEACHER_SCORE_TOP:-24}"
CANDIDATE_TOP="${CANDIDATE_TOP:-24}"
POSITION_CHUNK_SIZE="${POSITION_CHUNK_SIZE:-16}"
JOBS="${JOBS:-2}"
MAX_ABS_ROOT_SCORE="${MAX_ABS_ROOT_SCORE:-3000}"

EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
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
LISTWISE_HARD_NEGATIVE_WEIGHT="${LISTWISE_HARD_NEGATIVE_WEIGHT:-0.05}"
LISTWISE_HARD_NEGATIVE_MIN_REGRET_CP="${LISTWISE_HARD_NEGATIVE_MIN_REGRET_CP:-15}"
GAME_TEACHER_MARGIN_WEIGHT="${GAME_TEACHER_MARGIN_WEIGHT:-0.05}"
GAME_TEACHER_MAX_REGRET_CP="${GAME_TEACHER_MAX_REGRET_CP:-150}"
GAME_TEACHER_MIN_BAD_REGRET_CP="${GAME_TEACHER_MIN_BAD_REGRET_CP:-15}"
CURRENT_TOP_MARGIN_WEIGHT="${CURRENT_TOP_MARGIN_WEIGHT:-0.05}"
CURRENT_TOP_MIN_BAD_REGRET_CP="${CURRENT_TOP_MIN_BAD_REGRET_CP:-15}"
BEST_METRIC="${BEST_METRIC:-p95-regret}"
ANCHOR_L2="${ANCHOR_L2:-0.0002}"
MAX_WEIGHT_DELTA="${MAX_WEIGHT_DELTA:-0.005}"
STREAM_TRAIN="${STREAM_TRAIN:-0}"
STREAM_TRAIN_EVAL_MAX_SAMPLES="${STREAM_TRAIN_EVAL_MAX_SAMPLES:-0}"

RERANK_INPUT="${RERANK_INPUT:-}"
RERANK_MAX_POSITIONS="${RERANK_MAX_POSITIONS:-500}"
RERANK_REQUIRE_MEAN_REGRET_IMPROVEMENT_CP="${RERANK_REQUIRE_MEAN_REGRET_IMPROVEMENT_CP:-0.0}"
RERANK_REQUIRE_P90_REGRET_IMPROVEMENT_CP="${RERANK_REQUIRE_P90_REGRET_IMPROVEMENT_CP:-0.0}"
RERANK_REQUIRE_P95_REGRET_IMPROVEMENT_CP="${RERANK_REQUIRE_P95_REGRET_IMPROVEMENT_CP:-0.0}"
RERANK_REQUIRE_MATCH_RATE_IMPROVEMENT_PCT="${RERANK_REQUIRE_MATCH_RATE_IMPROVEMENT_PCT:-0.0}"
KEEP_CANDIDATE_RAW="${KEEP_CANDIDATE_RAW:-0}"
BLEND_RATIOS="${BLEND_RATIOS:-}"

for path in "$WEIGHTS" "$TEACHER_WEIGHTS" "$CANDIDATE_WEIGHTS" "$SOURCE_TRAIN" "$SOURCE_VALID"; do
  if [[ ! -f "$path" ]]; then
    echo "missing required file: $path" >&2
    exit 1
  fi
done
case "$TRAIN_MODE" in
  mixed|refresh-only) ;;
  *)
    echo "TRAIN_MODE must be mixed or refresh-only, got: $TRAIN_MODE" >&2
    exit 1
    ;;
esac

FREE_KB="$(df -Pk . | awk 'NR==2 {print $4}')"
MIN_FREE_KB=$((MIN_FREE_GB * 1024 * 1024))
if (( FREE_KB < MIN_FREE_KB )); then
  echo "free disk is too low: $((FREE_KB / 1024 / 1024))GB < ${MIN_FREE_GB}GB" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"

echo "Starting MMTO candidate refresh pipeline."
echo "SOURCE_RUN_DIR=$SOURCE_RUN_DIR"
echo "RUN_DIR=$RUN_DIR"
echo "WEIGHTS=$WEIGHTS"
echo "TEACHER_WEIGHTS=$TEACHER_WEIGHTS"
echo "CANDIDATE_WEIGHTS=$CANDIDATE_WEIGHTS"
echo "REFRESH_INPUTS=$REFRESH_INPUTS"
echo "REFRESH_MAX_POSITIONS=$REFRESH_MAX_POSITIONS REFRESH_VALID_PERCENT=$REFRESH_VALID_PERCENT TRAIN_MODE=$TRAIN_MODE"
echo "STREAM_TRAIN=$STREAM_TRAIN CURRENT_TOP_MARGIN_WEIGHT=$CURRENT_TOP_MARGIN_WEIGHT LISTWISE_HARD_NEGATIVE_WEIGHT=$LISTWISE_HARD_NEGATIVE_WEIGHT"

env RUST_FONTCONFIG_DLOPEN=1 cargo build --release \
  --bin mmto_tree_dump \
  --bin rank_stats \
  --bin mmto_tree_train \
  --bin mmto_score_gate \
  --bin mmto_rerank_gate \
  --bin adjust_weights

dump_args=(
  --student-weights "$CANDIDATE_WEIGHTS"
  --teacher-weights "$TEACHER_WEIGHTS"
  --train-output "$RUN_DIR/refresh.train.tree.jsonl"
  --valid-output "$RUN_DIR/refresh.valid.tree.jsonl"
  --teacher-depth "$TEACHER_DEPTH"
  --student-depth "$STUDENT_DEPTH"
  --teacher-score-top "$TEACHER_SCORE_TOP"
  --candidate-top "$CANDIDATE_TOP"
  --position-chunk-size "$POSITION_CHUNK_SIZE"
  --valid-percent "$REFRESH_VALID_PERCENT"
  --seed 9821
  --jobs "$JOBS"
  --min-legal-moves 2
  --exclude-in-check
  --max-abs-root-score "$MAX_ABS_ROOT_SCORE"
  --max-positions "$REFRESH_MAX_POSITIONS"
)
for input in $REFRESH_INPUTS; do
  dump_args+=(--input "$input")
done
if [[ "$REFRESH_SCORE_ALL_LEGAL_FOR_VALID" == "1" ]]; then
  dump_args+=(--score-all-legal-for-valid)
fi

env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_tree_dump "${dump_args[@]}" \
  | tee "$RUN_DIR/refresh_dump_stdout.log"

if [[ ! -s "$RUN_DIR/refresh.train.tree.jsonl" ]]; then
  echo "refresh dump produced no train samples. Rejecting this stage."
  rm -f "$RUN_DIR/candidate.raw.binary" "$RUN_DIR/best.raw.binary"
  echo "RUN_DIR=$RUN_DIR"
  exit 1
fi
if [[ ! -s "$RUN_DIR/refresh.valid.tree.jsonl" ]]; then
  echo "refresh dump produced no valid samples. Increase REFRESH_MAX_POSITIONS or REFRESH_VALID_PERCENT." >&2
  rm -f "$RUN_DIR/candidate.raw.binary" "$RUN_DIR/best.raw.binary"
  echo "RUN_DIR=$RUN_DIR"
  exit 1
fi

env RUST_FONTCONFIG_DLOPEN=1 target/release/rank_stats \
  --input "$RUN_DIR/refresh.train.tree.jsonl" \
  --input "$RUN_DIR/refresh.valid.tree.jsonl" \
  --json-output "$RUN_DIR/refresh_rank_stats.json" \
  | tee "$RUN_DIR/refresh_rank_stats_stdout.log"

BASE_TRAIN_PATH="$SOURCE_TRAIN"
BASE_VALID_PATH="$SOURCE_VALID"
if (( BASE_TRAIN_LINES > 0 )); then
  BASE_TRAIN_PATH="$RUN_DIR/base.train.tree.jsonl"
  head -n "$BASE_TRAIN_LINES" "$SOURCE_TRAIN" > "$BASE_TRAIN_PATH"
fi
if (( BASE_VALID_LINES > 0 )); then
  BASE_VALID_PATH="$RUN_DIR/base.valid.tree.jsonl"
  head -n "$BASE_VALID_LINES" "$SOURCE_VALID" > "$BASE_VALID_PATH"
fi

case "$TRAIN_MODE" in
  mixed)
    cat "$BASE_TRAIN_PATH" "$RUN_DIR/refresh.train.tree.jsonl" > "$RUN_DIR/train.tree.jsonl"
    cat "$BASE_VALID_PATH" "$RUN_DIR/refresh.valid.tree.jsonl" > "$RUN_DIR/valid.tree.jsonl"
    ;;
  refresh-only)
    cp "$RUN_DIR/refresh.train.tree.jsonl" "$RUN_DIR/train.tree.jsonl"
    cp "$RUN_DIR/refresh.valid.tree.jsonl" "$RUN_DIR/valid.tree.jsonl"
    ;;
esac
wc -l "$RUN_DIR/train.tree.jsonl" "$RUN_DIR/valid.tree.jsonl" | tee "$RUN_DIR/train_counts.txt"

SCORE_POSITIONS="$SOURCE_SCORE_POSITIONS"
if [[ ! -f "$SCORE_POSITIONS" ]]; then
  SCORE_POSITIONS="$RUN_DIR/score_positions.sfen"
  python3 - "$RUN_DIR/train.tree.jsonl" "$RUN_DIR/valid.tree.jsonl" "$SCORE_POSITIONS" <<'PY'
import json
import sys

train_path, valid_path, output_path = sys.argv[1:4]
seen = set()
with open(output_path, "w", encoding="utf-8") as out:
    for path in (train_path, valid_path):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                sfen = json.loads(line)["sfen"].strip()
                if sfen and sfen not in seen:
                    seen.add(sfen)
                    out.write(sfen + "\n")
print(f"score_positions={len(seen)}")
PY
fi

train_args=(
  --weights "$WEIGHTS"
  --train "$RUN_DIR/train.tree.jsonl"
  --valid "$RUN_DIR/valid.tree.jsonl"
  --extra-valid "refresh=$RUN_DIR/refresh.valid.tree.jsonl"
  --extra-valid-best-weight 0.25
  --output "$RUN_DIR/candidate.raw.binary"
  --best-checkpoint-path "$RUN_DIR/best.raw.binary"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --learning-rate "$LEARNING_RATE"
  --teacher-top-k "$TEACHER_TOP_K"
  --student-bad-top-k "$STUDENT_BAD_TOP_K"
  --bad-candidate-scope "$BAD_CANDIDATE_SCOPE"
  --min-regret-cp "$MIN_REGRET_CP"
  --max-pairs-per-sample "$MAX_PAIRS_PER_SAMPLE"
  --pair-mining "$PAIR_MINING"
  --pair-weight-mode "$PAIR_WEIGHT_MODE"
  --pair-weight-scale-cp "$PAIR_WEIGHT_SCALE_CP"
  --max-pair-weight "$MAX_PAIR_WEIGHT"
  --optimizer adagrad
  --loss-mode "$LOSS_MODE"
  --listwise-feature-source "$LISTWISE_FEATURE_SOURCE"
  --listwise-hard-negative-weight "$LISTWISE_HARD_NEGATIVE_WEIGHT"
  --listwise-hard-negative-min-regret-cp "$LISTWISE_HARD_NEGATIVE_MIN_REGRET_CP"
  --game-teacher-margin-weight "$GAME_TEACHER_MARGIN_WEIGHT"
  --game-teacher-max-regret-cp "$GAME_TEACHER_MAX_REGRET_CP"
  --game-teacher-min-bad-regret-cp "$GAME_TEACHER_MIN_BAD_REGRET_CP"
  --current-top-margin-weight "$CURRENT_TOP_MARGIN_WEIGHT"
  --current-top-min-bad-regret-cp "$CURRENT_TOP_MIN_BAD_REGRET_CP"
  --margin-cp 50
  --softplus-temp-cp 100
  --bad-regret-cp 300
  --bad-regret-thresholds-cp 50,100,200,300
  --best-metric "$BEST_METRIC"
  --selected-regret-cap-cp 300
  --freeze-material
  --anchor-l2 "$ANCHOR_L2"
  --max-weight-delta "$MAX_WEIGHT_DELTA"
  --log-path "$RUN_DIR/train.csv"
)
if [[ "$STREAM_TRAIN" == "1" ]]; then
  train_args+=(--stream-train --stream-train-eval-max-samples "$STREAM_TRAIN_EVAL_MAX_SAMPLES")
fi

env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_tree_train "${train_args[@]}" \
  | tee "$RUN_DIR/train_stdout.log"

BEST_EPOCH="$(
  grep -o 'best_epoch=[0-9]*' "$RUN_DIR/train_stdout.log" \
    | tail -1 \
    | cut -d= -f2
)"

if [[ -z "$BEST_EPOCH" || "$BEST_EPOCH" == "0" ]]; then
  echo "best_epoch=${BEST_EPOCH:-none}: baseline is still best. Rejecting this stage."
  rm -f "$RUN_DIR/candidate.raw.binary" "$RUN_DIR/best.raw.binary"
  echo "RUN_DIR=$RUN_DIR"
  exit 0
fi

set +e
env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_score_gate \
  --baseline-weights "$WEIGHTS" \
  --candidate-weights "$RUN_DIR/best.raw.binary" \
  --input "$SCORE_POSITIONS" \
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
  echo "score gate failed. Rejecting this stage."
  if [[ "$KEEP_CANDIDATE_RAW" != "1" ]]; then
    rm -f "$RUN_DIR/candidate.raw.binary" "$RUN_DIR/best.raw.binary"
  fi
  echo "RUN_DIR=$RUN_DIR"
  exit "$score_status"
fi

if [[ -z "$RERANK_INPUT" ]]; then
  RERANK_INPUT="$BASE_VALID_PATH"
fi
set +e
env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_rerank_gate \
  --baseline-weights "$WEIGHTS" \
  --candidate-weights "$RUN_DIR/best.raw.binary" \
  --teacher-weights "$TEACHER_WEIGHTS" \
  --input "$RERANK_INPUT" \
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

echo "refresh offline gates passed"
echo "RUN_DIR=$RUN_DIR"
ls -lh "$RUN_DIR"

#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script. Run it as: SOURCE_RUN_DIR=... bash tools/run_mmto_dagger_from_run.sh"
  return 2
fi

cd "$(dirname "$0")/.."

SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-}"
if [[ -z "$SOURCE_RUN_DIR" ]]; then
  echo "SOURCE_RUN_DIR is required. It should point to a prior MMTO run with train/valid dumps and hard_positions.sfen." >&2
  exit 1
fi

WEIGHTS="${WEIGHTS:-policy_weights_v2.1.0.binary}"
TEACHER_WEIGHTS="${TEACHER_WEIGHTS:-$WEIGHTS}"
CANDIDATE_WEIGHTS="${CANDIDATE_WEIGHTS:-$SOURCE_RUN_DIR/best.raw.binary}"
BASE_TRAIN="${BASE_TRAIN:-$SOURCE_RUN_DIR/train.tree.jsonl}"
BASE_VALID="${BASE_VALID:-$SOURCE_RUN_DIR/valid.tree.jsonl}"
DAGGER_INPUT="${DAGGER_INPUT:-$SOURCE_RUN_DIR/hard_positions.sfen}"
RERANK_JSON="${RERANK_JSON:-$SOURCE_RUN_DIR/rerank_gate.json}"
USE_EXPLICIT_HARD_PAIRS="${USE_EXPLICIT_HARD_PAIRS:-1}"
EXPLICIT_MIN_REGRET_DELTA_CP="${EXPLICIT_MIN_REGRET_DELTA_CP:-0}"
EXPLICIT_MIN_CANDIDATE_REGRET_CP="${EXPLICIT_MIN_CANDIDATE_REGRET_CP:-0}"
EXPLICIT_WEIGHT_MODE="${EXPLICIT_WEIGHT_MODE:-combined}"
EXPLICIT_WEIGHT_SCALE_CP="${EXPLICIT_WEIGHT_SCALE_CP:-100}"
EXPLICIT_MAX_SAMPLE_WEIGHT="${EXPLICIT_MAX_SAMPLE_WEIGHT:-5}"
RUN_DIR="${RUN_DIR:-data/mmto/runs/mmto_dagger_$(date -u +%Y%m%d_%H%M%S)}"
MIN_FREE_GB="${MIN_FREE_GB:-6}"

TRAIN_LINES="${TRAIN_LINES:-0}"
VALID_LINES="${VALID_LINES:-0}"
SCORE_POSITIONS="${SCORE_POSITIONS:-$SOURCE_RUN_DIR/score_positions.sfen}"

TEACHER_DEPTH="${TEACHER_DEPTH:-4}"
STUDENT_DEPTH="${STUDENT_DEPTH:-3}"
RERANK_TEACHER_DEPTH="${RERANK_TEACHER_DEPTH:-5}"
TEACHER_SCORE_TOP="${TEACHER_SCORE_TOP:-24}"
CANDIDATE_TOP="${CANDIDATE_TOP:-24}"
POSITION_CHUNK_SIZE="${POSITION_CHUNK_SIZE:-16}"
JOBS="${JOBS:-2}"
DAGGER_MAX_POSITIONS="${DAGGER_MAX_POSITIONS:-1000}"
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
REPLAY_WEIGHT="${REPLAY_WEIGHT:-0.10}"
REPLAY_MAX_SAMPLES="${REPLAY_MAX_SAMPLES:-0}"
EXTRA_VALID_BEST_WEIGHT="${EXTRA_VALID_BEST_WEIGHT:-0.25}"
BEST_GUARD_MAX_REGRET_INCREASE_CP="${BEST_GUARD_MAX_REGRET_INCREASE_CP:--1}"
BEST_GUARD_BAD100_INCREASE="${BEST_GUARD_BAD100_INCREASE:--1}"
BEST_METRIC="${BEST_METRIC:-p95-regret}"
ANCHOR_L2="${ANCHOR_L2:-0.0002}"
MAX_WEIGHT_DELTA="${MAX_WEIGHT_DELTA:-0.005}"

RERANK_MAX_POSITIONS="${RERANK_MAX_POSITIONS:-500}"
RERANK_REQUIRE_MEAN_REGRET_IMPROVEMENT_CP="${RERANK_REQUIRE_MEAN_REGRET_IMPROVEMENT_CP:-0.0}"
RERANK_REQUIRE_P90_REGRET_IMPROVEMENT_CP="${RERANK_REQUIRE_P90_REGRET_IMPROVEMENT_CP:-0.0}"
RERANK_REQUIRE_P95_REGRET_IMPROVEMENT_CP="${RERANK_REQUIRE_P95_REGRET_IMPROVEMENT_CP:-0.0}"
RERANK_REQUIRE_MATCH_RATE_IMPROVEMENT_PCT="${RERANK_REQUIRE_MATCH_RATE_IMPROVEMENT_PCT:-0.0}"
KEEP_CANDIDATE_RAW="${KEEP_CANDIDATE_RAW:-0}"
BLEND_RATIOS="${BLEND_RATIOS:-}"

for path in "$WEIGHTS" "$TEACHER_WEIGHTS" "$CANDIDATE_WEIGHTS" "$BASE_TRAIN" "$BASE_VALID"; do
  if [[ ! -f "$path" ]]; then
    echo "missing required file: $path" >&2
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

if [[ "$USE_EXPLICIT_HARD_PAIRS" == "1" ]]; then
  if [[ ! -f "$RERANK_JSON" ]]; then
    echo "missing rerank json for explicit hard pairs: $RERANK_JSON" >&2
    exit 1
  fi
  DAGGER_INPUT="$RUN_DIR/dagger_input_pairs.jsonl"
  tools/extract_rerank_hard_pairs.py \
    --input "$RERANK_JSON" \
    --output "$DAGGER_INPUT" \
    --min-regret-delta-cp "$EXPLICIT_MIN_REGRET_DELTA_CP" \
    --min-candidate-regret-cp "$EXPLICIT_MIN_CANDIDATE_REGRET_CP" \
    --weight-mode "$EXPLICIT_WEIGHT_MODE" \
    --weight-scale-cp "$EXPLICIT_WEIGHT_SCALE_CP" \
    --max-sample-weight "$EXPLICIT_MAX_SAMPLE_WEIGHT"
fi
if [[ ! -f "$DAGGER_INPUT" ]]; then
  echo "missing dagger input: $DAGGER_INPUT" >&2
  exit 1
fi
if [[ ! -s "$DAGGER_INPUT" ]]; then
  echo "dagger input is empty: $DAGGER_INPUT" >&2
  exit 1
fi

echo "Starting MMTO DAgger replay stage."
echo "SOURCE_RUN_DIR=$SOURCE_RUN_DIR"
echo "RUN_DIR=$RUN_DIR"
echo "WEIGHTS=$WEIGHTS"
echo "TEACHER_WEIGHTS=$TEACHER_WEIGHTS"
echo "CANDIDATE_WEIGHTS=$CANDIDATE_WEIGHTS"
echo "BASE_TRAIN=$BASE_TRAIN BASE_VALID=$BASE_VALID"
echo "DAGGER_INPUT=$DAGGER_INPUT DAGGER_MAX_POSITIONS=$DAGGER_MAX_POSITIONS USE_EXPLICIT_HARD_PAIRS=$USE_EXPLICIT_HARD_PAIRS"
echo "EXPLICIT_WEIGHT_MODE=$EXPLICIT_WEIGHT_MODE EXPLICIT_WEIGHT_SCALE_CP=$EXPLICIT_WEIGHT_SCALE_CP EXPLICIT_MAX_SAMPLE_WEIGHT=$EXPLICIT_MAX_SAMPLE_WEIGHT"
echo "REPLAY_WEIGHT=$REPLAY_WEIGHT CURRENT_TOP_MARGIN_WEIGHT=$CURRENT_TOP_MARGIN_WEIGHT LISTWISE_HARD_NEGATIVE_WEIGHT=$LISTWISE_HARD_NEGATIVE_WEIGHT"

env RUST_FONTCONFIG_DLOPEN=1 cargo build --release \
  --bin mmto_tree_dump \
  --bin rank_stats \
  --bin mmto_tree_train \
  --bin mmto_score_gate \
  --bin mmto_rerank_gate \
  --bin adjust_weights

TRAIN_PATH="$BASE_TRAIN"
VALID_PATH="$BASE_VALID"
if (( TRAIN_LINES > 0 )); then
  TRAIN_PATH="$RUN_DIR/base.train.tree.jsonl"
  head -n "$TRAIN_LINES" "$BASE_TRAIN" > "$TRAIN_PATH"
fi
if (( VALID_LINES > 0 )); then
  VALID_PATH="$RUN_DIR/base.valid.tree.jsonl"
  head -n "$VALID_LINES" "$BASE_VALID" > "$VALID_PATH"
fi
wc -l "$TRAIN_PATH" "$VALID_PATH" | tee "$RUN_DIR/base_counts.txt"

if [[ ! -f "$SCORE_POSITIONS" ]]; then
  SCORE_POSITIONS="$RUN_DIR/score_positions.sfen"
  python3 - "$TRAIN_PATH" "$VALID_PATH" "$SCORE_POSITIONS" <<'PY'
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

env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_tree_dump \
  --student-weights "$CANDIDATE_WEIGHTS" \
  --teacher-weights "$TEACHER_WEIGHTS" \
  --input "$DAGGER_INPUT" \
  --train-output "$RUN_DIR/dagger.train.tree.jsonl" \
  --valid-output "$RUN_DIR/dagger.valid.tree.jsonl" \
  --teacher-depth "$TEACHER_DEPTH" \
  --student-depth "$STUDENT_DEPTH" \
  --teacher-score-top "$TEACHER_SCORE_TOP" \
  --candidate-top "$CANDIDATE_TOP" \
  --position-chunk-size "$POSITION_CHUNK_SIZE" \
  --valid-percent 0 \
  --seed 9731 \
  --jobs "$JOBS" \
  --min-legal-moves 2 \
  --exclude-in-check \
  --max-abs-root-score "$MAX_ABS_ROOT_SCORE" \
  --max-positions "$DAGGER_MAX_POSITIONS" \
  | tee "$RUN_DIR/dagger_dump_stdout.log"

if [[ ! -s "$RUN_DIR/dagger.train.tree.jsonl" ]]; then
  echo "dagger dump produced no train samples. Rejecting this stage."
  rm -f "$RUN_DIR/candidate.raw.binary" "$RUN_DIR/best.raw.binary"
  echo "RUN_DIR=$RUN_DIR"
  exit 1
fi

env RUST_FONTCONFIG_DLOPEN=1 target/release/rank_stats \
  --input "$RUN_DIR/dagger.train.tree.jsonl" \
  --json-output "$RUN_DIR/dagger_rank_stats.json" \
  | tee "$RUN_DIR/dagger_rank_stats_stdout.log"

train_args=(
  --weights "$WEIGHTS"
  --train "$TRAIN_PATH"
  --valid "$VALID_PATH"
  --extra-valid "dagger=$RUN_DIR/dagger.train.tree.jsonl"
  --extra-valid-best-weight "$EXTRA_VALID_BEST_WEIGHT"
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
  --best-guard-max-regret-increase-cp "$BEST_GUARD_MAX_REGRET_INCREASE_CP"
  --best-guard-bad100-increase "$BEST_GUARD_BAD100_INCREASE"
  --selected-regret-cap-cp 300
  --freeze-material
  --anchor-l2 "$ANCHOR_L2"
  --max-weight-delta "$MAX_WEIGHT_DELTA"
  --replay-train "$RUN_DIR/dagger.train.tree.jsonl"
  --replay-weight "$REPLAY_WEIGHT"
  --log-path "$RUN_DIR/train.csv"
)
if (( REPLAY_MAX_SAMPLES > 0 )); then
  train_args+=(--replay-max-samples "$REPLAY_MAX_SAMPLES")
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

set +e
env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_rerank_gate \
  --baseline-weights "$WEIGHTS" \
  --candidate-weights "$RUN_DIR/best.raw.binary" \
  --teacher-weights "$TEACHER_WEIGHTS" \
  --input "$VALID_PATH" \
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

echo "dagger offline gates passed"
echo "RUN_DIR=$RUN_DIR"
ls -lh "$RUN_DIR"

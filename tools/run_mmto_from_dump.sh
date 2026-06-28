#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script. Run it as: SOURCE_RUN_DIR=... bash tools/run_mmto_from_dump.sh"
  return 2
fi

cd "$(dirname "$0")/.."

SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-}"
if [[ -z "$SOURCE_RUN_DIR" ]]; then
  echo "SOURCE_RUN_DIR is required. Example: SOURCE_RUN_DIR=data/mmto/runs/mmto_rerank_long_... bash tools/run_mmto_from_dump.sh" >&2
  exit 1
fi

SOURCE_TRAIN="${SOURCE_TRAIN:-$SOURCE_RUN_DIR/train.tree.jsonl}"
SOURCE_VALID="${SOURCE_VALID:-$SOURCE_RUN_DIR/valid.tree.jsonl}"
TRAIN_LINES="${TRAIN_LINES:-9000}"
VALID_LINES="${VALID_LINES:-1000}"

WEIGHTS="${WEIGHTS:-policy_weights_v2.1.0.binary}"
TEACHER_WEIGHTS="${TEACHER_WEIGHTS:-$WEIGHTS}"
STUDENT_DEPTH="${STUDENT_DEPTH:-4}"
RERANK_TEACHER_DEPTH="${RERANK_TEACHER_DEPTH:-5}"
RERANK_MAX_POSITIONS="${RERANK_MAX_POSITIONS:-1000}"
RERANK_ALLOW_MEAN_REGRET_INCREASE_CP="${RERANK_ALLOW_MEAN_REGRET_INCREASE_CP:-0.0}"
RERANK_ALLOW_P90_REGRET_INCREASE_CP="${RERANK_ALLOW_P90_REGRET_INCREASE_CP:-0.0}"
RERANK_ALLOW_P95_REGRET_INCREASE_CP="${RERANK_ALLOW_P95_REGRET_INCREASE_CP:-0.0}"
RERANK_ALLOW_BAD_RATIO_INCREASE="${RERANK_ALLOW_BAD_RATIO_INCREASE:-0.0}"
RERANK_ALLOW_PHASE_MEAN_REGRET_INCREASE_CP="${RERANK_ALLOW_PHASE_MEAN_REGRET_INCREASE_CP:--1}"
RERANK_ALLOW_PHASE_P90_REGRET_INCREASE_CP="${RERANK_ALLOW_PHASE_P90_REGRET_INCREASE_CP:--1}"
RERANK_ALLOW_PHASE_P95_REGRET_INCREASE_CP="${RERANK_ALLOW_PHASE_P95_REGRET_INCREASE_CP:--1}"
RERANK_ALLOW_PHASE_BAD_RATIO_INCREASE="${RERANK_ALLOW_PHASE_BAD_RATIO_INCREASE:--1}"
RERANK_REQUIRE_MEAN_REGRET_IMPROVEMENT_CP="${RERANK_REQUIRE_MEAN_REGRET_IMPROVEMENT_CP:-0.5}"
RERANK_REQUIRE_P90_REGRET_IMPROVEMENT_CP="${RERANK_REQUIRE_P90_REGRET_IMPROVEMENT_CP:-0.0}"
RERANK_REQUIRE_P95_REGRET_IMPROVEMENT_CP="${RERANK_REQUIRE_P95_REGRET_IMPROVEMENT_CP:-0.0}"
RERANK_REQUIRE_MATCH_RATE_IMPROVEMENT_PCT="${RERANK_REQUIRE_MATCH_RATE_IMPROVEMENT_PCT:-0.0}"
RERANK_DEDUPE_SFEN="${RERANK_DEDUPE_SFEN:-0}"

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-0.00012}"
TEACHER_TOP_K="${TEACHER_TOP_K:-2}"
STUDENT_BAD_TOP_K="${STUDENT_BAD_TOP_K:-12}"
BAD_CANDIDATE_SCOPE="${BAD_CANDIDATE_SCOPE:-model-top}"
MIN_REGRET_CP="${MIN_REGRET_CP:-15}"
MAX_PAIRS_PER_SAMPLE="${MAX_PAIRS_PER_SAMPLE:-32}"
PAIR_MINING="${PAIR_MINING:-loss-top}"
PAIR_WEIGHT_MODE="${PAIR_WEIGHT_MODE:-bad-regret}"
PAIR_WEIGHT_SCALE_CP="${PAIR_WEIGHT_SCALE_CP:-100}"
MAX_PAIR_WEIGHT="${MAX_PAIR_WEIGHT:-3}"
OPTIMIZER="${OPTIMIZER:-adagrad}"
ADAGRAD_EPSILON="${ADAGRAD_EPSILON:-1e-6}"
MAX_WEIGHT_DELTA="${MAX_WEIGHT_DELTA:-0.03}"
ANCHOR_L2="${ANCHOR_L2:-0.0002}"
LOSS_MODE="${LOSS_MODE:-pairwise}"
LISTWISE_FEATURE_SOURCE="${LISTWISE_FEATURE_SOURCE:-teacher-leaf}"
LISTWISE_HARD_NEGATIVE_WEIGHT="${LISTWISE_HARD_NEGATIVE_WEIGHT:-0.0}"
LISTWISE_HARD_NEGATIVE_MIN_REGRET_CP="${LISTWISE_HARD_NEGATIVE_MIN_REGRET_CP:-50}"
TEACHER_TOP_CE_WEIGHT="${TEACHER_TOP_CE_WEIGHT:-0}"
GAME_TEACHER_MARGIN_WEIGHT="${GAME_TEACHER_MARGIN_WEIGHT:-0.0}"
GAME_TEACHER_MAX_REGRET_CP="${GAME_TEACHER_MAX_REGRET_CP:-150}"
GAME_TEACHER_MIN_BAD_REGRET_CP="${GAME_TEACHER_MIN_BAD_REGRET_CP:-15}"
CURRENT_TOP_MARGIN_WEIGHT="${CURRENT_TOP_MARGIN_WEIGHT:-0.0}"
CURRENT_TOP_MIN_BAD_REGRET_CP="${CURRENT_TOP_MIN_BAD_REGRET_CP:-15}"
FEEDBACK_JSON="${FEEDBACK_JSON:-}"
FEEDBACK_WEIGHT="${FEEDBACK_WEIGHT:-0}"
FEEDBACK_GOOD_MOVE="${FEEDBACK_GOOD_MOVE:-baseline}"
FEEDBACK_MIN_REGRET_DELTA_CP="${FEEDBACK_MIN_REGRET_DELTA_CP:-0}"
FEEDBACK_MIN_CANDIDATE_REGRET_CP="${FEEDBACK_MIN_CANDIDATE_REGRET_CP:-0}"
FEEDBACK_WEIGHT_SCALE_CP="${FEEDBACK_WEIGHT_SCALE_CP:-100}"
FEEDBACK_MAX_SAMPLE_WEIGHT="${FEEDBACK_MAX_SAMPLE_WEIGHT:-5}"
FEEDBACK_LIMIT="${FEEDBACK_LIMIT:-0}"
FEEDBACK_DEDUPE_SFEN="${FEEDBACK_DEDUPE_SFEN:-0}"
POLICY_ANCHOR_WEIGHTS="${POLICY_ANCHOR_WEIGHTS:-}"
POLICY_ANCHOR_WEIGHT="${POLICY_ANCHOR_WEIGHT:-0}"
POLICY_ANCHOR_TEMPERATURE_CP="${POLICY_ANCHOR_TEMPERATURE_CP:-100}"
POLICY_ANCHOR_FEATURE_SOURCE="${POLICY_ANCHOR_FEATURE_SOURCE:-move}"
POLICY_ANCHOR_MARGIN_WEIGHT="${POLICY_ANCHOR_MARGIN_WEIGHT:-0}"
POLICY_ANCHOR_MARGIN_CP="${POLICY_ANCHOR_MARGIN_CP:-50}"
POLICY_ANCHOR_MARGIN_SOFTPLUS_TEMP_CP="${POLICY_ANCHOR_MARGIN_SOFTPLUS_TEMP_CP:-100}"
SEPARATE_AUX_ADAGRAD="${SEPARATE_AUX_ADAGRAD:-0}"
BEST_METRIC="${BEST_METRIC:-p95-regret}"
BEST_GUARD_MAX_REGRET_INCREASE_CP="${BEST_GUARD_MAX_REGRET_INCREASE_CP:--1}"
BEST_GUARD_BAD100_INCREASE="${BEST_GUARD_BAD100_INCREASE:--1}"
BEST_GUARD_TEACHER_MATCH_DROP_PCT="${BEST_GUARD_TEACHER_MATCH_DROP_PCT:--1}"

BLEND_RATIOS="${BLEND_RATIOS-0.02 0.05}"
KEEP_CANDIDATE_RAW="${KEEP_CANDIDATE_RAW:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-6}"
RUN_DIR="${RUN_DIR:-data/mmto/runs/mmto_from_dump_$(date -u +%Y%m%d_%H%M%S)}"

if [[ ! -f "$SOURCE_TRAIN" ]]; then
  echo "missing source train dump: $SOURCE_TRAIN" >&2
  exit 1
fi
if [[ ! -f "$SOURCE_VALID" ]]; then
  echo "missing source valid dump: $SOURCE_VALID" >&2
  exit 1
fi
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

echo "Starting MMTO training from existing dump."
echo "SOURCE_RUN_DIR=$SOURCE_RUN_DIR"
echo "SOURCE_TRAIN=$SOURCE_TRAIN"
echo "SOURCE_VALID=$SOURCE_VALID"
echo "TRAIN_LINES=$TRAIN_LINES VALID_LINES=$VALID_LINES"
echo "RUN_DIR=$RUN_DIR"
echo "WEIGHTS=$WEIGHTS"
echo "TEACHER_WEIGHTS=$TEACHER_WEIGHTS"
echo "BAD_CANDIDATE_SCOPE=$BAD_CANDIDATE_SCOPE MIN_REGRET_CP=$MIN_REGRET_CP MAX_PAIRS_PER_SAMPLE=$MAX_PAIRS_PER_SAMPLE"
echo "PAIR_MINING=$PAIR_MINING PAIR_WEIGHT_MODE=$PAIR_WEIGHT_MODE PAIR_WEIGHT_SCALE_CP=$PAIR_WEIGHT_SCALE_CP MAX_PAIR_WEIGHT=$MAX_PAIR_WEIGHT"
echo "OPTIMIZER=$OPTIMIZER ADAGRAD_EPSILON=$ADAGRAD_EPSILON"
echo "LOSS_MODE=$LOSS_MODE LISTWISE_FEATURE_SOURCE=$LISTWISE_FEATURE_SOURCE"
echo "TEACHER_TOP_CE_WEIGHT=$TEACHER_TOP_CE_WEIGHT BEST_METRIC=$BEST_METRIC"
echo "CURRENT_TOP_MARGIN_WEIGHT=$CURRENT_TOP_MARGIN_WEIGHT CURRENT_TOP_MIN_BAD_REGRET_CP=$CURRENT_TOP_MIN_BAD_REGRET_CP"
echo "FEEDBACK_JSON=$FEEDBACK_JSON FEEDBACK_WEIGHT=$FEEDBACK_WEIGHT FEEDBACK_GOOD_MOVE=$FEEDBACK_GOOD_MOVE"
echo "FEEDBACK_DEDUPE_SFEN=$FEEDBACK_DEDUPE_SFEN"
echo "POLICY_ANCHOR_WEIGHTS=$POLICY_ANCHOR_WEIGHTS POLICY_ANCHOR_WEIGHT=$POLICY_ANCHOR_WEIGHT POLICY_ANCHOR_FEATURE_SOURCE=$POLICY_ANCHOR_FEATURE_SOURCE"
echo "POLICY_ANCHOR_MARGIN_WEIGHT=$POLICY_ANCHOR_MARGIN_WEIGHT POLICY_ANCHOR_MARGIN_CP=$POLICY_ANCHOR_MARGIN_CP"
echo "SEPARATE_AUX_ADAGRAD=$SEPARATE_AUX_ADAGRAD"
echo "RERANK_DEDUPE_SFEN=$RERANK_DEDUPE_SFEN"
echo "RERANK_ALLOW_MEAN/P90/P95/BAD=$RERANK_ALLOW_MEAN_REGRET_INCREASE_CP/$RERANK_ALLOW_P90_REGRET_INCREASE_CP/$RERANK_ALLOW_P95_REGRET_INCREASE_CP/$RERANK_ALLOW_BAD_RATIO_INCREASE"
echo "RERANK_ALLOW_PHASE_MEAN/P90/P95/BAD50=$RERANK_ALLOW_PHASE_MEAN_REGRET_INCREASE_CP/$RERANK_ALLOW_PHASE_P90_REGRET_INCREASE_CP/$RERANK_ALLOW_PHASE_P95_REGRET_INCREASE_CP/$RERANK_ALLOW_PHASE_BAD_RATIO_INCREASE"

head -n "$TRAIN_LINES" "$SOURCE_TRAIN" > "$RUN_DIR/train.tree.jsonl"
head -n "$VALID_LINES" "$SOURCE_VALID" > "$RUN_DIR/valid.tree.jsonl"
wc -l "$RUN_DIR/train.tree.jsonl" "$RUN_DIR/valid.tree.jsonl" | tee "$RUN_DIR/subset_counts.txt"

python3 - "$RUN_DIR/train.tree.jsonl" "$RUN_DIR/valid.tree.jsonl" "$RUN_DIR/score_positions.sfen" <<'PY'
import json
import sys

train_path, valid_path, output_path = sys.argv[1:4]
seen = set()
with open(output_path, "w", encoding="utf-8") as out:
    for path in (train_path, valid_path):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                sfen = json.loads(line)["sfen"].strip()
                if sfen and sfen not in seen:
                    seen.add(sfen)
                    out.write(sfen + "\n")
print(f"score_positions={len(seen)}")
PY

env RUST_FONTCONFIG_DLOPEN=1 cargo build --release \
  --bin rank_stats \
  --bin mmto_tree_train \
  --bin mmto_score_gate \
  --bin mmto_rerank_gate \
  --bin adjust_weights

env RUST_FONTCONFIG_DLOPEN=1 target/release/rank_stats \
  --input "$RUN_DIR/train.tree.jsonl" \
  --input "$RUN_DIR/valid.tree.jsonl" \
  --json-output "$RUN_DIR/rank_stats.json" \
  | tee "$RUN_DIR/rank_stats_stdout.log"

feedback_args=()
for path in $FEEDBACK_JSON; do
  if [[ ! -f "$path" ]]; then
    echo "missing feedback json: $path" >&2
    exit 1
  fi
  feedback_args+=(--feedback-json "$path")
done
policy_anchor_args=()
if [[ -n "$POLICY_ANCHOR_WEIGHTS" ]]; then
  if [[ ! -f "$POLICY_ANCHOR_WEIGHTS" ]]; then
    echo "missing policy anchor weights: $POLICY_ANCHOR_WEIGHTS" >&2
    exit 1
  fi
  policy_anchor_args+=(--policy-anchor-weights "$POLICY_ANCHOR_WEIGHTS")
fi
separate_aux_adagrad_args=()
if [[ "$SEPARATE_AUX_ADAGRAD" == "1" ]]; then
  separate_aux_adagrad_args+=(--separate-aux-adagrad)
fi
feedback_dedupe_args=()
if [[ "$FEEDBACK_DEDUPE_SFEN" == "1" ]]; then
  feedback_dedupe_args+=(--feedback-dedupe-sfen)
fi

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
  --optimizer "$OPTIMIZER" \
  --adagrad-epsilon "$ADAGRAD_EPSILON" \
  --loss-mode "$LOSS_MODE" \
  --listwise-feature-source "$LISTWISE_FEATURE_SOURCE" \
  --listwise-hard-negative-weight "$LISTWISE_HARD_NEGATIVE_WEIGHT" \
  --listwise-hard-negative-min-regret-cp "$LISTWISE_HARD_NEGATIVE_MIN_REGRET_CP" \
  --teacher-top-ce-weight "$TEACHER_TOP_CE_WEIGHT" \
  --game-teacher-margin-weight "$GAME_TEACHER_MARGIN_WEIGHT" \
  --game-teacher-max-regret-cp "$GAME_TEACHER_MAX_REGRET_CP" \
  --game-teacher-min-bad-regret-cp "$GAME_TEACHER_MIN_BAD_REGRET_CP" \
  --current-top-margin-weight "$CURRENT_TOP_MARGIN_WEIGHT" \
  --current-top-min-bad-regret-cp "$CURRENT_TOP_MIN_BAD_REGRET_CP" \
  "${feedback_args[@]}" \
  --feedback-weight "$FEEDBACK_WEIGHT" \
  --feedback-good-move "$FEEDBACK_GOOD_MOVE" \
  --feedback-min-regret-delta-cp "$FEEDBACK_MIN_REGRET_DELTA_CP" \
  --feedback-min-candidate-regret-cp "$FEEDBACK_MIN_CANDIDATE_REGRET_CP" \
  --feedback-weight-scale-cp "$FEEDBACK_WEIGHT_SCALE_CP" \
  --feedback-max-sample-weight "$FEEDBACK_MAX_SAMPLE_WEIGHT" \
  --feedback-limit "$FEEDBACK_LIMIT" \
  "${feedback_dedupe_args[@]}" \
  "${policy_anchor_args[@]}" \
  --policy-anchor-weight "$POLICY_ANCHOR_WEIGHT" \
  --policy-anchor-temperature-cp "$POLICY_ANCHOR_TEMPERATURE_CP" \
  --policy-anchor-feature-source "$POLICY_ANCHOR_FEATURE_SOURCE" \
  --policy-anchor-margin-weight "$POLICY_ANCHOR_MARGIN_WEIGHT" \
  --policy-anchor-margin-cp "$POLICY_ANCHOR_MARGIN_CP" \
  --policy-anchor-margin-softplus-temp-cp "$POLICY_ANCHOR_MARGIN_SOFTPLUS_TEMP_CP" \
  "${separate_aux_adagrad_args[@]}" \
  --margin-cp 50 \
  --softplus-temp-cp 100 \
  --bad-regret-cp 300 \
  --bad-regret-thresholds-cp 50,100,200,300 \
  --best-metric "$BEST_METRIC" \
  --best-guard-max-regret-increase-cp="$BEST_GUARD_MAX_REGRET_INCREASE_CP" \
  --best-guard-bad100-increase="$BEST_GUARD_BAD100_INCREASE" \
  --best-guard-teacher-match-drop-pct="$BEST_GUARD_TEACHER_MATCH_DROP_PCT" \
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
  --input "$RUN_DIR/score_positions.sfen" \
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
rerank_dedupe_arg=()
if [[ "$RERANK_DEDUPE_SFEN" == "1" ]]; then
  rerank_dedupe_arg=(--dedupe-sfen)
fi
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
  --allow-mean-regret-increase-cp "$RERANK_ALLOW_MEAN_REGRET_INCREASE_CP" \
  --allow-p90-regret-increase-cp "$RERANK_ALLOW_P90_REGRET_INCREASE_CP" \
  --allow-p95-regret-increase-cp "$RERANK_ALLOW_P95_REGRET_INCREASE_CP" \
  --allow-bad-ratio-increase "$RERANK_ALLOW_BAD_RATIO_INCREASE" \
  --allow-phase-mean-regret-increase-cp "$RERANK_ALLOW_PHASE_MEAN_REGRET_INCREASE_CP" \
  --allow-phase-p90-regret-increase-cp "$RERANK_ALLOW_PHASE_P90_REGRET_INCREASE_CP" \
  --allow-phase-p95-regret-increase-cp "$RERANK_ALLOW_PHASE_P95_REGRET_INCREASE_CP" \
  --allow-phase-bad-ratio-increase "$RERANK_ALLOW_PHASE_BAD_RATIO_INCREASE" \
  --require-mean-regret-improvement-cp "$RERANK_REQUIRE_MEAN_REGRET_IMPROVEMENT_CP" \
  --require-p90-regret-improvement-cp "$RERANK_REQUIRE_P90_REGRET_IMPROVEMENT_CP" \
  --require-p95-regret-improvement-cp "$RERANK_REQUIRE_P95_REGRET_IMPROVEMENT_CP" \
  --require-match-rate-improvement-pct "$RERANK_REQUIRE_MATCH_RATE_IMPROVEMENT_PCT" \
  "${rerank_dedupe_arg[@]}" \
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

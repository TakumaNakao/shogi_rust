#!/usr/bin/env bash
set -Eeuo pipefail

# End-to-end large HalfKP-64 training. PROFILE=smoke is a quick pipeline check;
# the default PROFILE=full builds the release dataset and can run for many hours.
export RUST_FONTCONFIG_DLOPEN="${RUST_FONTCONFIG_DLOPEN:-1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PROFILE="${PROFILE:-full}"
STAGE="${STAGE:-all}" # prepare, train, gate, or all
SEED="${SEED:-20260718}"
WORK_DIR="${WORK_DIR:-data/halfkp_search_learning_large_v1}"
INIT_WEIGHTS="${INIT_WEIGHTS:-data/policy_weights_halfkp64_kpp_distilled.binary}"
WDOOR_ROOT="${WDOOR_ROOT:-data/wdoor/extract}"
POSITIONS="${POSITIONS:-taya36.sfen}"
OPTIMIZER="${OPTIMIZER:-adagrad}"
THREADS="${THREADS:-0}"
GENERATOR_JOBS="${GENERATOR_JOBS:-0}"
FORCE_RAW="${FORCE_RAW:-0}"
FORCE_TEACHERS="${FORCE_TEACHERS:-0}"
RESUME="${RESUME:-0}"

case "$PROFILE" in
  full)
    TRAIN_OPENING="${TRAIN_OPENING:-250000}"
    TRAIN_MIDDLE="${TRAIN_MIDDLE:-500000}"
    TRAIN_ENDGAME="${TRAIN_ENDGAME:-250000}"
    EVAL_OPENING="${EVAL_OPENING:-25000}"
    EVAL_MIDDLE="${EVAL_MIDDLE:-40000}"
    EVAL_ENDGAME="${EVAL_ENDGAME:-15000}"
    SHARD_LINES="${SHARD_LINES:-25000}"
    RANDOM_EVERY="${RANDOM_EVERY:-4}"
    EPOCHS="${EPOCHS:-8}"
    GATE_GAMES="${GATE_GAMES:-200}"
    GATE_TIME_MS="${GATE_TIME_MS:-3000}"
    SELECTION_DEPTH="${SELECTION_DEPTH:-2}"
    TEACHER_DEPTH="${TEACHER_DEPTH:-4}"
    HARD_TEACHER_DEPTH="${HARD_TEACHER_DEPTH:-5}"
    CANDIDATE_TOP="${CANDIDATE_TOP:-8}"
    TACTICAL_CANDIDATE_LIMIT="${TACTICAL_CANDIDATE_LIMIT:-2}"
    ;;
  smoke)
    TRAIN_OPENING=80
    TRAIN_MIDDLE=120
    TRAIN_ENDGAME=80
    EVAL_OPENING=20
    EVAL_MIDDLE=30
    EVAL_ENDGAME=20
    SHARD_LINES=100
    RANDOM_EVERY=2
    EPOCHS="${EPOCHS:-1}"
    GATE_GAMES="${GATE_GAMES:-2}"
    GATE_TIME_MS="${GATE_TIME_MS:-50}"
    SELECTION_DEPTH="${SELECTION_DEPTH:-1}"
    TEACHER_DEPTH="${TEACHER_DEPTH:-2}"
    HARD_TEACHER_DEPTH="${HARD_TEACHER_DEPTH:-2}"
    CANDIDATE_TOP="${CANDIDATE_TOP:-4}"
    TACTICAL_CANDIDATE_LIMIT="${TACTICAL_CANDIDATE_LIMIT:-1}"
    ;;
  *)
    echo "PROFILE must be full or smoke" >&2
    exit 2
    ;;
esac

case "$STAGE" in
  prepare|train|gate|all) ;;
  *) echo "STAGE must be prepare, train, gate, or all" >&2; exit 2 ;;
esac
case "$OPTIMIZER" in
  adagrad|schedule-free) ;;
  *) echo "OPTIMIZER must be adagrad or schedule-free" >&2; exit 2 ;;
esac

RAW_DIR="$WORK_DIR/raw"
JSON_SHARD_DIR="$WORK_DIR/json_shards"
TEACHER_DIR="$WORK_DIR/teachers"
CHECKPOINT_DIR="$WORK_DIR/checkpoint_${OPTIMIZER}"
OUTPUT_WEIGHTS="$WORK_DIR/policy_weights_halfkp64_large_${OPTIMIZER}.binary"
TRAINING_LOG="$WORK_DIR/training_${OPTIMIZER}.csv"
GATE_DIR="$WORK_DIR/gate_${OPTIMIZER}"

TRAIN_INPUTS=(
  "$WDOOR_ROOT/2023"
  "$WDOOR_ROOT/2024"
  "$WDOOR_ROOT/2025"
)
EVAL_INPUTS=("$WDOOR_ROOT/2026")

[[ -s "$INIT_WEIGHTS" ]] || {
  echo "initial HalfKP-64 weights not found: $INIT_WEIGHTS" >&2
  exit 2
}
for path in "${TRAIN_INPUTS[@]}" "${EVAL_INPUTS[@]}"; do
  [[ -d "$path" ]] || { echo "CSA directory not found: $path" >&2; exit 2; }
done
mkdir -p "$RAW_DIR" "$JSON_SHARD_DIR" "$TEACHER_DIR"

available_kib="$(df -Pk "$WORK_DIR" | awk 'NR==2 {print $4}')"
if [[ "$PROFILE" == "full" && "$available_kib" -lt 15728640 ]]; then
  echo "At least 15 GiB of free disk is required; available: $((available_kib / 1048576)) GiB" >&2
  exit 2
fi

echo "[build] Compiling dataset, teacher, trainer, and gate binaries"
cargo build --release --features halfkp64 \
  --bin dataset_build --bin halfkp_search_teacher --bin halfkp_search_train \
  --bin usi_engine --bin usi_benchmark

build_raw_datasets() {
  if [[ "$FORCE_RAW" == "1" || ! -f "$RAW_DIR/train/.complete" ]]; then
    rm -rf "$RAW_DIR/train"
    local input_args=()
    for path in "${TRAIN_INPUTS[@]}"; do input_args+=(--input "$path"); done
    target/release/dataset_build \
      "${input_args[@]}" \
      --output-dir "$RAW_DIR/train" \
      --seed "$SEED" \
      --shuffle-games \
      --valid-percent 0 \
      --test-percent 0 \
      --target-opening-records "$TRAIN_OPENING" \
      --target-middle-records "$TRAIN_MIDDLE" \
      --target-late-records "$TRAIN_ENDGAME" \
      --require-targets \
      --target-minimum-percent 95 \
      --phase-records-per-game 5 10 10 \
      --min-ply 8 \
      --max-ply 200 \
      --min-player-rate 4000 \
      --min-opponent-rate 4000 \
      --known-result-only
    touch "$RAW_DIR/train/.complete"
  else
    echo "[data] Reusing $RAW_DIR/train"
  fi

  if [[ "$FORCE_RAW" == "1" || ! -f "$RAW_DIR/eval/.complete" ]]; then
    rm -rf "$RAW_DIR/eval"
    local input_args=()
    for path in "${EVAL_INPUTS[@]}"; do input_args+=(--input "$path"); done
    target/release/dataset_build \
      "${input_args[@]}" \
      --output-dir "$RAW_DIR/eval" \
      --seed "$((SEED + 1))" \
      --shuffle-games \
      --valid-percent 50 \
      --test-percent 50 \
      --target-opening-records "$EVAL_OPENING" \
      --target-middle-records "$EVAL_MIDDLE" \
      --target-late-records "$EVAL_ENDGAME" \
      --require-targets \
      --target-minimum-percent 95 \
      --phase-records-per-game 3 5 5 \
      --min-ply 8 \
      --max-ply 200 \
      --min-player-rate 4000 \
      --min-opponent-rate 4000 \
      --known-result-only
    touch "$RAW_DIR/eval/.complete"
  else
    echo "[data] Reusing $RAW_DIR/eval"
  fi
}

make_shards() {
  local input="$1"
  local prefix="$2"
  if [[ "$FORCE_RAW" == "1" ]] || ! compgen -G "${prefix}*.jsonl" >/dev/null; then
    rm -f "${prefix}"*.jsonl
    split --lines="$SHARD_LINES" --numeric-suffixes=0 --suffix-length=4 \
      --additional-suffix=.jsonl "$input" "$prefix"
  fi
}

generate_teacher() {
  local input="$1"
  local output="$2"
  local random_plies="$3"
  local seed="$4"
  if [[ "$FORCE_TEACHERS" != "1" && -s "$output" ]]; then
    echo "[teacher] Reusing $output"
    return
  fi
  local temporary="${output}.tmp"
  rm -f "$temporary"
  target/release/halfkp_search_teacher \
    --weights "$INIT_WEIGHTS" \
    --input "$input" \
    --output "$temporary" \
    --selection-depth "$SELECTION_DEPTH" \
    --teacher-depth "$TEACHER_DEPTH" \
    --candidate-top "$CANDIDATE_TOP" \
    --tactical-candidate-limit "$TACTICAL_CANDIDATE_LIMIT" \
    --hard-teacher-depth "$HARD_TEACHER_DEPTH" \
    --hard-percent 20 \
    --hard-score-gap-cp 100 \
    --hard-in-check \
    --hard-endgame \
    --randomize-max-plies "$random_plies" \
    --chunk-size 128 \
    --jobs "$GENERATOR_JOBS" \
    --seed "$seed"
  mv "$temporary" "$output"
}

prepare() {
  build_raw_datasets
  make_shards "$RAW_DIR/train/train.jsonl" "$JSON_SHARD_DIR/train-"
  make_shards "$RAW_DIR/eval/valid.jsonl" "$JSON_SHARD_DIR/valid-"
  make_shards "$RAW_DIR/eval/test.jsonl" "$JSON_SHARD_DIR/test-"

  local index=0
  for input in "$JSON_SHARD_DIR"/train-*.jsonl; do
    local stem="${input##*/}"
    stem="${stem%.jsonl}"
    generate_teacher "$input" "$TEACHER_DIR/${stem}.hkst" 0 "$((SEED + index))"
    if (( index % RANDOM_EVERY == 0 )); then
      generate_teacher "$input" "$TEACHER_DIR/${stem}-random.hkst" 4 \
        "$((SEED + 100000 + index))"
    fi
    index=$((index + 1))
  done
  for kind in valid test; do
    index=0
    for input in "$JSON_SHARD_DIR"/${kind}-*.jsonl; do
      local stem="${input##*/}"
      stem="${stem%.jsonl}"
      generate_teacher "$input" "$TEACHER_DIR/${stem}.hkst" 0 \
        "$((SEED + 200000 + index))"
      index=$((index + 1))
    done
  done
}

train() {
  local train_args=()
  local valid_args=()
  local test_args=()
  for path in "$TEACHER_DIR"/train-*.hkst; do train_args+=(--train "$path"); done
  for path in "$TEACHER_DIR"/valid-*.hkst; do valid_args+=(--valid "$path"); done
  for path in "$TEACHER_DIR"/test-*.hkst; do test_args+=(--test "$path"); done
  (( ${#train_args[@]} > 0 && ${#valid_args[@]} > 0 && ${#test_args[@]} > 0 )) || {
    echo "teacher shards are missing; run STAGE=prepare first" >&2
    exit 2
  }
  local resume_args=()
  if [[ "$RESUME" == "1" ]]; then resume_args+=(--resume); fi
  target/release/halfkp_search_train \
    "${train_args[@]}" \
    "${valid_args[@]}" \
    "${test_args[@]}" \
    --init "$INIT_WEIGHTS" \
    --output "$OUTPUT_WEIGHTS" \
    --optimizer "$OPTIMIZER" \
    --epochs "$EPOCHS" \
    --batch-size 128 \
    --learning-rate 0.0003 \
    --output-learning-rate 0.00003 \
    --search-mix 0.75 \
    --search-mix-end 0.50 \
    --loss-power 2.5 \
    --rank-weight 0.10 \
    --game-rank-weight 0.025 \
    --early-stop-patience 3 \
    --swa-start-epoch 4 \
    --fit-kappa \
    --threads "$THREADS" \
    --seed "$SEED" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --log "$TRAINING_LOG" \
    "${resume_args[@]}"
}

gate() {
  [[ -s "$OUTPUT_WEIGHTS" ]] || {
    echo "trained weights not found: $OUTPUT_WEIGHTS" >&2
    exit 2
  }
  [[ -s "$POSITIONS" ]] || { echo "gate positions not found: $POSITIONS" >&2; exit 2; }
  mkdir -p "$GATE_DIR"
  target/release/usi_benchmark \
    --new-engine target/release/usi_engine \
    --baseline-engine target/release/usi_engine \
    --new-weights "$OUTPUT_WEIGHTS" \
    --baseline-weights "$INIT_WEIGHTS" \
    --positions "$POSITIONS" \
    --games "$GATE_GAMES" \
    --depth 32 \
    --time-limit-ms "$GATE_TIME_MS" \
    --new-threads 0 \
    --baseline-threads 0 \
    --max-plies 256 \
    --jobs 1 \
    --seed "$((SEED + 300000))" \
    --record-dir "$GATE_DIR/games" | tee "$GATE_DIR/match.log"
}

case "$STAGE" in
  prepare) prepare ;;
  train) train ;;
  gate) gate ;;
  all) prepare; train; gate ;;
esac

echo "Weights: $OUTPUT_WEIGHTS"
echo "Training log: $TRAINING_LOG"
echo "Work directory: $WORK_DIR"

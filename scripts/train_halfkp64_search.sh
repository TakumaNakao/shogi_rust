#!/usr/bin/env bash
set -Eeuo pipefail

export RUST_FONTCONFIG_DLOPEN="${RUST_FONTCONFIG_DLOPEN:-1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TRAIN_INPUT="${TRAIN_INPUT:-data/wdoor/search_quality_phase0/pools/dev.jsonl}"
VALID_INPUT="${VALID_INPUT:-data/wdoor/search_quality_phase0/pools/holdout.jsonl}"
INIT_WEIGHTS="${INIT_WEIGHTS:-data/policy_weights_halfkp64_kpp_distilled.binary}"
OUT_DIR="${OUT_DIR:-data/halfkp_search_learning_v1}"
TRAIN_DATA="${TRAIN_DATA:-$OUT_DIR/train_depth3_v2.hkst}"
VALID_DATA="${VALID_DATA:-$OUT_DIR/valid_depth3_v2.hkst}"
OUTPUT_WEIGHTS="${OUTPUT_WEIGHTS:-$OUT_DIR/policy_weights_halfkp64_search_v1.binary}"
TRAINING_LOG="${TRAINING_LOG:-$OUT_DIR/training.csv}"

SELECTION_DEPTH="${SELECTION_DEPTH:-1}"
TEACHER_DEPTH="${TEACHER_DEPTH:-3}"
CANDIDATE_TOP="${CANDIDATE_TOP:-4}"
GENERATOR_JOBS="${GENERATOR_JOBS:-0}"
GENERATOR_CHUNK="${GENERATOR_CHUNK:-256}"
TRAIN_THREADS="${TRAIN_THREADS:-0}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SEED="${SEED:-20260718}"
FORCE_DATA="${FORCE_DATA:-0}"

for path in "$TRAIN_INPUT" "$VALID_INPUT" "$INIT_WEIGHTS"; do
  [[ -s "$path" ]] || { echo "required input not found: $path" >&2; exit 2; }
done
mkdir -p "$OUT_DIR"

echo "[1/4] Building HalfKP-64 teacher generator and trainer"
cargo build --release --features halfkp64,training-tools \
  --bin halfkp_search_teacher --bin halfkp_search_train

generate_teacher_data() {
  local input="$1"
  local output="$2"
  local seed="$3"
  local reuse_args=()
  if [[ "$FORCE_DATA" != "1" ]]; then
    reuse_args+=(--reuse-if-matches)
  fi
  target/release/halfkp_search_teacher \
    --weights "$INIT_WEIGHTS" \
    --input "$input" \
    --output "$output" \
    "${reuse_args[@]}" \
    --selection-depth "$SELECTION_DEPTH" \
    --teacher-depth "$TEACHER_DEPTH" \
    --candidate-top "$CANDIDATE_TOP" \
    --chunk-size "$GENERATOR_CHUNK" \
    --jobs "$GENERATOR_JOBS" \
    --seed "$seed"
}

echo "[2/4] Generating training teachers"
generate_teacher_data "$TRAIN_INPUT" "$TRAIN_DATA" "$SEED"

echo "[3/4] Generating independent validation teachers"
generate_teacher_data "$VALID_INPUT" "$VALID_DATA" "$((SEED + 1))"

echo "[4/4] Training HalfKP-64 with validation checkpointing"
target/release/halfkp_search_train \
  --train "$TRAIN_DATA" \
  --valid "$VALID_DATA" \
  --init "$INIT_WEIGHTS" \
  --output "$OUTPUT_WEIGHTS" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --threads "$TRAIN_THREADS" \
  --seed "$SEED" \
  --log "$TRAINING_LOG"

echo "Weights: $OUTPUT_WEIGHTS"
echo "Log: $TRAINING_LOG"

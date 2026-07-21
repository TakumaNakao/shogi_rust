#!/usr/bin/env bash
set -euo pipefail

# Train the 64-wide HalfKP model against the project's v2.1.0 KPP teacher.
# All paths can be overridden from the environment before invoking the script.
SHARD_DIR="${SHARD_DIR:-data/halfkp_distill_shards_v1}"
VALID_DATA="${VALID_DATA:-data/halfkp_distill_pool_v1/holdout_2026_kpp.jsonl}"
OUTPUT="${OUTPUT:-data/policy_weights_halfkp64_kpp_distilled.binary}"
LOG="${LOG:-data/halfkp64_kpp_training.csv}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-data/halfkp64_kpp_checkpoint}"
INIT="${INIT:-}"
RESUME="${RESUME:-0}"
EPOCHS="${EPOCHS:-12}"
LEARNING_RATE="${LEARNING_RATE:-0.003}"
OUTPUT_LEARNING_RATE="${OUTPUT_LEARNING_RATE:-0.0003}"
HUBER_DELTA="${HUBER_DELTA:-0.1}"
OUTPUT_LIMIT="${OUTPUT_LIMIT:-0.05}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-3}"
MIN_VALID_IMPROVEMENT_CP="${MIN_VALID_IMPROVEMENT_CP:-0.02}"
BATCH_SIZE="${BATCH_SIZE:-256}"
THREADS="${THREADS:-0}"

if [[ ! -s "$SHARD_DIR/manifest.json" ]]; then
  echo "sharded training data not found: $SHARD_DIR" >&2
  exit 1
fi
if [[ ! -s "$VALID_DATA" ]]; then
  echo "validation data not found: $VALID_DATA" >&2
  exit 1
fi

INIT_ARGS=()
if [[ -n "$INIT" ]]; then
  INIT_ARGS=(--init "$INIT")
fi
RESUME_ARGS=()
if [[ "$RESUME" == "1" ]]; then
  RESUME_ARGS=(--resume)
fi

export RUST_FONTCONFIG_DLOPEN=1
cargo build --release --features halfkp64,training-tools --bin halfkp_kpp_train

target/release/halfkp_kpp_train \
  --train "$SHARD_DIR/rank_train.jsonl" \
  --train-dir "$SHARD_DIR/train/search" \
  --train-dir "$SHARD_DIR/train/random" \
  --train-dir "$SHARD_DIR/train/mainline" \
  --train-dir "$SHARD_DIR/train/mainline" \
  --rank-train "$SHARD_DIR/rank_train_groups.jsonl" \
  --rank-pairs-per-root 4 \
  --train-dir "$SHARD_DIR/train/mainline" \
  --train-dir "$SHARD_DIR/train/mainline" \
  --valid "$VALID_DATA" \
  --valid-search "$SHARD_DIR/holdout/search.jsonl" \
  --valid-random "$SHARD_DIR/holdout/random.jsonl" \
  --valid-rank "$SHARD_DIR/holdout/rank_groups.jsonl" \
  --output "$OUTPUT" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --epochs "$EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --output-learning-rate "$OUTPUT_LEARNING_RATE" \
  --huber-delta "$HUBER_DELTA" \
  --output-limit "$OUTPUT_LIMIT" \
  --early-stop-patience "$EARLY_STOP_PATIENCE" \
  --min-valid-improvement-cp "$MIN_VALID_IMPROVEMENT_CP" \
  --batch-size "$BATCH_SIZE" \
  --threads "$THREADS" \
  --log "$LOG" \
  "${INIT_ARGS[@]}" \
  "${RESUME_ARGS[@]}"

echo "64-wide HalfKP weights: $OUTPUT"

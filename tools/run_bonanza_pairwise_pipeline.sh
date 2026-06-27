#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script. Run it as: bash tools/run_bonanza_pairwise_pipeline.sh"
  return 2
fi

cd "$(dirname "$0")/.."

WEIGHTS="${WEIGHTS:-./policy_weights_v2.1.0.binary}"
SEED="${SEED:-9601}"
RUN_DIR="${RUN_DIR:-data/bonanza_pairwise_runs/$(date -u +%Y%m%d_%H%M%S)}"
INPUTS="${INPUTS:-data/wdoor/extract/2026}"

MAX_RECORDS="${MAX_RECORDS:-2000}"
VALID_PERCENT="${VALID_PERCENT:-10}"
MIN_PLY="${MIN_PLY:-16}"
MAX_PLY="${MAX_PLY:-}"
WINNER_ONLY="${WINNER_ONLY:-0}"
DECISIVE_ONLY="${DECISIVE_ONLY:-0}"
FREEZE_MATERIAL="${FREEZE_MATERIAL:-1}"

EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-0.02}"
L2_LAMBDA="${L2_LAMBDA:-0.0}"
ANCHOR_L2="${ANCHOR_L2:-0.0}"
MAX_WEIGHT_DELTA="${MAX_WEIGHT_DELTA:-0}"
HARD_NEGATIVES="${HARD_NEGATIVES:-4}"
MARGIN_CP="${MARGIN_CP:-0.5}"
SOFTPLUS_TEMP_CP="${SOFTPLUS_TEMP_CP:-100.0}"
VALID_MAX_SAMPLES="${VALID_MAX_SAMPLES:-}"
TRAIN_OUTPUT="${RUN_DIR}/train.jsonl"
VALID_OUTPUT="${RUN_DIR}/valid.jsonl"

mkdir -p "$RUN_DIR"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "missing weights: $WEIGHTS" >&2
  exit 1
fi
if [[ ! -d "$(dirname "$RUN_DIR")" ]]; then
  mkdir -p "$(dirname "$RUN_DIR")"
fi

echo "Starting Bonanza pairwise pipeline."
echo "RUN_DIR=$RUN_DIR"
echo "WEIGHTS=$WEIGHTS"
echo "INPUTS=$INPUTS"
echo "MAX_RECORDS=$MAX_RECORDS VALID_PERCENT=$VALID_PERCENT MIN_PLY=$MIN_PLY MAX_PLY=${MAX_PLY:-none}"

env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin csa_policy_dump --bin bonanza_pairwise_train

read -r -a input_list <<< "$INPUTS"
if [[ ${#input_list[@]} -eq 0 || -z "${input_list[0]}" ]]; then
  echo "-- INPUTS is empty. Specify input directories/files with INPUTS env var." >&2
  exit 1
fi

dump_args=(
  --train-output "$TRAIN_OUTPUT"
  --valid-output "$VALID_OUTPUT"
  --seed "$SEED"
  --valid-percent "$VALID_PERCENT"
  --max-records "$MAX_RECORDS"
  --min-ply "$MIN_PLY"
)
if [[ -n "$MAX_PLY" && "$MAX_PLY" != "0" ]]; then
  dump_args+=(--max-ply "$MAX_PLY")
fi
if [[ "$WINNER_ONLY" == "1" ]]; then
  dump_args+=(--winner-only)
fi
if [[ "$DECISIVE_ONLY" == "1" ]]; then
  dump_args+=(--decisive-only)
fi
for input in "${input_list[@]}"; do
  dump_args+=(--input "$input")
done

env RUST_FONTCONFIG_DLOPEN=1 target/release/csa_policy_dump "${dump_args[@]}" \
  | tee "$RUN_DIR/csa_policy_dump_stdout.log"

train_args=(
  --weights "$WEIGHTS"
  --train "$TRAIN_OUTPUT"
  --valid "$VALID_OUTPUT"
  --output "$RUN_DIR/final.binary"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --learning-rate "$LEARNING_RATE"
  --l2-lambda "$L2_LAMBDA"
  --hard-negatives "$HARD_NEGATIVES"
  --margin-cp "$MARGIN_CP"
  --softplus-temp-cp "$SOFTPLUS_TEMP_CP"
  --seed "$SEED"
  --log-path "$RUN_DIR/train.csv"
  --best-checkpoint-path "$RUN_DIR/best.binary"
)
if [[ -n "$ANCHOR_L2" ]]; then
  train_args+=(--anchor-l2 "$ANCHOR_L2")
fi
if [[ "$MAX_WEIGHT_DELTA" != "0" && -n "$MAX_WEIGHT_DELTA" ]]; then
  train_args+=(--max-weight-delta "$MAX_WEIGHT_DELTA")
fi
if [[ "$VALID_MAX_SAMPLES" != "" ]]; then
  train_args+=(--valid-max-samples "$VALID_MAX_SAMPLES")
fi

if [[ "$FREEZE_MATERIAL" == "0" ]]; then
  train_args+=(--freeze-material false)
fi

env RUST_FONTCONFIG_DLOPEN=1 target/release/bonanza_pairwise_train "${train_args[@]}" \
  | tee "$RUN_DIR/train_stdout.log"

echo "RUN_DIR=$RUN_DIR"

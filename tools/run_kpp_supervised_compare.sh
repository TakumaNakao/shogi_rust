#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script. Run it as: bash tools/run_kpp_supervised_compare.sh"
  return 2
fi

cd "$(dirname "$0")/.."

YEARS="${YEARS:-2023 2024 2025 2026}"
RUN_KIND="${RUN_KIND:-both}" # scratch, warm, or both
BASELINE_WEIGHTS="${BASELINE_WEIGHTS:-policy_weights_v2.1.0.binary}"
RUN_ROOT="${RUN_ROOT:-data/wdoor/runs/kpp_supervised_compare_$(date -u +%Y%m%d_%H%M%S)}"

EPOCHS="${EPOCHS:-4}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
CHUNK_SIZE="${CHUNK_SIZE:-20000}"
LOAD_FILE_BATCH_SIZE="${LOAD_FILE_BATCH_SIZE:-256}"
VALID_PERCENT="${VALID_PERCENT:-5}"
VALID_MAX_FILES="${VALID_MAX_FILES:-500}"
MIN_PLAYER_RATE="${MIN_PLAYER_RATE:-4000}"
SOFTMAX_TEMPERATURE="${SOFTMAX_TEMPERATURE:-150}"
LEARNING_RATE="${LEARNING_RATE:-0.005}"
L2_LAMBDA="${L2_LAMBDA:-0.0001}"
SCRATCH_MATERIAL_COEFF="${SCRATCH_MATERIAL_COEFF:-0.14564776}"
SEED="${SEED:-20260625}"
RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-4}"

DECISIVE_ONLY="${DECISIVE_ONLY:-1}"
EXCLUDE_LOSER_AFTER_PLY="${EXCLUDE_LOSER_AFTER_PLY:-100}"
CHECKPOINT_EVERY_BATCHES="${CHECKPOINT_EVERY_BATCHES:-5000}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-0}"

case "$RUN_KIND" in
  scratch|warm|both) ;;
  *)
    echo "RUN_KIND must be scratch, warm, or both: $RUN_KIND" >&2
    exit 1
    ;;
esac

if [[ ! -f "$BASELINE_WEIGHTS" ]]; then
  echo "missing baseline weights: $BASELINE_WEIGHTS" >&2
  exit 1
fi

input_args=()
for year in $YEARS; do
  dir="data/wdoor/extract/$year"
  if [[ ! -d "$dir" ]]; then
    echo "missing Wdoor extract dir: $dir" >&2
    echo "Download it first, for example: tools/download_wdoor_kifu.sh $year" >&2
    exit 1
  fi
  input_args+=(--input-dir "$dir")
done

mkdir -p "$RUN_ROOT"

echo "RUN_ROOT=$RUN_ROOT"
echo "RUN_KIND=$RUN_KIND"
echo "YEARS=$YEARS"
echo "BASELINE_WEIGHTS=$BASELINE_WEIGHTS"
echo "SCRATCH_MATERIAL_COEFF=$SCRATCH_MATERIAL_COEFF"

env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --features training-tools --bin kpp_learn

run_one() {
  local kind="$1"
  local run_dir="$RUN_ROOT/$kind"
  mkdir -p "$run_dir/checkpoints"

  local init_args=()
  local output_name="policy_weights_${kind}_wdoor_${YEARS// /_}_r${MIN_PLAYER_RATE}_ce.binary"
  if [[ "$kind" == "scratch" ]]; then
    init_args=(
      --init-mode scratch
      --weights "$BASELINE_WEIGHTS"
      --scratch-material-coeff "$SCRATCH_MATERIAL_COEFF"
    )
  else
    init_args=(
      --init-mode load
      --weights "$BASELINE_WEIGHTS"
    )
  fi

  echo "Starting $kind run: $run_dir"
  env RAYON_NUM_THREADS="$RAYON_NUM_THREADS" RUST_FONTCONFIG_DLOPEN=1 target/release/kpp_learn \
    "${input_args[@]}" \
    "${init_args[@]}" \
    --output "$run_dir/$output_name" \
    --loss ce \
    --softmax-temperature "$SOFTMAX_TEMPERATURE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --chunk-size "$CHUNK_SIZE" \
    --load-file-batch-size "$LOAD_FILE_BATCH_SIZE" \
    --valid-percent "$VALID_PERCENT" \
    --valid-max-files "$VALID_MAX_FILES" \
    --min-player-rate "$MIN_PLAYER_RATE" \
    --learning-rate "$LEARNING_RATE" \
    --l2-lambda "$L2_LAMBDA" \
    --freeze-material \
    --checkpoint-dir "$run_dir/checkpoints" \
    --checkpoint-every-batches "$CHECKPOINT_EVERY_BATCHES" \
    --best-checkpoint-path "$run_dir/best.binary" \
    --no-graph \
    $(if [[ "$DECISIVE_ONLY" == "1" ]]; then printf '%s' "--decisive-only"; fi) \
    $(if [[ -n "$EXCLUDE_LOSER_AFTER_PLY" ]]; then printf '%s %s' "--exclude-loser-after-ply" "$EXCLUDE_LOSER_AFTER_PLY"; fi) \
    2>&1 | tee "$run_dir/train_stdout.log"

  if [[ "$KEEP_CHECKPOINTS" != "1" ]]; then
    find "$run_dir/checkpoints" -maxdepth 1 -type f -name '*.binary' -delete
  fi
  echo "$kind run finished: $run_dir"
}

case "$RUN_KIND" in
  scratch)
    run_one scratch
    ;;
  warm)
    run_one warm
    ;;
  both)
    run_one scratch
    run_one warm
    ;;
esac

echo "All requested runs finished."
echo "RUN_ROOT=$RUN_ROOT"

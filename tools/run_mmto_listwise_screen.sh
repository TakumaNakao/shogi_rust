#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script. Run it as: bash tools/run_mmto_listwise_screen.sh"
  return 2
fi

cd "$(dirname "$0")/.."

SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-data/mmto/runs/pv_sibling_feedback_20260628_190857}"
SOURCE_TRAIN="${SOURCE_TRAIN:-$SOURCE_RUN_DIR/train.pv.tree.jsonl}"
SOURCE_VALID="${SOURCE_VALID:-$SOURCE_RUN_DIR/valid.pv.tree.jsonl}"
SCREEN_NAME="${SCREEN_NAME:-current_top}"
SCREEN_DIR="${SCREEN_DIR:-data/mmto/runs/listwise_screen_${SCREEN_NAME}_$(date -u +%Y%m%d_%H%M%S)}"
SEEDS="${SEEDS:-7301 7401}"

TRAIN_LINES="${TRAIN_LINES:-1200}"
VALID_LINES="${VALID_LINES:-240}"
EPOCHS="${EPOCHS:-3}"
RERANK_MAX_POSITIONS="${RERANK_MAX_POSITIONS:-240}"
MIN_FREE_GB="${MIN_FREE_GB:-1}"

mkdir -p "$SCREEN_DIR"

echo "Starting MMTO listwise screen."
echo "SCREEN_DIR=$SCREEN_DIR"
echo "SCREEN_NAME=$SCREEN_NAME"
echo "SEEDS=$SEEDS"
echo "SOURCE_TRAIN=$SOURCE_TRAIN"
echo "SOURCE_VALID=$SOURCE_VALID"

run_dirs=()
for seed in $SEEDS; do
  run_dir="$SCREEN_DIR/seed_${seed}"
  run_dirs+=("$run_dir")
  echo "screen seed=$seed run_dir=$run_dir"

  set +e
  SOURCE_RUN_DIR="$SOURCE_RUN_DIR" \
    SOURCE_TRAIN="$SOURCE_TRAIN" \
    SOURCE_VALID="$SOURCE_VALID" \
    RUN_DIR="$run_dir" \
    TRAIN_LINES="$TRAIN_LINES" \
    VALID_LINES="$VALID_LINES" \
    SUBSET_MODE=shuffle \
    SUBSET_SEED="$seed" \
    STREAM_TRAIN=1 \
    EPOCHS="$EPOCHS" \
    RERANK_MAX_POSITIONS="$RERANK_MAX_POSITIONS" \
    MIN_FREE_GB="$MIN_FREE_GB" \
    KEEP_CANDIDATE_RAW=0 \
    BLEND_RATIOS="" \
    bash tools/run_mmto_from_dump.sh \
    > "$run_dir.pipeline_stdout.log" 2>&1
  status=$?
  set -e
  echo "seed=$seed exit_code=$status" | tee "$run_dir.exit"
done

python3 tools/summarize_mmto_runs.py "${run_dirs[@]}" \
  --json-output "$SCREEN_DIR/summary.json" \
  | tee "$SCREEN_DIR/summary.md"

echo "SCREEN_DIR=$SCREEN_DIR"

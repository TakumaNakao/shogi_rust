#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script. Run it as: bash tools/collect_rerank_feedback_pool.sh"
  return 2
fi

cd "$(dirname "$0")/.."

RUN_DIR="${RUN_DIR:-data/mmto/feedback/rerank_feedback_$(date -u +%Y%m%d_%H%M%S)}"
SEARCH_ROOT="${SEARCH_ROOT:-data/mmto/runs}"
OUTPUT="${OUTPUT:-$RUN_DIR/feedback.json}"
MIN_REGRET_DELTA_CP="${MIN_REGRET_DELTA_CP:-5}"
MIN_CANDIDATE_REGRET_CP="${MIN_CANDIDATE_REGRET_CP:-30}"
LIMIT="${LIMIT:-500}"
INCLUDE_WORST_DELTA="${INCLUDE_WORST_DELTA:-1}"
INCLUDE_WORST_CANDIDATE="${INCLUDE_WORST_CANDIDATE:-1}"
DEDUPE_SFEN="${DEDUPE_SFEN:-1}"

mkdir -p "$RUN_DIR"

mapfile -t JSONS < <(find "$SEARCH_ROOT" -path '*/rerank_gate.json' -type f | sort)
if (( ${#JSONS[@]} == 0 )); then
  echo "no rerank_gate.json files found under $SEARCH_ROOT" >&2
  exit 1
fi

echo "RUN_DIR=$RUN_DIR"
echo "SEARCH_ROOT=$SEARCH_ROOT"
echo "OUTPUT=$OUTPUT"
echo "INPUT_JSONS=${#JSONS[@]}"
echo "MIN_REGRET_DELTA_CP=$MIN_REGRET_DELTA_CP MIN_CANDIDATE_REGRET_CP=$MIN_CANDIDATE_REGRET_CP LIMIT=$LIMIT"

env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin rerank_feedback_collect

collect_args=()
for path in "${JSONS[@]}"; do
  collect_args+=(--input "$path")
done
collect_args+=(
  --output "$OUTPUT"
  --min-regret-delta-cp "$MIN_REGRET_DELTA_CP"
  --min-candidate-regret-cp "$MIN_CANDIDATE_REGRET_CP"
  --limit "$LIMIT"
  --dedupe-sfen "$DEDUPE_SFEN"
)
if [[ "$INCLUDE_WORST_DELTA" == "1" ]]; then
  collect_args+=(--include-worst-delta)
fi
if [[ "$INCLUDE_WORST_CANDIDATE" == "1" ]]; then
  collect_args+=(--include-worst-candidate)
fi

env RUST_FONTCONFIG_DLOPEN=1 target/release/rerank_feedback_collect "${collect_args[@]}" \
  | tee "$RUN_DIR/collect_stdout.log"

echo "FEEDBACK_JSON=$OUTPUT"

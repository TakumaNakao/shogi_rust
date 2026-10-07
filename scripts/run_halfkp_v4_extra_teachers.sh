#!/usr/bin/env bash
set -Eeuo pipefail

# Resume-safe supplemental teacher generation for T07.  Inputs are the
# explicitly documented 2016 no-rate branch plus one additional replay shard;
# this script never uses legacy weights or external evaluations.
export RUST_FONTCONFIG_DLOPEN="${RUST_FONTCONFIG_DLOPEN:-1}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_DIR="${RUN_DIR:-$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen5-20260910}"
EXTRA_DIR="${EXTRA_DIR:-$ROOT_DIR/runs/halfkp64-scratch-v1/v4-t07-extra-2016-20260910}"
REPLAY_DIR="${REPLAY_DIR:-$ROOT_DIR/runs/halfkp64-scratch-v1/v4-t07-extra-replay-20260910}"
OUT_DIR="${OUT_DIR:-$RUN_DIR/teachers-extra-100k}"
PARENT="$ROOT_DIR/runs/halfkp64-scratch-v1/phase3-2m-preserved-target-20260825/model-2m-100k.bin"
PARENT_MANIFEST="$PARENT.manifest.json"
REGISTRY="$ROOT_DIR/training/halfkp64-scratch-v1/promotion_registry.json"
GATE_REPORT="$ROOT_DIR/runs/halfkp64-scratch-v1/phase3-2m-preserved-target-20260825/match-20k-1000g-gen5-reserve-20260825/report.json"
WORKERS="${WORKERS:-12}"
SEED="${SEED:-20262000}"

[[ -s "$PARENT" && -s "$PARENT_MANIFEST" && -s "$REGISTRY" && -s "$GATE_REPORT" ]] || {
  echo "accepted gen5 lineage artifacts are missing" >&2; exit 2;
}
actual="$(sha256sum "$PARENT" | awk '{print $1}')"
[[ "$actual" == "61c05fe287cb0c0f9b434194802e07c835476b737a37bb9d4063143c63e28637" ]] || {
  echo "unexpected parent SHA: $actual" >&2; exit 2
}
mkdir -p "$OUT_DIR"

index=0
for input in "$EXTRA_DIR"/new*.jsonl "$REPLAY_DIR"/replay0000.jsonl; do
  [[ -e "$input" ]] || { echo "missing input: $input" >&2; exit 2; }
  stem="$(basename "$input" .jsonl)"
  output="$OUT_DIR/${stem}.hkst"
  if [[ -s "$output.manifest.json" ]]; then
    index=$((index + 1)); continue
  fi
  if [[ -e "$output" || -e "$output.tmp" ]]; then
    echo "refusing to overwrite partial teacher: $output" >&2; exit 1
  fi
  target/release/halfkp_search_teacher \
    --weights "$PARENT" \
    --input "$input" \
    --parent-manifest "$PARENT_MANIFEST" \
    --model-manifest "$PARENT_MANIFEST" \
    --promotion-registry "$REGISTRY" \
    --gate-report "$GATE_REPORT" \
    --parent-generation 5 \
    --output "$output" \
    --value-only --qsearch-leaf --teacher-depth 8 \
    --min-completed-depth 1 --max-nodes-per-search 100000 \
    --chunk-size 128 --jobs "$WORKERS" --seed "$((SEED + index))"
  index=$((index + 1))
done
echo "supplemental gen5 teachers complete: $OUT_DIR"

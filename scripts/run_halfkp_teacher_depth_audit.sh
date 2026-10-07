#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/runs/halfkp64-scratch-v1/h02-teacher-depth-audit-20260915}"
ROOTS="$RUN_DIR/roots-5001.jsonl"
PARENT="$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen5-20260910/candidate/model-epoch24.bin"
PARENT_MANIFEST="$PARENT.manifest.json"
REGISTRY="$ROOT_DIR/training/halfkp64-scratch-v1/promotion_registry.json"
GATE="$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen5-20260910/match-confirm-20k-250pair/report.json"
JOBS="${JOBS:-12}"

for path in "$ROOTS" "$PARENT" "$PARENT_MANIFEST" "$REGISTRY" "$GATE"; do
  [[ -s "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done
[[ "$(sha256sum "$PARENT" | awk '{print $1}')" == 162e17bacea3de80ecdbbdae765ca1157246c4febb6fa6a15f8d36b900d011dc ]] || exit 2
mkdir -p "$RUN_DIR"

run_teacher() {
  local label="$1" nodes="$2" depth="$3"
  local output="$RUN_DIR/teacher-${label}.hkst"
  [[ -s "$output.manifest.json" ]] && return
  target/release/halfkp_search_teacher \
    --weights "$PARENT" --input "$ROOTS" --parent-manifest "$PARENT_MANIFEST" \
    --model-manifest "$PARENT_MANIFEST" --promotion-registry "$REGISTRY" \
    --gate-report "$GATE" --parent-generation 6 --output "$output" \
    --value-only --qsearch-leaf --teacher-depth "$depth" --min-completed-depth 1 \
    --max-nodes-per-search "$nodes" --max-positions 5001 --chunk-size 128 \
    --jobs "$JOBS" --seed 20260915 > "$RUN_DIR/teacher-${label}.log" 2>&1
}

run_teacher 100k 100000 8
run_teacher 300k 300000 12
echo "teacher depth audit complete: $RUN_DIR"

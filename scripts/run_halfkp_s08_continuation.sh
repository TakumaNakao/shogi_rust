#!/usr/bin/env bash
set -Eeuo pipefail

# S08: freeze S07 outer8, continue the same AdaGrad run through outer12 and
# outer16, and preserve an immutable bundle at each milestone.  This script
# never uses v2.5.0/KPP weights as an input.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

S07_RUN="$ROOT_DIR/runs/halfkp64-scratch-v1/s07-retry-rate2500-v3-20260922-soft-target-1m-8pass"
S08_BASE="$ROOT_DIR/runs/halfkp64-scratch-v1/s08-retry-rate2500-v3-20260929"
FREEZE_RUN="$S08_BASE/frozen-outer8"
CONTINUE_RUN="$S08_BASE/continue-outer16"
OUTER12_RUN="$S08_BASE/outer12"
OUTER16_RUN="$S08_BASE/outer16"

TRAIN_BIN="$ROOT_DIR/target/release/halfkp_mmto_train"
DATASET="$ROOT_DIR/runs/halfkp64-scratch-v1/s07-retry-rate2500-v3-20260922-selected-1m"
INIT_MODEL="$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen5-20260910/candidate/model-epoch24.bin"
MODEL_MANIFEST="$INIT_MODEL.manifest.json"
REGISTRY="$ROOT_DIR/training/halfkp64-scratch-v1/promotion_registry.json"
GATE_REPORT="$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen5-20260910/match-confirm-20k-250pair/report.json"
DATASET_SHA="5595e9eae6dd79e2e062aaf7cffa6a09e41c95cc7a2a9a793c789d2c28066eca"

need_file() { [[ -s "$1" ]] || { echo "missing required file: $1" >&2; exit 2; }; }
need_file "$S07_RUN/checkpoint.json"
need_file "$S07_RUN/model.bin"
need_file "$S07_RUN/optimizer.bin"
need_file "$DATASET/manifest.json"
need_file "$INIT_MODEL"
need_file "$MODEL_MANIFEST"
need_file "$REGISTRY"
need_file "$GATE_REPORT"

if [[ "$(sha256sum "$DATASET/manifest.json" | awk '{print $1}')" != "$DATASET_SHA" ]]; then
  echo "dataset manifest SHA mismatch" >&2
  exit 2
fi

completed_outer() {
  python3 - "$1/checkpoint.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    print(int(json.load(f)["completed_outer"]))
PY
}

copy_bundle_once() {
  local source="$1" destination="$2"
  if [[ -e "$destination" ]]; then
    [[ -f "$destination/checkpoint.json" ]] || { echo "incomplete bundle: $destination" >&2; exit 2; }
    return
  fi
  mkdir -p "$(dirname "$destination")"
  cp -a "$source" "$destination"
}

copy_bundle_once "$S07_RUN" "$FREEZE_RUN"
[[ "$(completed_outer "$FREEZE_RUN")" == 8 ]] || { echo "S07 freeze is not outer8" >&2; exit 2; }

copy_bundle_once "$FREEZE_RUN" "$CONTINUE_RUN"

if [[ ! -x "$TRAIN_BIN" ]]; then
  cargo build --release --locked --features halfkp64,training-tools --bin halfkp_mmto_train
fi

run_until() {
  local target="$1"
  local current
  current="$(completed_outer "$CONTINUE_RUN")"
  if (( current >= target )); then
    return
  fi
  "$TRAIN_BIN" \
    --train "$DATASET/train.jsonl" \
    --valid "$DATASET/valid.jsonl" \
    --dataset-manifest "$DATASET/manifest.json" \
    --expected-dataset-manifest-sha256 "$DATASET_SHA" \
    --init "$INIT_MODEL" \
    --model-manifest "$MODEL_MANIFEST" \
    --promotion-registry "$REGISTRY" \
    --gate-report "$GATE_REPORT" \
    --parent-generation 6 \
    --output-dir "$CONTINUE_RUN" \
    --outer-iterations "$target" \
    --search-block-roots 1024 \
    --jobs 12 \
    --pair-batch-size 8192 \
    --learning-rate 0.0001 \
    --output-learning-rate-ratio 0.1 \
    --gradient-clip-norm 1.0 \
    --tau-cp 300 \
    --max-train-roots 1000000 \
    --max-valid-roots 12000 \
    --pv-probe-roots 256 \
    --checkpoint-roots 25000 \
    --qsearch-node-limit 2000000 \
    --resume
}

run_until 12
[[ "$(completed_outer "$CONTINUE_RUN")" -ge 12 ]] || { echo "outer12 did not complete" >&2; exit 1; }
copy_bundle_once "$CONTINUE_RUN" "$OUTER12_RUN"

run_until 16
[[ "$(completed_outer "$CONTINUE_RUN")" -ge 16 ]] || { echo "outer16 did not complete" >&2; exit 1; }
copy_bundle_once "$CONTINUE_RUN" "$OUTER16_RUN"

echo "S08 continuation complete"
echo "outer12: $OUTER12_RUN"
echo "outer16: $OUTER16_RUN"

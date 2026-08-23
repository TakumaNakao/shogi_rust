#!/usr/bin/env bash
set -Eeuo pipefail

# Resume-safe continuation for the one allowed 500k preserved-target escalation.
# The teacher process may be started separately; this script waits for its
# immutable HKST manifest, then runs valid generation, overlap filtering,
# training, and the fixed-node paired Gate without overwriting artifacts.

export RUST_FONTCONFIG_DLOPEN="${RUST_FONTCONFIG_DLOPEN:-1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_DIR="${RUN_DIR:-$ROOT_DIR/runs/halfkp64-scratch-v1/phase3-500k-preserved-target-qleaf-20260824}"
PARENT="$ROOT_DIR/runs/halfkp64-scratch-v1/unattended-mmto-full-lr1e-4-20260808-v1/archives/phase-c/outer_0004/model.bin"
PARENT_MANIFEST="$PARENT.manifest.json"
REGISTRY="$ROOT_DIR/training/halfkp64-scratch-v1/promotion_registry.json"
GATE_REPORT="$ROOT_DIR/training/halfkp64-scratch-v1/outer4_internal_gate_20260817.json"
VALID_INPUT="$ROOT_DIR/runs/halfkp64-scratch-v1/wdoor-highrate4000-1m-v2/valid.jsonl"
OPENINGS="$ROOT_DIR/training/halfkp64-scratch-v1/mmto_dev_openings_seed20260731_250/openings.jsonl"

mkdir -p "$RUN_DIR"

wait_for_train_teacher() {
  local manifest="$RUN_DIR/train.hkst.manifest.json"
  while [[ ! -s "$manifest" ]]; do
    if ! pgrep -f '[t]arget/release/halfkp_search_teacher' >/dev/null; then
      echo "teacher exited before producing $manifest" >&2
      return 1
    fi
    sleep 60
  done
}

wait_for_train_teacher

if [[ ! -s "$RUN_DIR/valid.hkst.manifest.json" ]]; then
  if [[ -e "$RUN_DIR/valid.hkst" || -e "$RUN_DIR/valid.hkst.tmp" ]]; then
    echo "refusing to overwrite partial valid teacher artifact" >&2
    exit 2
  fi
  target/release/halfkp_search_teacher \
    --weights "$PARENT" \
    --input "$VALID_INPUT" \
    --parent-manifest "$PARENT_MANIFEST" \
    --model-manifest "$PARENT_MANIFEST" \
    --promotion-registry "$REGISTRY" \
    --gate-report "$GATE_REPORT" \
    --parent-generation 3 \
    --output "$RUN_DIR/valid.hkst" \
    --value-only --qsearch-leaf \
    --teacher-depth 8 --min-completed-depth 1 \
    --max-nodes-per-search 200000 \
    --max-positions 50000 --jobs 8 --chunk-size 128 --seed 20260901 \
    2>&1 | tee "$RUN_DIR/teacher-valid.log"
fi

if [[ ! -s "$RUN_DIR/valid.filtered.hkst.manifest.json" ]]; then
  if [[ -e "$RUN_DIR/valid.filtered.hkst" || -e "$RUN_DIR/valid.filtered.hkst.tmp" ]]; then
    echo "refusing to overwrite partial filtered valid artifact" >&2
    exit 2
  fi
  target/release/halfkp_search_split_filter \
    --train "$RUN_DIR/train.hkst" \
    --valid "$RUN_DIR/valid.hkst" \
    --output "$RUN_DIR/valid.filtered.hkst" \
    --report "$RUN_DIR/valid.filter.report.json" \
    2>&1 | tee "$RUN_DIR/filter.stdout.log"
fi

if [[ ! -s "$RUN_DIR/model.bin.manifest.json" ]]; then
  if [[ -e "$RUN_DIR/model.bin" ]]; then
    echo "refusing to overwrite partial trained model" >&2
    exit 2
  fi
  mkdir -p "$RUN_DIR/checkpoints"
  target/release/halfkp_search_train \
    --train "$RUN_DIR/train.hkst" \
    --valid "$RUN_DIR/valid.filtered.hkst" \
    --init "$PARENT" \
    --model-manifest "$PARENT_MANIFEST" \
    --promotion-registry "$REGISTRY" \
    --gate-report "$GATE_REPORT" \
    --parent-generation 3 \
    --output "$RUN_DIR/model.bin" \
    --epochs 16 \
    --batch-size 128 \
    --learning-rate 0.0003 \
    --output-learning-rate 0.00003 \
    --kappa-cp 1352.7871 \
    --search-mix 0.8 --search-mix-end 0.8 \
    --value-weight 1 \
    --rank-weight 0 \
    --game-rank-weight 0 \
    --game-vs-search-weight 0 \
    --loss-power 2.5 \
    --gradient-clip-norm 1 \
    --output-limit 0.20 \
    --optimizer adagrad \
    --threads 8 \
    --seed 20260902 \
    --checkpoint-dir "$RUN_DIR/checkpoints" \
    --save-last-epoch \
    --log "$RUN_DIR/train.log" \
    2>&1 | tee "$RUN_DIR/train.stdout.log"
fi

MATCH_DIR="$RUN_DIR/match-20k-500g"
if [[ ! -s "$MATCH_DIR/report.json" ]]; then
  if [[ -e "$MATCH_DIR" ]]; then
    echo "resuming existing paired match directory"
  fi
  python3 scripts/run_halfkp_paired_match.py \
    --binary target/release/halfkp_promotion_match \
    --candidate "$RUN_DIR/model.bin" \
    --candidate-manifest "$RUN_DIR/model.bin.manifest.json" \
    --parent "$PARENT" \
    --parent-manifest "$PARENT_MANIFEST" \
    --registry "$REGISTRY" \
    --openings "$OPENINGS" \
    --openings-format dataset-jsonl \
    --opening-selection first-unique \
    --opening-seed 0 \
    --pairs 250 \
    --nodes 20000 \
    --max-depth 64 \
    --max-plies 256 \
    --output-dir "$MATCH_DIR" \
    --bootstrap-resamples 100000 \
    --bootstrap-seed 20260903
fi

echo "phase3 continuation complete: $RUN_DIR"

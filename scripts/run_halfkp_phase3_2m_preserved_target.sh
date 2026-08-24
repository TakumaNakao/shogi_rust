#!/usr/bin/env bash
set -Eeuo pipefail

# Resume-safe Phase 3 pipeline.  It waits for the immutable WDoor root set,
# labels every shard with the accepted scratch teacher, filters validation
# network-input overlap, and trains one candidate without touching v2.5.0.

export RUST_FONTCONFIG_DLOPEN="${RUST_FONTCONFIG_DLOPEN:-1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_DIR="${RUN_DIR:-$ROOT_DIR/runs/halfkp64-scratch-v1/phase3-2m-preserved-target-20260825}"
RAW_DIR="$RUN_DIR/raw"
SHARD_DIR="$RUN_DIR/shards-50k"
TEACHER_DIR="$RUN_DIR/teachers-100k"
PARENT="$ROOT_DIR/runs/halfkp64-scratch-v1/phase3-500k-preserved-target-qleaf-20260824/model.bin"
PARENT_MANIFEST="$PARENT.manifest.json"
REGISTRY="$ROOT_DIR/training/halfkp64-scratch-v1/promotion_registry.json"
GATE_REPORT="$ROOT_DIR/runs/halfkp64-scratch-v1/phase3-500k-preserved-target-qleaf-20260824/match-20k-1000g-finalreserve-20260824/report.json"
MODEL="$RUN_DIR/model-2m-100k.bin"
VALID_FILTERED="$RUN_DIR/valid-2m-100k.filtered.hkst"

[[ -s "$PARENT" && -s "$PARENT_MANIFEST" && -s "$REGISTRY" && -s "$GATE_REPORT" ]] || {
  echo "accepted scratch parent artifacts are missing" >&2
  exit 2
}

wait_for_raw() {
  while [[ ! -s "$RAW_DIR/manifest.json" ]]; do
    if [[ -s "$RUN_DIR/raw-build.pid" ]]; then
      local pid
      pid="$(cat "$RUN_DIR/raw-build.pid")"
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "raw dataset builder exited before manifest" >&2
        return 1
      fi
    fi
    sleep 60
  done
}

make_shards() {
  mkdir -p "$SHARD_DIR"
  for kind in train valid test; do
    local manifest="$SHARD_DIR/${kind}-manifest.json"
    if [[ -s "$manifest" ]]; then
      continue
    fi
    if compgen -G "$SHARD_DIR/${kind}-*.jsonl" >/dev/null; then
      echo "refusing to overwrite partial $kind shard set" >&2
      return 1
    fi
    target/release/jsonl_shard \
      --input "$RAW_DIR/$kind.jsonl" \
      --output-prefix "$SHARD_DIR/${kind}-" \
      --lines 50000 --suffix-width 4 \
      --parent-manifest "$RAW_DIR/manifest.json"
  done
}

generate_teachers() {
  mkdir -p "$TEACHER_DIR"
  local index=0 input stem output seed
  for input in "$SHARD_DIR"/train-*.jsonl; do
    stem="${input##*/}"; stem="${stem%.jsonl}"
    output="$TEACHER_DIR/${stem}.hkst"
    if [[ -s "$output.manifest.json" ]]; then
      index=$((index + 1)); continue
    fi
    if [[ -e "$output" || -e "$output.tmp" ]]; then
      echo "refusing to overwrite partial teacher $output" >&2
      return 1
    fi
    seed=$((20260825 + index))
    target/release/halfkp_search_teacher \
      --weights "$PARENT" --input "$input" \
      --parent-manifest "$PARENT_MANIFEST" --model-manifest "$PARENT_MANIFEST" \
      --promotion-registry "$REGISTRY" --gate-report "$GATE_REPORT" \
      --parent-generation 4 --output "$output" --value-only --qsearch-leaf \
      --teacher-depth 8 --min-completed-depth 1 --max-nodes-per-search 100000 \
      --chunk-size 128 --jobs 8 --seed "$seed"
    index=$((index + 1))
  done

  for kind in valid test; do
    index=0
    for input in "$SHARD_DIR"/${kind}-*.jsonl; do
      stem="${input##*/}"; stem="${stem%.jsonl}"
      output="$TEACHER_DIR/${stem}.hkst"
      if [[ -s "$output.manifest.json" ]]; then
        index=$((index + 1)); continue
      fi
      if [[ -e "$output" || -e "$output.tmp" ]]; then
        echo "refusing to overwrite partial teacher $output" >&2
        return 1
      fi
      seed=$((20270825 + index))
      target/release/halfkp_search_teacher \
        --weights "$PARENT" --input "$input" \
        --parent-manifest "$PARENT_MANIFEST" --model-manifest "$PARENT_MANIFEST" \
        --promotion-registry "$REGISTRY" --gate-report "$GATE_REPORT" \
        --parent-generation 4 --output "$output" --value-only --qsearch-leaf \
        --teacher-depth 8 --min-completed-depth 1 --max-nodes-per-search 100000 \
        --chunk-size 128 --jobs 8 --seed "$seed"
      index=$((index + 1))
    done
  done
}

filter_validation() {
  if [[ -s "$VALID_FILTERED.manifest.json" ]]; then
    return
  fi
  if [[ -e "$VALID_FILTERED" || -e "$VALID_FILTERED.tmp" ]]; then
    echo "refusing to overwrite partial filtered validation" >&2
    return 1
  fi
  local args=(--output "$VALID_FILTERED" --report "$RUN_DIR/valid-2m-100k.filter.json")
  local path
  for path in "$TEACHER_DIR"/train-*.hkst; do args+=(--train "$path"); done
  for path in "$TEACHER_DIR"/valid-*.hkst; do args+=(--valid "$path"); done
  target/release/halfkp_search_split_filter "${args[@]}"
}

train_model() {
  if [[ -s "$MODEL.manifest.json" ]]; then
    return
  fi
  if [[ -e "$MODEL" ]]; then
    echo "refusing to overwrite partial model" >&2
    return 1
  fi
  mkdir -p "$RUN_DIR/checkpoints-2m-100k"
  local args=()
  local path
  for path in "$TEACHER_DIR"/train-*.hkst; do args+=(--train "$path"); done
  args+=(--valid "$VALID_FILTERED")
  for path in "$TEACHER_DIR"/test-*.hkst; do args+=(--test "$path"); done
  target/release/halfkp_search_train \
    "${args[@]}" --init "$PARENT" --model-manifest "$PARENT_MANIFEST" \
    --promotion-registry "$REGISTRY" --gate-report "$GATE_REPORT" --parent-generation 4 \
    --output "$MODEL" --epochs 16 --batch-size 128 \
    --learning-rate 0.0003 --output-learning-rate 0.00003 \
    --kappa-cp 1352.7871 --search-mix 0.8 --search-mix-end 0.8 \
    --value-weight 1 --rank-weight 0 --game-rank-weight 0 --game-vs-search-weight 0 \
    --loss-power 2.5 --gradient-clip-norm 1 --output-limit 0.20 \
    --optimizer adagrad --threads 8 --seed 20260925 \
    --checkpoint-dir "$RUN_DIR/checkpoints-2m-100k" --save-last-epoch \
    --log "$RUN_DIR/train-2m-100k.log"
}

wait_for_raw
make_shards
generate_teachers
filter_validation
train_model
echo "phase3 2m preserved-target pipeline complete: $RUN_DIR"

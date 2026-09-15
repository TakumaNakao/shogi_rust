#!/usr/bin/env bash
set -Eeuo pipefail

# H05: compare teacher search budgets on one fixed 500k-root sample. Both
# models start from accepted scratch gen6 and use the same trainer recipe;
# only teacher node budget/depth differs. Every expensive stage is resumable.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/runs/halfkp64-scratch-v1/h05-teacher-ab-500k-20260915}"
SOURCE_DIR="$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen6-4m-20260913/shards-50k"
PARENT="$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen5-20260910/candidate/model-epoch24.bin"
PARENT_MANIFEST="$PARENT.manifest.json"
REGISTRY="$ROOT_DIR/training/halfkp64-scratch-v1/promotion_registry.json"
GATE="$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen5-20260910/match-confirm-20k-250pair/report.json"
VALID="$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen6-4m-20260913/valid-4m.filtered.hkst"
TEST="$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen6-4m-20260913/test-4m.filtered.hkst"
TEACHER_BIN="$ROOT_DIR/target/release/halfkp_search_teacher"
FILTER_BIN="$ROOT_DIR/target/release/halfkp_search_split_filter"
TRAIN_BIN="$ROOT_DIR/target/release/halfkp_search_train"
ROOTS="$RUN_DIR/roots-500k.jsonl"
ROOTS_MANIFEST="$RUN_DIR/roots-500k.manifest.json"
VALID_H05="$RUN_DIR/valid-h05-common.hkst"
TEST_H05="$RUN_DIR/test-h05-common.hkst"
JOBS="${JOBS:-4}"
WORKERS="${WORKERS:-12}"
SEED="${SEED:-20260920}"
EPOCHS="${EPOCHS:-16}"

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/checkpoints"
exec 9>"$RUN_DIR/.lock"
flock -n 9 || { echo "H05 is already running: $RUN_DIR"; exit 0; }

need() { [[ -s "$1" ]] || { echo "missing input: $1" >&2; exit 2; }; }
for path in "$PARENT" "$PARENT_MANIFEST" "$REGISTRY" "$GATE" "$VALID" "$TEST" "$TEACHER_BIN" "$FILTER_BIN" "$TRAIN_BIN"; do
  need "$path"
done
actual_parent="$(sha256sum "$PARENT" | awk '{print $1}')"
[[ "$actual_parent" == 162e17bacea3de80ecdbbdae765ca1157246c4febb6fa6a15f8d36b900d011dc ]] || {
  echo "unexpected parent SHA: $actual_parent" >&2; exit 2;
}

freeze_roots() {
  [[ -s "$ROOTS" && -s "$ROOTS_MANIFEST" ]] && return
  mapfile -t shards < <(find "$SOURCE_DIR" -maxdepth 1 -type f -name 'train-*.jsonl' | sort | head -10)
  (( ${#shards[@]} == 10 )) || { echo "expected ten 50k source shards" >&2; exit 2; }
  python3 scripts/freeze_halfkp_h05_roots.py --output "$ROOTS" --manifest "$ROOTS_MANIFEST" "${shards[@]}"
}

teacher_stage() {
  local label="$1" nodes="$2" depth="$3"
  local output="$RUN_DIR/teacher-${label}.hkst"
  if [[ -s "$output" && -s "$output.manifest.json" ]]; then return; fi
  "$TEACHER_BIN" \
    --weights "$PARENT" --input "$ROOTS" \
    --parent-manifest "$PARENT_MANIFEST" --model-manifest "$PARENT_MANIFEST" \
    --promotion-registry "$REGISTRY" --gate-report "$GATE" --parent-generation 6 \
    --output "$output" --value-only --qsearch-leaf \
    --teacher-depth "$depth" --min-completed-depth 1 \
    --max-nodes-per-search "$nodes" --max-positions 500000 \
    --chunk-size 128 --jobs "$JOBS" --seed "$SEED" \
    >"$RUN_DIR/logs/teacher-${label}.log" 2>&1
  need "$output"; need "$output.manifest.json"
}

filter_eval() {
  if [[ ! -s "$VALID_H05.manifest.json" ]]; then
    "$FILTER_BIN" \
      --train "$RUN_DIR/teacher-100k.hkst" --train "$RUN_DIR/teacher-300k.hkst" \
      --valid "$VALID" --output "$VALID_H05" \
      --report "$RUN_DIR/valid-h05-common.filter.json"
  fi
  if [[ ! -s "$TEST_H05.manifest.json" ]]; then
    "$FILTER_BIN" \
      --train "$RUN_DIR/teacher-100k.hkst" --train "$RUN_DIR/teacher-300k.hkst" \
      --train "$VALID_H05" --valid "$TEST" --output "$TEST_H05" \
      --report "$RUN_DIR/test-h05-common.filter.json"
  fi
  need "$VALID_H05"; need "$VALID_H05.manifest.json"
  need "$TEST_H05"; need "$TEST_H05.manifest.json"
}

train_stage() {
  local label="$1"
  local teacher="$RUN_DIR/teacher-${label}.hkst"
  local stage="$RUN_DIR/model-${label}.bin"
  local checkpoints="$RUN_DIR/checkpoints/${label}"
  local log="$RUN_DIR/train-${label}.csv"
  local marker="$checkpoints/selected-through-${EPOCHS}.bin"
  [[ -s "$marker" ]] && return
  mkdir -p "$checkpoints"
  local args=(
    --init "$PARENT" --model-manifest "$PARENT_MANIFEST"
    --promotion-registry "$REGISTRY" --gate-report "$GATE" --parent-generation 6
    --output "$stage" --epochs "$EPOCHS" --batch-size 128
    --learning-rate .0003 --output-learning-rate .00003 --kappa-cp 1352.7871
    --search-mix .8 --search-mix-end .8 --value-weight 1
    --rank-weight 0 --game-rank-weight 0 --game-vs-search-weight 0
    --loss-power 2.5 --gradient-clip-norm 1 --output-limit .20
    --optimizer adagrad --threads "$WORKERS" --seed "$SEED"
    --early-stop-patience 1000000 --min-valid-improvement .000001
    --swa-start-epoch 1000000 --checkpoint-dir "$checkpoints" --log "$log"
    --train "$teacher" --max-train-records 500000 --valid "$VALID_H05" --test "$TEST_H05"
  )
  if [[ -s "$stage" && -s "$stage.manifest.json" && -s "$checkpoints/current.binary" ]]; then
    args+=(--resume)
  fi
  "$TRAIN_BIN" "${args[@]}"
  need "$stage"; need "$stage.manifest.json"; need "$log"
  cp "$stage" "$marker"
  cp "$stage.manifest.json" "$marker.manifest.json"
}

freeze_roots
teacher_stage 100k 100000 8
teacher_stage 300k 300000 12
filter_eval
train_stage 100k
train_stage 300k
python3 scripts/summarize_halfkp_h05_teacher_ab.py \
  --output "$RUN_DIR/summary.json" \
  --low-manifest "$RUN_DIR/model-100k.bin.manifest.json" \
  --high-manifest "$RUN_DIR/model-300k.bin.manifest.json"
echo "H05 teacher A/B complete: $RUN_DIR"

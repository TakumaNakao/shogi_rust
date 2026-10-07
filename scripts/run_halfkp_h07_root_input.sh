#!/usr/bin/env bash
set -Eeuo pipefail

# H07: compare root-position input with H05's qsearch-leaf input.  The root
# sample, parent, teacher budget, trainer, and evaluation policy stay fixed;
# only --qsearch-leaf is removed from teacher generation.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/runs/halfkp64-scratch-v1/h07-root-input-100k-20260915}"
H05_DIR="$ROOT_DIR/runs/halfkp64-scratch-v1/h05-teacher-ab-500k-20260915"
PARENT="$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen5-20260910/candidate/model-epoch24.bin"
PARENT_MANIFEST="$PARENT.manifest.json"
REGISTRY="$ROOT_DIR/training/halfkp64-scratch-v1/promotion_registry.json"
GATE="$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen5-20260910/match-confirm-20k-250pair/report.json"
ROOTS="$H05_DIR/roots-500k.jsonl"
ROOTS_MANIFEST="$H05_DIR/roots-500k.manifest.json"
VALID_SOURCE="$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen6-4m-20260913/valid-4m.filtered.hkst"
TEST_SOURCE="$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen6-4m-20260913/test-4m.filtered.hkst"
TEACHER_BIN="$ROOT_DIR/target/release/halfkp_search_teacher"
FILTER_BIN="$ROOT_DIR/target/release/halfkp_search_split_filter"
TRAIN_BIN="$ROOT_DIR/target/release/halfkp_search_train"
COMPARE_BIN="$ROOT_DIR/target/release/halfkp_value_compare"
JOBS="${JOBS:-12}"
WORKERS="${WORKERS:-12}"
SEED="${SEED:-20260920}"
EPOCHS="${EPOCHS:-16}"

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/checkpoints" "$RUN_DIR/diagnostic"
exec 9>"$RUN_DIR/.lock"
flock -n 9 || { echo "H07 is already running: $RUN_DIR"; exit 0; }

need() { [[ -s "$1" ]] || { echo "missing input: $1" >&2; exit 2; }; }
for path in "$PARENT" "$PARENT_MANIFEST" "$REGISTRY" "$GATE" "$ROOTS" \
  "$ROOTS_MANIFEST" "$VALID_SOURCE" "$TEST_SOURCE" "$TEACHER_BIN" \
  "$FILTER_BIN" "$TRAIN_BIN" "$COMPARE_BIN"; do
  need "$path"
done
actual_parent="$(sha256sum "$PARENT" | awk '{print $1}')"
[[ "$actual_parent" == 162e17bacea3de80ecdbbdae765ca1157246c4febb6fa6a15f8d36b900d011dc ]] || {
  echo "unexpected parent SHA: $actual_parent" >&2
  exit 2
}

teacher="$RUN_DIR/teacher-root-100k.hkst"
if [[ ! -s "$teacher.manifest.json" ]]; then
  "$TEACHER_BIN" \
    --weights "$PARENT" --input "$ROOTS" \
    --parent-manifest "$PARENT_MANIFEST" --model-manifest "$PARENT_MANIFEST" \
    --promotion-registry "$REGISTRY" --gate-report "$GATE" --parent-generation 6 \
    --output "$teacher" --value-only \
    --teacher-depth 8 --min-completed-depth 1 \
    --max-nodes-per-search 100000 --max-positions 500000 \
    --chunk-size 128 --jobs "$JOBS" --seed "$SEED" \
    >"$RUN_DIR/logs/teacher-root-100k.log" 2>&1
fi
need "$teacher"; need "$teacher.manifest.json"

valid="$RUN_DIR/valid-root-common.hkst"
testset="$RUN_DIR/test-root-common.hkst"
if [[ ! -s "$valid.manifest.json" ]]; then
  "$FILTER_BIN" --train "$teacher" --valid "$VALID_SOURCE" --output "$valid" \
    --report "$RUN_DIR/valid-root-common.filter.json"
fi
if [[ ! -s "$testset.manifest.json" ]]; then
  "$FILTER_BIN" --train "$teacher" --train "$valid" --valid "$TEST_SOURCE" \
    --output "$testset" --report "$RUN_DIR/test-root-common.filter.json"
fi
need "$valid"; need "$valid.manifest.json"; need "$testset"; need "$testset.manifest.json"

stage="$RUN_DIR/model-root-input.bin"
checkpoints="$RUN_DIR/checkpoints/root-input"
log="$RUN_DIR/train-root-input.csv"
marker="$checkpoints/selected-through-$EPOCHS.bin"
mkdir -p "$checkpoints"
if [[ ! -s "$marker" ]]; then
  args=(
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
    --train "$teacher" --max-train-records 500000 --valid "$valid" --test "$testset"
  )
  if [[ -s "$stage" && -s "$stage.manifest.json" && -s "$checkpoints/current.binary" ]]; then
    args+=(--resume)
  fi
  "$TRAIN_BIN" "${args[@]}"
  need "$stage"; need "$stage.manifest.json"; need "$log"
  cp "$stage" "$marker"; cp "$stage.manifest.json" "$marker.manifest.json"
fi

candidate="$checkpoints/current.binary"
need "$candidate"
report="$RUN_DIR/diagnostic/compare-current-root-input.json"
if [[ ! -s "$report" ]]; then
  "$COMPARE_BIN" --dataset "$valid" --baseline-model "$PARENT" \
    --candidate-model "$candidate" --kappa-cp 1352.7871 --search-mix .8 \
    --output "$report"
fi
sha256sum "$teacher" "$teacher.manifest.json" "$stage" "$stage.manifest.json" \
  "$candidate" "$report" > "$RUN_DIR/artifacts.sha256"
echo "H07 root-input complete: $RUN_DIR"

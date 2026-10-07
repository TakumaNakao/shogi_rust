#!/usr/bin/env bash
set -Eeuo pipefail

# H06: isolate the search/result mixture after the H05 teacher-depth A/B.
# Reuses the already frozen 100k teacher and the H05 common evaluation split;
# the only training change is search_mix 0.8 -> 0.6.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/runs/halfkp64-scratch-v1/h06-mix60-100k-20260915}"
H05_DIR="$ROOT_DIR/runs/halfkp64-scratch-v1/h05-teacher-ab-500k-20260915"
PARENT="$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen5-20260910/candidate/model-epoch24.bin"
PARENT_MANIFEST="$PARENT.manifest.json"
REGISTRY="$ROOT_DIR/training/halfkp64-scratch-v1/promotion_registry.json"
GATE="$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen5-20260910/match-confirm-20k-250pair/report.json"
TEACHER="$H05_DIR/teacher-100k.hkst"
VALID="$H05_DIR/valid-h05-common.hkst"
TEST="$H05_DIR/test-h05-common.hkst"
TRAIN_BIN="$ROOT_DIR/target/release/halfkp_search_train"
WORKERS="${WORKERS:-12}"
SEED="${SEED:-20260920}"
EPOCHS="${EPOCHS:-16}"

mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/diagnostic"
exec 9>"$RUN_DIR/.lock"
flock -n 9 || { echo "H06 is already running: $RUN_DIR"; exit 0; }

need() { [[ -s "$1" ]] || { echo "missing input: $1" >&2; exit 2; }; }
for path in "$PARENT" "$PARENT_MANIFEST" "$REGISTRY" "$GATE" "$TEACHER" \
  "$TEACHER.manifest.json" "$VALID" "$VALID.manifest.json" "$TEST" \
  "$TEST.manifest.json" "$TRAIN_BIN"; do
  need "$path"
done
actual_parent="$(sha256sum "$PARENT" | awk '{print $1}')"
[[ "$actual_parent" == 162e17bacea3de80ecdbbdae765ca1157246c4febb6fa6a15f8d36b900d011dc ]] || {
  echo "unexpected parent SHA: $actual_parent" >&2
  exit 2
}

stage="$RUN_DIR/model-mix60.bin"
checkpoints="$RUN_DIR/checkpoints/mix60"
log="$RUN_DIR/train-mix60.csv"
marker="$checkpoints/selected-through-$EPOCHS.bin"
mkdir -p "$checkpoints"
if [[ ! -s "$marker" ]]; then
  args=(
    --init "$PARENT" --model-manifest "$PARENT_MANIFEST"
    --promotion-registry "$REGISTRY" --gate-report "$GATE" --parent-generation 6
    --output "$stage" --epochs "$EPOCHS" --batch-size 128
    --learning-rate .0003 --output-learning-rate .00003 --kappa-cp 1352.7871
    --search-mix .6 --search-mix-end .6 --value-weight 1
    --rank-weight 0 --game-rank-weight 0 --game-vs-search-weight 0
    --loss-power 2.5 --gradient-clip-norm 1 --output-limit .20
    --optimizer adagrad --threads "$WORKERS" --seed "$SEED"
    --early-stop-patience 1000000 --min-valid-improvement .000001
    --swa-start-epoch 1000000 --checkpoint-dir "$checkpoints" --log "$log"
    --train "$TEACHER" --max-train-records 500000 --valid "$VALID" --test "$TEST"
  )
  if [[ -s "$stage" && -s "$stage.manifest.json" && -s "$checkpoints/current.binary" ]]; then
    args+=(--resume)
  fi
  "$TRAIN_BIN" "${args[@]}"
  need "$stage"
  need "$stage.manifest.json"
  need "$log"
  cp "$stage" "$marker"
  cp "$stage.manifest.json" "$marker.manifest.json"
fi

baseline="$PARENT"
candidate="$checkpoints/current.binary"
need "$candidate"
report="$RUN_DIR/diagnostic/compare-current-mix60.json"
if [[ ! -s "$report" ]]; then
  "$ROOT_DIR/target/release/halfkp_value_compare" \
    --dataset "$VALID" \
    --baseline-model "$baseline" \
    --candidate-model "$candidate" \
    --kappa-cp 1352.7871 --search-mix .6 \
    --output "$report"
fi
sha256sum "$stage" "$stage.manifest.json" "$candidate" "$report" > "$RUN_DIR/artifacts.sha256"
echo "H06 mix60 complete: $RUN_DIR"

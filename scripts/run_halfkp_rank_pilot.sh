#!/usr/bin/env bash
set -euo pipefail

# Small, resumable pilot for checking that game-rank supervision is actually
# present in the current preserved-target pipeline.  This is an experiment
# only; it never touches the promotion registry or an existing artifact.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$ROOT/runs/halfkp64-scratch-v1/phase3-2m-highrate-20260829/rank-pilot-20260830"
TEACHER="$ROOT/target/release/halfkp_search_teacher"
PREPARE="$ROOT/target/release/halfkp_dataset_prepare"
TRAINER="$ROOT/target/release/halfkp_search_train"
OBJECTIVE="$ROOT/target/release/halfkp_search_objective_eval"
PARENT="$ROOT/runs/halfkp64-scratch-v1/phase3-500k-preserved-target-qleaf-20260824/model.bin"
PARENT_MANIFEST="$PARENT.manifest.json"
REGISTRY="$ROOT/training/halfkp64-scratch-v1/promotion_registry.json"
GATE="$ROOT/runs/halfkp64-scratch-v1/phase3-500k-preserved-target-qleaf-20260824/match-20k-1000g-finalreserve-20260824/report.json"
TRAIN_SOURCE="$ROOT/runs/halfkp64-scratch-v1/phase3-2m-highrate-20260829/raw/train.jsonl"
VALID_SOURCE="$ROOT/runs/halfkp64-scratch-v1/phase3-2m-highrate-20260829/raw/valid.jsonl"

mkdir -p "$RUN/logs" "$RUN/checkpoints"
cd "$ROOT"

if [[ ! -s "$RUN/train.hkst" || ! -s "$RUN/train.hkst.manifest.json" ]]; then
  "$TEACHER" \
    --weights "$PARENT" --parent-manifest "$PARENT_MANIFEST" \
    --model-manifest "$PARENT_MANIFEST" --promotion-registry "$REGISTRY" \
    --gate-report "$GATE" --parent-generation 4 \
    --input "$TRAIN_SOURCE" --output "$RUN/train.hkst" \
    --value-only --treestrap-lite --teacher-depth 8 --min-completed-depth 1 \
    --max-nodes-per-search 20000 --max-positions 25000 \
    --jobs 8 --chunk-size 128 --seed 20260830 \
    >"$RUN/logs/teacher-train.log" 2>&1
fi

if [[ ! -s "$RUN/valid-raw.hkst" || ! -s "$RUN/valid-raw.hkst.manifest.json" ]]; then
  "$TEACHER" \
    --weights "$PARENT" --parent-manifest "$PARENT_MANIFEST" \
    --model-manifest "$PARENT_MANIFEST" --promotion-registry "$REGISTRY" \
    --gate-report "$GATE" --parent-generation 4 \
    --input "$VALID_SOURCE" --output "$RUN/valid-raw.hkst" \
    --value-only --treestrap-lite --teacher-depth 8 --min-completed-depth 1 \
    --max-nodes-per-search 20000 --max-positions 5000 \
    --jobs 8 --chunk-size 128 --seed 20260831 \
    >"$RUN/logs/teacher-valid.log" 2>&1
fi

if [[ ! -s "$RUN/valid.hkst" || ! -s "$RUN/valid.hkst.manifest.json" ]]; then
  "$PREPARE" filter-hkst-network-overlap \
    --train "$RUN/train.hkst" --input "$RUN/valid-raw.hkst" \
    --output "$RUN/valid.hkst" --max-records 5000 \
    >"$RUN/logs/filter-valid.log" 2>&1
fi

if [[ ! -s "$RUN/model.bin" || ! -s "$RUN/model.bin.manifest.json" ]]; then
  "$TRAINER" \
    --train "$RUN/train.hkst" --valid "$RUN/valid.hkst" \
    --init "$PARENT" --model-manifest "$PARENT_MANIFEST" \
    --promotion-registry "$REGISTRY" --gate-report "$GATE" \
    --parent-generation 4 --output "$RUN/model.bin" \
    --epochs 2 --batch-size 128 --learning-rate 0.0003 \
    --output-learning-rate 0.00003 --kappa-cp 1352.7871 \
    --search-mix 0.8 --search-mix-end 0.8 --value-weight 1 \
    --rank-weight 0 --game-rank-weight 0.1 --game-vs-search-weight 0 \
    --rank-margin-cp 20 --rank-temperature-cp 100 --game-regret-cap-cp 300 \
    --gradient-clip-norm 1 --optimizer adagrad --threads 8 \
    --seed 20260832 --checkpoint-dir "$RUN/checkpoints" \
    --save-last-epoch --log "$RUN/logs/train.log"
fi

if [[ ! -s "$RUN/objective-game-rank.json" ]]; then
  "$OBJECTIVE" --game-rank --dataset "$RUN/valid.hkst" \
    --baseline-model "$PARENT" --candidate-model "$RUN/model.bin" \
    --game-regret-cap-cp 300 --min-rank-gap-cp 15 \
    --rank-margin-cp 20 --rank-temperature-cp 100 \
    --output "$RUN/objective-game-rank.json" \
    >"$RUN/logs/objective.log" 2>&1
fi

date -Is > "$RUN/complete"
echo "rank pilot complete: $RUN"

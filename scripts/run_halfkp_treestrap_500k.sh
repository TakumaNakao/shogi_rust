#!/usr/bin/env bash
set -euo pipefail

# Resumable Phase 4 scale-up.  This intentionally keeps the accepted 100k
# teacher/training recipe unchanged and only increases the WDoor root count.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$ROOT/runs/halfkp64-scratch-v1/phase5-treestrap-500k-20260822"
TEACHER="$ROOT/target/release/halfkp_search_teacher"
PREPARE="$ROOT/target/release/halfkp_dataset_prepare"
TRAINER="$ROOT/target/release/halfkp_search_train"
MATCH="$ROOT/scripts/run_halfkp_paired_match.py"
WEIGHTS="$ROOT/runs/halfkp64-scratch-v1/phase4-treestrap-100k-20260822/model.bin"
PARENT_MANIFEST="$ROOT/runs/halfkp64-scratch-v1/phase4-treestrap-100k-20260822/model.bin.manifest.json"
REGISTRY="$ROOT/training/halfkp64-scratch-v1/promotion_registry.json"
GATE="$ROOT/runs/halfkp64-scratch-v1/phase4-treestrap-100k-20260822/match-20k-500g/report.json"
TRAIN_SOURCE="$ROOT/runs/halfkp64-scratch-v1/wdoor-highrate4000-1m-v2/train.jsonl"
VALID_SOURCE="$ROOT/runs/halfkp64-scratch-v1/wdoor-highrate4000-1m-v2/valid.jsonl"
OPENINGS="$ROOT/training/halfkp64-scratch-v1/mmto_dev_openings_seed20260731_250/openings.jsonl"

mkdir -p "$RUN/logs" "$RUN/checkpoints"
cd "$ROOT"

if [[ ! -s "$RUN/train.hkst" || ! -s "$RUN/train.hkst.manifest.json" ]]; then
  "$TEACHER" \
    --weights "$WEIGHTS" --parent-manifest "$PARENT_MANIFEST" \
    --model-manifest "$PARENT_MANIFEST" --promotion-registry "$REGISTRY" \
    --gate-report "$GATE" --parent-generation 5 \
    --input "$TRAIN_SOURCE" --output "$RUN/train.hkst" \
    --value-only --treestrap-lite --teacher-depth 8 --min-completed-depth 1 \
    --max-nodes-per-search 200000 --max-positions 500000 \
    --jobs 8 --chunk-size 128 --seed 20260827 \
    >"$RUN/logs/teacher-train.log" 2>&1
fi

if [[ ! -s "$RUN/valid-raw.hkst" || ! -s "$RUN/valid-raw.hkst.manifest.json" ]]; then
  "$TEACHER" \
    --weights "$WEIGHTS" --parent-manifest "$PARENT_MANIFEST" \
    --model-manifest "$PARENT_MANIFEST" --promotion-registry "$REGISTRY" \
    --gate-report "$GATE" --parent-generation 5 \
    --input "$VALID_SOURCE" --output "$RUN/valid-raw.hkst" \
    --value-only --treestrap-lite --teacher-depth 8 --min-completed-depth 1 \
    --max-nodes-per-search 200000 --max-positions 20000 \
    --jobs 8 --chunk-size 128 --seed 20260828 \
    >"$RUN/logs/teacher-valid.log" 2>&1
fi

if [[ ! -s "$RUN/valid.hkst" || ! -s "$RUN/valid.hkst.manifest.json" ]]; then
  "$PREPARE" filter-hkst-network-overlap \
    --train "$RUN/train.hkst" --input "$RUN/valid-raw.hkst" \
    --output "$RUN/valid.hkst" --max-records 20000 \
    >"$RUN/logs/filter-valid.log" 2>&1
fi

if [[ ! -s "$RUN/model.bin" || ! -s "$RUN/model.bin.manifest.json" ]]; then
  "$TRAINER" \
    --train "$RUN/train.hkst" --valid "$RUN/valid.hkst" \
    --init "$WEIGHTS" --model-manifest "$PARENT_MANIFEST" \
    --promotion-registry "$REGISTRY" --gate-report "$GATE" \
    --parent-generation 5 --output "$RUN/model.bin" \
    --epochs 2 --batch-size 128 --learning-rate 0.0003 \
    --output-learning-rate 0.00003 --kappa-cp 1352.7871 \
    --search-mix 1.0 --search-mix-end 1.0 --value-weight 1 \
    --rank-weight 0 --game-rank-weight 0 --output-limit 0.20 \
    --gradient-clip-norm 1 --optimizer adagrad --threads 8 \
    --seed 20260827 --checkpoint-dir "$RUN/checkpoints" \
    --log "$RUN/logs/train.log" --save-last-epoch
fi

if [[ ! -s "$RUN/match-20k-500g/report.json" ]]; then
  python3 "$MATCH" \
    --candidate "$RUN/model.bin" --candidate-manifest "$RUN/model.bin.manifest.json" \
    --parent "$WEIGHTS" --parent-manifest "$PARENT_MANIFEST" \
    --registry "$REGISTRY" --openings "$OPENINGS" --pairs 250 \
    --nodes 20000 --max-depth 64 --max-plies 256 \
    --bootstrap-resamples 100000 --bootstrap-seed 20260829 \
    --output-dir "$RUN/match-20k-500g" \
    >"$RUN/logs/match.log" 2>&1
fi

date -Is > "$RUN/complete"

#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/runs/halfkp64-scratch-v1/h01-global-mix-4m-20260915}"
SHUFFLE_DIR="$RUN_DIR/train-shards"
SOURCE_RUN="$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen6-4m-20260913"
PARENT="$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen5-20260910/candidate/model-epoch24.bin"
PARENT_MANIFEST="$PARENT.manifest.json"
REGISTRY="$ROOT_DIR/training/halfkp64-scratch-v1/promotion_registry.json"
GATE="$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen5-20260910/match-confirm-20k-250pair/report.json"
VALID="$SOURCE_RUN/valid-4m.filtered.hkst"; TEST="$SOURCE_RUN/test-4m.filtered.hkst"
MODEL="$RUN_DIR/model.bin"; LOG="$RUN_DIR/train.csv"; CHECKPOINTS="$RUN_DIR/checkpoints"
WORKERS="${WORKERS:-12}"; SEED="${SEED:-20260915}"

need() { [[ -s "$1" ]] || { echo "missing: $1" >&2; exit 2; }; }
need "$PARENT"; need "$PARENT_MANIFEST"; need "$REGISTRY"; need "$GATE"; need "$VALID"; need "$TEST"
need "$RUN_DIR/shuffle-report.json"
mapfile -t TRAIN_SHARDS < <(find "$SHUFFLE_DIR" -maxdepth 1 -type f -name 'train-*.hkst' | sort)
(( ${#TRAIN_SHARDS[@]} == 32 )) || { echo "expected 32 mixed shards" >&2; exit 2; }

if [[ ! -s "$RUN_DIR/config.json" ]]; then
  python3 - "$RUN_DIR/config.json" "$VALID" "$TEST" "${TRAIN_SHARDS[@]}" <<'PY'
import hashlib, json, pathlib, sys
out, valid, test, *train = map(pathlib.Path, sys.argv[1:])
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
x = {
    "schema": "halfkp_h01_global_mix_train_v1",
    "seed": 20260915,
    "parent_model_sha256": "162e17bacea3de80ecdbbdae765ca1157246c4febb6fa6a15f8d36b900d011dc",
    "train_records": 4000000,
    "train_shards": [{"path": str(path), "sha256": sha(path)} for path in train],
    "valid": {"path": str(valid), "sha256": sha(valid)},
    "test": {"path": str(test), "sha256": sha(test)},
    "shuffle_report": {"path": "shuffle-report.json", "sha256": sha(out.parent / "shuffle-report.json")},
    "epochs_primary": 16,
    "epochs_secondary": 32,
    "workers": int(__import__("os").environ.get("WORKERS", "12")),
    "teacher_target": {"search_mix": 0.8, "kappa_cp": 1352.7871, "loss_power": 2.5},
}
out.write_text(json.dumps(x, indent=2) + "\n")
PY
fi

run_stage() {
  local target="$1"
  local marker="$CHECKPOINTS/selected-through-$target.bin"
  [[ -s "$marker" ]] && return
  mkdir -p "$CHECKPOINTS"
  local args=(--init "$PARENT" --model-manifest "$PARENT_MANIFEST" --promotion-registry "$REGISTRY"
    --gate-report "$GATE" --parent-generation 6 --output "$MODEL" --epochs "$target"
    --batch-size 128 --learning-rate .0003 --output-learning-rate .00003 --kappa-cp 1352.7871
    --search-mix .8 --search-mix-end .8 --value-weight 1 --rank-weight 0 --game-rank-weight 0
    --game-vs-search-weight 0 --loss-power 2.5 --gradient-clip-norm 1 --output-limit .20
    --optimizer adagrad --threads "$WORKERS" --seed "$SEED" --early-stop-patience 1000000
    --min-valid-improvement .000001 --swa-start-epoch 1000000 --checkpoint-dir "$CHECKPOINTS"
    --log "$LOG" --max-train-records 4000000 --valid "$VALID" --test "$TEST")
  for shard in "${TRAIN_SHARDS[@]}"; do args+=(--train "$shard"); done
  if [[ -s "$MODEL.manifest.json" && -s "$CHECKPOINTS/current.binary" ]]; then
    args+=(--resume)
  fi
  target/release/halfkp_search_train "${args[@]}"
  need "$MODEL"; need "$MODEL.manifest.json"
  cp "$MODEL" "$marker"; cp "$MODEL.manifest.json" "$marker.manifest.json"
}

run_stage 16
run_stage 32
echo "mixed HalfKP training complete: $RUN_DIR"

#!/usr/bin/env bash
set -Eeuo pipefail

# Resume-safe v4 value-training pipeline.  T06/T07 provide raw JSONL and
# frozen HKST inputs; this script owns sharding, provenance, staged training,
# and checkpoints. It never falls back to v2.5.0/KPP artifacts.
export RUST_FONTCONFIG_DLOPEN="${RUST_FONTCONFIG_DLOPEN:-1}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_DIR="${RUN_DIR:-$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen5-20260910}"
RAW_DIR="$RUN_DIR/raw"
SHARD_DIR="$RUN_DIR/shards-50k"
TEACHER_DIR="$RUN_DIR/teachers-100k"
CHECKPOINT_DIR="$RUN_DIR/checkpoints"
PARENT="$ROOT_DIR/runs/halfkp64-scratch-v1/phase3-2m-preserved-target-20260825/model-2m-100k.bin"
PARENT_MANIFEST="$PARENT.manifest.json"
REGISTRY="$ROOT_DIR/training/halfkp64-scratch-v1/promotion_registry.json"
GATE_REPORT="$ROOT_DIR/runs/halfkp64-scratch-v1/phase3-2m-preserved-target-20260825/match-20k-1000g-gen5-reserve-20260825/report.json"
VALID_FILTERED="$RUN_DIR/valid.filtered.hkst"
LOG="$RUN_DIR/train.csv"
CONFIG="$RUN_DIR/config.json"
MODEL="$RUN_DIR/model.bin"
WORKERS="${WORKERS:-12}"
MAX_EPOCH="${MAX_EPOCH:-16}"
SEED="${SEED:-20260910}"

require_file() {
  [[ -s "$1" ]] || { echo "required artifact is missing or empty: $1" >&2; exit 2; }
}

validate_parent() {
  require_file "$PARENT"; require_file "$PARENT_MANIFEST"
  require_file "$REGISTRY"; require_file "$GATE_REPORT"
  local actual
  actual="$(sha256sum "$PARENT" | awk '{print $1}')"
  [[ "$actual" == "61c05fe287cb0c0f9b434194802e07c835476b737a37bb9d4063143c63e28637" ]] || {
    echo "unexpected v4 parent SHA: $actual" >&2; exit 2;
  }
}

validate_raw() {
  require_file "$RAW_DIR/manifest.json"
  python3 - "$RAW_DIR/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
stats = manifest.get("stats", {})
train = int(stats.get("train", {}).get("records", 0))
if train < 1_250_000:
    raise SystemExit(f"v4 raw train records are {train}; need at least 1250000")
for name in ("train", "valid", "test"):
    path = manifest_path.parent / f"{name}.jsonl"
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"raw dataset is missing non-empty {path}")
if manifest.get("freeze_base_eval") is not True:
    raise SystemExit("v4 raw dataset must freeze the prior evaluation split")
PY
}

write_config() {
  mkdir -p "$RUN_DIR"
  python3 - "$CONFIG" "$RUN_DIR" "$PARENT" "$PARENT_MANIFEST" "$REGISTRY" "$GATE_REPORT" "$WORKERS" "$MAX_EPOCH" "$SEED" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

out, run_dir, parent, parent_manifest, registry, gate_report, workers, max_epoch, seed = sys.argv[1:]
def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

payload = {
    "schema": "halfkp_v4_value_runner_config_v1",
    "lineage_id": "halfkp64-scratch-v1",
    "parent_generation": 5,
    "parent_model": {"path": parent, "sha256": sha(parent)},
    "parent_manifest": {"path": parent_manifest, "sha256": sha(parent_manifest)},
    "promotion_registry": {"path": registry, "sha256": sha(registry)},
    "parent_gate_report": {"path": gate_report, "sha256": sha(gate_report)},
    "feature_profile": "halfkp64",
    "teacher_semantics": {"version": 4, "id": "deep-search-target-v1"},
    "teacher": {"value_only": True, "qsearch_leaf": True, "teacher_depth": 8,
                 "min_completed_depth": 1, "max_nodes_per_search": 100000,
                 "jobs": int(workers)},
    "training": {"optimizer": "adagrad", "feature_learning_rate": 3e-4,
                  "output_learning_rate": 3e-5, "batch_size": 128,
                  "search_mix": 0.8, "value_weight": 1.0, "rank_weight": 0.0,
                  "game_rank_weight": 0.0, "game_vs_search_weight": 0.0,
                  "kappa_cp": 1352.7871, "loss_power": 2.5,
                  "gradient_clip_norm": 1.0, "output_limit": 0.20,
                  "seed": int(seed), "workers": int(workers),
                  "staged_targets": [4, 8, 16, 32], "requested_max_epoch": int(max_epoch)},
    "run_dir": str(Path(run_dir)),
    "external_evaluation_teacher": False,
}
Path(out).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

build_tools() {
  local missing=0 binary
  for binary in jsonl_shard halfkp_search_teacher halfkp_search_split_filter halfkp_search_train; do
    [[ -x "$ROOT_DIR/target/release/$binary" ]] || missing=1
  done
  if [[ "$missing" == 1 ]]; then
    cargo build --release --locked --features halfkp64,training-tools \
      --bin jsonl_shard --bin halfkp_search_teacher --bin halfkp_search_split_filter --bin halfkp_search_train
  fi
}

make_shards() {
  mkdir -p "$SHARD_DIR"
  local kind manifest input
  for kind in train valid test; do
    manifest="$SHARD_DIR/${kind}-manifest.json"
    [[ -s "$manifest" ]] && continue
    if compgen -G "$SHARD_DIR/${kind}-*.jsonl" >/dev/null; then
      echo "refusing to overwrite partial $kind shard set" >&2; return 1
    fi
    input="$RAW_DIR/$kind.jsonl"
    target/release/jsonl_shard --input "$input" --output-prefix "$SHARD_DIR/${kind}-" \
      --lines 50000 --suffix-width 4 --parent-manifest "$RAW_DIR/manifest.json"
  done
}

generate_teachers() {
  mkdir -p "$TEACHER_DIR"
  local index=0 kind input stem output
  for kind in train valid test; do
    for input in "$SHARD_DIR"/${kind}-*.jsonl; do
      [[ -e "$input" ]] || continue
      stem="${input##*/}"; stem="${stem%.jsonl}"; output="$TEACHER_DIR/${stem}.hkst"
      if [[ -s "$output.manifest.json" ]]; then index=$((index + 1)); continue; fi
      if [[ -e "$output" || -e "$output.tmp" ]]; then
        echo "refusing to overwrite partial teacher $output" >&2; return 1
      fi
      target/release/halfkp_search_teacher --weights "$PARENT" --input "$input" \
        --parent-manifest "$PARENT_MANIFEST" --model-manifest "$PARENT_MANIFEST" \
        --promotion-registry "$REGISTRY" --gate-report "$GATE_REPORT" --parent-generation 5 \
        --output "$output" --value-only --qsearch-leaf --teacher-depth 8 \
        --min-completed-depth 1 --max-nodes-per-search 100000 --chunk-size 128 \
        --jobs "$WORKERS" --seed "$((SEED + index))"
      index=$((index + 1))
    done
  done
}

filter_validation() {
  [[ -s "$VALID_FILTERED.manifest.json" ]] && return
  [[ ! -e "$VALID_FILTERED" && ! -e "$VALID_FILTERED.tmp" ]] || {
    echo "refusing to overwrite partial filtered validation" >&2; return 1;
  }
  local args=(--output "$VALID_FILTERED" --report "$RUN_DIR/valid.filter.json") path
  for path in "$TEACHER_DIR"/train-*.hkst; do [[ -e "$path" ]] && args+=(--train "$path"); done
  for path in "$TEACHER_DIR"/valid-*.hkst; do [[ -e "$path" ]] && args+=(--valid "$path"); done
  ((${#args[@]} > 4)) || { echo "teacher shards are missing" >&2; return 1; }
  target/release/halfkp_search_split_filter "${args[@]}"
}

train_to_epoch() {
  local target_epoch="$1" resume="$2" snapshot="$CHECKPOINT_DIR/model-epoch-${1}.bin"
  [[ -s "$snapshot" ]] && return
  mkdir -p "$CHECKPOINT_DIR"
  local args=() path
  for path in "$TEACHER_DIR"/train-*.hkst; do [[ -e "$path" ]] && args+=(--train "$path"); done
  args+=(--valid "$VALID_FILTERED")
  for path in "$TEACHER_DIR"/test-*.hkst; do [[ -e "$path" ]] && args+=(--test "$path"); done
  args+=(--init "$PARENT" --model-manifest "$PARENT_MANIFEST" --promotion-registry "$REGISTRY"
    --gate-report "$GATE_REPORT" --parent-generation 5 --output "$MODEL" --epochs "$target_epoch"
    --batch-size 128 --learning-rate 0.0003 --output-learning-rate 0.00003 --kappa-cp 1352.7871
    --search-mix 0.8 --search-mix-end 0.8 --value-weight 1 --rank-weight 0 --game-rank-weight 0
    --game-vs-search-weight 0 --loss-power 2.5 --gradient-clip-norm 1 --output-limit 0.20
    --optimizer adagrad --threads "$WORKERS" --seed "$SEED" --checkpoint-dir "$CHECKPOINT_DIR"
    --save-last-epoch --log "$LOG")
  if [[ "$resume" == 1 ]]; then args+=(--resume)
  elif [[ -e "$MODEL" || -e "$MODEL.tmp" ]]; then
    echo "refusing to overwrite partial model $MODEL" >&2; return 1
  fi
  target/release/halfkp_search_train "${args[@]}"
  require_file "$MODEL"
  cp "$MODEL" "$snapshot"
  cp "$MODEL.manifest.json" "$snapshot.manifest.json"
}

decide_epoch32() {
  python3 - "$LOG" "$RUN_DIR/epoch-32-decision.json" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

log_path, decision_path = sys.argv[1:]
rows = {}
with open(log_path, newline="", encoding="utf-8") as source:
    for row in csv.DictReader(source):
        rows[int(row["epoch"])] = row
decision = {"schema": "halfkp_v4_epoch32_decision_v1", "eligible": False, "reasons": []}
for epoch in (8, 16):
    if epoch not in rows:
        decision["reasons"].append(f"epoch {epoch} metrics are missing")
if not decision["reasons"]:
    r8, r16 = rows[8], rows[16]
    checks = {
        "valid_value_loss_improves": float(r16["valid_value_loss"]) < float(r8["valid_value_loss"]) - 1e-6,
        "model_brier_not_worse": float(r16["valid_model_result_brier"]) <= float(r8["valid_model_result_brier"]) + 1e-4,
        "model_logloss_not_worse": float(r16["valid_model_result_logloss"]) <= float(r8["valid_model_result_logloss"]) + 1e-4,
    }
    decision["checks"] = checks
    decision["eligible"] = all(checks.values()) and all(
        math.isfinite(float(r16[key])) for key in ("valid_value_loss", "valid_model_result_brier", "valid_model_result_logloss")
    )
    if not decision["eligible"]:
        decision["reasons"].append("fixed 16-vs-8 continuation criteria were not met")
Path(decision_path).write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("epoch32 eligible" if decision["eligible"] else "epoch32 skipped: " + "; ".join(decision["reasons"]))
PY
}

validate_parent
validate_raw
build_tools
write_config
make_shards
generate_teachers
filter_validation
train_to_epoch 4 0
if ((MAX_EPOCH >= 8)); then train_to_epoch 8 1; fi
if ((MAX_EPOCH >= 16)); then train_to_epoch 16 1; fi
if ((MAX_EPOCH >= 32)); then
  if decide_epoch32 | tee "$RUN_DIR/epoch-32-decision.log" | grep -q '^epoch32 eligible'; then
    train_to_epoch 32 1
  fi
fi
echo "v4 gen5 value pipeline complete: $RUN_DIR"

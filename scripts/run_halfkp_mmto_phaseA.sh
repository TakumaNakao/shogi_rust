#!/usr/bin/env bash
set -euo pipefail

# Phase A: meaningful-dose PV-regenerated MMTO from the accepted scratch head.
#
# This script deliberately names every immutable input.  It never reads a
# v2.5.0/KPP artifact and refuses to replace an existing run directory.  The
# trainer is invoked one outer at a time so that a restart can resume the
# active checkpoint and the supervisor can stop after two fresh-valid loss
# regressions.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${HALFKP_MMTO_PHASEA_RUN_DIR:-${ROOT}/runs/halfkp64-scratch-v1/phaseA-mmto-500k-gen5-20260830}"
DATA_DIR="${ROOT}/runs/halfkp64-scratch-v1/phase3-2m-highrate-20260829/raw"
TRAIN="${DATA_DIR}/train.jsonl"
VALID="${DATA_DIR}/valid.jsonl"
DATASET_MANIFEST="${DATA_DIR}/manifest.json"
INIT="${ROOT}/runs/halfkp64-scratch-v1/phase3-2m-preserved-target-20260825/model-2m-100k.bin"
MODEL_MANIFEST="${INIT}.manifest.json"
REGISTRY="${ROOT}/training/halfkp64-scratch-v1/promotion_registry.json"
GATE_REPORT="${ROOT}/runs/halfkp64-scratch-v1/phase3-2m-preserved-target-20260825/match-20k-1000g-gen5-reserve-20260825/report.json"
BIN="${ROOT}/target/release/halfkp_mmto_train"
JOBS="${HALFKP_MMTO_PHASEA_JOBS:-8}"
MAX_TRAIN_ROOTS=500000
MAX_VALID_ROOTS=50000
CHECKPOINT_ROOTS=50000
OUTER_TOTAL=4
EXPECTED_PARENT_SHA="61c05fe287cb0c0f9b434194802e07c835476b737a37bb9d4063143c63e28637"
EXPECTED_GATE_SHA="3e8ed317d96ca64d60b188824be953cfd0656f8e1b0a3ad2a2bb1f2337627fab"
EXPECTED_DATASET_SHA="a0f9b65f1ab9fd2178157b18eb551e0f08ab0bda8a0d7990517f4ee437791269"

sha256() {
  sha256sum "$1" | awk '{print $1}'
}

require_file() {
  test -f "$1" || { echo "missing required input: $1" >&2; exit 1; }
}

for path in "$TRAIN" "$VALID" "$DATASET_MANIFEST" "$INIT" "$MODEL_MANIFEST" \
  "$REGISTRY" "$GATE_REPORT" "$BIN"; do
  require_file "$path"
done

[[ "$(sha256 "$INIT")" == "$EXPECTED_PARENT_SHA" ]] || {
  echo "accepted gen5 model SHA mismatch" >&2
  exit 1
}
[[ "$(sha256 "$GATE_REPORT")" == "$EXPECTED_GATE_SHA" ]] || {
  echo "accepted gen5 Gate report SHA mismatch" >&2
  exit 1
}
[[ "$(sha256 "$DATASET_MANIFEST")" == "$EXPECTED_DATASET_SHA" ]] || {
  echo "Phase A dataset manifest SHA mismatch" >&2
  exit 1
}

if [[ -e "$RUN_DIR" ]]; then
  # A previous invocation is resumable only when its checkpoint exists.  No
  # flag can make this script overwrite a different or incomplete run.
  if [[ ! -f "$RUN_DIR/checkpoint.json" ]]; then
    echo "refusing to reuse an uninitialised run directory: $RUN_DIR" >&2
    exit 1
  fi
else
  mkdir -p "$RUN_DIR"
fi

if [[ -f "$RUN_DIR/phaseA-stopped.json" ]]; then
  echo "Phase A is marked stopped; inspect $RUN_DIR/phaseA-stopped.json" >&2
  exit 0
fi

DATASET_SHA="$(sha256 "$DATASET_MANIFEST")"
PARENT_MANIFEST_SHA="$(sha256 "$MODEL_MANIFEST")"
REGISTRY_SHA="$(sha256 "$REGISTRY")"
BIN_SHA="$(sha256 "$BIN")"

# The source JSONL order and the root limits define a frozen, reproducible
# subset.  Record the rule and verify that the selected prefix has no exact
# (game_key,SFEN) duplicates before searching it.  The trainer itself remains
# the authority for schema, legal-move, split, and artifact validation.
python3 - "$RUN_DIR/subset-policy.json" "$TRAIN" "$VALID" "$DATASET_SHA" <<'PY'
import json
import sys
from pathlib import Path

out, train_path, valid_path, manifest_sha = sys.argv[1:]
limits = ((train_path, "train", 500_000), (valid_path, "valid", 50_000))
summary = []
for path, split, limit in limits:
    seen = set()
    rows = 0
    games = set()
    with Path(path).open() as handle:
        for line in handle:
            if rows >= limit:
                break
            record = json.loads(line)
            key = (record.get("game_key"), record.get("sfen"))
            if key in seen:
                raise SystemExit(f"duplicate {split} root in frozen prefix: {key!r}")
            seen.add(key)
            games.add(record.get("game_key"))
            rows += 1
    summary.append({"split": split, "prefix_records": rows, "unique_roots": len(seen), "games": len(games)})

payload = {
    "schema": "halfkp_mmto_phaseA_subset_policy_v1",
    "selection": "source JSONL order, first max_train_roots/max_valid_roots eligible records",
    "train_limit": 500_000,
    "valid_limit": 50_000,
    "dataset_manifest_sha256": manifest_sha,
    "source_train": str(Path(train_path).resolve()),
    "source_valid": str(Path(valid_path).resolve()),
    "prefix_summary": summary,
}
Path(out).write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, sort_keys=True))
PY

if [[ ! -f "$RUN_DIR/input-sha256.txt" ]]; then
  {
    echo "dataset_manifest $DATASET_SHA"
    echo "parent_model $(sha256 "$INIT")"
    echo "parent_model_manifest $PARENT_MANIFEST_SHA"
    echo "parent_gate_report $(sha256 "$GATE_REPORT")"
    echo "promotion_registry $REGISTRY_SHA"
    echo "trainer_binary $BIN_SHA"
  } > "$RUN_DIR/input-sha256.txt"
fi

if [[ "${HALFKP_MMTO_PHASEA_DRY_RUN:-0}" == 1 ]]; then
  echo "Phase A preflight passed (dry run; no trainer started)"
  exit 0
fi

LOG="$RUN_DIR/phaseA-supervisor.log"
metric() {
  python3 - "$1" <<'PY'
import json
import sys
value = json.loads(open(sys.argv[1]).read())["valid"]["mean_loss_per_pair"]
if not isinstance(value, (int, float)):
    raise SystemExit("mean_loss_per_pair is not numeric")
print(f"{value:.12g}")
PY
}

completed_outer() {
  if [[ ! -f "$RUN_DIR/checkpoint.json" ]]; then
    echo 0
    return
  fi
  python3 - "$RUN_DIR/checkpoint.json" <<'PY'
import json
import sys
print(int(json.loads(open(sys.argv[1]).read()).get("completed_outer", 0)))
PY
}

run_outer() {
  local outer="$1"
  local resume=()
  if [[ -f "$RUN_DIR/checkpoint.json" ]]; then
    resume=(--resume)
  fi
  echo "[$(date --iso-8601=seconds)] starting outer=${outer} resume=${#resume[@]}" | tee -a "$LOG"
  "$BIN" \
    --train "$TRAIN" \
    --valid "$VALID" \
    --dataset-manifest "$DATASET_MANIFEST" \
    --expected-dataset-manifest-sha256 "$DATASET_SHA" \
    --init "$INIT" \
    --model-manifest "$MODEL_MANIFEST" \
    --promotion-registry "$REGISTRY" \
    --gate-report "$GATE_REPORT" \
    --parent-generation 5 \
    --output-dir "$RUN_DIR" \
    --outer-iterations "$outer" \
    --search-block-roots 512 \
    --jobs "$JOBS" \
    --pair-batch-size 8192 \
    --learning-rate 0.0001 \
    --output-learning-rate-ratio 0.1 \
    --gradient-clip-norm 1.0 \
    --tau-cp 36.63 \
    --loss-weight 1.0 \
    --wdl-weight 0.0 \
    --feature-limit 1.0 \
    --output-limit 0.20 \
    --max-feature-delta-per-outer 0.02 \
    --max-output-delta-per-outer 0.004 \
    --max-train-roots "$MAX_TRAIN_ROOTS" \
    --max-valid-roots "$MAX_VALID_ROOTS" \
    --pv-probe-roots 1024 \
    --checkpoint-roots "$CHECKPOINT_ROOTS" \
    --objective mmto-only \
    "${resume[@]}" 2>&1 | tee -a "$LOG"
}

for outer in $(seq 1 "$OUTER_TOTAL"); do
  done_outer="$(completed_outer)"
  if (( done_outer < outer )); then
    run_outer "$outer"
  fi
  [[ -f "$RUN_DIR/metrics/outer_$(printf '%04d' "$outer").json" ]] || {
    echo "outer ${outer} completed without metrics" >&2
    exit 1
  }
  current="$(metric "$RUN_DIR/metrics/outer_$(printf '%04d' "$outer").json")"
  echo "[$(date --iso-8601=seconds)] outer=${outer} valid_fresh_loss=${current}" | tee -a "$LOG"

  # Stop only after two consecutive increases (outer n-1 > n-2 and outer n >
  # n-1).  This is an audit stop, not a candidate rejection or promotion.
  if (( outer >= 3 )); then
    previous="$(metric "$RUN_DIR/metrics/outer_$(printf '%04d' "$((outer - 1))").json")"
    before_previous="$(metric "$RUN_DIR/metrics/outer_$(printf '%04d' "$((outer - 2))").json")"
    if python3 - "$before_previous" "$previous" "$current" <<'PY'
import sys
a, b, c = map(float, sys.argv[1:])
raise SystemExit(0 if (b > a and c > b) else 1)
PY
    then
      python3 - "$RUN_DIR/phaseA-stopped.json" "$outer" "$before_previous" "$previous" "$current" <<'PY'
import json
import sys
from pathlib import Path
out, outer, a, b, c = sys.argv[1:]
Path(out).write_text(json.dumps({
    "schema": "halfkp_mmto_phaseA_stopped_v1",
    "reason": "fresh-valid mean_loss_per_pair increased for two consecutive outer transitions",
    "outer": int(outer),
    "losses": [float(a), float(b), float(c)],
}, indent=2) + "\n")
PY
      echo "stopping after two consecutive fresh-valid regressions" | tee -a "$LOG"
      exit 0
    fi
  fi
done

echo "[$(date --iso-8601=seconds)] Phase A completed all ${OUTER_TOTAL} outer iterations" | tee -a "$LOG"

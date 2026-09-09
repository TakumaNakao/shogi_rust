#!/usr/bin/env bash
set -euo pipefail

# Freeze the best safe Phase-A checkpoint and run the strict scratch-parent
# Gate.  The paired wrapper is intentionally sequential: the Gate's fixed
# node budget and deterministic opening-pair semantics take precedence over
# throughput.  The 12-worker setting is used by the subsequent training phase.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASEA="${ROOT}/runs/halfkp64-scratch-v1/phaseA-mmto-500k-gen5-20260830"
FREEZE_DIR="${PHASEA}/candidate-outer4"
GATE_DIR="${FREEZE_DIR}/match-20k-500g"
SOURCE_MODEL="${PHASEA}/model.bin"
SOURCE_METRICS="${PHASEA}/metrics/outer_0004.json"
CANDIDATE="${FREEZE_DIR}/model.bin"
CANDIDATE_MANIFEST="${FREEZE_DIR}/model.bin.manifest.json"
SELECTION="${FREEZE_DIR}/selection.json"
PARENT="${ROOT}/runs/halfkp64-scratch-v1/phase3-2m-preserved-target-20260825/model-2m-100k.bin"
PARENT_MANIFEST="${PARENT}.manifest.json"
REGISTRY="${ROOT}/training/halfkp64-scratch-v1/promotion_registry.json"
OPENINGS="${ROOT}/training/halfkp64-scratch-v1/mmto_dev_openings_seed20260731_250/openings.jsonl"
MATCH="${ROOT}/scripts/run_halfkp_paired_match.py"
BINARY="${ROOT}/target/release/halfkp_promotion_match"
EXPECTED_CANDIDATE_SHA="9a1d477ba94137134ec2359243d8e2a45a5928045e65a029894551d7450a3425"
EXPECTED_PARENT_SHA="61c05fe287cb0c0f9b434194802e07c835476b737a37bb9d4063143c63e28637"
EXPECTED_PAIRS=250

sha256() { sha256sum "$1" | awk '{print $1}'; }
require_file() { test -f "$1" || { echo "missing required input: $1" >&2; exit 1; }; }

for path in "$SOURCE_MODEL" "$SOURCE_METRICS" "$PARENT" "$PARENT_MANIFEST" "$REGISTRY" \
  "$OPENINGS" "$MATCH" "$BINARY"; do
  require_file "$path"
done

[[ "$(sha256 "$SOURCE_MODEL")" == "$EXPECTED_CANDIDATE_SHA" ]] || {
  echo "Phase A outer-4 model SHA mismatch" >&2
  exit 1
}
[[ "$(sha256 "$PARENT")" == "$EXPECTED_PARENT_SHA" ]] || {
  echo "gen5 parent SHA mismatch" >&2
  exit 1
}

mkdir -p "$FREEZE_DIR"
if [[ ! -e "$CANDIDATE" ]]; then
  cp --reflink=auto "$SOURCE_MODEL" "$CANDIDATE"
  chmod a-w "$CANDIDATE"
elif [[ "$(sha256 "$CANDIDATE")" != "$EXPECTED_CANDIDATE_SHA" ]]; then
  echo "frozen candidate exists with unexpected SHA" >&2
  exit 1
fi

if [[ ! -e "$CANDIDATE_MANIFEST" ]]; then
  CANDIDATE_SHA="$(sha256 "$CANDIDATE")" \
  PARENT_SHA="$(sha256 "$PARENT")" \
  PARENT_MANIFEST_SHA="$(sha256 "$PARENT_MANIFEST")" \
  REGISTRY_SHA="$(sha256 "$REGISTRY")" \
  python3 - "$CANDIDATE_MANIFEST" "$CANDIDATE" "$SOURCE_METRICS" "$PHASEA" <<'PY'
import json
import os
import sys
from pathlib import Path

out, model, metrics, phasea = sys.argv[1:]
metrics_data = json.loads(Path(metrics).read_text())
candidate_sha = os.environ["CANDIDATE_SHA"]
payload = {
    "schema_version": 1,
    "schema": "halfkp_mmto_phaseA_candidate_manifest_v1",
    "lineage_id": "halfkp64-scratch-v1",
    "status": "offline_best_frozen_pending_gate",
    "source_run": str(Path(phasea).resolve()),
    "source_outer": 4,
    "source_metrics": str(Path(metrics).resolve()),
    "source_metrics_sha256": __import__("hashlib").sha256(Path(metrics).read_bytes()).hexdigest(),
    "parent_generation": 5,
    "parent_model_sha256": os.environ["PARENT_SHA"],
    "parent_manifest_sha256": os.environ["PARENT_MANIFEST_SHA"],
    "promotion_registry_sha256": os.environ["REGISTRY_SHA"],
    "model": {
        "path": str(Path(model).resolve()),
        "sha256": candidate_sha,
        "bytes": Path(model).stat().st_size,
    },
    "selection": {
        "criterion": "minimum fresh-valid mean_loss_per_pair among safe Phase-A outers",
        "fresh_valid_mean_loss_per_pair": metrics_data["valid"]["mean_loss_per_pair"],
        "fresh_valid_agreement": metrics_data["valid"]["agreement"],
        "safety_rejected_bounds": metrics_data["valid"]["rejected_bounds"],
        "pv_probe_best_move_unchanged": metrics_data["pv_probe_best_move_unchanged"],
    },
}
Path(out).write_text(json.dumps(payload, indent=2) + "\n")
PY
  chmod a-w "$CANDIDATE_MANIFEST"
else
  python3 - "$CANDIDATE_MANIFEST" "$CANDIDATE" "$EXPECTED_CANDIDATE_SHA" <<'PY'
import json, sys
from pathlib import Path
manifest, model, expected = sys.argv[1:]
payload = json.loads(Path(manifest).read_text())
if payload.get("lineage_id") != "halfkp64-scratch-v1" or payload.get("status") != "offline_best_frozen_pending_gate":
    raise SystemExit("frozen candidate manifest provenance mismatch")
if payload.get("model", {}).get("sha256") != expected:
    raise SystemExit("frozen candidate manifest SHA mismatch")
PY
fi

if [[ ! -e "$SELECTION" ]]; then
  python3 - "$SELECTION" "$SOURCE_METRICS" "$CANDIDATE" "$CANDIDATE_MANIFEST" <<'PY'
import hashlib, json, sys
from pathlib import Path
out, metrics, model, manifest = sys.argv[1:]
payload = {
    "schema": "halfkp_mmto_phaseA_candidate_selection_v1",
    "candidate": str(Path(model).resolve()),
    "candidate_sha256": hashlib.sha256(Path(model).read_bytes()).hexdigest(),
    "candidate_manifest": str(Path(manifest).resolve()),
    "source_metrics": str(Path(metrics).resolve()),
    "source_metrics_sha256": hashlib.sha256(Path(metrics).read_bytes()).hexdigest(),
    "selected_outer": 4,
    "criterion": "minimum fresh-valid mean_loss_per_pair; all parameter/safety checks must be zero",
}
Path(out).write_text(json.dumps(payload, indent=2) + "\n")
PY
  chmod a-w "$SELECTION"
fi

if [[ "${HALFKP_PHASEA_GATE_DRY_RUN:-0}" == 1 ]]; then
  echo "Phase-A candidate freeze passed (dry run; Gate not started)"
  exit 0
fi

if [[ -f "$GATE_DIR/completed.json" ]]; then
  echo "Phase-A candidate Gate already complete: $GATE_DIR"
  exit 0
fi

mkdir -p "$GATE_DIR"
exec python3 "$MATCH" \
  --binary "$BINARY" \
  --candidate "$CANDIDATE" \
  --candidate-manifest "$CANDIDATE_MANIFEST" \
  --parent "$PARENT" \
  --parent-manifest "$PARENT_MANIFEST" \
  --registry "$REGISTRY" \
  --openings "$OPENINGS" \
  --openings-format dataset-jsonl \
  --opening-selection first-unique \
  --opening-seed 0 \
  --pairs "$EXPECTED_PAIRS" \
  --nodes 20000 \
  --max-depth 64 \
  --max-plies 256 \
  --bootstrap-resamples 100000 \
  --bootstrap-seed 20260909 \
  --output-dir "$GATE_DIR"

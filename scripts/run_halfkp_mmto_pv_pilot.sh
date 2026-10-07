#!/usr/bin/env bash
set -euo pipefail

# Bounded diagnostic for the next training route in the scratch lineage.
# The MMTO trainer refreshes qsearch PV leaves from the immutable model
# snapshot at every outer iteration; no KPP/v2.5.0 artifact is an input.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${HALFKP_MMTO_PV_RUN_DIR:-${ROOT}/runs/halfkp64-scratch-v1/phase4-mmto-pv-pilot-20260830}"
DATA_DIR="${ROOT}/runs/halfkp64-scratch-v1/phase3-2m-highrate-20260829/raw"
TRAIN="${DATA_DIR}/train.jsonl"
VALID="${DATA_DIR}/valid.jsonl"
DATASET_MANIFEST="${DATA_DIR}/manifest.json"
INIT="${ROOT}/runs/halfkp64-scratch-v1/unattended-mmto-full-lr1e-4-20260808-v1/archives/phase-c/outer_0004/model.bin"
MODEL_MANIFEST="${INIT}.manifest.json"
REGISTRY="${ROOT}/training/halfkp64-scratch-v1/promotion_registry.json"
GATE_REPORT="${ROOT}/training/halfkp64-scratch-v1/outer4_internal_gate_20260817.json"
JOBS="${HALFKP_MMTO_PV_JOBS:-8}"

for path in "${TRAIN}" "${VALID}" "${DATASET_MANIFEST}" "${INIT}" "${MODEL_MANIFEST}" "${REGISTRY}" "${GATE_REPORT}"; do
  test -f "${path}" || { echo "missing required input: ${path}" >&2; exit 1; }
done

if [[ -e "${RUN_DIR}" ]]; then
  echo "refusing to reuse existing run directory: ${RUN_DIR}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}"
sha256sum "${DATASET_MANIFEST}" > "${RUN_DIR}/dataset-manifest.sha256"
sha256sum "${INIT}" > "${RUN_DIR}/parent-model.sha256"
sha256sum "${MODEL_MANIFEST}" > "${RUN_DIR}/parent-model.manifest.sha256"
sha256sum "${REGISTRY}" > "${RUN_DIR}/promotion-registry.sha256"
sha256sum "${GATE_REPORT}" > "${RUN_DIR}/parent-gate-report.sha256"

exec "${ROOT}/target/release/halfkp_mmto_train" \
  --train "${TRAIN}" \
  --valid "${VALID}" \
  --dataset-manifest "${DATASET_MANIFEST}" \
  --expected-dataset-manifest-sha256 "$(cut -d' ' -f1 "${RUN_DIR}/dataset-manifest.sha256")" \
  --init "${INIT}" \
  --model-manifest "${MODEL_MANIFEST}" \
  --promotion-registry "${REGISTRY}" \
  --gate-report "${GATE_REPORT}" \
  --parent-generation 3 \
  --output-dir "${RUN_DIR}" \
  --outer-iterations 1 \
  --search-block-roots 512 \
  --jobs "${JOBS}" \
  --pair-batch-size 8192 \
  --learning-rate 0.0001 \
  --output-learning-rate-ratio 0.1 \
  --gradient-clip-norm 1.0 \
  --tau-cp 36.63 \
  --feature-limit 1.0 \
  --output-limit 0.20 \
  --max-feature-delta-per-outer 0.02 \
  --max-output-delta-per-outer 0.004 \
  --max-train-roots 10000 \
  --max-valid-roots 5000 \
  --pv-probe-roots 1024 \
  --checkpoint-roots 1000

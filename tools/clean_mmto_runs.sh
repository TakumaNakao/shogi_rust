#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script. Run it as: bash tools/clean_mmto_runs.sh"
  return 2
fi

cd "$(dirname "$0")/.."

RUNS_DIR="data/mmto/runs"

if [[ ! -d "$RUNS_DIR" ]]; then
  echo "$RUNS_DIR does not exist."
  exit 0
fi

echo "Target: $RUNS_DIR"
du -sh "$RUNS_DIR" || true
echo
echo "This removes generated MMTO run logs, dumps, checkpoints, and candidate weights."
echo "It does not remove policy_weights_v2.1.0.binary."

if [[ "${1:-}" != "--yes" ]]; then
  read -r -p "Delete everything under $RUNS_DIR ? Type DELETE to continue: " ANSWER
  if [[ "$ANSWER" != "DELETE" ]]; then
    echo "Canceled."
    exit 0
  fi
fi

find "$RUNS_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
mkdir -p "$RUNS_DIR"

echo "Cleaned $RUNS_DIR"
du -sh "$RUNS_DIR" || true

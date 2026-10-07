#!/usr/bin/env bash
set -Eeuo pipefail

# Wait for the strict 1.18M-position Gate. If it is not promoted, build a
# second immutable corpus with a wider WDoor rating filter and resume the same
# preserved-target teacher/training pipeline. The script is one-shot and
# fail-closed: an unexpected match exit never starts training.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MATCH_DIR="$ROOT_DIR/runs/halfkp64-scratch-v1/phase3-2m-preserved-target-20260825/match-20k-1000g-gen5-reserve-20260825"
MATCH_PID_FILE="$MATCH_DIR/match.pid"
REPORT="$MATCH_DIR/report.json"
NEXT_RUN="$ROOT_DIR/runs/halfkp64-scratch-v1/phase3-2m-wide-20260825"

while [[ ! -s "$REPORT" ]]; do
  if [[ -s "$MATCH_PID_FILE" ]] && ! kill -0 "$(cat "$MATCH_PID_FILE")" 2>/dev/null; then
    echo "match exited without an aggregate report" >&2
    exit 1
  fi
  sleep 60
done

if python3 - "$REPORT" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if report.get("promotion_gate_passed") is True else 1)
PY
then
  echo "1.18M candidate passed the strict Gate; no expansion required"
  exit 0
fi

marker="$NEXT_RUN/.expansion-started"
mkdir -p "$NEXT_RUN"
if ! mkdir "$marker" 2>/dev/null; then
  echo "2M expansion is already started: $NEXT_RUN"
  exit 0
fi

mkdir -p "$NEXT_RUN/raw"
nohup setsid target/release/dataset_build \
  --input data/wdoor/extract/2018 --input data/wdoor/extract/2019 \
  --input data/wdoor/extract/2020 --input data/wdoor/extract/2021 \
  --input data/wdoor/extract/2022 --input data/wdoor/extract/2023 \
  --input data/wdoor/extract/2024 --input data/wdoor/extract/2025 \
  --output-dir "$NEXT_RUN/raw" \
  --license-id user-authorized:wdoor-public-archive \
  --source-uri https://wdoor.c.u-tokyo.ac.jp/shogi/ \
  --seed 20260826 --shuffle-games --valid-percent 5 --test-percent 5 \
  --target-opening-records 500000 --target-middle-records 1000000 \
  --target-late-records 500000 --require-targets --target-minimum-percent 95 \
  --phase-records-per-game 5 10 10 --min-ply 8 --max-ply 200 \
  --min-player-rate 3000 --min-opponent-rate 3000 --known-result-only \
  > "$NEXT_RUN/raw-build.log" 2>&1 < /dev/null &
echo $! > "$NEXT_RUN/raw-build.pid"

RUN_DIR="$NEXT_RUN" MIN_RAW_RECORDS=1900000 \
  nohup setsid scripts/run_halfkp_phase3_2m_preserved_target.sh \
  > "$NEXT_RUN/phase3-pipeline.log" 2>&1 < /dev/null &
echo $! > "$NEXT_RUN/phase3-pipeline.pid"
echo "started wider 2M corpus and resumable pipeline: $NEXT_RUN"

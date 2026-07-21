#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script. Run it as: bash tools/make_wdoor_mmto_positions.sh"
  return 2
fi

cd "$(dirname "$0")/.."

INPUT_ROOT="${INPUT_ROOT:-data/wdoor/extract}"
YEARS="${YEARS:-2023 2024 2025 2026}"
MIN_PLAYER_RATE="${MIN_PLAYER_RATE:-4000}"
MIN_PLY="${MIN_PLY:-16}"
MAX_PLY="${MAX_PLY:-120}"
MAX_RECORDS="${MAX_RECORDS:-200000}"
SEED="${SEED:-9601}"
OUT="${OUT:-data/mmto/positions/wdoor2023_2026_r4000_p16_120.sfen}"
TMP_DIR="${TMP_DIR:-data/mmto/positions/tmp_wdoor_mmto_$(date -u +%Y%m%d_%H%M%S)}"

mkdir -p "$TMP_DIR" "$(dirname "$OUT")"

inputs=()
for year in $YEARS; do
  path="$INPUT_ROOT/$year"
  if [[ -d "$path" ]]; then
    inputs+=(--input "$path")
  else
    echo "missing Wdoor year directory: $path" >&2
  fi
done

if [[ "${#inputs[@]}" == "0" ]]; then
  echo "no Wdoor input directories found under $INPUT_ROOT" >&2
  exit 1
fi

echo "Creating MMTO position set from Wdoor CSA."
echo "YEARS=$YEARS"
echo "MIN_PLAYER_RATE=$MIN_PLAYER_RATE MIN_PLY=$MIN_PLY MAX_PLY=$MAX_PLY MAX_RECORDS=$MAX_RECORDS"
echo "OUT=$OUT"

env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --features training-tools --bin csa_policy_dump

env RUST_FONTCONFIG_DLOPEN=1 target/release/csa_policy_dump \
  "${inputs[@]}" \
  --train-output "$TMP_DIR/policy_records.jsonl" \
  --valid-output "$TMP_DIR/unused_valid.jsonl" \
  --seed "$SEED" \
  --valid-percent 0 \
  --max-records "$MAX_RECORDS" \
  --min-ply "$MIN_PLY" \
  --max-ply "$MAX_PLY" \
  --min-player-rate "$MIN_PLAYER_RATE" \
  --decisive-only

python3 - "$TMP_DIR/policy_records.jsonl" "$OUT" <<'PY'
import json
import sys

source, output = sys.argv[1], sys.argv[2]
seen = set()
written = 0

with open(source, "r", encoding="utf-8") as handle, open(output, "w", encoding="utf-8") as out:
    for line in handle:
        if not line.strip():
            continue
        try:
            sfen = json.loads(line)["sfen"].strip()
        except Exception:
            continue
        if not sfen or sfen in seen:
            continue
        seen.add(sfen)
        out.write(sfen + "\n")
        written += 1

print(f"unique_sfens={written}")
PY

rm -rf "$TMP_DIR"
wc -l "$OUT"
echo "Wrote $OUT"

#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script. Run it as: bash tools/run_bench_failure_mining_suite.sh"
  return 2
fi

cd "$(dirname "$0")/.."

RUN_DIR="${RUN_DIR:-/tmp/bench_failure_mining_suite_$(date -u +%Y%m%d_%H%M%S)}"
WEIGHTS="${WEIGHTS:-policy_weights_v2.1.0.binary}"
TEACHER_WEIGHTS="${TEACHER_WEIGHTS:-$WEIGHTS}"
TAIL_PLIES="${TAIL_PLIES:-10}"
TIMED_DEPTH="${TIMED_DEPTH:-4}"
TEACHER_DEPTH="${TEACHER_DEPTH:-5}"
TIME_LIMIT_MS="${TIME_LIMIT_MS:-100}"
BAD_REGRET_CP="${BAD_REGRET_CP:-200}"
ROOT_RESCUE_GOOD_REGRET_CP="${ROOT_RESCUE_GOOD_REGRET_CP:-80}"
ROOT_RESCUE_MIN_IMPROVEMENT_CP="${ROOT_RESCUE_MIN_IMPROVEMENT_CP:-200}"
TOP="${TOP:-20}"
MAX_RECORDS="${MAX_RECORDS:-0}"

DEFAULT_RECORD_DIRS="
/tmp/shogi_weight_bench_pv_sibling_3k_ultrasafe_11101
/tmp/shogi_weight_bench_mmto_replay100k_hard_10001
/tmp/shogi_weight_bench_pv_sibling_3k_ultrasafe_11001
/tmp/shogi_weight_bench_mmto_replay100k_hard_9901
/tmp/shogi_weight_bench_bonanza_root_500_stronger_leaf_11401
/tmp/shogi_weight_bench_bonanza_root_500_stronger_leaf_11301
"
RECORD_DIRS="${RECORD_DIRS:-$DEFAULT_RECORD_DIRS}"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "missing weights: $WEIGHTS" >&2
  exit 1
fi
if [[ ! -f "$TEACHER_WEIGHTS" ]]; then
  echo "missing teacher weights: $TEACHER_WEIGHTS" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"

echo "Starting bench failure mining suite."
echo "RUN_DIR=$RUN_DIR"
echo "WEIGHTS=$WEIGHTS"
echo "TEACHER_WEIGHTS=$TEACHER_WEIGHTS"
echo "TAIL_PLIES=$TAIL_PLIES TIMED_DEPTH=$TIMED_DEPTH TEACHER_DEPTH=$TEACHER_DEPTH TIME_LIMIT_MS=$TIME_LIMIT_MS"
echo "BAD_REGRET_CP=$BAD_REGRET_CP ROOT_RESCUE=$ROOT_RESCUE_GOOD_REGRET_CP/$ROOT_RESCUE_MIN_IMPROVEMENT_CP"

env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin bench_failure_miner

: > "$RUN_DIR/all_failures.jsonl"
: > "$RUN_DIR/all_timed_bad.sfen"
: > "$RUN_DIR/all_root_rescue.sfen"
: > "$RUN_DIR/record_dirs.txt"

run_count=0
while IFS= read -r record_dir; do
  [[ -z "$record_dir" ]] && continue
  if [[ ! -d "$record_dir" ]]; then
    echo "skip missing record-dir: $record_dir"
    continue
  fi
  usi_count="$(find "$record_dir" -maxdepth 1 -type f -name '*.usi' | wc -l)"
  if [[ "$usi_count" = "0" ]]; then
    echo "skip empty record-dir: $record_dir"
    continue
  fi

  label="$(basename "$record_dir")"
  out_dir="$RUN_DIR/$label"
  mkdir -p "$out_dir"
  echo "$record_dir" >> "$RUN_DIR/record_dirs.txt"
  echo "Mining $record_dir ($usi_count records)"

  cmd=(
    env RUST_FONTCONFIG_DLOPEN=1 target/release/bench_failure_miner
    --record-dir "$record_dir"
    --weights "$WEIGHTS"
    --teacher-weights "$TEACHER_WEIGHTS"
    --tail-plies "$TAIL_PLIES"
    --timed-depth "$TIMED_DEPTH"
    --teacher-depth "$TEACHER_DEPTH"
    --time-limit-ms "$TIME_LIMIT_MS"
    --bad-regret-cp "$BAD_REGRET_CP"
    --root-rescue-good-regret-cp "$ROOT_RESCUE_GOOD_REGRET_CP"
    --root-rescue-min-improvement-cp "$ROOT_RESCUE_MIN_IMPROVEMENT_CP"
    --top "$TOP"
    --jsonl-output "$out_dir/failures.jsonl"
    --export-timed-bad-sfens "$out_dir/timed_bad.sfen"
    --export-root-rescue-sfens "$out_dir/root_rescue.sfen"
  )
  if [[ "$MAX_RECORDS" != "0" ]]; then
    cmd+=(--max-records "$MAX_RECORDS")
  fi

  "${cmd[@]}" 2>&1 | tee "$out_dir/summary.txt"

  if [[ -s "$out_dir/failures.jsonl" ]]; then
    cat "$out_dir/failures.jsonl" >> "$RUN_DIR/all_failures.jsonl"
  fi
  if [[ -s "$out_dir/timed_bad.sfen" ]]; then
    cat "$out_dir/timed_bad.sfen" >> "$RUN_DIR/all_timed_bad.sfen"
  fi
  if [[ -s "$out_dir/root_rescue.sfen" ]]; then
    cat "$out_dir/root_rescue.sfen" >> "$RUN_DIR/all_root_rescue.sfen"
  fi
  run_count=$((run_count + 1))
done <<< "$RECORD_DIRS"

sort -u "$RUN_DIR/all_timed_bad.sfen" -o "$RUN_DIR/all_timed_bad.sfen"
sort -u "$RUN_DIR/all_root_rescue.sfen" -o "$RUN_DIR/all_root_rescue.sfen"

{
  echo "RUN_DIR=$RUN_DIR"
  echo "runs=$run_count"
  echo "all_failures_lines=$(wc -l < "$RUN_DIR/all_failures.jsonl")"
  echo "unique_timed_bad_sfens=$(grep -cve '^$' "$RUN_DIR/all_timed_bad.sfen" || true)"
  echo "unique_root_rescue_sfens=$(grep -cve '^$' "$RUN_DIR/all_root_rescue.sfen" || true)"
  if command -v jq >/dev/null 2>&1 && [[ -s "$RUN_DIR/all_failures.jsonl" ]]; then
    jq -s '{
      samples: length,
      actual_bad: map(select(.actual_bad == true)) | length,
      timed_bad: map(select(.timed_bad == true)) | length,
      root_rescuable: map(select(.root_rescuable == true)) | length,
      in_check: map(select(.in_check == true)) | length,
      legal_moves_le_3: map(select(.legal_moves <= 3)) | length,
      baseline_sweep_start: map(select(.baseline_sweep_start == true)) | length,
      actual_minus_timed_ge_200: map(select(.actual_minus_timed_regret_cp >= 200)) | length
    }' "$RUN_DIR/all_failures.jsonl"
  fi
} | tee "$RUN_DIR/suite_summary.txt"

echo "Done."

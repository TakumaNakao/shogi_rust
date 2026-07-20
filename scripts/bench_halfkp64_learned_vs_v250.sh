#!/usr/bin/env bash
set -Eeuo pipefail

export RUST_FONTCONFIG_DLOPEN="${RUST_FONTCONFIG_DLOPEN:-1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NEW_WEIGHTS="${NEW_WEIGHTS:-$ROOT_DIR/data/halfkp_search_learning_large_v1/policy_weights_halfkp64_large_adagrad.binary}"
BASELINE_WEIGHTS="${BASELINE_WEIGHTS:-$ROOT_DIR/policy_weights_halfkp64_kpp_distilled_v2.5.0.binary}"
POSITIONS="${POSITIONS:-$ROOT_DIR/taya36.sfen}"
GAMES="${GAMES:-200}"
TIME_LIMIT_MS="${TIME_LIMIT_MS:-1000}"
DEPTH="${DEPTH:-32}"
MAX_PLIES="${MAX_PLIES:-256}"
SEED="${SEED:-20260720}"
THREADS="${THREADS:-0}"
JOBS="${JOBS:-1}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/data/bench_halfkp64_learned_vs_v250_${TIME_LIMIT_MS}ms_${GAMES}g_seed${SEED}}"
FORCE="${FORCE:-0}"

if (( GAMES < 2 || GAMES % 2 != 0 )); then
  echo "GAMES must be an even number >= 2 for paired side swaps" >&2
  exit 2
fi
if (( THREADS < 0 || JOBS < 1 )); then
  echo "THREADS must be >= 0 and JOBS must be >= 1" >&2
  exit 2
fi
for path in "$NEW_WEIGHTS" "$BASELINE_WEIGHTS" "$POSITIONS"; do
  [[ -s "$path" ]] || { echo "required input not found: $path" >&2; exit 2; }
done
if [[ "$NEW_WEIGHTS" -ef "$BASELINE_WEIGHTS" ]]; then
  echo "NEW_WEIGHTS and BASELINE_WEIGHTS must be different files" >&2
  exit 2
fi
if [[ -e "$OUT_DIR/match.log" && "$FORCE" != "1" ]]; then
  echo "result already exists: $OUT_DIR/match.log" >&2
  echo "set FORCE=1 to replace it, or choose another OUT_DIR" >&2
  exit 2
fi

if [[ "$FORCE" == "1" ]]; then
  rm -rf "$OUT_DIR"
fi
mkdir -p "$OUT_DIR"

echo "[1/2] Building the corrected HalfKP-64 engine and benchmark harness"
cargo build --release --features halfkp64 --bin usi_engine --bin usi_benchmark

ENGINE="$ROOT_DIR/target/release/usi_engine"
BENCH="$ROOT_DIR/target/release/usi_benchmark"
RECORD_DIR="$OUT_DIR/games"

{
  echo "new_weights=$NEW_WEIGHTS"
  echo "new_sha256=$(sha256sum "$NEW_WEIGHTS" | awk '{print $1}')"
  echo "baseline_weights=$BASELINE_WEIGHTS"
  echo "baseline_sha256=$(sha256sum "$BASELINE_WEIGHTS" | awk '{print $1}')"
  echo "positions=$POSITIONS"
  echo "games=$GAMES"
  echo "time_limit_ms=$TIME_LIMIT_MS"
  echo "depth=$DEPTH"
  echo "max_plies=$MAX_PLIES"
  echo "seed=$SEED"
  echo "threads=$THREADS"
  echo "jobs=$JOBS"
  echo "git_rev=$(git rev-parse HEAD)"
} >"$OUT_DIR/config.txt"

echo "[2/2] Running $GAMES paired games at ${TIME_LIMIT_MS} ms/move"
"$BENCH" \
  --new-engine "$ENGINE" \
  --baseline-engine "$ENGINE" \
  --new-weights "$NEW_WEIGHTS" \
  --baseline-weights "$BASELINE_WEIGHTS" \
  --positions "$POSITIONS" \
  --games "$GAMES" \
  --depth "$DEPTH" \
  --time-limit-ms "$TIME_LIMIT_MS" \
  --new-threads "$THREADS" \
  --baseline-threads "$THREADS" \
  --max-plies "$MAX_PLIES" \
  --jobs "$JOBS" \
  --seed "$SEED" \
  --record-dir "$RECORD_DIR" | tee "$OUT_DIR/match.log"

echo "Results: $OUT_DIR"

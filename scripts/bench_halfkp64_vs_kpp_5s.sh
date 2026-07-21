#!/usr/bin/env bash
set -Eeuo pipefail

# plotters/fontconfig is an optional runtime dependency of the benchmark binary.
export RUST_FONTCONFIG_DLOPEN="${RUST_FONTCONFIG_DLOPEN:-1}"

# Paired 5-second match: games 1/2 use the same start position with sides swapped.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

POSITIONS="${POSITIONS:-$ROOT_DIR/taya36.sfen}"
GAMES="${GAMES:-20}"
TIME_LIMIT_MS="${TIME_LIMIT_MS:-5000}"
DEPTH="${DEPTH:-32}"
MAX_PLIES="${MAX_PLIES:-256}"
SEED="${SEED:-20260716}"
JOBS="${JOBS:-1}"
HALFKP_THREADS="${HALFKP_THREADS:-1}"
KPP_THREADS="${KPP_THREADS:-1}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/data/bench_halfkp64_vs_kpp_5s}"
HALFKP_WEIGHTS="${HALFKP_WEIGHTS:-$ROOT_DIR/data/policy_weights_halfkp64_kpp_distilled.binary}"
KPP_WEIGHTS="${KPP_WEIGHTS:-$ROOT_DIR/policy_weights_v2.1.0.binary}"

if (( GAMES < 2 || GAMES % 2 != 0 )); then
  echo "GAMES must be an even number >= 2 (paired side swaps)" >&2
  exit 2
fi
if (( HALFKP_THREADS < 0 || KPP_THREADS < 0 )); then
  echo "HALFKP_THREADS and KPP_THREADS must be >= 0 (0 selects all available threads)" >&2
  exit 2
fi
[[ -s "$POSITIONS" ]] || { echo "positions file not found: $POSITIONS" >&2; exit 2; }
[[ -s "$HALFKP_WEIGHTS" ]] || { echo "HalfKP weights not found: $HALFKP_WEIGHTS" >&2; exit 2; }
[[ -s "$KPP_WEIGHTS" ]] || { echo "KPP weights not found: $KPP_WEIGHTS" >&2; exit 2; }

KPP_TARGET="${KPP_TARGET:-$ROOT_DIR/target/bench-kpp}"
HALFKP_TARGET="${HALFKP_TARGET:-$ROOT_DIR/target/bench-halfkp64}"
mkdir -p "$OUT_DIR"

echo "[1/4] Building KPP engine and benchmark harness"
CARGO_TARGET_DIR="$KPP_TARGET" cargo build --release --features benchmark-tools --bin usi_engine --bin usi_benchmark --bin search_profile
echo "[2/4] Building HalfKP-64 engine"
CARGO_TARGET_DIR="$HALFKP_TARGET" cargo build --release --features halfkp64,benchmark-tools --bin usi_engine --bin search_profile

KPP_ENGINE="$KPP_TARGET/release/usi_engine"
HALFKP_ENGINE="$HALFKP_TARGET/release/usi_engine"
BENCH="$KPP_TARGET/release/usi_benchmark"
KPP_PROFILE="$KPP_TARGET/release/search_profile"
HALFKP_PROFILE="$HALFKP_TARGET/release/search_profile"
MATCH_LOG="$OUT_DIR/match.log"
KPP_SPEED_LOG="$OUT_DIR/kpp_speed.log"
HALFKP_SPEED_LOG="$OUT_DIR/halfkp64_speed.log"
RECORD_DIR="$OUT_DIR/games"

echo "[3/4] Running $GAMES paired games at ${TIME_LIMIT_MS} ms/move"
"$BENCH" \
  --new-engine "$HALFKP_ENGINE" \
  --baseline-engine "$KPP_ENGINE" \
  --new-weights "$HALFKP_WEIGHTS" \
  --baseline-weights "$KPP_WEIGHTS" \
  --positions "$POSITIONS" \
  --games "$GAMES" \
  --depth "$DEPTH" \
  --time-limit-ms "$TIME_LIMIT_MS" \
  --new-threads "$HALFKP_THREADS" \
  --baseline-threads "$KPP_THREADS" \
  --max-plies "$MAX_PLIES" \
  --jobs "$JOBS" \
  --seed "$SEED" \
  --record-dir "$RECORD_DIR" | tee "$MATCH_LOG"

echo "[4/4] Measuring search speed with the same time limit"
"$KPP_PROFILE" --weights "$KPP_WEIGHTS" --positions "$POSITIONS" --samples "$GAMES" \
  --depth "$DEPTH" --time-limit-ms "$TIME_LIMIT_MS" --threads "$KPP_THREADS" \
  --seed "$SEED" | tee "$KPP_SPEED_LOG"
"$HALFKP_PROFILE" --halfkp-weights "$HALFKP_WEIGHTS" --positions "$POSITIONS" --samples "$GAMES" \
  --depth "$DEPTH" --time-limit-ms "$TIME_LIMIT_MS" --threads "$HALFKP_THREADS" \
  --seed "$SEED" | tee "$HALFKP_SPEED_LOG"

echo "Results: $OUT_DIR"

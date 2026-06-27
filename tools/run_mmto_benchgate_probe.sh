#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script. Run it as: bash tools/run_mmto_benchgate_probe.sh"
  return 2
fi

cd "$(dirname "$0")/.."

SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-data/mmto/runs/bonanza_root_pergame_2k_leaf_gt010_20260627_001929}"
LOOP_DIR="${LOOP_DIR:-data/mmto/runs/mmto_refresh_loop_guarded200_benchgate_$(date -u +%Y%m%d_%H%M%S)}"
WEIGHTS="${WEIGHTS:-policy_weights_v2.1.0.binary}"
POSITIONS="${POSITIONS:-taya36.sfen}"
BENCH_SEEDS="${BENCH_SEEDS:-12001 12101 12201}"
BENCH_GAMES="${BENCH_GAMES:-20}"
BENCH_DEPTH="${BENCH_DEPTH:-10}"
BENCH_TIME_LIMIT_MS="${BENCH_TIME_LIMIT_MS:-100}"
BENCH_JOBS="${BENCH_JOBS:-2}"
MIN_SCORE_RATE_PCT="${MIN_SCORE_RATE_PCT:-50}"
KEEP_FAILED_CANDIDATE="${KEEP_FAILED_CANDIDATE:-0}"

REFRESH_MAX_POSITIONS="${REFRESH_MAX_POSITIONS:-200}"
REFRESH_VALID_PERCENT="${REFRESH_VALID_PERCENT:-10}"
BASE_TRAIN_LINES="${BASE_TRAIN_LINES:-1800}"
BASE_VALID_LINES="${BASE_VALID_LINES:-200}"
REFRESH_EPOCHS="${REFRESH_EPOCHS:-2}"
REFRESH_BATCH_SIZE="${REFRESH_BATCH_SIZE:-64}"
REFRESH_LEARNING_RATE="${REFRESH_LEARNING_RATE:-0.0002}"
REFRESH_MAX_WEIGHT_DELTA="${REFRESH_MAX_WEIGHT_DELTA:-0.001}"
REFRESH_ANCHOR_L2="${REFRESH_ANCHOR_L2:-0.0005}"
REFRESH_RERANK_MAX_POSITIONS="${REFRESH_RERANK_MAX_POSITIONS:-300}"
BEST_GUARD_MAX_REGRET_INCREASE_CP="${BEST_GUARD_MAX_REGRET_INCREASE_CP:-0}"
BEST_GUARD_BAD100_INCREASE="${BEST_GUARD_BAD100_INCREASE:-0}"
BEST_GUARD_TEACHER_MATCH_DROP_PCT="${BEST_GUARD_TEACHER_MATCH_DROP_PCT:-0}"

RERANK_MAX_POSITIONS="${RERANK_MAX_POSITIONS:-80}"
RERANK_BASELINE_DEPTH="${RERANK_BASELINE_DEPTH:-4}"
RERANK_CANDIDATE_DEPTH="${RERANK_CANDIDATE_DEPTH:-4}"
RERANK_TEACHER_DEPTH="${RERANK_TEACHER_DEPTH:-6}"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "missing weights: $WEIGHTS" >&2
  exit 1
fi
if [[ ! -d "$SOURCE_RUN_DIR" ]]; then
  echo "missing source run dir: $SOURCE_RUN_DIR" >&2
  exit 1
fi

mkdir -p "$LOOP_DIR"

echo "Starting MMTO bench-aligned gate probe."
echo "LOOP_DIR=$LOOP_DIR"
echo "SOURCE_RUN_DIR=$SOURCE_RUN_DIR"
echo "BENCH_SEEDS=$BENCH_SEEDS BENCH_GAMES=$BENCH_GAMES BENCH_DEPTH=$BENCH_DEPTH"
echo "REFRESH_MAX_POSITIONS=$REFRESH_MAX_POSITIONS REFRESH_EPOCHS=$REFRESH_EPOCHS"
echo "REFRESH_RERANK_MAX_POSITIONS=$REFRESH_RERANK_MAX_POSITIONS"
echo "BEST_GUARD_MAX/BAD100/TEACHER_MATCH=$BEST_GUARD_MAX_REGRET_INCREASE_CP/$BEST_GUARD_BAD100_INCREASE/$BEST_GUARD_TEACHER_MATCH_DROP_PCT"
echo "RERANK_MAX_POSITIONS=$RERANK_MAX_POSITIONS RERANK_DEPTHS=$RERANK_BASELINE_DEPTH/$RERANK_CANDIDATE_DEPTH/$RERANK_TEACHER_DEPTH"

env RUST_FONTCONFIG_DLOPEN=1 cargo build --release \
  --bin usi_engine \
  --bin usi_benchmark \
  --bin record_analyze \
  --bin mmto_rerank_gate

env RUST_FONTCONFIG_DLOPEN=1 \
  SOURCE_RUN_DIR="$SOURCE_RUN_DIR" \
  LOOP_DIR="$LOOP_DIR" \
  INITIAL_CANDIDATE_WEIGHTS="$WEIGHTS" \
  ITERATIONS=1 \
  KEEP_PASSED_WEIGHTS=1 \
  REFRESH_MAX_POSITIONS="$REFRESH_MAX_POSITIONS" \
  REFRESH_VALID_PERCENT="$REFRESH_VALID_PERCENT" \
  BASE_TRAIN_LINES="$BASE_TRAIN_LINES" \
  BASE_VALID_LINES="$BASE_VALID_LINES" \
  EPOCHS="$REFRESH_EPOCHS" \
  BATCH_SIZE="$REFRESH_BATCH_SIZE" \
  LEARNING_RATE="$REFRESH_LEARNING_RATE" \
  MAX_WEIGHT_DELTA="$REFRESH_MAX_WEIGHT_DELTA" \
  ANCHOR_L2="$REFRESH_ANCHOR_L2" \
  LOSS_MODE=listwise-leaf \
  LISTWISE_FEATURE_SOURCE=teacher-leaf \
  LISTWISE_HARD_NEGATIVE_WEIGHT=0.05 \
  CURRENT_TOP_MARGIN_WEIGHT=0.05 \
  GAME_TEACHER_MARGIN_WEIGHT=0.05 \
  BEST_METRIC=p95-regret \
  BEST_GUARD_MAX_REGRET_INCREASE_CP="$BEST_GUARD_MAX_REGRET_INCREASE_CP" \
  BEST_GUARD_BAD100_INCREASE="$BEST_GUARD_BAD100_INCREASE" \
  BEST_GUARD_TEACHER_MATCH_DROP_PCT="$BEST_GUARD_TEACHER_MATCH_DROP_PCT" \
  RERANK_MAX_POSITIONS="$REFRESH_RERANK_MAX_POSITIONS" \
  RERANK_DEDUPE_SFEN=1 \
  RERANK_ALLOW_MEAN_REGRET_INCREASE_CP=0.05 \
  RERANK_ALLOW_P90_REGRET_INCREASE_CP=0 \
  RERANK_ALLOW_P95_REGRET_INCREASE_CP=0 \
  RERANK_ALLOW_BAD_RATIO_INCREASE=0 \
  HARD_FEEDBACK_ON_FAILURE=1 \
  DAGGER_MAX_POSITIONS=50 \
  EXPLICIT_WEIGHT_MODE=combined \
  EXPLICIT_WEIGHT_SCALE_CP=50 \
  MIN_FREE_GB=6 \
  bash tools/run_mmto_refresh_loop.sh 2>&1 | tee "$LOOP_DIR/pipeline_stdout.log"

if [[ ! -s "$LOOP_DIR/current_candidate_path.txt" ]]; then
  {
    echo "NO_CANDIDATE"
    if [[ -f "$LOOP_DIR/pipeline_stdout.log" ]]; then
      grep -E 'best_epoch=|best_guard|did not pass offline gates|FINAL_CANDIDATE=|failed with status=|refresh offline gates passed|hard feedback' \
        "$LOOP_DIR/pipeline_stdout.log" | tail -40
    fi
  } | tee "$LOOP_DIR/benchgate_summary.txt"
  find "$LOOP_DIR" -type f -name '*.binary' -delete
  du -sh "$LOOP_DIR" | tee "$LOOP_DIR/du_after.txt"
  exit 0
fi

CAND="$(cat "$LOOP_DIR/current_candidate_path.txt")"
if [[ "$CAND" == "$WEIGHTS" || ! -f "$CAND" ]]; then
  {
    echo "NO_NEW_CANDIDATE CAND=$CAND"
    if [[ -f "$LOOP_DIR/pipeline_stdout.log" ]]; then
      grep -E 'best_epoch=|best_guard|did not pass offline gates|FINAL_CANDIDATE=|failed with status=|refresh offline gates passed|hard feedback' \
        "$LOOP_DIR/pipeline_stdout.log" | tail -40
    fi
  } | tee "$LOOP_DIR/benchgate_summary.txt"
  find "$LOOP_DIR" -type f -name '*.binary' -delete
  du -sh "$LOOP_DIR" | tee "$LOOP_DIR/du_after.txt"
  exit 0
fi

total_new=0
total_baseline=0
total_draw=0
for seed in $BENCH_SEEDS; do
  bench_dir="$LOOP_DIR/bench${BENCH_GAMES}_seed${seed}"
  rm -rf "$bench_dir"
  env RUST_FONTCONFIG_DLOPEN=1 target/release/usi_benchmark \
    --new-engine target/release/usi_engine \
    --baseline-engine target/release/usi_engine \
    --new-weights "$CAND" \
    --baseline-weights "$WEIGHTS" \
    --positions "$POSITIONS" \
    --games "$BENCH_GAMES" \
    --depth "$BENCH_DEPTH" \
    --time-limit-ms "$BENCH_TIME_LIMIT_MS" \
    --max-plies 200 \
    --adjudicate-at-max-plies \
    --jobs "$BENCH_JOBS" \
    --seed "$seed" \
    --record-dir "$bench_dir" 2>&1 | tee "$bench_dir.stdout.log"

  env RUST_FONTCONFIG_DLOPEN=1 target/release/record_analyze \
    --record-dir "$bench_dir" \
    --weights "$WEIGHTS" \
    --top-drops 20 \
    --export-baseline-sweep-starts "$bench_dir/baseline_sweep_starts.sfen" \
    --export-drop-windows "$bench_dir/drop_windows.sfen" \
    > "$bench_dir.record_analyze.log"

  new_wins="$(awk '/new wins:/ {print $3}' "$bench_dir.stdout.log" | tail -1)"
  baseline_wins="$(awk '/baseline wins:/ {print $3}' "$bench_dir.stdout.log" | tail -1)"
  draws="$(awk '/draws:/ {print $2}' "$bench_dir.stdout.log" | tail -1)"
  total_new=$((total_new + ${new_wins:-0}))
  total_baseline=$((total_baseline + ${baseline_wins:-0}))
  total_draw=$((total_draw + ${draws:-0}))
done

shopt -s nullglob
gate_inputs=("$POSITIONS" "$LOOP_DIR"/bench${BENCH_GAMES}_seed*/baseline_sweep_starts.sfen "$LOOP_DIR"/bench${BENCH_GAMES}_seed*/drop_windows.sfen)
cat "${gate_inputs[@]}" > "$LOOP_DIR/bench_aligned_gate.sfen"

set +e
env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_rerank_gate \
  --baseline-weights "$WEIGHTS" \
  --candidate-weights "$CAND" \
  --teacher-weights "$WEIGHTS" \
  --input "$LOOP_DIR/bench_aligned_gate.sfen" \
  --dedupe-sfen \
  --max-positions "$RERANK_MAX_POSITIONS" \
  --baseline-depth "$RERANK_BASELINE_DEPTH" \
  --candidate-depth "$RERANK_CANDIDATE_DEPTH" \
  --teacher-depth "$RERANK_TEACHER_DEPTH" \
  --allow-mean-regret-increase-cp 0 \
  --allow-p90-regret-increase-cp 0 \
  --allow-p95-regret-increase-cp 0 \
  --allow-bad-ratio-increase 0 \
  --json-output "$LOOP_DIR/bench_aligned_rerank.json" \
  2>&1 | tee "$LOOP_DIR/bench_aligned_rerank_stdout.log"
rerank_status="${PIPESTATUS[0]}"
set -e

score_rate_num=$((total_new * 2 + total_draw))
score_rate_den=$(((total_new + total_baseline + total_draw) * 2))
{
  echo "LOOP_DIR=$LOOP_DIR"
  echo "CAND=$CAND"
  echo "total_new=$total_new total_baseline=$total_baseline total_draw=$total_draw"
  echo "score_rate_num=$score_rate_num score_rate_den=$score_rate_den"
  echo "min_score_rate_pct=$MIN_SCORE_RATE_PCT"
  echo "bench_aligned_rerank_status=$rerank_status"
} | tee "$LOOP_DIR/benchgate_summary.txt"

if (( score_rate_num * 100 < score_rate_den * MIN_SCORE_RATE_PCT )) || (( rerank_status != 0 )); then
  if [[ "$KEEP_FAILED_CANDIDATE" != "1" ]]; then
    rm -f "$CAND"
    echo "candidate deleted" | tee -a "$LOOP_DIR/benchgate_summary.txt"
  else
    echo "candidate kept despite failed gate" | tee -a "$LOOP_DIR/benchgate_summary.txt"
  fi
else
  echo "candidate kept for further evaluation" | tee -a "$LOOP_DIR/benchgate_summary.txt"
fi

find "$LOOP_DIR" -type f -name '*.binary' -printf '%s %p\n' | tee "$LOOP_DIR/binary_files_after.txt"
du -sh "$LOOP_DIR" | tee "$LOOP_DIR/du_after.txt"
df -h . | tee "$LOOP_DIR/df_after.txt"

#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script. Run it as: bash tools/run_mmto_bench_feedback_probe.sh"
  return 2
fi

cd "$(dirname "$0")/.."

SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-data/mmto/runs/bonanza_root_pergame_2k_leaf_gt010_20260627_001929}"
RUN_DIR="${RUN_DIR:-data/mmto/runs/mmto_bench_feedback_$(date -u +%Y%m%d_%H%M%S)}"
WEIGHTS="${WEIGHTS:-policy_weights_v2.1.0.binary}"
TEACHER_WEIGHTS="${TEACHER_WEIGHTS:-$WEIGHTS}"
POSITIONS="${POSITIONS:-taya36.sfen}"
BENCH_SEEDS="${BENCH_SEEDS:-12001 12101 12201}"
BENCH_GAMES="${BENCH_GAMES:-20}"
BENCH_DEPTH="${BENCH_DEPTH:-10}"
BENCH_TIME_LIMIT_MS="${BENCH_TIME_LIMIT_MS:-100}"
BENCH_JOBS="${BENCH_JOBS:-2}"
MIN_SCORE_RATE_PCT="${MIN_SCORE_RATE_PCT:-50}"
MIN_FREE_GB="${MIN_FREE_GB:-6}"

INITIAL_REFRESH_MAX_POSITIONS="${INITIAL_REFRESH_MAX_POSITIONS:-200}"
INITIAL_REFRESH_EPOCHS="${INITIAL_REFRESH_EPOCHS:-2}"
INITIAL_BASE_TRAIN_LINES="${INITIAL_BASE_TRAIN_LINES:-1800}"
INITIAL_BASE_VALID_LINES="${INITIAL_BASE_VALID_LINES:-200}"
INITIAL_REFRESH_LEARNING_RATE="${INITIAL_REFRESH_LEARNING_RATE:-0.0002}"
INITIAL_REFRESH_MAX_WEIGHT_DELTA="${INITIAL_REFRESH_MAX_WEIGHT_DELTA:-0.001}"
INITIAL_REFRESH_ANCHOR_L2="${INITIAL_REFRESH_ANCHOR_L2:-0.0005}"
INITIAL_REFRESH_RERANK_MAX_POSITIONS="${INITIAL_REFRESH_RERANK_MAX_POSITIONS:-300}"
INITIAL_BEST_GUARD_MAX_REGRET_INCREASE_CP="${INITIAL_BEST_GUARD_MAX_REGRET_INCREASE_CP:-0}"
INITIAL_BEST_GUARD_BAD100_INCREASE="${INITIAL_BEST_GUARD_BAD100_INCREASE:-0}"
INITIAL_BEST_GUARD_TEACHER_MATCH_DROP_PCT="${INITIAL_BEST_GUARD_TEACHER_MATCH_DROP_PCT:-0}"

FEEDBACK_START_WEIGHTS="${FEEDBACK_START_WEIGHTS:-candidate}"
FEEDBACK_MAX_POSITIONS="${FEEDBACK_MAX_POSITIONS:-200}"
FEEDBACK_EPOCHS="${FEEDBACK_EPOCHS:-2}"
FEEDBACK_LEARNING_RATE="${FEEDBACK_LEARNING_RATE:-0.0001}"
FEEDBACK_REPLAY_WEIGHT="${FEEDBACK_REPLAY_WEIGHT:-0.15}"
FEEDBACK_MAX_WEIGHT_DELTA="${FEEDBACK_MAX_WEIGHT_DELTA:-0.001}"
FEEDBACK_ANCHOR_L2="${FEEDBACK_ANCHOR_L2:-0.0005}"
FEEDBACK_BASE_TRAIN_LINES="${FEEDBACK_BASE_TRAIN_LINES:-1800}"
FEEDBACK_BASE_VALID_LINES="${FEEDBACK_BASE_VALID_LINES:-200}"
FEEDBACK_RERANK_MAX_POSITIONS="${FEEDBACK_RERANK_MAX_POSITIONS:-300}"
FEEDBACK_BEST_GUARD_MAX_REGRET_INCREASE_CP="${FEEDBACK_BEST_GUARD_MAX_REGRET_INCREASE_CP:-0}"
FEEDBACK_BEST_GUARD_BAD100_INCREASE="${FEEDBACK_BEST_GUARD_BAD100_INCREASE:-0}"

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
if [[ "$FEEDBACK_START_WEIGHTS" != "candidate" && "$FEEDBACK_START_WEIGHTS" != "baseline" ]]; then
  echo "FEEDBACK_START_WEIGHTS must be candidate or baseline, got: $FEEDBACK_START_WEIGHTS" >&2
  exit 1
fi

FREE_KB="$(df -Pk . | awk 'NR==2 {print $4}')"
MIN_FREE_KB=$((MIN_FREE_GB * 1024 * 1024))
if (( FREE_KB < MIN_FREE_KB )); then
  echo "free disk is too low: $((FREE_KB / 1024 / 1024))GB < ${MIN_FREE_GB}GB" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"

echo "Starting MMTO bench-feedback probe."
echo "RUN_DIR=$RUN_DIR"
echo "SOURCE_RUN_DIR=$SOURCE_RUN_DIR"
echo "WEIGHTS=$WEIGHTS TEACHER_WEIGHTS=$TEACHER_WEIGHTS"
echo "BENCH_SEEDS=$BENCH_SEEDS BENCH_GAMES=$BENCH_GAMES BENCH_DEPTH=$BENCH_DEPTH"
echo "INITIAL_REFRESH_MAX_POSITIONS=$INITIAL_REFRESH_MAX_POSITIONS INITIAL_REFRESH_EPOCHS=$INITIAL_REFRESH_EPOCHS"
echo "INITIAL_REFRESH_RERANK_MAX_POSITIONS=$INITIAL_REFRESH_RERANK_MAX_POSITIONS"
echo "INITIAL_BEST_GUARD_MAX/BAD100/TEACHER_MATCH=$INITIAL_BEST_GUARD_MAX_REGRET_INCREASE_CP/$INITIAL_BEST_GUARD_BAD100_INCREASE/$INITIAL_BEST_GUARD_TEACHER_MATCH_DROP_PCT"
echo "FEEDBACK_START_WEIGHTS=$FEEDBACK_START_WEIGHTS FEEDBACK_MAX_POSITIONS=$FEEDBACK_MAX_POSITIONS"
echo "FEEDBACK_RERANK_MAX_POSITIONS=$FEEDBACK_RERANK_MAX_POSITIONS"
echo "FEEDBACK_BEST_GUARD_MAX/BAD100=$FEEDBACK_BEST_GUARD_MAX_REGRET_INCREASE_CP/$FEEDBACK_BEST_GUARD_BAD100_INCREASE"

env RUST_FONTCONFIG_DLOPEN=1 cargo build --release \
  --bin usi_engine \
  --bin usi_benchmark \
  --bin record_analyze \
  --bin mmto_rerank_gate \
  --bin mmto_tree_dump \
  --bin rank_stats \
  --bin mmto_tree_train \
  --bin mmto_score_gate \
  --bin adjust_weights

run_bench_set() {
  local label="$1"
  local candidate_weights="$2"
  local out_dir="$3"
  mkdir -p "$out_dir"

  local total_new=0
  local total_baseline=0
  local total_draw=0
  for seed in $BENCH_SEEDS; do
    local bench_dir="$out_dir/${label}_bench${BENCH_GAMES}_seed${seed}"
    rm -rf "$bench_dir"
    env RUST_FONTCONFIG_DLOPEN=1 target/release/usi_benchmark \
      --new-engine target/release/usi_engine \
      --baseline-engine target/release/usi_engine \
      --new-weights "$candidate_weights" \
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

    local new_wins baseline_wins draws
    new_wins="$(awk '/new wins:/ {print $3}' "$bench_dir.stdout.log" | tail -1)"
    baseline_wins="$(awk '/baseline wins:/ {print $3}' "$bench_dir.stdout.log" | tail -1)"
    draws="$(awk '/draws:/ {print $2}' "$bench_dir.stdout.log" | tail -1)"
    total_new=$((total_new + ${new_wins:-0}))
    total_baseline=$((total_baseline + ${baseline_wins:-0}))
    total_draw=$((total_draw + ${draws:-0}))
  done

  local score_rate_num score_rate_den
  score_rate_num=$((total_new * 2 + total_draw))
  score_rate_den=$(((total_new + total_baseline + total_draw) * 2))
  {
    echo "label=$label"
    echo "candidate=$candidate_weights"
    echo "total_new=$total_new total_baseline=$total_baseline total_draw=$total_draw"
    echo "score_rate_num=$score_rate_num score_rate_den=$score_rate_den"
  } | tee "$out_dir/${label}_bench_summary.txt"
}

make_hard_input() {
  local input_dir="$1"
  local output_path="$2"
  shopt -s nullglob
  local files=(
    "$input_dir"/bench${BENCH_GAMES}_seed*/baseline_sweep_starts.sfen
    "$input_dir"/bench${BENCH_GAMES}_seed*/drop_windows.sfen
    "$input_dir"/*_bench${BENCH_GAMES}_seed*/baseline_sweep_starts.sfen
    "$input_dir"/*_bench${BENCH_GAMES}_seed*/drop_windows.sfen
  )
  if (( ${#files[@]} == 0 )); then
    : > "$output_path"
    return 0
  fi
  cat "${files[@]}" | awk 'NF && !seen[$0]++' > "$output_path"
}

run_bench_aligned_rerank() {
  local label="$1"
  local candidate_weights="$2"
  local hard_input="$3"
  local out_dir="$4"
  local rerank_input="$out_dir/${label}_bench_aligned_gate.sfen"

  if [[ -s "$hard_input" ]]; then
    cat "$POSITIONS" "$hard_input" | awk 'NF && !seen[$0]++' > "$rerank_input"
  else
    cp "$POSITIONS" "$rerank_input"
  fi

  set +e
  env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_rerank_gate \
    --baseline-weights "$WEIGHTS" \
    --candidate-weights "$candidate_weights" \
    --teacher-weights "$TEACHER_WEIGHTS" \
    --input "$rerank_input" \
    --dedupe-sfen \
    --max-positions "$RERANK_MAX_POSITIONS" \
    --baseline-depth "$RERANK_BASELINE_DEPTH" \
    --candidate-depth "$RERANK_CANDIDATE_DEPTH" \
    --teacher-depth "$RERANK_TEACHER_DEPTH" \
    --allow-mean-regret-increase-cp 0 \
    --allow-p90-regret-increase-cp 0 \
    --allow-p95-regret-increase-cp 0 \
    --allow-bad-ratio-increase 0 \
    --json-output "$out_dir/${label}_bench_aligned_rerank.json" \
    2>&1 | tee "$out_dir/${label}_bench_aligned_rerank_stdout.log"
  local status="${PIPESTATUS[0]}"
  set -e
  echo "$status" > "$out_dir/${label}_bench_aligned_rerank.status"
}

passes_score_gate() {
  local summary_path="$1"
  local score_num score_den
  score_num="$(awk -F= '/score_rate_num=/ {print $2}' "$summary_path" | tail -1)"
  score_den="$(awk -F= '/score_rate_den=/ {print $2}' "$summary_path" | tail -1)"
  [[ -n "$score_num" && -n "$score_den" ]] || return 1
  (( score_num * 100 >= score_den * MIN_SCORE_RATE_PCT ))
}

write_artifact_summary() {
  find "$RUN_DIR" -type f -name '*.binary' -printf '%s %p\n' | tee "$RUN_DIR/binary_files_after.txt"
  du -sh "$RUN_DIR" | tee "$RUN_DIR/du_after.txt"
  df -h . | tee "$RUN_DIR/df_after.txt"
}

INITIAL_DIR="$RUN_DIR/initial_benchgate"
KEEP_FAILED_CANDIDATE=1 \
  LOOP_DIR="$INITIAL_DIR" \
  SOURCE_RUN_DIR="$SOURCE_RUN_DIR" \
  WEIGHTS="$WEIGHTS" \
  POSITIONS="$POSITIONS" \
  BENCH_SEEDS="$BENCH_SEEDS" \
  BENCH_GAMES="$BENCH_GAMES" \
  BENCH_DEPTH="$BENCH_DEPTH" \
  BENCH_TIME_LIMIT_MS="$BENCH_TIME_LIMIT_MS" \
  BENCH_JOBS="$BENCH_JOBS" \
  MIN_SCORE_RATE_PCT="$MIN_SCORE_RATE_PCT" \
  REFRESH_MAX_POSITIONS="$INITIAL_REFRESH_MAX_POSITIONS" \
  REFRESH_EPOCHS="$INITIAL_REFRESH_EPOCHS" \
  BASE_TRAIN_LINES="$INITIAL_BASE_TRAIN_LINES" \
  BASE_VALID_LINES="$INITIAL_BASE_VALID_LINES" \
  REFRESH_LEARNING_RATE="$INITIAL_REFRESH_LEARNING_RATE" \
  REFRESH_MAX_WEIGHT_DELTA="$INITIAL_REFRESH_MAX_WEIGHT_DELTA" \
  REFRESH_ANCHOR_L2="$INITIAL_REFRESH_ANCHOR_L2" \
  REFRESH_RERANK_MAX_POSITIONS="$INITIAL_REFRESH_RERANK_MAX_POSITIONS" \
  BEST_GUARD_MAX_REGRET_INCREASE_CP="$INITIAL_BEST_GUARD_MAX_REGRET_INCREASE_CP" \
  BEST_GUARD_BAD100_INCREASE="$INITIAL_BEST_GUARD_BAD100_INCREASE" \
  BEST_GUARD_TEACHER_MATCH_DROP_PCT="$INITIAL_BEST_GUARD_TEACHER_MATCH_DROP_PCT" \
  RERANK_MAX_POSITIONS="$RERANK_MAX_POSITIONS" \
  RERANK_BASELINE_DEPTH="$RERANK_BASELINE_DEPTH" \
  RERANK_CANDIDATE_DEPTH="$RERANK_CANDIDATE_DEPTH" \
  RERANK_TEACHER_DEPTH="$RERANK_TEACHER_DEPTH" \
  bash tools/run_mmto_benchgate_probe.sh 2>&1 | tee "$RUN_DIR/initial_benchgate_stdout.log"

if [[ ! -s "$INITIAL_DIR/benchgate_summary.txt" ]]; then
  echo "initial benchgate summary is missing" >&2
  exit 1
fi
INITIAL_CAND="$(awk -F= '/^CAND=/ {print $2}' "$INITIAL_DIR/benchgate_summary.txt" | tail -1)"
if [[ -z "$INITIAL_CAND" || "$INITIAL_CAND" == "$WEIGHTS" || ! -f "$INITIAL_CAND" ]]; then
  echo "No initial candidate available for feedback."
  {
    echo "RUN_DIR=$RUN_DIR"
    echo "final_status=no_initial_candidate"
    cat "$INITIAL_DIR/benchgate_summary.txt"
  } | tee "$RUN_DIR/final_summary.txt"
  write_artifact_summary
  exit 0
fi

if grep -q "candidate kept for further evaluation" "$INITIAL_DIR/benchgate_summary.txt"; then
  echo "Initial candidate passed benchgate; no feedback repair needed."
  {
    echo "RUN_DIR=$RUN_DIR"
    echo "final_status=initial_candidate_kept"
    echo "FINAL_CANDIDATE=$INITIAL_CAND"
    cat "$INITIAL_DIR/benchgate_summary.txt"
  } | tee "$RUN_DIR/final_summary.txt"
  write_artifact_summary
  exit 0
fi

HARD_INPUT="$RUN_DIR/bench_feedback_input.sfen"
make_hard_input "$INITIAL_DIR" "$HARD_INPUT"
HARD_LINES="$(wc -l < "$HARD_INPUT" | tr -d ' ')"
echo "bench_feedback_positions=$HARD_LINES" | tee "$RUN_DIR/feedback_input_summary.txt"
if (( HARD_LINES == 0 )); then
  echo "No bench hard positions were exported. Deleting failed initial candidate."
  rm -f "$INITIAL_CAND"
  {
    echo "RUN_DIR=$RUN_DIR"
    echo "final_status=no_bench_hard_positions"
    echo "INITIAL_CAND=$INITIAL_CAND"
    echo "bench_feedback_positions=$HARD_LINES"
    cat "$INITIAL_DIR/benchgate_summary.txt"
  } | tee "$RUN_DIR/final_summary.txt"
  write_artifact_summary
  exit 0
fi

case "$FEEDBACK_START_WEIGHTS" in
  candidate) FEEDBACK_WEIGHTS="$INITIAL_CAND" ;;
  baseline) FEEDBACK_WEIGHTS="$WEIGHTS" ;;
esac

FEEDBACK_DIR="$RUN_DIR/bench_feedback_stage"
set +e
SOURCE_RUN_DIR="$SOURCE_RUN_DIR" \
  WEIGHTS="$FEEDBACK_WEIGHTS" \
  TEACHER_WEIGHTS="$TEACHER_WEIGHTS" \
  CANDIDATE_WEIGHTS="$INITIAL_CAND" \
  BASE_TRAIN="$SOURCE_RUN_DIR/train.tree.jsonl" \
  BASE_VALID="$SOURCE_RUN_DIR/valid.tree.jsonl" \
  SCORE_POSITIONS="$SOURCE_RUN_DIR/score_positions.sfen" \
  DAGGER_INPUT="$HARD_INPUT" \
  USE_EXPLICIT_HARD_PAIRS=0 \
  RUN_DIR="$FEEDBACK_DIR" \
  TRAIN_LINES="$FEEDBACK_BASE_TRAIN_LINES" \
  VALID_LINES="$FEEDBACK_BASE_VALID_LINES" \
  DAGGER_MAX_POSITIONS="$FEEDBACK_MAX_POSITIONS" \
  EPOCHS="$FEEDBACK_EPOCHS" \
  BATCH_SIZE=64 \
  LEARNING_RATE="$FEEDBACK_LEARNING_RATE" \
  REPLAY_WEIGHT="$FEEDBACK_REPLAY_WEIGHT" \
  MAX_WEIGHT_DELTA="$FEEDBACK_MAX_WEIGHT_DELTA" \
  ANCHOR_L2="$FEEDBACK_ANCHOR_L2" \
  LOSS_MODE=listwise-leaf \
  LISTWISE_FEATURE_SOURCE=teacher-leaf \
  LISTWISE_HARD_NEGATIVE_WEIGHT=0.05 \
  CURRENT_TOP_MARGIN_WEIGHT=0.05 \
  GAME_TEACHER_MARGIN_WEIGHT=0.05 \
  BEST_METRIC=p95-regret \
  BEST_GUARD_MAX_REGRET_INCREASE_CP="$FEEDBACK_BEST_GUARD_MAX_REGRET_INCREASE_CP" \
  BEST_GUARD_BAD100_INCREASE="$FEEDBACK_BEST_GUARD_BAD100_INCREASE" \
  RERANK_MAX_POSITIONS="$FEEDBACK_RERANK_MAX_POSITIONS" \
  RERANK_DEDUPE_SFEN=1 \
  RERANK_ALLOW_MEAN_REGRET_INCREASE_CP=0.05 \
  RERANK_ALLOW_P90_REGRET_INCREASE_CP=0 \
  RERANK_ALLOW_P95_REGRET_INCREASE_CP=0 \
  RERANK_ALLOW_BAD_RATIO_INCREASE=0 \
  MIN_FREE_GB="$MIN_FREE_GB" \
  bash tools/run_mmto_dagger_from_run.sh 2>&1 | tee "$RUN_DIR/bench_feedback_stage_stdout.log"
feedback_status="${PIPESTATUS[0]}"
set -e

FEEDBACK_CAND="$FEEDBACK_DIR/best.raw.binary"
if (( feedback_status != 0 )) || [[ ! -f "$FEEDBACK_CAND" ]]; then
  echo "bench feedback did not produce a gated candidate. Deleting failed initial candidate."
  rm -f "$INITIAL_CAND" "$FEEDBACK_DIR/candidate.raw.binary" "$FEEDBACK_CAND"
  {
    echo "RUN_DIR=$RUN_DIR"
    echo "final_status=feedback_rejected_no_candidate"
    echo "INITIAL_CAND=$INITIAL_CAND"
    echo "FEEDBACK_CAND=$FEEDBACK_CAND"
    echo "bench_feedback_positions=$HARD_LINES"
    echo "feedback_status=$feedback_status"
    cat "$INITIAL_DIR/benchgate_summary.txt"
  } | tee "$RUN_DIR/final_summary.txt"
  write_artifact_summary
  exit 0
fi
rm -f "$FEEDBACK_DIR/candidate.raw.binary"

FEEDBACK_BENCH_DIR="$RUN_DIR/feedback_candidate_gate"
run_bench_set "feedback" "$FEEDBACK_CAND" "$FEEDBACK_BENCH_DIR"
FEEDBACK_HARD_INPUT="$RUN_DIR/feedback_bench_hard_input.sfen"
make_hard_input "$FEEDBACK_BENCH_DIR" "$FEEDBACK_HARD_INPUT"
run_bench_aligned_rerank "feedback" "$FEEDBACK_CAND" "$FEEDBACK_HARD_INPUT" "$FEEDBACK_BENCH_DIR"

FEEDBACK_SUMMARY="$FEEDBACK_BENCH_DIR/feedback_bench_summary.txt"
FEEDBACK_RERANK_STATUS="$(cat "$FEEDBACK_BENCH_DIR/feedback_bench_aligned_rerank.status")"
{
  echo "RUN_DIR=$RUN_DIR"
  echo "final_status=feedback_candidate_evaluated"
  echo "INITIAL_CAND=$INITIAL_CAND"
  echo "FEEDBACK_CAND=$FEEDBACK_CAND"
  echo "bench_feedback_positions=$HARD_LINES"
  cat "$FEEDBACK_SUMMARY"
  echo "feedback_bench_aligned_rerank_status=$FEEDBACK_RERANK_STATUS"
} | tee "$RUN_DIR/final_summary.txt"

if passes_score_gate "$FEEDBACK_SUMMARY" && [[ "$FEEDBACK_RERANK_STATUS" == "0" ]]; then
  sed -i 's/^final_status=.*/final_status=feedback_candidate_kept/' "$RUN_DIR/final_summary.txt"
  echo "feedback candidate kept for further evaluation" | tee -a "$RUN_DIR/final_summary.txt"
  rm -f "$INITIAL_CAND"
else
  sed -i 's/^final_status=.*/final_status=feedback_candidate_rejected/' "$RUN_DIR/final_summary.txt"
  echo "feedback candidate rejected and deleted" | tee -a "$RUN_DIR/final_summary.txt"
  rm -f "$INITIAL_CAND" "$FEEDBACK_CAND"
fi

write_artifact_summary

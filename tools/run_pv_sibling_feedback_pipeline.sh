#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script. Run it as: bash tools/run_pv_sibling_feedback_pipeline.sh"
  return 2
fi

cd "$(dirname "$0")/.."

WEIGHTS="${WEIGHTS:-policy_weights_v2.1.0.binary}"
TEACHER_WEIGHTS="${TEACHER_WEIGHTS:-$WEIGHTS}"
POSITIONS="${POSITIONS:-data/mmto/positions/wdoor2023_2026_r4000_p16_120.sfen}"
if [[ ! -f "$POSITIONS" ]]; then
  POSITIONS="${FALLBACK_POSITIONS:-converted_records2016_10818.sfen}"
fi

RUN_DIR="${RUN_DIR:-data/mmto/runs/pv_sibling_feedback_$(date -u +%Y%m%d_%H%M%S)}"
MIN_FREE_GB="${MIN_FREE_GB:-6}"

MAX_POSITIONS="${MAX_POSITIONS:-3000}"
POSITION_CHUNK_SIZE="${POSITION_CHUNK_SIZE:-128}"
VALID_PERCENT="${VALID_PERCENT:-10}"
JOBS="${JOBS:-4}"
TEACHER_DEPTH="${TEACHER_DEPTH:-3}"
STUDENT_DEPTH="${STUDENT_DEPTH:-2}"
TEACHER_SCORE_TOP="${TEACHER_SCORE_TOP:-24}"
CANDIDATE_TOP="${CANDIDATE_TOP:-24}"
MAX_ABS_ROOT_SCORE="${MAX_ABS_ROOT_SCORE:-30000}"
PV_SIBLING_MAX_PLIES="${PV_SIBLING_MAX_PLIES:-2}"
PV_SIBLING_SAMPLE_WEIGHT="${PV_SIBLING_SAMPLE_WEIGHT:-0.25}"
PV_SIBLING_TOTAL_WEIGHT_CAP="${PV_SIBLING_TOTAL_WEIGHT_CAP:-0.25}"

FEEDBACK_MIN_CANDIDATE_REGRET_CP="${FEEDBACK_MIN_CANDIDATE_REGRET_CP:-30}"
FEEDBACK_MIN_REGRET_DELTA_CP="${FEEDBACK_MIN_REGRET_DELTA_CP:-10}"
FEEDBACK_MAX_GOOD_REGRET_CP="${FEEDBACK_MAX_GOOD_REGRET_CP:-30}"
FEEDBACK_LIMIT="${FEEDBACK_LIMIT:-5000}"
FEEDBACK_GUARD_PERCENT="${FEEDBACK_GUARD_PERCENT:-25}"
FEEDBACK_WEIGHT="${FEEDBACK_WEIGHT:-0.5}"

EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-0.00005}"
MAX_WEIGHT_DELTA="${MAX_WEIGHT_DELTA:-0.01}"
ANCHOR_L2="${ANCHOR_L2:-0.0003}"
BEST_GUARD_FEEDBACK_VIOLATION_INCREASE="${BEST_GUARD_FEEDBACK_VIOLATION_INCREASE:-0}"
BEST_GUARD_FEEDBACK_LOSS_INCREASE="${BEST_GUARD_FEEDBACK_LOSS_INCREASE:--1}"
VALID_LINES="${VALID_LINES:-1000}"

RERANK_MAX_POSITIONS="${RERANK_MAX_POSITIONS:-1000}"
RERANK_ALLOW_MEAN_REGRET_INCREASE_CP="${RERANK_ALLOW_MEAN_REGRET_INCREASE_CP:-0}"
RERANK_ALLOW_P90_REGRET_INCREASE_CP="${RERANK_ALLOW_P90_REGRET_INCREASE_CP:-0}"
RERANK_ALLOW_P95_REGRET_INCREASE_CP="${RERANK_ALLOW_P95_REGRET_INCREASE_CP:-0}"
RERANK_ALLOW_BAD_RATIO_INCREASE="${RERANK_ALLOW_BAD_RATIO_INCREASE:-0}"
RERANK_ALLOW_PHASE_MEAN_REGRET_INCREASE_CP="${RERANK_ALLOW_PHASE_MEAN_REGRET_INCREASE_CP:--1}"
RERANK_ALLOW_PHASE_P90_REGRET_INCREASE_CP="${RERANK_ALLOW_PHASE_P90_REGRET_INCREASE_CP:--1}"
RERANK_ALLOW_PHASE_P95_REGRET_INCREASE_CP="${RERANK_ALLOW_PHASE_P95_REGRET_INCREASE_CP:--1}"
RERANK_ALLOW_PHASE_BAD_RATIO_INCREASE="${RERANK_ALLOW_PHASE_BAD_RATIO_INCREASE:--1}"

RUN_BENCH="${RUN_BENCH:-1}"
BENCH_POSITIONS="${BENCH_POSITIONS:-taya36.sfen}"
BENCH_GAMES="${BENCH_GAMES:-20}"
BENCH_DEPTH="${BENCH_DEPTH:-5}"
BENCH_TIME_LIMIT_MS="${BENCH_TIME_LIMIT_MS:-100}"
BENCH_MAX_PLIES="${BENCH_MAX_PLIES:-200}"
BENCH_JOBS="${BENCH_JOBS:-2}"
BENCH_SEED="${BENCH_SEED:-35001}"
KEEP_MIN_NEW_WINS="${KEEP_MIN_NEW_WINS:-12}"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "missing weights: $WEIGHTS" >&2
  exit 1
fi
if [[ ! -f "$TEACHER_WEIGHTS" ]]; then
  echo "missing teacher weights: $TEACHER_WEIGHTS" >&2
  exit 1
fi
if [[ ! -f "$POSITIONS" ]]; then
  echo "missing positions: $POSITIONS" >&2
  exit 1
fi

FREE_KB="$(df -Pk . | awk 'NR==2 {print $4}')"
MIN_FREE_KB=$((MIN_FREE_GB * 1024 * 1024))
if (( FREE_KB < MIN_FREE_KB )); then
  echo "free disk is too low: $((FREE_KB / 1024 / 1024))GB < ${MIN_FREE_GB}GB" >&2
  echo "Clean old generated runs first: bash tools/clean_mmto_runs.sh" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"

echo "Starting PV sibling feedback pipeline."
echo "RUN_DIR=$RUN_DIR"
echo "WEIGHTS=$WEIGHTS"
echo "TEACHER_WEIGHTS=$TEACHER_WEIGHTS"
echo "POSITIONS=$POSITIONS"
echo "MAX_POSITIONS=$MAX_POSITIONS POSITION_CHUNK_SIZE=$POSITION_CHUNK_SIZE JOBS=$JOBS"
echo "TEACHER_DEPTH=$TEACHER_DEPTH STUDENT_DEPTH=$STUDENT_DEPTH"
echo "FEEDBACK_MIN_CANDIDATE_REGRET_CP=$FEEDBACK_MIN_CANDIDATE_REGRET_CP"
echo "FEEDBACK_MIN_REGRET_DELTA_CP=$FEEDBACK_MIN_REGRET_DELTA_CP"
echo "FEEDBACK_MAX_GOOD_REGRET_CP=$FEEDBACK_MAX_GOOD_REGRET_CP"
echo "FEEDBACK_WEIGHT=$FEEDBACK_WEIGHT EPOCHS=$EPOCHS LEARNING_RATE=$LEARNING_RATE"

env RUST_FONTCONFIG_DLOPEN=1 cargo build --release \
  --bin mmto_tree_dump \
  --bin tree_feedback_collect \
  --bin rank_stats \
  --bin mmto_tree_train \
  --bin mmto_score_gate \
  --bin mmto_rerank_gate \
  --bin usi_engine \
  --bin usi_benchmark \
  --bin record_analyze

env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_tree_dump \
  --student-weights "$WEIGHTS" \
  --teacher-weights "$TEACHER_WEIGHTS" \
  --input "$POSITIONS" \
  --train-output "$RUN_DIR/train.pv.tree.jsonl" \
  --valid-output "$RUN_DIR/valid.pv.tree.jsonl" \
  --teacher-depth "$TEACHER_DEPTH" \
  --student-depth "$STUDENT_DEPTH" \
  --teacher-score-top "$TEACHER_SCORE_TOP" \
  --candidate-top "$CANDIDATE_TOP" \
  --max-positions "$MAX_POSITIONS" \
  --position-chunk-size "$POSITION_CHUNK_SIZE" \
  --valid-percent "$VALID_PERCENT" \
  --jobs "$JOBS" \
  --exclude-in-check \
  --min-legal-moves 2 \
  --max-abs-root-score "$MAX_ABS_ROOT_SCORE" \
  --score-all-legal-for-valid \
  --emit-pv-sibling-nodes \
  --pv-sibling-max-plies "$PV_SIBLING_MAX_PLIES" \
  --pv-sibling-sample-weight "$PV_SIBLING_SAMPLE_WEIGHT" \
  --pv-sibling-total-weight-cap "$PV_SIBLING_TOTAL_WEIGHT_CAP" \
  | tee "$RUN_DIR/dump_stdout.log"

env RUST_FONTCONFIG_DLOPEN=1 target/release/rank_stats \
  --input "$RUN_DIR/train.pv.tree.jsonl" \
  --input "$RUN_DIR/valid.pv.tree.jsonl" \
  --json-output "$RUN_DIR/rank_stats.json" \
  | tee "$RUN_DIR/rank_stats_stdout.log"

env RUST_FONTCONFIG_DLOPEN=1 target/release/tree_feedback_collect \
  --input "$RUN_DIR/train.pv.tree.jsonl" \
  --input "$RUN_DIR/valid.pv.tree.jsonl" \
  --output "$RUN_DIR/pv_feedback_train.json" \
  --guard-output "$RUN_DIR/pv_feedback_guard.json" \
  --guard-percent "$FEEDBACK_GUARD_PERCENT" \
  --min-candidate-regret-cp "$FEEDBACK_MIN_CANDIDATE_REGRET_CP" \
  --min-regret-delta-cp "$FEEDBACK_MIN_REGRET_DELTA_CP" \
  --max-good-regret-cp "$FEEDBACK_MAX_GOOD_REGRET_CP" \
  --limit "$FEEDBACK_LIMIT" \
  | tee "$RUN_DIR/feedback_collect_stdout.log"

python3 - "$RUN_DIR/pv_feedback_train.json" "$RUN_DIR/pv_feedback_guard.json" \
  > "$RUN_DIR/feedback_summary.txt" <<'PY'
import json
import sys

for path in sys.argv[1:]:
    payload = json.load(open(path, encoding="utf-8"))
    records = payload.get("hard_positions") or payload.get("samples") or []
    print(f"{path}: records={len(records)}")
    if records:
        regrets = [float(item.get("candidate_regret", 0.0)) for item in records]
        print(
            "  candidate_regret_mean={:.2f} min={:.2f} max={:.2f}".format(
                sum(regrets) / len(regrets), min(regrets), max(regrets)
            )
        )
PY

: > "$RUN_DIR/train.empty.tree.jsonl"
head -n "$VALID_LINES" "$RUN_DIR/valid.pv.tree.jsonl" > "$RUN_DIR/valid.top.tree.jsonl"
wc -l "$RUN_DIR/train.empty.tree.jsonl" "$RUN_DIR/valid.top.tree.jsonl" \
  | tee "$RUN_DIR/subset_counts.txt"

env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_tree_train \
  --weights "$WEIGHTS" \
  --train "$RUN_DIR/train.empty.tree.jsonl" \
  --valid "$RUN_DIR/valid.top.tree.jsonl" \
  --output "$RUN_DIR/candidate.raw.binary" \
  --best-checkpoint-path "$RUN_DIR/best.raw.binary" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --max-weight-delta "$MAX_WEIGHT_DELTA" \
  --anchor-l2 "$ANCHOR_L2" \
  --best-metric feedback-loss \
  --best-guard-feedback-violation-increase "$BEST_GUARD_FEEDBACK_VIOLATION_INCREASE" \
  --best-guard-feedback-loss-increase "$BEST_GUARD_FEEDBACK_LOSS_INCREASE" \
  --feedback-json "$RUN_DIR/pv_feedback_train.json" \
  --feedback-guard-json "$RUN_DIR/pv_feedback_guard.json" \
  --feedback-weight "$FEEDBACK_WEIGHT" \
  --feedback-min-regret-delta-cp 0 \
  --feedback-min-candidate-regret-cp 0 \
  --feedback-good-move baseline \
  --separate-aux-adagrad \
  --allow-empty-train \
  --freeze-material \
  --selected-regret-cap-cp 300 \
  --bad-regret-cp 300 \
  --bad-regret-thresholds-cp 50,100,200,300 \
  --log-path "$RUN_DIR/train.csv" \
  | tee "$RUN_DIR/train_stdout.log"

BEST_EPOCH="$(
  grep -o 'best_epoch=[0-9]*' "$RUN_DIR/train_stdout.log" \
    | tail -1 \
    | cut -d= -f2
)"

if [[ -z "$BEST_EPOCH" || "$BEST_EPOCH" == "0" ]]; then
  echo "best_epoch=${BEST_EPOCH:-none}: baseline is still best. Rejecting this run."
  rm -f "$RUN_DIR/candidate.raw.binary" "$RUN_DIR/best.raw.binary"
  echo "kept_best_raw=0" > "$RUN_DIR/final_binary_status.txt"
  echo "RUN_DIR=$RUN_DIR"
  exit 0
fi

rm -f "$RUN_DIR/candidate.raw.binary"

python3 - "$RUN_DIR/valid.top.tree.jsonl" "$RUN_DIR/score_positions.sfen" <<'PY'
import json
import sys

input_path, output_path = sys.argv[1:3]
seen = set()
with open(input_path, encoding="utf-8") as handle, open(output_path, "w", encoding="utf-8") as out:
    for line in handle:
        if not line.strip():
            continue
        sfen = json.loads(line)["sfen"].strip()
        if sfen and sfen not in seen:
            seen.add(sfen)
            out.write(sfen + "\n")
print(f"score_positions={len(seen)}")
PY

set +e
env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_score_gate \
  --baseline-weights "$WEIGHTS" \
  --candidate-weights "$RUN_DIR/best.raw.binary" \
  --input "$RUN_DIR/score_positions.sfen" \
  --max-positions 1000 \
  --seed 7202 \
  --p95-limit-cp 50 \
  --max-limit-cp 200 \
  --mean-limit-cp 10 \
  --json-output "$RUN_DIR/score_gate.json" \
  | tee "$RUN_DIR/score_gate_stdout.log"
score_status="${PIPESTATUS[0]}"
set -e

if [[ "$score_status" != "0" ]]; then
  echo "score gate failed. Rejecting this run."
  rm -f "$RUN_DIR/best.raw.binary"
  echo "kept_best_raw=0" > "$RUN_DIR/final_binary_status.txt"
  echo "RUN_DIR=$RUN_DIR"
  exit "$score_status"
fi

set +e
env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_rerank_gate \
  --baseline-weights "$WEIGHTS" \
  --candidate-weights "$RUN_DIR/best.raw.binary" \
  --teacher-weights "$TEACHER_WEIGHTS" \
  --input "$RUN_DIR/valid.pv.tree.jsonl" \
  --max-positions "$RERANK_MAX_POSITIONS" \
  --baseline-depth "$STUDENT_DEPTH" \
  --candidate-depth "$STUDENT_DEPTH" \
  --teacher-depth "$TEACHER_DEPTH" \
  --seed 7202 \
  --jobs "$JOBS" \
  --bad-regret-thresholds-cp 50,100,200,300 \
  --allow-mean-regret-increase-cp "$RERANK_ALLOW_MEAN_REGRET_INCREASE_CP" \
  --allow-p90-regret-increase-cp "$RERANK_ALLOW_P90_REGRET_INCREASE_CP" \
  --allow-p95-regret-increase-cp "$RERANK_ALLOW_P95_REGRET_INCREASE_CP" \
  --allow-bad-ratio-increase "$RERANK_ALLOW_BAD_RATIO_INCREASE" \
  --allow-phase-mean-regret-increase-cp="$RERANK_ALLOW_PHASE_MEAN_REGRET_INCREASE_CP" \
  --allow-phase-p90-regret-increase-cp="$RERANK_ALLOW_PHASE_P90_REGRET_INCREASE_CP" \
  --allow-phase-p95-regret-increase-cp="$RERANK_ALLOW_PHASE_P95_REGRET_INCREASE_CP" \
  --allow-phase-bad-ratio-increase="$RERANK_ALLOW_PHASE_BAD_RATIO_INCREASE" \
  --hard-position-limit 1000 \
  --print-worst 20 \
  --json-output "$RUN_DIR/rerank_gate.json" \
  | tee "$RUN_DIR/rerank_gate_stdout.log"
rerank_status="${PIPESTATUS[0]}"
set -e

if [[ -f "$RUN_DIR/rerank_gate.json" ]]; then
  python3 - "$RUN_DIR/rerank_gate.json" > "$RUN_DIR/hard_positions.sfen" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
for pos in payload.get("hard_positions", []):
    print(pos["sfen"])
PY
fi

if [[ "$rerank_status" != "0" ]]; then
  echo "rerank gate failed. Rejecting this run."
  rm -f "$RUN_DIR/best.raw.binary"
  echo "kept_best_raw=0" > "$RUN_DIR/final_binary_status.txt"
  echo "RUN_DIR=$RUN_DIR"
  exit "$rerank_status"
fi

if [[ "$RUN_BENCH" == "1" ]]; then
  BENCH_DIR="$RUN_DIR/bench${BENCH_GAMES}"
  env RUST_FONTCONFIG_DLOPEN=1 target/release/usi_benchmark \
    --new-engine target/release/usi_engine \
    --baseline-engine target/release/usi_engine \
    --new-weights "$RUN_DIR/best.raw.binary" \
    --baseline-weights "$WEIGHTS" \
    --positions "$BENCH_POSITIONS" \
    --games "$BENCH_GAMES" \
    --depth "$BENCH_DEPTH" \
    --time-limit-ms "$BENCH_TIME_LIMIT_MS" \
    --max-plies "$BENCH_MAX_PLIES" \
    --adjudicate-at-max-plies \
    --jobs "$BENCH_JOBS" \
    --seed "$BENCH_SEED" \
    --record-dir "$BENCH_DIR" \
    | tee "$RUN_DIR/bench_stdout.log"

  env RUST_FONTCONFIG_DLOPEN=1 target/release/record_analyze \
    --record-dir "$BENCH_DIR" \
    --weights "$WEIGHTS" \
    > "$RUN_DIR/record_analyze.log"

  NEW_WINS="$(awk '/^new wins:/ {print $3}' "$RUN_DIR/bench_stdout.log" | tail -1)"
  NEW_WINS="${NEW_WINS:-0}"
  if (( NEW_WINS >= KEEP_MIN_NEW_WINS )); then
    echo "kept_best_raw=1" > "$RUN_DIR/final_binary_status.txt"
  else
    rm -f "$RUN_DIR/best.raw.binary"
    echo "kept_best_raw=0" > "$RUN_DIR/final_binary_status.txt"
  fi
else
  echo "kept_best_raw=1" > "$RUN_DIR/final_binary_status.txt"
fi

echo "PV sibling feedback pipeline finished."
echo "RUN_DIR=$RUN_DIR"
cat "$RUN_DIR/final_binary_status.txt"
find "$RUN_DIR" -maxdepth 1 -type f -name '*.binary' -printf '%s %p\n'

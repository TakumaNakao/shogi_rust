#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script. Run it as: bash tools/run_protected_feedback_from_existing.sh"
  return 2
fi

cd "$(dirname "$0")/.."

WEIGHTS="${WEIGHTS:-policy_weights_v2.1.0.binary}"
TEACHER_WEIGHTS="${TEACHER_WEIGHTS:-$WEIGHTS}"
PV_SOURCE_RUN_DIR="${PV_SOURCE_RUN_DIR:-data/mmto/runs/pv_sibling_feedback_20260628_190857}"
STRONG_RUN_DIR="${STRONG_RUN_DIR:-data/mmto/runs/pv_sibling_strong_teacher_20260628_203933}"
FILTERED_RUN_DIR="${FILTERED_RUN_DIR:-data/mmto/runs/pv_sibling_strong_teacher_filtered_20260628_212449}"
SOURCE_VALID_TREE="${SOURCE_VALID_TREE:-$STRONG_RUN_DIR/valid.strong.tree.jsonl}"
FEEDBACK_JSON="${FEEDBACK_JSON:-$FILTERED_RUN_DIR/train_feedback.json}"
FEEDBACK_GUARD_JSON="${FEEDBACK_GUARD_JSON:-$FILTERED_RUN_DIR/feedback_guard.json}"
RUN_DIR="${RUN_DIR:-data/mmto/runs/protected_feedback_$(date -u +%Y%m%d_%H%M%S)}"
MIN_FREE_GB="${MIN_FREE_GB:-4}"

PROTECTION_SOURCES="${PROTECTION_SOURCES:-$PV_SOURCE_RUN_DIR/train.pv.tree.jsonl $PV_SOURCE_RUN_DIR/valid.pv.tree.jsonl}"
PROTECTION_LINES="${PROTECTION_LINES:-1200}"
PROTECTION_SEED="${PROTECTION_SEED:-7402}"
VALID_LINES="${VALID_LINES:-1000}"

LOSS_MODE="${LOSS_MODE:-aux-only}"
FEEDBACK_WEIGHT="${FEEDBACK_WEIGHT:-0.5}"
FEEDBACK_MIN_REGRET_DELTA_CP="${FEEDBACK_MIN_REGRET_DELTA_CP:-10}"
FEEDBACK_MIN_CANDIDATE_REGRET_CP="${FEEDBACK_MIN_CANDIDATE_REGRET_CP:-30}"
FEEDBACK_GOOD_MOVE="${FEEDBACK_GOOD_MOVE:-baseline}"
INCUMBENT_PROTECTION_WEIGHT="${INCUMBENT_PROTECTION_WEIGHT:-0.05}"
INCUMBENT_PROTECTION_MAX_REGRET_CP="${INCUMBENT_PROTECTION_MAX_REGRET_CP:-80}"
INCUMBENT_PROTECTION_ALLOW_TEACHER_BETTER_CP="${INCUMBENT_PROTECTION_ALLOW_TEACHER_BETTER_CP:-50}"

EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-0.00005}"
MAX_WEIGHT_DELTA="${MAX_WEIGHT_DELTA:-0.01}"
ANCHOR_L2="${ANCHOR_L2:-0.0003}"
BEST_GUARD_FEEDBACK_VIOLATION_INCREASE="${BEST_GUARD_FEEDBACK_VIOLATION_INCREASE:-0}"
BEST_GUARD_FEEDBACK_LOSS_INCREASE="${BEST_GUARD_FEEDBACK_LOSS_INCREASE:--1}"

STUDENT_DEPTH="${STUDENT_DEPTH:-2}"
TEACHER_DEPTH="${TEACHER_DEPTH:-4}"
RERANK_MAX_POSITIONS="${RERANK_MAX_POSITIONS:-800}"
JOBS="${JOBS:-4}"

RUN_BENCH="${RUN_BENCH:-1}"
BENCH_POSITIONS="${BENCH_POSITIONS:-taya36.sfen}"
BENCH_GAMES="${BENCH_GAMES:-20}"
BENCH_DEPTH="${BENCH_DEPTH:-5}"
BENCH_TIME_LIMIT_MS="${BENCH_TIME_LIMIT_MS:-100}"
BENCH_MAX_PLIES="${BENCH_MAX_PLIES:-200}"
BENCH_JOBS="${BENCH_JOBS:-2}"
BENCH_SEED="${BENCH_SEED:-37111}"
KEEP_MIN_NEW_WINS="${KEEP_MIN_NEW_WINS:-12}"
BENCH_FAILURE_MINE_ON_REJECT="${BENCH_FAILURE_MINE_ON_REJECT:-1}"
BENCH_FAILURE_TAIL_PLIES="${BENCH_FAILURE_TAIL_PLIES:-16}"
BENCH_FAILURE_TEACHER_DEPTH="${BENCH_FAILURE_TEACHER_DEPTH:-6}"
BENCH_FAILURE_BAD_REGRET_CP="${BENCH_FAILURE_BAD_REGRET_CP:-300}"
BENCH_FAILURE_FEEDBACK_MIN_TIMED_REGRET_CP="${BENCH_FAILURE_FEEDBACK_MIN_TIMED_REGRET_CP:-150}"
BENCH_FAILURE_FEEDBACK_MAX_TIMED_REGRET_CP="${BENCH_FAILURE_FEEDBACK_MAX_TIMED_REGRET_CP:-100000}"
BENCH_FAILURE_FEEDBACK_LIMIT="${BENCH_FAILURE_FEEDBACK_LIMIT:-200}"

for path in "$WEIGHTS" "$TEACHER_WEIGHTS" "$SOURCE_VALID_TREE"; do
  if [[ ! -f "$path" ]]; then
    echo "missing required file: $path" >&2
    exit 1
  fi
done
for path in $FEEDBACK_JSON $FEEDBACK_GUARD_JSON $PROTECTION_SOURCES; do
  if [[ ! -f "$path" ]]; then
    echo "missing required file: $path" >&2
    exit 1
  fi
done

FREE_KB="$(df -Pk . | awk 'NR==2 {print $4}')"
MIN_FREE_KB=$((MIN_FREE_GB * 1024 * 1024))
if (( FREE_KB < MIN_FREE_KB )); then
  echo "free disk is too low: $((FREE_KB / 1024 / 1024))GB < ${MIN_FREE_GB}GB" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"

echo "Starting protected feedback run from existing dumps."
echo "RUN_DIR=$RUN_DIR"
echo "WEIGHTS=$WEIGHTS"
echo "SOURCE_VALID_TREE=$SOURCE_VALID_TREE"
echo "FEEDBACK_JSON=$FEEDBACK_JSON"
echo "FEEDBACK_GUARD_JSON=$FEEDBACK_GUARD_JSON"
echo "PROTECTION_LINES=$PROTECTION_LINES LOSS_MODE=$LOSS_MODE"
echo "FEEDBACK_WEIGHT=$FEEDBACK_WEIGHT INCUMBENT_PROTECTION_WEIGHT=$INCUMBENT_PROTECTION_WEIGHT"
echo "BENCH_FAILURE_FEEDBACK_REGRET_RANGE=$BENCH_FAILURE_FEEDBACK_MIN_TIMED_REGRET_CP..$BENCH_FAILURE_FEEDBACK_MAX_TIMED_REGRET_CP"

env RUST_FONTCONFIG_DLOPEN=1 cargo build --release \
  --bin mmto_tree_train \
  --bin mmto_score_gate \
  --bin mmto_rerank_gate \
  --bin usi_engine \
  --bin usi_benchmark \
  --bin record_analyze \
  --bin bench_failure_miner \
  --bin bench_failure_feedback

feedback_args=()
for path in $FEEDBACK_JSON; do
  if [[ ! -f "$path" ]]; then
    echo "missing feedback json: $path" >&2
    exit 1
  fi
  feedback_args+=(--feedback-json "$path")
done

feedback_guard_args=()
for path in $FEEDBACK_GUARD_JSON; do
  if [[ ! -f "$path" ]]; then
    echo "missing feedback guard json: $path" >&2
    exit 1
  fi
  feedback_guard_args+=(--feedback-guard-json "$path")
done

python3 - "$RUN_DIR/train.protection.tree.jsonl" "$PROTECTION_LINES" "$PROTECTION_SEED" $PROTECTION_SOURCES <<'PY'
import json
import random
import sys
from collections import defaultdict

out_path = sys.argv[1]
limit = int(sys.argv[2])
seed = int(sys.argv[3])
sources = sys.argv[4:]
rng = random.Random(seed)

records = []
seen = set()
for path in sources:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            sfen = str(item.get("sfen", "")).strip()
            if not sfen or sfen in seen:
                continue
            seen.add(sfen)
            records.append((item, line.rstrip("\n")))

selected = []
selected_sfens = set()

def add_from_bucket(bucket, quota):
    rng.shuffle(bucket)
    added = 0
    for item, raw in bucket:
        if added >= quota:
            break
        sfen = str(item.get("sfen", "")).strip()
        if sfen in selected_sfens:
            continue
        selected.append(raw)
        selected_sfens.add(sfen)
        added += 1
    return added

in_check = [(item, raw) for item, raw in records if item.get("in_check")]
low_legal = [
    (item, raw)
    for item, raw in records
    if int(item.get("legal_moves") or 999) <= 20
]
special_quota = max(0, limit // 5)
add_from_bucket(in_check, special_quota // 2)
add_from_bucket(low_legal, special_quota - len(selected))

phase_buckets = defaultdict(list)
for item, raw in records:
    phase_buckets[str(item.get("phase") or "unknown")].append((item, raw))

phases = ["opening", "middle", "late", "endgame", "unknown"]
while len(selected) < limit:
    before = len(selected)
    remaining = limit - len(selected)
    per_phase = max(1, (remaining + len(phases) - 1) // len(phases))
    for phase in phases:
        if len(selected) >= limit:
            break
        add_from_bucket(phase_buckets.get(phase, []), min(per_phase, limit - len(selected)))
    if len(selected) == before:
        break

if len(selected) < limit:
    add_from_bucket(records, limit - len(selected))

with open(out_path, "w", encoding="utf-8") as out:
    for raw in selected:
        out.write(raw + "\n")

phase_counts = defaultdict(int)
special_counts = {"in_check": 0, "low_legal": 0}
for raw in selected:
    item = json.loads(raw)
    phase_counts[str(item.get("phase") or "unknown")] += 1
    if item.get("in_check"):
        special_counts["in_check"] += 1
    if int(item.get("legal_moves") or 999) <= 20:
        special_counts["low_legal"] += 1
print(f"protection_records={len(selected)}")
for key in sorted(phase_counts):
    print(f"protection_phase[{key}]={phase_counts[key]}")
for key, value in special_counts.items():
    print(f"protection_{key}={value}")
PY

head -n "$VALID_LINES" "$SOURCE_VALID_TREE" > "$RUN_DIR/valid.top.tree.jsonl"
wc -l "$RUN_DIR/train.protection.tree.jsonl" "$RUN_DIR/valid.top.tree.jsonl" \
  | tee "$RUN_DIR/subset_counts.txt"

env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_tree_train \
  --weights "$WEIGHTS" \
  --train "$RUN_DIR/train.protection.tree.jsonl" \
  --valid "$RUN_DIR/valid.top.tree.jsonl" \
  --output "$RUN_DIR/candidate.raw.binary" \
  --best-checkpoint-path "$RUN_DIR/best.raw.binary" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --loss-mode "$LOSS_MODE" \
  --max-weight-delta "$MAX_WEIGHT_DELTA" \
  --anchor-l2 "$ANCHOR_L2" \
  --best-metric feedback-loss \
  --best-guard-feedback-violation-increase="$BEST_GUARD_FEEDBACK_VIOLATION_INCREASE" \
  --best-guard-feedback-loss-increase="$BEST_GUARD_FEEDBACK_LOSS_INCREASE" \
  "${feedback_args[@]}" \
  "${feedback_guard_args[@]}" \
  --feedback-weight "$FEEDBACK_WEIGHT" \
  --feedback-min-regret-delta-cp "$FEEDBACK_MIN_REGRET_DELTA_CP" \
  --feedback-min-candidate-regret-cp "$FEEDBACK_MIN_CANDIDATE_REGRET_CP" \
  --feedback-good-move "$FEEDBACK_GOOD_MOVE" \
  --incumbent-protection-weight "$INCUMBENT_PROTECTION_WEIGHT" \
  --incumbent-protection-max-regret-cp "$INCUMBENT_PROTECTION_MAX_REGRET_CP" \
  --incumbent-protection-allow-teacher-better-cp "$INCUMBENT_PROTECTION_ALLOW_TEACHER_BETTER_CP" \
  --separate-aux-adagrad \
  --freeze-material \
  --selected-regret-cap-cp 300 \
  --bad-regret-cp 300 \
  --bad-regret-thresholds-cp 50,100,200,300 \
  --log-path "$RUN_DIR/train.csv" \
  | tee "$RUN_DIR/train_stdout.log"

BEST_EPOCH="$(
  sed -n 's/.*best_epoch=\([0-9][0-9]*\).*/\1/p' "$RUN_DIR/train_stdout.log" \
    | tail -1
)"

rm -f "$RUN_DIR/candidate.raw.binary"
if [[ -z "$BEST_EPOCH" || "$BEST_EPOCH" == "0" ]]; then
  echo "best_epoch=${BEST_EPOCH:-none}: baseline is still best. Rejecting this run."
  rm -f "$RUN_DIR/best.raw.binary"
  echo "kept_best_raw=0" > "$RUN_DIR/final_binary_status.txt"
  echo "RUN_DIR=$RUN_DIR"
  exit 0
fi

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
  --seed 7501 \
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
  --input "$SOURCE_VALID_TREE" \
  --max-positions "$RERANK_MAX_POSITIONS" \
  --baseline-depth "$STUDENT_DEPTH" \
  --candidate-depth "$STUDENT_DEPTH" \
  --teacher-depth "$TEACHER_DEPTH" \
  --seed 7501 \
  --jobs "$JOBS" \
  --bad-regret-thresholds-cp 50,100,200,300 \
  --allow-mean-regret-increase-cp 0 \
  --allow-p90-regret-increase-cp 0 \
  --allow-p95-regret-increase-cp 0 \
  --allow-bad-ratio-increase 0 \
  --allow-phase-mean-regret-increase-cp=-1 \
  --allow-phase-p90-regret-increase-cp=-1 \
  --allow-phase-p95-regret-increase-cp=-1 \
  --allow-phase-bad-ratio-increase=-1 \
  --hard-position-limit 1000 \
  --print-worst 20 \
  --json-output "$RUN_DIR/rerank_gate.json" \
  | tee "$RUN_DIR/rerank_gate_stdout.log"
rerank_status="${PIPESTATUS[0]}"
set -e

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
    if [[ "$BENCH_FAILURE_MINE_ON_REJECT" == "1" ]]; then
      set +e
      env RUST_FONTCONFIG_DLOPEN=1 target/release/bench_failure_miner \
        --weights "$RUN_DIR/best.raw.binary" \
        --record-dir "$BENCH_DIR" \
        --tail-plies "$BENCH_FAILURE_TAIL_PLIES" \
        --timed-depth "$BENCH_DEPTH" \
        --time-limit-ms "$BENCH_TIME_LIMIT_MS" \
        --teacher-depth "$BENCH_FAILURE_TEACHER_DEPTH" \
        --bad-regret-cp "$BENCH_FAILURE_BAD_REGRET_CP" \
        --jsonl-output "$RUN_DIR/bench_failure.jsonl" \
        --export-timed-bad-sfens "$RUN_DIR/bench_failure_timed_bad.sfen" \
        --export-root-rescue-sfens "$RUN_DIR/bench_failure_root_rescue.sfen" \
        > "$RUN_DIR/bench_failure_miner.log" 2>&1
      failure_miner_status="$?"
      if [[ "$failure_miner_status" == "0" ]]; then
        env RUST_FONTCONFIG_DLOPEN=1 target/release/bench_failure_feedback \
          --input "$RUN_DIR/bench_failure.jsonl" \
          --output "$RUN_DIR/bench_failure_feedback.json" \
          --min-timed-regret-cp "$BENCH_FAILURE_FEEDBACK_MIN_TIMED_REGRET_CP" \
          --max-timed-regret-cp "$BENCH_FAILURE_FEEDBACK_MAX_TIMED_REGRET_CP" \
          --limit "$BENCH_FAILURE_FEEDBACK_LIMIT" \
          > "$RUN_DIR/bench_failure_feedback.log" 2>&1
        failure_feedback_status="$?"
      else
        failure_feedback_status="skipped"
      fi
      set -e
      {
        echo "bench_failure_miner_status=$failure_miner_status"
        echo "bench_failure_feedback_status=$failure_feedback_status"
      } > "$RUN_DIR/bench_failure_status.txt"
    fi
    rm -f "$RUN_DIR/best.raw.binary"
    echo "kept_best_raw=0" > "$RUN_DIR/final_binary_status.txt"
  fi
else
  echo "kept_best_raw=1" > "$RUN_DIR/final_binary_status.txt"
fi

echo "Protected feedback run finished."
echo "RUN_DIR=$RUN_DIR"
cat "$RUN_DIR/final_binary_status.txt"
find "$RUN_DIR" -maxdepth 1 -type f -name '*.binary' -printf '%s %p\n'

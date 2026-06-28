#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Do not source this script. Run it as: bash tools/run_pv_sibling_strong_teacher_feedback.sh"
  return 2
fi

cd "$(dirname "$0")/.."

SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-data/mmto/runs/pv_sibling_feedback_20260628_190857}"
WEIGHTS="${WEIGHTS:-policy_weights_v2.1.0.binary}"
TEACHER_WEIGHTS="${TEACHER_WEIGHTS:-$WEIGHTS}"
RUN_DIR="${RUN_DIR:-data/mmto/runs/pv_sibling_strong_teacher_$(date -u +%Y%m%d_%H%M%S)}"
MIN_FREE_GB="${MIN_FREE_GB:-5}"

HARD_LIMIT="${HARD_LIMIT:-800}"
HARD_SOURCES="${HARD_SOURCES:-$SOURCE_RUN_DIR/pv_feedback_train.json $SOURCE_RUN_DIR/pv_feedback_guard.json}"
TEACHER_DEPTH="${TEACHER_DEPTH:-4}"
STUDENT_DEPTH="${STUDENT_DEPTH:-2}"
TEACHER_SCORE_TOP="${TEACHER_SCORE_TOP:-24}"
CANDIDATE_TOP="${CANDIDATE_TOP:-24}"
POSITION_CHUNK_SIZE="${POSITION_CHUNK_SIZE:-64}"
VALID_PERCENT="${VALID_PERCENT:-20}"
JOBS="${JOBS:-4}"
MAX_ABS_ROOT_SCORE="${MAX_ABS_ROOT_SCORE:-30000}"
PV_SIBLING_MAX_PLIES="${PV_SIBLING_MAX_PLIES:-1}"
PV_SIBLING_SAMPLE_WEIGHT="${PV_SIBLING_SAMPLE_WEIGHT:-0.20}"
PV_SIBLING_TOTAL_WEIGHT_CAP="${PV_SIBLING_TOTAL_WEIGHT_CAP:-0.20}"

FEEDBACK_MIN_CANDIDATE_REGRET_CP="${FEEDBACK_MIN_CANDIDATE_REGRET_CP:-30}"
FEEDBACK_MAX_CANDIDATE_REGRET_CP="${FEEDBACK_MAX_CANDIDATE_REGRET_CP:-600}"
FEEDBACK_MIN_REGRET_DELTA_CP="${FEEDBACK_MIN_REGRET_DELTA_CP:-10}"
FEEDBACK_MAX_REGRET_DELTA_CP="${FEEDBACK_MAX_REGRET_DELTA_CP:-600}"
FEEDBACK_MAX_GOOD_REGRET_CP="${FEEDBACK_MAX_GOOD_REGRET_CP:-30}"
FEEDBACK_LIMIT="${FEEDBACK_LIMIT:-2500}"
FEEDBACK_GUARD_PERCENT="${FEEDBACK_GUARD_PERCENT:-25}"
FEEDBACK_WEIGHT="${FEEDBACK_WEIGHT:-0.5}"

PROTECTION_SOURCES="${PROTECTION_SOURCES:-$SOURCE_RUN_DIR/train.pv.tree.jsonl $SOURCE_RUN_DIR/valid.pv.tree.jsonl}"
PROTECTION_LINES="${PROTECTION_LINES:-1200}"
PROTECTION_SEED="${PROTECTION_SEED:-7402}"
LOSS_MODE="${LOSS_MODE:-aux-only}"
INCUMBENT_PROTECTION_WEIGHT="${INCUMBENT_PROTECTION_WEIGHT:-0}"
INCUMBENT_PROTECTION_MAX_REGRET_CP="${INCUMBENT_PROTECTION_MAX_REGRET_CP:-80}"
INCUMBENT_PROTECTION_ALLOW_TEACHER_BETTER_CP="${INCUMBENT_PROTECTION_ALLOW_TEACHER_BETTER_CP:-50}"
POLICY_ANCHOR_WEIGHTS="${POLICY_ANCHOR_WEIGHTS:-}"
POLICY_ANCHOR_WEIGHT="${POLICY_ANCHOR_WEIGHT:-0}"
POLICY_ANCHOR_TEMPERATURE_CP="${POLICY_ANCHOR_TEMPERATURE_CP:-100}"
POLICY_ANCHOR_FEATURE_SOURCE="${POLICY_ANCHOR_FEATURE_SOURCE:-move}"
POLICY_ANCHOR_MARGIN_WEIGHT="${POLICY_ANCHOR_MARGIN_WEIGHT:-0}"
POLICY_ANCHOR_MARGIN_CP="${POLICY_ANCHOR_MARGIN_CP:-50}"
POLICY_ANCHOR_MARGIN_SOFTPLUS_TEMP_CP="${POLICY_ANCHOR_MARGIN_SOFTPLUS_TEMP_CP:-100}"

EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-0.00005}"
MAX_WEIGHT_DELTA="${MAX_WEIGHT_DELTA:-0.01}"
ANCHOR_L2="${ANCHOR_L2:-0.0003}"
BEST_GUARD_FEEDBACK_VIOLATION_INCREASE="${BEST_GUARD_FEEDBACK_VIOLATION_INCREASE:-0}"
BEST_GUARD_FEEDBACK_LOSS_INCREASE="${BEST_GUARD_FEEDBACK_LOSS_INCREASE:--1}"
VALID_LINES="${VALID_LINES:-1000}"

RERANK_MAX_POSITIONS="${RERANK_MAX_POSITIONS:-800}"
RUN_BENCH="${RUN_BENCH:-1}"
BENCH_POSITIONS="${BENCH_POSITIONS:-taya36.sfen}"
BENCH_GAMES="${BENCH_GAMES:-20}"
BENCH_DEPTH="${BENCH_DEPTH:-5}"
BENCH_TIME_LIMIT_MS="${BENCH_TIME_LIMIT_MS:-100}"
BENCH_MAX_PLIES="${BENCH_MAX_PLIES:-200}"
BENCH_JOBS="${BENCH_JOBS:-2}"
BENCH_SEED="${BENCH_SEED:-37001}"
KEEP_MIN_NEW_WINS="${KEEP_MIN_NEW_WINS:-12}"

for path in "$WEIGHTS" "$TEACHER_WEIGHTS"; do
  if [[ ! -f "$path" ]]; then
    echo "missing required file: $path" >&2
    exit 1
  fi
done

FREE_KB="$(df -Pk . | awk 'NR==2 {print $4}')"
MIN_FREE_KB=$((MIN_FREE_GB * 1024 * 1024))
if (( FREE_KB < MIN_FREE_KB )); then
  echo "free disk is too low: $((FREE_KB / 1024 / 1024))GB < ${MIN_FREE_GB}GB" >&2
  echo "Clean old generated runs first: bash tools/clean_mmto_runs.sh" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"

echo "Starting PV sibling strong-teacher feedback run."
echo "RUN_DIR=$RUN_DIR"
echo "SOURCE_RUN_DIR=$SOURCE_RUN_DIR"
echo "HARD_LIMIT=$HARD_LIMIT"
echo "TEACHER_DEPTH=$TEACHER_DEPTH STUDENT_DEPTH=$STUDENT_DEPTH"
echo "FEEDBACK_REGRET_RANGE=$FEEDBACK_MIN_CANDIDATE_REGRET_CP..$FEEDBACK_MAX_CANDIDATE_REGRET_CP"
echo "FEEDBACK_WEIGHT=$FEEDBACK_WEIGHT EPOCHS=$EPOCHS LEARNING_RATE=$LEARNING_RATE"
echo "PROTECTION_LINES=$PROTECTION_LINES LOSS_MODE=$LOSS_MODE"
echo "INCUMBENT_PROTECTION_WEIGHT=$INCUMBENT_PROTECTION_WEIGHT MAX_REGRET=$INCUMBENT_PROTECTION_MAX_REGRET_CP ALLOW_TEACHER_BETTER=$INCUMBENT_PROTECTION_ALLOW_TEACHER_BETTER_CP"
echo "POLICY_ANCHOR_WEIGHTS=$POLICY_ANCHOR_WEIGHTS POLICY_ANCHOR_WEIGHT=$POLICY_ANCHOR_WEIGHT POLICY_ANCHOR_MARGIN_WEIGHT=$POLICY_ANCHOR_MARGIN_WEIGHT"

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

python3 - "$RUN_DIR/hard_relabel_input.jsonl" "$HARD_LIMIT" $HARD_SOURCES <<'PY'
import json
import sys

out_path = sys.argv[1]
limit = int(sys.argv[2])
sources = sys.argv[3:]
seen = set()
written = 0
with open(out_path, "w", encoding="utf-8") as out:
    for path in sources:
        try:
            payload = json.load(open(path, encoding="utf-8"))
        except FileNotFoundError:
            continue
        records = payload.get("hard_positions") or payload.get("samples") or []
        for item in records:
            sfen = str(item.get("sfen", "")).strip()
            if not sfen or sfen in seen:
                continue
            teacher_move = (
                item.get("teacher_best_move")
                or item.get("baseline_move")
                or item.get("teacher_move")
            )
            bad_move = (
                item.get("candidate_move")
                or item.get("bad_move")
                or item.get("selected_move")
            )
            record = {"sfen": sfen}
            if teacher_move:
                record["teacher_best_move"] = teacher_move
            if bad_move:
                record["candidate_move"] = bad_move
            regret = item.get("candidate_regret") or item.get("regret_delta") or 0.0
            try:
                regret = max(0.0, float(regret))
            except (TypeError, ValueError):
                regret = 0.0
            record["sample_weight"] = min(3.0, 1.0 + regret / 200.0)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            seen.add(sfen)
            written += 1
            if written >= limit:
                print(f"hard_relabel_positions={written}")
                raise SystemExit(0)
print(f"hard_relabel_positions={written}")
if written == 0:
    raise SystemExit("no hard positions extracted")
PY

env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_tree_dump \
  --student-weights "$WEIGHTS" \
  --teacher-weights "$TEACHER_WEIGHTS" \
  --input "$RUN_DIR/hard_relabel_input.jsonl" \
  --train-output "$RUN_DIR/train.strong.tree.jsonl" \
  --valid-output "$RUN_DIR/valid.strong.tree.jsonl" \
  --teacher-depth "$TEACHER_DEPTH" \
  --student-depth "$STUDENT_DEPTH" \
  --teacher-score-top "$TEACHER_SCORE_TOP" \
  --candidate-top "$CANDIDATE_TOP" \
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
  --input "$RUN_DIR/train.strong.tree.jsonl" \
  --input "$RUN_DIR/valid.strong.tree.jsonl" \
  --json-output "$RUN_DIR/rank_stats.json" \
  | tee "$RUN_DIR/rank_stats_stdout.log"

env RUST_FONTCONFIG_DLOPEN=1 target/release/tree_feedback_collect \
  --input "$RUN_DIR/train.strong.tree.jsonl" \
  --input "$RUN_DIR/valid.strong.tree.jsonl" \
  --output "$RUN_DIR/strong_feedback_train.json" \
  --guard-output "$RUN_DIR/strong_feedback_guard.json" \
  --guard-percent "$FEEDBACK_GUARD_PERCENT" \
  --min-candidate-regret-cp "$FEEDBACK_MIN_CANDIDATE_REGRET_CP" \
  --max-candidate-regret-cp "$FEEDBACK_MAX_CANDIDATE_REGRET_CP" \
  --min-regret-delta-cp "$FEEDBACK_MIN_REGRET_DELTA_CP" \
  --max-regret-delta-cp "$FEEDBACK_MAX_REGRET_DELTA_CP" \
  --max-good-regret-cp "$FEEDBACK_MAX_GOOD_REGRET_CP" \
  --limit "$FEEDBACK_LIMIT" \
  | tee "$RUN_DIR/feedback_collect_stdout.log"

python3 - "$RUN_DIR/strong_feedback_train.json" "$RUN_DIR/strong_feedback_guard.json" \
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
    try:
        handle = open(path, encoding="utf-8")
    except FileNotFoundError:
        continue
    with handle:
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

if limit > 0 and records:
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

head -n "$VALID_LINES" "$RUN_DIR/valid.strong.tree.jsonl" > "$RUN_DIR/valid.top.tree.jsonl"
wc -l "$RUN_DIR/train.protection.tree.jsonl" "$RUN_DIR/valid.top.tree.jsonl" \
  | tee "$RUN_DIR/subset_counts.txt"

policy_anchor_args=()
if [[ -n "$POLICY_ANCHOR_WEIGHTS" ]]; then
  if [[ ! -f "$POLICY_ANCHOR_WEIGHTS" ]]; then
    echo "missing policy anchor weights: $POLICY_ANCHOR_WEIGHTS" >&2
    exit 1
  fi
  policy_anchor_args+=(--policy-anchor-weights "$POLICY_ANCHOR_WEIGHTS")
fi

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
  --feedback-json "$RUN_DIR/strong_feedback_train.json" \
  --feedback-guard-json "$RUN_DIR/strong_feedback_guard.json" \
  --feedback-weight "$FEEDBACK_WEIGHT" \
  --feedback-min-regret-delta-cp 0 \
  --feedback-min-candidate-regret-cp 0 \
  --feedback-good-move baseline \
  --incumbent-protection-weight "$INCUMBENT_PROTECTION_WEIGHT" \
  --incumbent-protection-max-regret-cp "$INCUMBENT_PROTECTION_MAX_REGRET_CP" \
  --incumbent-protection-allow-teacher-better-cp "$INCUMBENT_PROTECTION_ALLOW_TEACHER_BETTER_CP" \
  "${policy_anchor_args[@]}" \
  --policy-anchor-weight "$POLICY_ANCHOR_WEIGHT" \
  --policy-anchor-temperature-cp "$POLICY_ANCHOR_TEMPERATURE_CP" \
  --policy-anchor-feature-source "$POLICY_ANCHOR_FEATURE_SOURCE" \
  --policy-anchor-margin-weight "$POLICY_ANCHOR_MARGIN_WEIGHT" \
  --policy-anchor-margin-cp "$POLICY_ANCHOR_MARGIN_CP" \
  --policy-anchor-margin-softplus-temp-cp "$POLICY_ANCHOR_MARGIN_SOFTPLUS_TEMP_CP" \
  --separate-aux-adagrad \
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
  --seed 7401 \
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
  --input "$RUN_DIR/valid.strong.tree.jsonl" \
  --max-positions "$RERANK_MAX_POSITIONS" \
  --baseline-depth "$STUDENT_DEPTH" \
  --candidate-depth "$STUDENT_DEPTH" \
  --teacher-depth "$TEACHER_DEPTH" \
  --seed 7401 \
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
    rm -f "$RUN_DIR/best.raw.binary"
    echo "kept_best_raw=0" > "$RUN_DIR/final_binary_status.txt"
  fi
else
  echo "kept_best_raw=1" > "$RUN_DIR/final_binary_status.txt"
fi

echo "PV sibling strong-teacher feedback run finished."
echo "RUN_DIR=$RUN_DIR"
cat "$RUN_DIR/final_binary_status.txt"
find "$RUN_DIR" -maxdepth 1 -type f -name '*.binary' -printf '%s %p\n'

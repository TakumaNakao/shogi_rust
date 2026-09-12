#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Reuse the audited T13 2M teacher corpus.  The accepted gen6 model is only
# the optimizer initialization; it is never used as a training target.
RUN_DIR="${RUN_DIR:-$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen6-2m-extended-seed20260911}"
DATA_DIR="$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen6-2m-20260911"
TRAIN="$DATA_DIR/train.dedup-expanded.hkst"
VALID="$DATA_DIR/valid.filtered.hkst"
TEST="$DATA_DIR/test.filtered.hkst"
PARENT="$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen5-20260910/candidate/model-epoch24.bin"
PARENT_MANIFEST="$PARENT.manifest.json"
REGISTRY="$ROOT_DIR/training/halfkp64-scratch-v1/promotion_registry.json"
GATE="$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen5-20260910/match-confirm-20k-250pair/report.json"
CHECKPOINTS="$RUN_DIR/checkpoints"
MODEL="$RUN_DIR/model.bin"
LOG="$RUN_DIR/train.csv"
DECISION="$RUN_DIR/extend-decision.json"
STAGE_STATE="$RUN_DIR/stage-state.json"
TEST_COMPARE="$RUN_DIR/test-value-compare.json"
CONFIG="$RUN_DIR/config.json"
WORKERS="${WORKERS:-12}"
SEED="${SEED:-20260911}"
MAX_EPOCH="${MAX_EPOCH:-32}"

mkdir -p "$RUN_DIR"
exec 9>"$RUN_DIR/.lock"
flock -n 9 || exit 0
exec > >(tee -a "$RUN_DIR/pipeline.log") 2>&1

need() { [[ -s "$1" ]] || { echo "missing: $1" >&2; exit 2; }; }

validate_inputs() {
  need "$TRAIN"; need "$VALID"; need "$TEST"
  need "$PARENT"; need "$PARENT_MANIFEST"; need "$REGISTRY"; need "$GATE"
  [[ "$(sha256sum "$PARENT" | awk '{print $1}')" == 162e17bacea3de80ecdbbdae765ca1157246c4febb6fa6a15f8d36b900d011dc ]] || {
    echo "accepted parent SHA mismatch" >&2; exit 2
  }
  python3 - "$TRAIN" "$VALID" "$TEST" <<'PY'
import json, pathlib, sys
for name in sys.argv[1:]:
    p = pathlib.Path(name)
    m = pathlib.Path(str(p) + '.manifest.json')
    if not m.is_file():
        raise SystemExit(f'missing manifest: {m}')
    x = json.loads(m.read_text())
    if int(x.get('records', 0)) <= 0:
        raise SystemExit(f'empty dataset: {p}')
print('validated T13 2M datasets')
PY
}

write_config() {
  [[ -s "$CONFIG" ]] && return
  python3 - "$CONFIG" "$TRAIN" "$VALID" "$TEST" <<'PY'
import json, os, pathlib, sys
out, train, valid, test = sys.argv[1:]
pathlib.Path(out).write_text(json.dumps({
  'schema': 'halfkp_v4_gen6_2m_extended_v1',
  'lineage_id': 'halfkp64-scratch-v1',
  'parent_generation': 6,
  'parent_model_sha256': '162e17bacea3de80ecdbbdae765ca1157246c4febb6fa6a15f8d36b900d011dc',
  'train': train, 'valid': valid, 'test': test,
  'optimizer': 'adagrad', 'feature_learning_rate': 3e-4,
  'output_learning_rate': 3e-5, 'batch_size': 128,
  'kappa_cp': 1352.7871, 'search_mix': 0.8, 'value_weight': 1.0,
  'rank_weight': 0.0, 'loss_power': 2.5, 'gradient_clip_norm': 1.0,
  'output_limit': 0.20, 'early_stop_patience': 1000000,
  'swa_start_epoch': 1000000, 'max_epoch': int(os.environ['MAX_EPOCH']),
  'workers': int(os.environ['WORKERS']), 'seed': int(os.environ['SEED']),
}, indent=2, sort_keys=True) + '\n')
PY
}

train_to() {
  local target="$1" snapshot="$CHECKPOINTS/model-epoch-$1.bin"
  [[ -s "$snapshot" ]] && { echo "stage epoch=$target already complete"; return; }
  mkdir -p "$CHECKPOINTS"
  local args=(
    --init "$PARENT" --model-manifest "$PARENT_MANIFEST"
    --promotion-registry "$REGISTRY" --gate-report "$GATE" --parent-generation 6
    --output "$MODEL" --epochs "$target" --batch-size 128
    --learning-rate .0003 --output-learning-rate .00003 --kappa-cp 1352.7871
    --search-mix .8 --search-mix-end .8 --value-weight 1
    --rank-weight 0 --game-rank-weight 0 --game-vs-search-weight 0
    --loss-power 2.5 --gradient-clip-norm 1 --output-limit .20
    --optimizer adagrad --threads "$WORKERS" --seed "$SEED"
    --early-stop-patience 1000000 --min-valid-improvement 0.000001
    --swa-start-epoch 1000000 --checkpoint-dir "$CHECKPOINTS" --log "$LOG"
    --train "$TRAIN" --valid "$VALID"
  )
  if [[ -s "$CHECKPOINTS/state.json" && -s "$MODEL" ]]; then
    args+=(--resume)
  elif [[ -e "$CHECKPOINTS/state.json" ]]; then
    echo "checkpoint state exists without model output" >&2
    exit 2
  else
    # Only clean incomplete files belonging to this new run.
    rm -f "$MODEL" "$MODEL.tmp" "$MODEL.manifest.json" "$MODEL.manifest.json.tmp"
  fi
  echo "starting training stage target_epoch=$target workers=$WORKERS"
  RUST_FONTCONFIG_DLOPEN=1 target/release/halfkp_search_train "${args[@]}"
  need "$MODEL"; need "$MODEL.manifest.json"
  local completed
  completed="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["completed_epoch"])' "$MODEL.manifest.json")"
  [[ "$completed" == "$target" ]] || {
    echo "stage target=$target completed=$completed" >&2; exit 2
  }
  cp "$MODEL" "$snapshot"
  cp "$MODEL.manifest.json" "$snapshot.manifest.json"
  echo "completed training stage epoch=$completed"
}

write_decision() {
  python3 - "$LOG" "$DECISION" <<'PY'
import csv, json, math, pathlib, sys
log, out = sys.argv[1:]
rows = []
with open(log, newline='') as f:
    for r in csv.DictReader(f):
        try:
            row = {k: float(r[k]) for k in ('epoch', 'score', 'valid_model_result_brier', 'valid_model_result_logloss')}
        except (KeyError, ValueError):
            continue
        if all(math.isfinite(v) for v in row.values()):
            rows.append(row)
if not rows:
    raise SystemExit('no finite training metrics')
def best(cutoff):
    x = [r for r in rows if r['epoch'] <= cutoff]
    return min(x, key=lambda r: (r['score'], r['epoch'])) if x else None
b16, b32 = best(16), best(32)
continue64 = False
reason = 'epoch32_not_available'
score_gain = brier_delta = logloss_delta = None
if b16 and b32:
    score_gain = b16['score'] - b32['score']
    brier_delta = b32['valid_model_result_brier'] - b16['valid_model_result_brier']
    logloss_delta = b32['valid_model_result_logloss'] - b16['valid_model_result_logloss']
    continue64 = score_gain >= 1e-6 and brier_delta <= 1e-4 and logloss_delta <= 1e-4
    reason = 'continue64' if continue64 else 'stop_at32'
pathlib.Path(out).write_text(json.dumps({
    'schema': 'halfkp_extended_decision_v1', 'reason': reason,
    'continue64': continue64, 'epoch16_best': b16, 'epoch32_best': b32,
    'score_gain_16_to_32': score_gain,
    'result_brier_delta_16_to_32': brier_delta,
    'result_logloss_delta_16_to_32': logloss_delta,
}, indent=2, sort_keys=True) + '\n')
print(reason)
PY
}

record_stage() {
  local target="$1"
  python3 - "$LOG" "$STAGE_STATE" "$target" <<'PY'
import csv, json, math, pathlib, sys
log, state_path, target = sys.argv[1], pathlib.Path(sys.argv[2]), int(sys.argv[3])
rows = []
with open(log, newline='') as f:
    for r in csv.DictReader(f):
        try:
            row = {k: float(r[k]) for k in ('epoch', 'score', 'valid_model_result_brier')}
        except (KeyError, ValueError):
            continue
        if int(row['epoch']) <= target and all(math.isfinite(v) for v in row.values()):
            rows.append(row)
if not rows:
    raise SystemExit('no finite stage metrics')
current = next(r for r in rows if int(r['epoch']) == target)
prior = [r for r in rows if int(r['epoch']) < target]
best_score = min((r['score'] for r in prior), default=current['score'])
best_brier = min((r['valid_model_result_brier'] for r in prior), default=current['valid_model_result_brier'])
bad = bool(prior and current['score'] > best_score + 1e-4 and current['valid_model_result_brier'] > best_brier + 1e-4)
state = json.loads(state_path.read_text()) if state_path.exists() else {'schema': 'halfkp_stage_state_v1', 'stages': []}
state['stages'].append({'epoch': target, 'score': current['score'], 'result_brier': current['valid_model_result_brier'], 'bad_against_prior_best': bad})
state['stages'] = state['stages'][-32:]
state['consecutive_bad_stages'] = 0
for item in reversed(state['stages']):
    if item['bad_against_prior_best']:
        state['consecutive_bad_stages'] += 1
    else:
        break
state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + '\n')
if state['consecutive_bad_stages'] >= 2:
    print(f'stage_stop epoch={target} consecutive_bad={state["consecutive_bad_stages"]}')
    raise SystemExit(10)
print(f'stage_ok epoch={target} bad={bad}')
PY
}

validate_inputs
export MAX_EPOCH WORKERS SEED
write_config
run_stage() {
  local target="$1" rc
  train_to "$target"
  record_stage "$target" || rc=$?
  if [[ "${rc:-0}" == 10 ]]; then
    echo "stopping at stage=$target after repeated degradation"
    STAGE_STOP=1
  elif [[ "${rc:-0}" != 0 ]]; then
    exit "${rc}"
  fi
}
STAGE_STOP=0
for target in 4 8 12 16 20 24 28 32; do
  run_stage "$target"
  [[ "$STAGE_STOP" == 0 ]] || break
done
write_decision
if [[ "$STAGE_STOP" == 0 ]] && python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("continue64") else 1)' "$DECISION"; then
  for target in 36 40 44 48 52 56 60 64; do
    run_stage "$target"
    [[ "$STAGE_STOP" == 0 ]] || break
  done
  write_decision
fi

need "$MODEL"
RUST_FONTCONFIG_DLOPEN=1 target/release/halfkp_value_compare \
  --dataset "$TEST" --baseline-model "$PARENT" --candidate-model "$MODEL" \
  --kappa-cp 1352.7871 --search-mix .8 --output "$TEST_COMPARE"
python3 - "$RUN_DIR/completion.json" "$MODEL.manifest.json" "$DECISION" "$TEST_COMPARE" <<'PY'
import json, pathlib, sys, time
out, manifest, decision, compare = sys.argv[1:]
m = json.loads(pathlib.Path(manifest).read_text())
pathlib.Path(out).write_text(json.dumps({
  'schema': 'halfkp_extended_completion_v1',
  'completed_epoch': m['completed_epoch'], 'model_sha256': m['model']['sha256'],
  'selected_kind': m['selected_kind'], 'selected_epoch': m['selected_epoch'],
  'selected_validation_score': m['selected_validation_score'],
  'decision': json.loads(pathlib.Path(decision).read_text()),
  'test_value_compare': json.loads(pathlib.Path(compare).read_text()),
  'completed_unix_secs': int(time.time()),
}, indent=2, sort_keys=True) + '\n')
PY
echo "extended 2M training complete: $RUN_DIR"

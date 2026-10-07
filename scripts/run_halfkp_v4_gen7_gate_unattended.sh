#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen6-2m-extended-seed20260911}"
TRAINING_COMPLETION="$RUN_DIR/completion.json"
GATE_DIR="$RUN_DIR/gate"
SCREEN_DIR="$GATE_DIR/screen-20k-250pair"
CONFIRM_DIR="$GATE_DIR/confirm-20k-250pair"
SCREEN_OPENINGS="$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen6-2m-20260911/match-screen-openings-501-750.jsonl"
OPENING_SOURCE="$ROOT_DIR/training/halfkp64-scratch-v1/final_openings_seed20260730_2000.jsonl"
CONFIRM_OPENINGS="$GATE_DIR/confirm-openings-1001-1250.jsonl"
PARENT="$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen5-20260910/candidate/model-epoch24.bin"
PARENT_MANIFEST="$PARENT.manifest.json"
REGISTRY="$ROOT_DIR/training/halfkp64-scratch-v1/promotion_registry.json"
BINARY="$ROOT_DIR/target/release/halfkp_promotion_match"
MATCH="$ROOT_DIR/scripts/run_halfkp_paired_match.py"
DECISION="$GATE_DIR/gate-decision.json"
GATE_COMPLETION="$GATE_DIR/gate-completed.json"
POLL_SECONDS="${POLL_SECONDS:-30}"

mkdir -p "$GATE_DIR"
exec 9>"$GATE_DIR/.lock"
flock -n 9 || exit 0
exec > >(tee -a "$GATE_DIR/pipeline.log") 2>&1

need() { [[ -s "$1" ]] || { echo "missing: $1" >&2; exit 2; }; }
write_terminal_artifact() {
  python3 - "$GATE_COMPLETION" "$DECISION" <<'PY'
import hashlib, json, pathlib, sys, time
completion, decision = map(pathlib.Path, sys.argv[1:])
payload = {
  'schema': 'halfkp_gate_completion_v1',
  'completed_unix_secs': int(time.time()),
  'decision_sha256': hashlib.sha256(decision.read_bytes()).hexdigest(),
}
temporary = completion.with_suffix(completion.suffix + '.tmp')
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
temporary.replace(completion)
print('gate terminal artifact:', completion)
PY
}
while [[ ! -s "$TRAINING_COMPLETION" ]]; do
  if [[ -s "$RUN_DIR/pipeline.failed" ]]; then
    echo "training pipeline reported failure" >&2
    exit 2
  fi
  echo "waiting for training completion: $TRAINING_COMPLETION"
  sleep "$POLL_SECONDS"
done

need "$BINARY"; need "$MATCH"; need "$PARENT"; need "$PARENT_MANIFEST"; need "$REGISTRY"
need "$SCREEN_OPENINGS"; need "$OPENING_SOURCE"
MODEL="$RUN_DIR/model.bin"
MODEL_MANIFEST="$MODEL.manifest.json"
need "$MODEL"; need "$MODEL_MANIFEST"
python3 - "$TRAINING_COMPLETION" "$MODEL" "$MODEL_MANIFEST" <<'PY'
import hashlib, json, pathlib, sys
completion, model, manifest = map(pathlib.Path, sys.argv[1:])
c = json.loads(completion.read_text())
m = json.loads(manifest.read_text())
actual = hashlib.sha256(model.read_bytes()).hexdigest()
if actual != c.get('model_sha256') or actual != m.get('model', {}).get('sha256'):
    raise SystemExit('completion/model/manifest SHA mismatch')
if int(m.get('completed_epoch', 0)) < 32:
    raise SystemExit('training completed before required epoch32')
print('candidate frozen', actual, 'epoch', m['completed_epoch'])
PY

if [[ ! -s "$CONFIRM_OPENINGS" ]]; then
  python3 - "$OPENING_SOURCE" "$CONFIRM_OPENINGS" <<'PY'
import json, pathlib, sys
source, out = map(pathlib.Path, sys.argv[1:])
lines = source.read_bytes().splitlines()
selected = lines[1000:1250]
if len(selected) != 250:
    raise SystemExit(f'expected 250 reserve openings, got {len(selected)}')
seen = set()
for raw in selected:
    x = json.loads(raw)
    family = x.get('opening_family') or x.get('family') or x.get('id')
    if family in seen:
        raise SystemExit(f'duplicate opening family: {family}')
    seen.add(family)
out.write_bytes(b'\n'.join(selected) + b'\n')
print('wrote independent reserve openings', out)
PY
fi

run_match() {
  local openings="$1" output="$2" seed="$3"
  mkdir -p "$output"
  if [[ -s "$output/report.json" && -s "$output/completed.json" ]]; then
    echo "match already complete: $output"
    return
  fi
  python3 "$MATCH" \
    --binary "$BINARY" \
    --candidate "$MODEL" --candidate-manifest "$MODEL_MANIFEST" \
    --parent "$PARENT" --parent-manifest "$PARENT_MANIFEST" \
    --registry "$REGISTRY" --openings "$openings" \
    --openings-format dataset-jsonl --opening-selection first-unique --opening-seed 0 \
    --pairs 250 --nodes 20000 --max-depth 64 --max-plies 256 \
    --bootstrap-resamples 100000 --bootstrap-seed "$seed" --output-dir "$output"
}

run_match "$SCREEN_OPENINGS" "$SCREEN_DIR" 20260916
python3 - "$SCREEN_DIR/report.json" "$DECISION" <<'PY'
import json, pathlib, sys
report = json.loads(pathlib.Path(sys.argv[1]).read_text())
safety = report.get('counts', {})
safe = all(int(safety.get(k, 0)) == 0 for k in ('search_failures', 'missing_root_results', 'illegal_moves', 'non_finite_scores'))
screen_pass = float(report.get('score', 0.0)) >= 0.525 and safe
decision = {
  'schema': 'halfkp_gen7_gate_decision_v1',
  'screen_pass': screen_pass,
  'screen_score': report.get('score'),
  'screen_paired_ci95': [report.get('paired_score_ci95_lower'), report.get('paired_score_ci95_upper')],
  'screen_safety': safe,
  'independent_confirmation_started': False,
  'independent_confirmation_pass': False,
}
pathlib.Path(sys.argv[2]).write_text(json.dumps(decision, indent=2, sort_keys=True) + '\n')
print('screen_pass', screen_pass, 'score', report.get('score'))
PY
if ! python3 - "$DECISION" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get('screen_pass') else 1)
PY
then
  write_terminal_artifact
  echo "screen gate failed; confirmation and later comparisons are not authorized"
  exit 10
fi

python3 - "$DECISION" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
decision = json.loads(path.read_text())
decision['independent_confirmation_started'] = True
path.write_text(json.dumps(decision, indent=2, sort_keys=True) + '\n')
PY
run_match "$CONFIRM_OPENINGS" "$CONFIRM_DIR" 20260917
python3 - "$CONFIRM_DIR/report.json" "$DECISION" <<'PY'
import json, pathlib, sys
report = json.loads(pathlib.Path(sys.argv[1]).read_text())
safety = report.get('counts', {})
safe = all(int(safety.get(k, 0)) == 0 for k in ('search_failures', 'missing_root_results', 'illegal_moves', 'non_finite_scores'))
passed = float(report.get('score', 0.0)) >= 0.525 and float(report.get('paired_score_ci95_lower', 0.0)) > 0.5 and safe
decision_path = pathlib.Path(sys.argv[2])
decision = json.loads(decision_path.read_text())
decision.update({
  'independent_confirmation_started': True,
  'independent_confirmation_pass': passed,
  'confirmation_score': report.get('score'),
  'confirmation_paired_ci95': [report.get('paired_score_ci95_lower'), report.get('paired_score_ci95_upper')],
  'confirmation_safety': safe,
})
decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + '\n')
print('confirmation_pass', passed, 'score', report.get('score'))
PY
echo "gen7 gate pipeline complete: $GATE_DIR"
write_terminal_artifact

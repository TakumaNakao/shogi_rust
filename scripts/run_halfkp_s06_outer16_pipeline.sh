#!/usr/bin/env bash
set -Eeuo pipefail

# S06-A: continue the scratch HalfKP MMTO run from S04 outer8 through outer16,
# then evaluate the frozen outer16 model.  No v2.5.0 artifact is read until the
# final comparison stage, and it is never supplied to the trainer.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_DIR="$ROOT_DIR/runs/halfkp64-scratch-v1/s04-soft-target-250k-8pass-20260916"
EVAL_DIR="$ROOT_DIR/runs/halfkp64-scratch-v1/s06-outer16-eval-20260919"
TRAIN="$ROOT_DIR/runs/halfkp64-scratch-v1/s04-selected-250k-v3-20260916/train.jsonl"
VALID="$ROOT_DIR/runs/halfkp64-scratch-v1/s04-selected-250k-v3-20260916/valid.jsonl"
DATASET_MANIFEST="$ROOT_DIR/runs/halfkp64-scratch-v1/s04-selected-250k-v3-20260916/manifest.json"
INIT="$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen5-20260910/candidate/model-epoch24.bin"
INIT_MANIFEST="$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen5-20260910/candidate/model-epoch24.bin.manifest.json"
REGISTRY="$ROOT_DIR/training/halfkp64-scratch-v1/promotion_registry.json"
GATE_REPORT="$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen5-20260910/match-confirm-20k-250pair/report.json"
BIN="$ROOT_DIR/target/release/halfkp_mmto_train"
MATCH="$ROOT_DIR/scripts/run_halfkp_paired_match.py"
MATCH_BIN="$ROOT_DIR/target/release/halfkp_promotion_match"
TAYA36="$ROOT_DIR/taya36.sfen"
OUTER8_SHA="b9666c6e54b0fcdef8cc4d680ad83f88daf54b14957703a5f0cfe55474c08e3d"
DATASET_SHA="43d52824f0cbef4b98da6f9f99803360ea323f876a1ca04597f4b14c92cf22b4"
DECISION="$EVAL_DIR/decision.json"

mkdir -p "$EVAL_DIR"
exec 9>"$EVAL_DIR/.lock"
flock -n 9 || { echo "S06 pipeline is already running"; exit 0; }
exec > >(tee -a "$EVAL_DIR/pipeline.log") 2>&1

for path in "$RUN_DIR/checkpoint.json" "$RUN_DIR/model.bin" "$RUN_DIR/optimizer.bin" \
  "$TRAIN" "$VALID" "$DATASET_MANIFEST" "$INIT" "$INIT_MANIFEST" "$REGISTRY" \
  "$GATE_REPORT" "$BIN" "$MATCH_BIN" "$TAYA36"; do
  [[ -s "$path" ]] || { echo "missing required input: $path"; exit 2; }
done

python3 - "$RUN_DIR/checkpoint.json" "$RUN_DIR/model.bin" "$RUN_DIR/optimizer.bin" "$DATASET_MANIFEST" "$OUTER8_SHA" "$DATASET_SHA" <<'PY'
import hashlib
import json
import pathlib
import sys

checkpoint_path, model_path, optimizer_path, dataset_path = map(pathlib.Path, sys.argv[1:5])
expected_model = sys.argv[5]
expected_dataset = sys.argv[6]
checkpoint = json.loads(checkpoint_path.read_text())

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

completed_outer = int(checkpoint.get('completed_outer', -1))
if completed_outer < 8 or completed_outer > 16:
    raise SystemExit(f"expected checkpoint between outer8 and outer16, found {completed_outer}")
if completed_outer == 8 and (checkpoint.get('model_sha256') != expected_model or sha(model_path) != expected_model):
    raise SystemExit('S04 model checkpoint SHA mismatch')
if completed_outer > 8 and checkpoint.get('model_sha256') != sha(model_path):
    raise SystemExit('current model does not match checkpoint SHA')
if checkpoint.get('optimizer_sha256') != sha(optimizer_path):
    raise SystemExit('S04 optimizer checkpoint SHA mismatch')
if sha(dataset_path) != expected_dataset:
    raise SystemExit('dataset manifest SHA mismatch')
print(f'S06 preflight: checkpoint outer{completed_outer} and dataset are verified')
PY

echo "[1/5] Resuming scratch MMTO from outer8 through outer16"
if [[ "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["completed_outer"])' "$RUN_DIR/checkpoint.json")" -lt 16 ]]; then
  "$BIN" \
    --train "$TRAIN" \
    --valid "$VALID" \
    --dataset-manifest "$DATASET_MANIFEST" \
    --expected-dataset-manifest-sha256 "$DATASET_SHA" \
    --init "$INIT" \
    --model-manifest "$INIT_MANIFEST" \
    --promotion-registry "$REGISTRY" \
    --gate-report "$GATE_REPORT" \
    --parent-generation 6 \
    --output-dir "$RUN_DIR" \
    --outer-iterations 16 \
    --search-block-roots 1024 \
    --jobs 12 \
    --pair-batch-size 8192 \
    --learning-rate 0.0001 \
    --output-learning-rate-ratio 0.1 \
    --gradient-clip-norm 1.0 \
    --tau-cp 300 \
    --max-train-roots 250000 \
    --max-valid-roots 6000 \
    --pv-probe-roots 256 \
    --checkpoint-roots 25000 \
    --qsearch-node-limit 2000000 \
    --resume
fi

python3 - "$RUN_DIR/checkpoint.json" <<'PY'
import json
import pathlib
import sys
checkpoint = json.loads(pathlib.Path(sys.argv[1]).read_text())
if checkpoint.get('completed_outer') != 16:
    raise SystemExit(f"S06 training did not reach outer16: {checkpoint.get('completed_outer')}")
print('S06 training complete: outer16')
PY

echo "[2/5] Recording outer8/12/16 validation comparison and freezing candidate manifest"
python3 - "$RUN_DIR" "$EVAL_DIR" "$DATASET_SHA" "$INIT" <<'PY'
import hashlib
import json
import pathlib
import sys

run_dir, eval_dir = map(pathlib.Path, sys.argv[1:3])
dataset_sha = sys.argv[3]
init_path = pathlib.Path(sys.argv[4])

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def metric(outer):
    data = json.loads((run_dir / 'metrics' / f'outer_{outer:04d}.json').read_text())
    valid = data['valid']
    return {
        'outer': outer,
        'mean_loss_per_pair': valid['mean_loss_per_pair'],
        'agreement': valid['agreement'],
        'phase_loss': [entry['mean_loss'] for entry in valid['phase']],
        'aborted_roots': valid['aborted_roots'],
        'terminal_roots': valid['terminal_roots'],
        'model_sha256': data['model_sha256'],
    }

metrics = [metric(outer) for outer in (8, 12, 16)]
(eval_dir / 'phase_comparison.json').write_text(json.dumps({
    'schema': 'halfkp_s06_phase_comparison_v1',
    'dataset_manifest_sha256': dataset_sha,
    'metrics': metrics,
}, indent=2, sort_keys=True) + '\n')
model = run_dir / 'model.bin'
model_sha = sha(model)
if model_sha != metrics[-1]['model_sha256']:
    raise SystemExit('outer16 model SHA does not match metric artifact')
manifest = {
    'schema_version': 1,
    'lineage_id': 'halfkp64-scratch-v1',
    'stage': 'halfkp_mmto_train_s06',
    'selected_outer': 16,
    'parent_model_sha256': sha(init_path),
    'dataset_manifest_sha256': dataset_sha,
    'model': {'path': str(model), 'sha256': model_sha, 'bytes': model.stat().st_size},
    'selection_reason': 'outer16 continuation of S04; phase metrics and S06 gates are recorded separately',
    'teacher_policy': 'no external engine labels; WDoor move-frequency soft targets only',
}
(eval_dir / 'candidate_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
print(json.dumps({'metrics': metrics, 'candidate_sha256': model_sha}, sort_keys=True))
PY

python3 - "$DECISION" "$EVAL_DIR/phase_comparison.json" <<'PY'
import json
import pathlib
import sys

decision_path, comparison_path = map(pathlib.Path, sys.argv[1:])
comparison = json.loads(comparison_path.read_text())['metrics']
outer8, outer12, outer16 = comparison
opening_improved = outer16['phase_loss'][0] < outer8['phase_loss'][0]
middle_improved = outer16['phase_loss'][1] < outer8['phase_loss'][1]
endgame_non_regressed = outer16['phase_loss'][2] <= outer8['phase_loss'][2]
decision = {
    'schema': 'halfkp_s06_decision_v1',
    'training': {'status': 'completed', 'completed_outer': 16},
    'phase_gate': {
        'opening_improved': opening_improved,
        'middlegame_improved': middle_improved,
        'endgame_non_regressed': endgame_non_regressed,
        'pass': opening_improved and middle_improved and endgame_non_regressed,
        'metrics': comparison,
    },
    'gen6_screen': None,
    'v250_comparison': None,
}
decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + '\n')
print('phase_gate', decision['phase_gate'])
if not decision['phase_gate']['pass']:
    raise SystemExit(10)
PY

echo "[3/5] Creating unused gen6 screen openings"
python3 - "$TAYA36" "$EVAL_DIR/gen6_screen_openings.sfen" <<'PY'
import pathlib
import sys

source, output = map(pathlib.Path, sys.argv[1:])
lines = [line for line in source.read_bytes().splitlines() if line.strip()]
selected = lines[125:150]
if len(selected) != 25:
    raise SystemExit(f'expected 25 unused screen openings, found {len(selected)}')
output.write_bytes(b'\n'.join(selected) + b'\n')
PY

python3 "$MATCH" \
  --binary "$MATCH_BIN" \
  --candidate "$RUN_DIR/model.bin" --candidate-manifest "$EVAL_DIR/candidate_manifest.json" \
  --parent "$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen5-20260910/candidate/model-epoch24.bin" \
  --parent-manifest "$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen5-20260910/candidate/model-epoch24.bin.manifest.json" \
  --registry "$REGISTRY" \
  --openings "$EVAL_DIR/gen6_screen_openings.sfen" --openings-format usi-lines --opening-selection first-unique \
  --pairs 25 --time-limit-ms 5000 --max-depth 64 --max-plies 256 \
  --bootstrap-resamples 100000 --bootstrap-seed 20260921 --output-dir "$EVAL_DIR/gen6-screen"

python3 - "$DECISION" "$EVAL_DIR/gen6-screen/report.json" <<'PY'
import json
import pathlib
import sys

decision_path, report_path = map(pathlib.Path, sys.argv[1:])
report = json.loads(report_path.read_text())
counts = report['counts']
safety = all(counts.get(key, 0) == 0 for key in ('search_failures', 'missing_root_results', 'illegal_moves', 'non_finite_scores'))
score = float(report['score'])
ci = report.get('opening_pair_bootstrap', {})
decision = json.loads(decision_path.read_text())
decision['gen6_screen'] = {
    'candidate_wins': counts.get('candidate_wins'),
    'parent_wins': counts.get('parent_wins'),
    'draws': counts.get('draws'),
    'score': score,
    'paired_ci95': [ci.get('ci95_lower'), ci.get('ci95_upper')],
    'safety': safety,
    'pass': score > 0.50 and safety,
}
decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + '\n')
print('gen6_screen', decision['gen6_screen'])
if not decision['gen6_screen']['pass']:
    raise SystemExit(11)
PY

echo "[4/5] Creating independent v2.5.0 comparison openings"
python3 - "$TAYA36" "$EVAL_DIR/v250_openings.sfen" <<'PY'
import pathlib
import sys

source, output = map(pathlib.Path, sys.argv[1:])
lines = [line for line in source.read_bytes().splitlines() if line.strip()]
selected = lines[150:250]
if len(selected) != 100:
    raise SystemExit(f'expected 100 unused v2.5.0 comparison openings, found {len(selected)}')
output.write_bytes(b'\n'.join(selected) + b'\n')
PY

RUST_FONTCONFIG_DLOPEN=1 \
NEW_WEIGHTS="$RUN_DIR/model.bin" \
BASELINE_WEIGHTS="$ROOT_DIR/policy_weights_halfkp64_kpp_distilled_v2.5.0.binary" \
POSITIONS="$EVAL_DIR/v250_openings.sfen" GAMES=100 TIME_LIMIT_MS=5000 DEPTH=32 MAX_PLIES=256 \
SEED=20260922 THREADS=1 JOBS=1 OUT_DIR="$EVAL_DIR/v250-100g" \
bash scripts/bench_halfkp64_learned_vs_v250.sh

python3 - "$DECISION" "$EVAL_DIR/v250-100g/match.log" <<'PY'
import json
import pathlib
import re
import sys

decision_path, log_path = map(pathlib.Path, sys.argv[1:])
log = log_path.read_text()

def find(pattern):
    match = re.search(pattern, log)
    return int(match.group(1)) if match else None

rate = re.search(r'new total score rate: ([0-9.]+)%', log)
decision = json.loads(decision_path.read_text())
decision['v250_comparison'] = {
    'candidate_wins': find(r'new wins: (\d+)'),
    'v250_wins': find(r'baseline wins: (\d+)'),
    'draws': find(r'draws: (\d+)'),
    'candidate_total_score_rate': float(rate.group(1)) / 100 if rate else None,
    'games': 100,
    'status': 'completed',
}
decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + '\n')
print('v250_comparison', decision['v250_comparison'])
PY

echo "[5/5] S06 pipeline complete: $DECISION"

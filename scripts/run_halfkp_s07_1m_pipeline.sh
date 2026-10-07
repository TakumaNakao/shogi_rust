#!/usr/bin/env bash
set -Eeuo pipefail

# S07 retry: build an independent WDoor corpus from 2015--2017, preserve
# observed move frequencies as soft targets, select one million roots, and run
# the same scratch HalfKP MMTO objective for eight outer passes.  The original
# S07 attempt used a 4000/4000 rate floor, but the historical 2015--2017
# ratings are on a lower scale and yielded only 2,482 records.  This retry uses
# a documented 2500/2500 floor and a larger per-game root cap; v2.5.0 remains
# evaluation-only and is never read by the trainer.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASE="$ROOT_DIR/runs/halfkp64-scratch-v1"
SOURCE_TAG="s07-retry-rate2500-v2-20260922"
RUN_TAG="s07-retry-rate2500-v3-20260922"
MIN_RATE=2500
MAX_ROOTS_PER_GAME=64
SEED=20260923
RAW_DIR="$BASE/${SOURCE_TAG}-raw"
FREQ_DIR="$BASE/${SOURCE_TAG}-frequency"
DATA_DIR="$BASE/${RUN_TAG}-selected-1m"
RUN_DIR="$BASE/${RUN_TAG}-soft-target-1m-8pass"
EVAL_DIR="$BASE/${RUN_TAG}-eval"
TRAIN="$DATA_DIR/train.jsonl"
VALID="$DATA_DIR/valid.jsonl"
DATASET_MANIFEST="$DATA_DIR/manifest.json"
INIT="$BASE/v4-value-gen5-20260910/candidate/model-epoch24.bin"
INIT_MANIFEST="$BASE/v4-value-gen5-20260910/candidate/model-epoch24.bin.manifest.json"
REGISTRY="$ROOT_DIR/training/halfkp64-scratch-v1/promotion_registry.json"
GATE_REPORT="$BASE/v4-value-gen5-20260910/match-confirm-20k-250pair/report.json"
DATASET_BUILD="$ROOT_DIR/target/release/dataset_build"
AGGREGATE="$ROOT_DIR/scripts/aggregate_halfkp_move_frequencies.py"
SELECT="$ROOT_DIR/scripts/select_halfkp_frequency_roots.py"
TRAINER="$ROOT_DIR/target/release/halfkp_mmto_train"
MATCH="$ROOT_DIR/scripts/run_halfkp_paired_match.py"
MATCH_BIN="$ROOT_DIR/target/release/halfkp_promotion_match"
TAYA36="$ROOT_DIR/taya36.sfen"
DECISION="$EVAL_DIR/decision.json"

mkdir -p "$EVAL_DIR"
exec 9>"$EVAL_DIR/.lock"
flock -n 9 || { echo "S07 pipeline is already running"; exit 0; }
exec > >(tee -a "$EVAL_DIR/pipeline.log") 2>&1

for path in "$DATASET_BUILD" "$AGGREGATE" "$SELECT" "$TRAINER" "$MATCH_BIN" \
  "$INIT" "$INIT_MANIFEST" "$REGISTRY" "$GATE_REPORT" "$TAYA36"; do
  [[ -s "$path" ]] || { echo "missing required input: $path"; exit 2; }
done

echo "[1/6] Building independent 2015--2017 WDoor raw positions"
if [[ ! -s "$RAW_DIR/manifest.json" ]]; then
  "$DATASET_BUILD" \
    --input "$ROOT_DIR/data/wdoor/extract/2015" \
    --input "$ROOT_DIR/data/wdoor/extract/2016" \
    --input "$ROOT_DIR/data/wdoor/extract/2017" \
    --output-dir "$RAW_DIR" \
    --license-id user-authorized:wdoor-public-archive \
    --source-uri https://wdoor.c.u-tokyo.ac.jp/shogi/ \
    --seed "$SEED" \
    --shuffle-games \
    --reuse-if-matches \
    --valid-percent 10 \
    --test-percent 10 \
    --opening-prefix-plies 16 \
    --max-records 1500000 \
    --target-opening-records 400000 \
    --target-middle-records 700000 \
    --target-late-records 300000 \
    --require-targets \
    --target-minimum-percent 95 \
    --max-records-per-game 24 \
    --phase-records-per-game 8 12 4 \
    --min-ply 8 \
    --max-ply 200 \
    --min-player-rate "$MIN_RATE" \
    --min-opponent-rate "$MIN_RATE" \
    --preserve-position-duplicates
else
  echo "reusing frozen raw dataset: $RAW_DIR"
fi

echo "[2/6] Aggregating observed moves into frequency soft targets"
if [[ ! -s "$FREQ_DIR/manifest.json" ]]; then
  python3 "$AGGREGATE" \
    --input "$RAW_DIR/train.jsonl" "$RAW_DIR/valid.jsonl" "$RAW_DIR/test.jsonl" \
    --output-dir "$FREQ_DIR" \
    --min-player-rate "$MIN_RATE" --min-opponent-rate "$MIN_RATE" \
    --shuffle-seed "$SEED"
  python3 - "$FREQ_DIR/manifest.json" <<'PY'
import json
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text())
data['min_player_rate'] = 2500
data['min_opponent_rate'] = 2500
data['source_policy'] = 'independent WDoor 2015-2017; historical rate floor 2500; observed move frequencies only'
path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
PY
else
  echo "reusing frozen frequency dataset: $FREQ_DIR"
fi

echo "[3/6] Selecting one million phase-balanced roots"
if [[ ! -s "$DATA_DIR/manifest.json" ]]; then
  python3 "$SELECT" \
    --input-dir "$FREQ_DIR" \
    --output-dir "$DATA_DIR" \
    --opening 208000 \
    --middlegame 552000 \
    --endgame 240000 \
    --seed "$SEED" \
    --max-roots-per-game "$MAX_ROOTS_PER_GAME"
  python3 - "$DATA_DIR/manifest.json" <<'PY'
import json
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text())
data['min_player_rate'] = 2500
data['min_opponent_rate'] = 2500
data['source_policy'] = 'independent WDoor 2015-2017; historical rate floor 2500; frequency soft targets; no S04 source years'
path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
PY
fi

python3 - "$DATA_DIR" <<'PY'
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / 'manifest.json').read_text())
if manifest.get('selected_train_records') != 1_000_000:
    raise SystemExit(f"expected exactly 1,000,000 selected roots, found {manifest.get('selected_train_records')}")
for split in ('train', 'valid', 'test'):
    if not (root / f'{split}.jsonl').is_file():
        raise SystemExit(f'missing selected {split}.jsonl')
print(json.dumps({
    'selected_train_records': manifest['selected_train_records'],
    'selected_train_by_phase': manifest.get('selected_train_by_phase'),
    'output_files': manifest.get('output_files'),
}, sort_keys=True))
PY

DATASET_SHA_EXPECTED="$(sha256sum "$DATASET_MANIFEST" | awk '{print $1}')"
echo "[4/6] Training one million roots for eight outer passes"
TRAIN_ARGS=(
  --train "$TRAIN"
  --valid "$VALID"
  --dataset-manifest "$DATASET_MANIFEST"
  --expected-dataset-manifest-sha256 "$DATASET_SHA_EXPECTED"
  --init "$INIT"
  --model-manifest "$INIT_MANIFEST"
  --promotion-registry "$REGISTRY"
  --gate-report "$GATE_REPORT"
  --parent-generation 6
  --output-dir "$RUN_DIR"
  --outer-iterations 8
  --search-block-roots 1024
  --jobs 12
  --pair-batch-size 8192
  --learning-rate 0.0001
  --output-learning-rate-ratio 0.1
  --gradient-clip-norm 1.0
  --tau-cp 300
  --max-train-roots 1000000
  --max-valid-roots 12000
  --pv-probe-roots 256
  --checkpoint-roots 25000
  --qsearch-node-limit 2000000
)
if [[ -s "$RUN_DIR/checkpoint.json" ]]; then
  echo "resuming existing S07 training checkpoint: $RUN_DIR"
  TRAIN_ARGS+=(--resume)
else
  echo "starting S07 training from accepted generation 6"
fi
"$TRAINER" "${TRAIN_ARGS[@]}"

python3 - "$RUN_DIR/checkpoint.json" <<'PY'
import json
import pathlib
import sys
checkpoint = json.loads(pathlib.Path(sys.argv[1]).read_text())
if checkpoint.get('completed_outer') != 8:
    raise SystemExit(f"S07 training checkpoint is not complete: {checkpoint.get('completed_outer')}")
print('S07 training complete: outer8')
PY

echo "[5/6] Freezing candidate and recording phase metrics"
python3 - "$RUN_DIR" "$EVAL_DIR" "$DATASET_SHA_EXPECTED" "$INIT" <<'PY'
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

metrics = []
for outer in (2, 4, 8):
    data = json.loads((run_dir / 'metrics' / f'outer_{outer:04d}.json').read_text())
    valid = data['valid']
    metrics.append({
        'outer': outer,
        'mean_loss_per_pair': valid['mean_loss_per_pair'],
        'agreement': valid['agreement'],
        'phase_loss': [item['mean_loss'] for item in valid['phase']],
        'aborted_roots': valid['aborted_roots'],
        'terminal_roots': valid['terminal_roots'],
        'model_sha256': data['model_sha256'],
    })
model = run_dir / 'model.bin'
model_sha = sha(model)
if model_sha != metrics[-1]['model_sha256']:
    raise SystemExit('S07 model SHA does not match outer8 metric')
(eval_dir / 'phase_metrics.json').write_text(json.dumps({
    'schema': 'halfkp_s07_phase_metrics_v1',
    'dataset_manifest_sha256': dataset_sha,
    'metrics': metrics,
}, indent=2, sort_keys=True) + '\n')
(eval_dir / 'candidate_manifest.json').write_text(json.dumps({
    'schema_version': 1,
    'lineage_id': 'halfkp64-scratch-v1',
    'stage': 'halfkp_mmto_train_s07',
    'selected_outer': 8,
    'parent_model_sha256': sha(init_path),
    'dataset_manifest_sha256': dataset_sha,
    'model': {'path': str(model), 'sha256': model_sha, 'bytes': model.stat().st_size},
    'selection_reason': 'S07 one-million-root independent WDoor corpus; phase metrics recorded separately',
    'teacher_policy': 'no external engine labels; WDoor move-frequency soft targets only',
}, indent=2, sort_keys=True) + '\n')
print(json.dumps({'candidate_sha256': model_sha, 'metrics': metrics}, sort_keys=True))
PY

echo "[6/6] Running gen6 screen and conditional v2.5.0 comparison"
python3 - "$TAYA36" "$EVAL_DIR/gen6_openings.sfen" "$EVAL_DIR/v250_openings.sfen" <<'PY'
import pathlib
import sys

source, gen6, v250 = map(pathlib.Path, sys.argv[1:])
lines = [line for line in source.read_bytes().splitlines() if line.strip()]
if len(lines[250:275]) != 25 or len(lines[275:375]) != 100:
    raise SystemExit('taya36 does not contain the reserved S07 opening ranges')
gen6.write_bytes(b'\n'.join(lines[250:275]) + b'\n')
v250.write_bytes(b'\n'.join(lines[275:375]) + b'\n')
PY

python3 "$MATCH" \
  --binary "$MATCH_BIN" \
  --candidate "$RUN_DIR/model.bin" --candidate-manifest "$EVAL_DIR/candidate_manifest.json" \
  --parent "$INIT" --parent-manifest "$INIT_MANIFEST" --registry "$REGISTRY" \
  --openings "$EVAL_DIR/gen6_openings.sfen" --openings-format usi-lines --opening-selection first-unique \
  --pairs 25 --time-limit-ms 5000 --max-depth 64 --max-plies 256 \
  --bootstrap-resamples 100000 --bootstrap-seed 20260923 --output-dir "$EVAL_DIR/gen6-screen"

python3 - "$EVAL_DIR/gen6-screen/report.json" "$DECISION" "$DATASET_SHA_EXPECTED" <<'PY'
import json
import pathlib
import sys

report_path, decision_path = map(pathlib.Path, sys.argv[1:3])
dataset_sha = sys.argv[3]
report = json.loads(report_path.read_text())
counts = report['counts']
safety = all(counts.get(key, 0) == 0 for key in ('search_failures', 'missing_root_results', 'illegal_moves', 'non_finite_scores'))
score = float(report['score'])
ci = report.get('opening_pair_bootstrap', {})
decision = {
    'schema': 'halfkp_s07_decision_v1',
    'dataset_manifest_sha256': dataset_sha,
    'training': {'status': 'completed', 'outer': 8},
    'gen6_screen': {
        'candidate_wins': counts.get('candidate_wins'),
        'parent_wins': counts.get('parent_wins'),
        'draws': counts.get('draws'),
        'score': score,
        'paired_ci95': [ci.get('ci95_lower'), ci.get('ci95_upper')],
        'safety': safety,
        'pass': score > 0.50 and safety,
    },
    'v250_comparison': None,
}
decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + '\n')
print('gen6_screen', decision['gen6_screen'])
if not decision['gen6_screen']['pass']:
    raise SystemExit(11)
PY

RUST_FONTCONFIG_DLOPEN=1 \
NEW_WEIGHTS="$RUN_DIR/model.bin" \
BASELINE_WEIGHTS="$ROOT_DIR/policy_weights_halfkp64_kpp_distilled_v2.5.0.binary" \
POSITIONS="$EVAL_DIR/v250_openings.sfen" GAMES=100 TIME_LIMIT_MS=5000 DEPTH=32 MAX_PLIES=256 \
SEED=20260924 THREADS=1 JOBS=1 OUT_DIR="$EVAL_DIR/v250-100g" \
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

echo "S07 pipeline complete: $DECISION"

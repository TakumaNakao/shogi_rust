#!/usr/bin/env bash
set -Eeuo pipefail

# S08 post-training evaluation.  The script waits for the continuation job,
# then runs the predeclared outer16-vs-outer8 and outer16-vs-v2.5.0 matches.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASE_DIR="$ROOT_DIR/runs/halfkp64-scratch-v1/s08-retry-rate2500-v3-20260929"
CONTINUE_RUN="$BASE_DIR/continue-outer16"
OUTER8_RUN="$BASE_DIR/frozen-outer8"
OUTER16_RUN="$BASE_DIR/outer16"
EVAL_DIR="$BASE_DIR/evaluation"
LOG="$ROOT_DIR/runs/halfkp64-scratch-v1/s08-retry-rate2500-v3-20260929-evaluation.log"
V250="$ROOT_DIR/policy_weights_halfkp64_kpp_distilled_v2.5.0.binary"
TAYA="$ROOT_DIR/taya36.sfen"

while [[ ! -s "$OUTER16_RUN/model.bin" ]]; do
  if [[ -f "$CONTINUE_RUN/checkpoint.json" ]]; then
    completed="$(python3 - "$CONTINUE_RUN/checkpoint.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))['completed_outer'])
PY
    )"
    if (( completed >= 16 )); then
      sleep 5
      continue
    fi
    if ! pgrep -x halfkp_mmto_train >/dev/null 2>&1; then
      echo "continuation stopped before outer16" >&2
      exit 1
    fi
  fi
  sleep 60
done

mkdir -p "$EVAL_DIR"
OPENING_8="$EVAL_DIR/outer16_vs_outer8.sfen"
OPENING_V250="$EVAL_DIR/outer16_vs_v250.sfen"
if [[ ! -e "$OPENING_8" ]]; then
  sed -n '376,400p' "$TAYA" > "$OPENING_8"
fi
if [[ ! -e "$OPENING_V250" ]]; then
  sed -n '401,450p' "$TAYA" > "$OPENING_V250"
fi

RUST_FONTCONFIG_DLOPEN=1 \
NEW_WEIGHTS="$OUTER16_RUN/model.bin" \
BASELINE_WEIGHTS="$OUTER8_RUN/model.bin" \
POSITIONS="$OPENING_8" GAMES=50 TIME_LIMIT_MS=5000 DEPTH=32 MAX_PLIES=256 \
SEED=20260929 THREADS=1 JOBS=1 OUT_DIR="$EVAL_DIR/outer16-vs-outer8-50g" \
bash scripts/bench_halfkp64_learned_vs_v250.sh

RUST_FONTCONFIG_DLOPEN=1 \
NEW_WEIGHTS="$OUTER16_RUN/model.bin" \
BASELINE_WEIGHTS="$V250" \
POSITIONS="$OPENING_V250" GAMES=100 TIME_LIMIT_MS=5000 DEPTH=32 MAX_PLIES=256 \
SEED=20260930 THREADS=1 JOBS=1 OUT_DIR="$EVAL_DIR/outer16-vs-v250-100g" \
bash scripts/bench_halfkp64_learned_vs_v250.sh

python3 - "$EVAL_DIR" <<'PY'
import json
import pathlib
import random
import sys

eval_dir = pathlib.Path(sys.argv[1])

def bootstrap(values, seed):
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(100_000))
    return [means[int(0.025 * len(means))], means[int(0.975 * len(means)) - 1]]

def summarize(name, games_dir, seed):
    rows = []
    for path in sorted(pathlib.Path(games_dir).glob('game_*.usi')):
        fields = {}
        for line in path.read_text().splitlines()[:4]:
            key, value = line.split(' ', 1)
            fields[key] = value
        fields['score'] = {'NewWin': 1.0, 'BaselineWin': 0.0, 'Draw': 0.5}[fields['result']]
        rows.append(fields)
    if len(rows) % 2 or not rows:
        raise SystemExit(f'{name}: invalid game count {len(rows)}')
    pairs = []
    for black, white in zip(rows[::2], rows[1::2]):
        if black['start_sfen'] != white['start_sfen']:
            raise SystemExit(f'{name}: color-swapped pair has different opening')
        pairs.append((black['score'] + white['score']) / 2.0)
    return {
        'games': len(rows),
        'pairs': len(pairs),
        'candidate_wins': sum(r['result'] == 'NewWin' for r in rows),
        'baseline_wins': sum(r['result'] == 'BaselineWin' for r in rows),
        'draws': sum(r['result'] == 'Draw' for r in rows),
        'all_resignations': all(r['reason'] == 'Resign' for r in rows),
        'pair_distribution': {str(v): pairs.count(v) for v in sorted(set(pairs))},
        'paired_bootstrap_ci95': bootstrap(pairs, seed),
        'bootstrap_resamples': 100_000,
        'bootstrap_seed': seed,
    }

summary = {
    'schema': 'halfkp_s08_evaluation_v1',
    'outer16_model_sha256': __import__('hashlib').sha256(
        (eval_dir.parent / 'outer16/model.bin').read_bytes()).hexdigest(),
    'outer8_model_sha256': __import__('hashlib').sha256(
        (eval_dir.parent / 'frozen-outer8/model.bin').read_bytes()).hexdigest(),
    'outer16_vs_outer8': summarize('outer16_vs_outer8', eval_dir / 'outer16-vs-outer8-50g/games', 20260929),
    'outer16_vs_v250': summarize('outer16_vs_v250', eval_dir / 'outer16-vs-v250-100g/games', 20260930),
}
(eval_dir / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "S08 evaluation complete: $EVAL_DIR/summary.json"

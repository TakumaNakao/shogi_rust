#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen6-2m-extended-seed20260911}"
GATE_DECISION="$RUN_DIR/gate/gate-decision.json"
MODEL="$RUN_DIR/model.bin"
OUT_DIR="$RUN_DIR/v250-taya36-5s-50g"
POLL_SECONDS="${POLL_SECONDS:-30}"

mkdir -p "$RUN_DIR/v250-watcher"
exec 9>"$RUN_DIR/v250-watcher/.lock"
flock -n 9 || exit 0
exec > >(tee -a "$RUN_DIR/v250-watcher/pipeline.log") 2>&1

need() { [[ -s "$1" ]] || { echo "missing: $1" >&2; exit 2; }; }
while [[ ! -s "$GATE_DECISION" ]]; do
  echo "waiting for independent gate decision: $GATE_DECISION"
  sleep "$POLL_SECONDS"
done

if ! python3 - "$GATE_DECISION" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
raise SystemExit(0 if x.get('independent_confirmation_pass') is True else 1)
PY
then
  echo "independent confirmation did not pass; v2.5.0 comparison is not authorized"
  exit 0
fi

need "$MODEL"
if [[ -s "$OUT_DIR/match.log" ]]; then
  echo "v2.5.0 comparison already complete: $OUT_DIR"
  exit 0
fi
if [[ -e "$OUT_DIR" ]]; then
  echo "incomplete comparison directory exists; refusing to overwrite: $OUT_DIR" >&2
  exit 2
fi

RUST_FONTCONFIG_DLOPEN=1 \
NEW_WEIGHTS="$MODEL" \
BASELINE_WEIGHTS="$ROOT_DIR/policy_weights_halfkp64_kpp_distilled_v2.5.0.binary" \
POSITIONS="$ROOT_DIR/taya36.sfen" \
GAMES=50 TIME_LIMIT_MS=5000 DEPTH=32 MAX_PLIES=256 SEED=20260918 THREADS=1 JOBS=1 \
OUT_DIR="$OUT_DIR" \
bash scripts/bench_halfkp64_learned_vs_v250.sh

python3 - "$OUT_DIR/match.log" "$OUT_DIR/result.json" <<'PY'
import json, pathlib, re, sys
log = pathlib.Path(sys.argv[1]).read_text()
def number(pattern):
    m = re.search(pattern, log)
    return int(m.group(1)) if m else None
rate_match = re.search(r'new total score rate: ([0-9.]+)%', log)
out = {
  'schema': 'halfkp_v250_taya36_5s_50g_result_v1',
  'new_wins': number(r'new wins: (\d+)'),
  'baseline_wins': number(r'baseline wins: (\d+)'),
  'draws': number(r'draws: (\d+)'),
  'new_total_score_rate': float(rate_match.group(1))/100 if rate_match else None,
  'decisive_ci95': next((line.split(':',1)[1].strip() for line in log.splitlines() if line.startswith('decisive win rate 95% CI:')), None),
  'total_score_ci95': next((line.split(':',1)[1].strip() for line in log.splitlines() if line.startswith('total score rate 95% CI:')), None),
}
pathlib.Path(sys.argv[2]).write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
print(json.dumps(out, sort_keys=True))
PY
echo "v2.5.0 comparison complete: $OUT_DIR"

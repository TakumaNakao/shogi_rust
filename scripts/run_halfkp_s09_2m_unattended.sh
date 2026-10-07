#!/usr/bin/env bash
set -Eeuo pipefail

# S09 unattended 1M control and 2M frequency-MMTO training from released v2.7.1.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
BASE="$ROOT_DIR/runs/halfkp64-scratch-v1/s09-frequency-2m-20261007"
RAW="$BASE/raw-2012-2014"
FREQ="$BASE/frequency-2012-2014"
DATA="$BASE/dataset-2m"
LINEAGE="$BASE/research-parent"
CONTROL="$BASE/control-1m-4pass"
RUN="$BASE/train-2m-8pass"
S07="$ROOT_DIR/runs/halfkp64-scratch-v1/s07-retry-rate2500-v3-20260922-selected-1m"
S08="$ROOT_DIR/runs/halfkp64-scratch-v1/s08-retry-rate2500-v3-20260929"
PARENT="$S08/outer16/model.bin"
OPTIMIZER="$S08/outer16/optimizer.bin"
SOURCE_MANIFEST="$S08/outer16/run_manifest.json"
GATE_SUMMARY="$S08/evaluation/summary.json"
REGISTRY="$ROOT_DIR/training/halfkp64-scratch-v1/promotion_registry.json"

mkdir -p "$BASE"
exec 9>"$BASE/.lock"
flock -n 9 || { echo "S09 is already running"; exit 0; }
exec > >(tee -a "$BASE/pipeline.log") 2>&1
trap 'rc=$?; echo "[$(date -u +%FT%TZ)] pipeline exit=$rc"' EXIT
phase() { echo "[$(date -u +%FT%TZ)] $*"; }
sha() { sha256sum "$1" | awk '{print $1}'; }

for path in "$S07/train.jsonl" "$S07/valid.jsonl" "$S07/test.jsonl" "$S07/manifest.json" "$PARENT" "$OPTIMIZER" "$SOURCE_MANIFEST" "$GATE_SUMMARY" "$REGISTRY"; do
  [[ -s "$path" ]] || { phase "missing input: $path"; exit 2; }
done

for year in 2012 2013 2014; do
  archive="$ROOT_DIR/data/wdoor/archive/wdoor${year}.7z"
  target="$ROOT_DIR/data/wdoor/extract/$year"
  case "$year" in
    2012) expected=8fbc7cafbad92539bc2a0fb22733e461e1e22aa4f64c3771c765aeaf270558be ;;
    2013) expected=bfd50bdb5acd570c96d7909d63b43db3ffc4ffcc2187d56fe5edc8120bf5a0ee ;;
    2014) expected=a49392b373ffc2809246f5e6319baeb4d9ee36995d7eb68d462ddd0e00cfcfac ;;
  esac
  [[ "$(sha "$archive")" == "$expected" ]] || { phase "archive SHA mismatch: $archive"; exit 2; }
  while pgrep -f "^/usr/lib/p7zip/7z x .*wdoor${year}\.7z$" >/dev/null; do phase "waiting for existing $year extraction"; sleep 60; done
  mkdir -p "$target"
  listed="$(7z l -slt "$archive" | awk '$1 == "Path" && $0 ~ /\.csa$/ {n++} END {print n+0}')"
  present="$(find "$target" -type f -name '*.csa' | wc -l)"
  if [[ "$listed" -ne "$present" ]]; then
    phase "extracting WDoor $year ($present/$listed CSA files present)"
    7z x -y -mmt=1 "-o$target" "$archive"
  else
    phase "WDoor $year extraction complete ($present CSA files)"
  fi
done

phase "building required release tools"
CARGO_BUILD_JOBS=1 cargo build --release --locked --features halfkp64,training-tools,benchmark-tools \
  --bin dataset_build --bin csa_rate_stats --bin halfkp_mmto_train --bin halfkp_promotion_match

if [[ ! -s "$BASE/rate-audit.txt" ]]; then
  phase "auditing historical rate metadata"
  target/release/csa_rate_stats --input data/wdoor/extract/2012 --input data/wdoor/extract/2013 \
    --input data/wdoor/extract/2014 --thresholds 2500,3000,3500,4000,4500 > "$BASE/rate-audit.txt"
  cat "$BASE/rate-audit.txt"
fi

if [[ ! -s "$RAW/manifest.json" ]]; then
  phase "building 2012-2014 raw corpus"
  target/release/dataset_build --input data/wdoor/extract/2012 --input data/wdoor/extract/2013 --input data/wdoor/extract/2014 \
    --output-dir "$RAW" --license-id user-authorized:wdoor-public-archive \
    --source-uri https://wdoor.c.u-tokyo.ac.jp/shogi/ --seed 20260923 --shuffle-games --reuse-if-matches \
    --valid-percent 10 --test-percent 10 --opening-prefix-plies 16 --max-records 2200000 \
    --target-opening-records 600000 --target-middle-records 1100000 --target-late-records 500000 \
    --require-targets --target-minimum-percent 80 --max-records-per-game 24 \
    --phase-records-per-game 8 12 4 --min-ply 8 --max-ply 200 --preserve-position-duplicates
else
  phase "reusing frozen raw corpus"
fi

if [[ ! -s "$FREQ/manifest.json" ]]; then
  phase "aggregating observed moves into frequency targets"
  python3 scripts/aggregate_halfkp_move_frequencies.py --input "$RAW/train.jsonl" "$RAW/valid.jsonl" "$RAW/test.jsonl" \
    --output-dir "$FREQ" --shuffle-seed 20260923
  python3 -c 'import json,pathlib,sys;p=pathlib.Path(sys.argv[1]);d=json.loads(p.read_text());d["min_player_rate"]=None;d["min_opponent_rate"]=None;d["source_policy"]="WDoor 2012-2014 independent historical cohort; observed-move frequencies only; rating metadata audited separately; no rate floor assumed";d["source_years"]=[2012,2013,2014];p.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n")' "$FREQ/manifest.json"
else
  phase "reusing frozen frequency corpus"
fi

if [[ ! -s "$DATA/manifest.json" ]]; then
  phase "freezing the combined 2M dataset and shared holdout"
  python3 scripts/prepare_halfkp_s09_2m_frequency.py --base-dir "$S07" --base-manifest "$S07/manifest.json" \
    --addition-dir "$FREQ" --addition-manifest "$FREQ/manifest.json" --output-dir "$DATA" --seed 20260923
else
  phase "reusing frozen 2M dataset"
fi
DATA_SHA="$(sha "$DATA/manifest.json")"
CONTROL_SHA="$(sha "$DATA/control/manifest.json")"
phase "frozen manifests: 2M=$DATA_SHA 1M-control=$CONTROL_SHA"

if [[ ! -s "$LINEAGE/research_parent_registry.json" ]]; then
  phase "registering the released v2.7.1 bundle as research generation 7"
  python3 scripts/prepare_halfkp_s09_research_parent.py --model "$PARENT" --optimizer "$OPTIMIZER" \
    --source-manifest "$SOURCE_MANIFEST" --gate-summary "$GATE_SUMMARY" --canonical-registry "$REGISTRY" --output-dir "$LINEAGE"
fi

train() {
  local data="$1" manifest_sha="$2" destination="$3" max_roots="$4" outer="$5" resume="$6"
  local args=(--train "$data/train.jsonl" --valid "$data/valid.jsonl" --dataset-manifest "$data/manifest.json"
    --expected-dataset-manifest-sha256 "$manifest_sha" --init "$PARENT" --init-optimizer "$OPTIMIZER"
    --model-manifest "$LINEAGE/v2.7.1.model.manifest.json" --promotion-registry "$LINEAGE/research_parent_registry.json"
    --gate-report "$LINEAGE/v2.7.1_gate_summary.json" --parent-generation 7 --output-dir "$destination"
    --outer-iterations "$outer" --search-block-roots 1024 --jobs 12 --pair-batch-size 8192
    --learning-rate 0.0001 --output-learning-rate-ratio 0.1 --gradient-clip-norm 1 --tau-cp 300
    --max-train-roots "$max_roots" --max-valid-roots 12000 --pv-probe-roots 256
    --checkpoint-roots 25000 --qsearch-node-limit 2000000)
  [[ "$resume" == yes ]] && args+=(--resume)
  target/release/halfkp_mmto_train "${args[@]}"
}

checkpoint_outer() {
  python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("completed_outer",0))' "$1" 2>/dev/null || echo 0
}
save_pass() {
  local src="$1" dst="$2" outer="$3"
  mkdir -p "$dst"
  cp --reflink=auto "$src/model.bin" "$dst/model.bin"
  cp --reflink=auto "$src/optimizer.bin" "$dst/optimizer.bin"
  cp --reflink=auto "$src/metrics/outer_$(printf '%04d' "$outer").json" "$dst/metric.json"
  (cd "$dst" && sha256sum model.bin optimizer.bin metric.json > SHA256SUMS)
}

if (( $(checkpoint_outer "$CONTROL/checkpoint.json") < 4 )); then
  resume=no; [[ -s "$CONTROL/checkpoint.json" ]] && resume=yes
  phase "training 1M x4 control from the released parent"
  train "$DATA/control" "$CONTROL_SHA" "$CONTROL" 1000000 4 "$resume"
fi
save_pass "$CONTROL" "$BASE/frozen-control-outer4" 4

for outer in 2 4 8; do
  current="$(checkpoint_outer "$RUN/checkpoint.json")"
  if (( current < outer )); then
    resume=no; [[ -s "$RUN/checkpoint.json" ]] && resume=yes
    phase "training 2M to outer$outer (currently outer$current)"
    train "$DATA" "$DATA_SHA" "$RUN" 2000000 "$outer" "$resume"
  fi
  save_pass "$RUN" "$BASE/frozen-2m-outer$outer" "$outer"
done

phase "training complete; 1M control outer4 and 2M outer2/4/8 are frozen"
phase "freezing candidate provenance and the unused paired-match openings"
python3 scripts/finalize_halfkp_s09_2m.py \
  --run-dir "$RUN" --dataset-manifest "$DATA/manifest.json" \
  --parent "$PARENT" --parent-manifest "$LINEAGE/v2.7.1.model.manifest.json" \
  --v250 "$ROOT_DIR/policy_weights_halfkp64_kpp_distilled_v2.5.0.binary" \
  --taya36 "$ROOT_DIR/taya36.sfen" --output-dir "$BASE/evaluation"

phase "running 50 paired games versus v2.7.1 (5 seconds per move)"
python3 scripts/run_halfkp_paired_match.py \
  --binary target/release/halfkp_promotion_match \
  --candidate "$RUN/model.bin" --candidate-manifest "$EVAL_DIR/candidate.manifest.json" \
  --parent "$PARENT" --parent-manifest "$LINEAGE/v2.7.1.model.manifest.json" \
  --registry "$LINEAGE/research_parent_registry.json" \
  --openings "$EVAL_DIR/v271_openings.sfen" --openings-format usi-lines --opening-selection first-unique \
  --pairs 25 --time-limit-ms 5000 --max-depth 64 --max-plies 256 \
  --bootstrap-resamples 100000 --bootstrap-seed 20261007 --output-dir "$EVAL_DIR/v271-50g"

phase "running 50 paired games versus v2.5.0 (evaluation-only baseline)"
python3 scripts/run_halfkp_paired_match.py \
  --binary target/release/halfkp_promotion_match \
  --candidate "$RUN/model.bin" --candidate-manifest "$EVAL_DIR/candidate.manifest.json" \
  --parent "$ROOT_DIR/policy_weights_halfkp64_kpp_distilled_v2.5.0.binary" \
  --parent-manifest "$EVAL_DIR/v250_parent.manifest.json" \
  --parent-authorization "$EVAL_DIR/v250_parent_authorization.json" \
  --openings "$EVAL_DIR/v250_openings.sfen" --openings-format usi-lines --opening-selection first-unique \
  --pairs 25 --time-limit-ms 5000 --max-depth 64 --max-plies 256 \
  --bootstrap-resamples 100000 --bootstrap-seed 20261008 --output-dir "$EVAL_DIR/v250-50g"

phase "S09 training and reserved 100-game screen are complete"
printf 'completed_at=%s\ncontrol_outer=4\ntrain_2m_outer=8\ndataset_manifest_sha256=%s\n' "$(now)" "$DATA_SHA" > "$BASE/COMPLETE"

#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen6-4m-20260913}"
BASE_RUN="$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen6-2m-20260911"
RAW_NEW="$RUN_DIR/raw-new"; SHARDS="$RUN_DIR/shards-50k"; TEACHERS="$RUN_DIR/teachers-100k"; CHECKPOINTS="$RUN_DIR/checkpoints"
PARENT="$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen5-20260910/candidate/model-epoch24.bin"
PARENT_MANIFEST="$PARENT.manifest.json"
REGISTRY="$ROOT_DIR/training/halfkp64-scratch-v1/promotion_registry.json"
GATE="$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen5-20260910/match-confirm-20k-250pair/report.json"
BASE_TRAIN="$BASE_RUN/train.dedup-expanded.hkst"
BASE_RAW_MANIFEST="$ROOT_DIR/runs/halfkp64-scratch-v1/phase3-2m-preserved-target-20260825/raw/manifest.json"
VALID="$BASE_RUN/valid.filtered.hkst"; TEST="$BASE_RUN/test.filtered.hkst"
MODEL="$RUN_DIR/model.bin"; LOG="$RUN_DIR/train.csv"; CONFIG="$RUN_DIR/config.json"
NEW_RAW_TARGET="${NEW_RAW_TARGET:-2300000}"; MAX_TRAIN_RECORDS="${MAX_TRAIN_RECORDS:-4000000}"
WORKERS="${WORKERS:-12}"; SEED="${SEED:-20260913}"; MAX_EPOCH="${MAX_EPOCH:-16}"

mkdir -p "$RUN_DIR"; exec 9>"$RUN_DIR/.lock"; flock -n 9 || exit 0
exec > >(tee -a "$RUN_DIR/pipeline.log") 2>&1
need() { [[ -s "$1" ]] || { echo "missing: $1" >&2; exit 2; }; }

validate_inputs() {
  for p in "$PARENT" "$PARENT_MANIFEST" "$REGISTRY" "$GATE" "$BASE_TRAIN" "$BASE_RAW_MANIFEST" "$VALID" "$TEST"; do need "$p"; done
  local actual; actual="$(sha256sum "$PARENT" | awk '{print $1}')"
  [[ "$actual" == 162e17bacea3de80ecdbbdae765ca1157246c4febb6fa6a15f8d36b900d011dc ]] || { echo "unexpected parent SHA $actual" >&2; exit 2; }
  python3 - "$BASE_TRAIN.manifest.json" "$BASE_TRAIN" <<'PY'
import hashlib,json,pathlib,sys
m,p=map(pathlib.Path,sys.argv[1:]); x=json.loads(m.read_text())
if x.get('records') != 2060908: raise SystemExit(f'base train records changed: {x.get("records")}')
if x.get('output',{}).get('sha256') != hashlib.sha256(p.read_bytes()).hexdigest(): raise SystemExit('base train SHA mismatch')
PY
}

build_tools() {
  local missing=0 b
  for b in dataset_build jsonl_shard halfkp_search_teacher halfkp_search_dedup halfkp_search_train; do [[ -x "$ROOT_DIR/target/release/$b" ]] || missing=1; done
  if ((missing)); then
    RUST_FONTCONFIG_DLOPEN=1 cargo build --release --locked --features halfkp64,training-tools \
      --bin dataset_build --bin jsonl_shard --bin halfkp_search_teacher --bin halfkp_search_dedup --bin halfkp_search_train
  fi
}

build_raw() {
  mkdir -p "$RAW_NEW"
  if [[ ! -s "$RAW_NEW/manifest.json" ]]; then
    target/release/dataset_build \
      --input "$ROOT_DIR/data/wdoor/extract/2018" --input "$ROOT_DIR/data/wdoor/extract/2019" \
      --input "$ROOT_DIR/data/wdoor/extract/2020" --input "$ROOT_DIR/data/wdoor/extract/2021" \
      --input "$ROOT_DIR/data/wdoor/extract/2022" --input "$ROOT_DIR/data/wdoor/extract/2023" \
      --input "$ROOT_DIR/data/wdoor/extract/2024" --input "$ROOT_DIR/data/wdoor/extract/2025" \
      --input "$ROOT_DIR/data/wdoor/extract/2026" --output-dir "$RAW_NEW" \
      --license-id user-authorized:wdoor-public-archive --source-uri https://wdoor.c.u-tokyo.ac.jp/shogi/ \
      --seed "$SEED" --shuffle-games --valid-percent 0 --test-percent 0 --max-records "$NEW_RAW_TARGET" \
      --max-records-per-game 24 --phase-records-per-game 8 8 8 --min-ply 8 --max-ply 200 --known-result-only \
      --base-dataset-manifest "$BASE_RAW_MANIFEST" --exclude-base-train --freeze-base-eval
  fi
  python3 - "$RAW_NEW/manifest.json" "$NEW_RAW_TARGET" <<'PY'
import json,sys
n=int(json.load(open(sys.argv[1]))['stats']['train']['records'])
if n<int(sys.argv[2]): raise SystemExit(f'new raw records {n} < target {sys.argv[2]}')
print('new raw records',n)
PY
}

make_shards() {
  mkdir -p "$SHARDS"
  [[ -s "$SHARDS/train-manifest.json" ]] || target/release/jsonl_shard --input "$RAW_NEW/train.jsonl" \
    --output-prefix "$SHARDS/train-" --lines 50000 --suffix-width 4 --parent-manifest "$RAW_NEW/manifest.json"
}

make_teachers() {
  mkdir -p "$TEACHERS"; local i=0 input stem output
  for input in "$SHARDS"/train-*.jsonl; do
    [[ -e "$input" ]] || continue
    stem="${input##*/}"; stem="${stem%.jsonl}"; output="$TEACHERS/$stem.hkst"
    if [[ -s "$output.manifest.json" ]]; then ((i+=1)); continue; fi
    target/release/halfkp_search_teacher --weights "$PARENT" --input "$input" \
      --parent-manifest "$PARENT_MANIFEST" --model-manifest "$PARENT_MANIFEST" \
      --promotion-registry "$REGISTRY" --gate-report "$GATE" --parent-generation 6 --output "$output" \
      --value-only --qsearch-leaf --teacher-depth 8 --min-completed-depth 1 --max-nodes-per-search 100000 \
      --chunk-size 128 --jobs "$WORKERS" --seed "$((SEED+i))"
    ((i+=1))
  done
}

deduplicate() {
  FINAL_TRAIN="$RUN_DIR/train-4m.dedup.hkst"; local report="$RUN_DIR/train-4m.dedup.json"
  if [[ ! -s "$FINAL_TRAIN.manifest.json" ]]; then
    local args=(--output "$FINAL_TRAIN" --report "$report" --input "$BASE_TRAIN") input
    for input in "$TEACHERS"/train-*.hkst; do [[ -e "$input" ]] && args+=(--input "$input"); done
    target/release/halfkp_search_dedup "${args[@]}"
  fi
  python3 - "$report" "$MAX_TRAIN_RECORDS" <<'PY'
import json,sys
n=int(json.load(open(sys.argv[1]))['retained_records']); lim=int(sys.argv[2])
if n<lim: raise SystemExit(f'combined dedup records {n} < required {lim}')
print('combined dedup records',n,'training limit',lim)
PY
}

write_config() {
  [[ -s "$CONFIG" ]] && return
  python3 - "$CONFIG" <<'PY'
import json,os,pathlib
x={'schema':'halfkp_v4_gen6_4m_unattended_config_v1','lineage_id':'halfkp64-scratch-v1','parent_generation':6,
   'parent_model_sha256':'162e17bacea3de80ecdbbdae765ca1157246c4febb6fa6a15f8d36b900d011dc',
   'base_train_records':2060908,'effective_train_records':int(os.environ['MAX_TRAIN_RECORDS']),
   'new_raw_target':int(os.environ['NEW_RAW_TARGET']),'workers':int(os.environ['WORKERS']),
   'seed':int(os.environ['SEED']),'max_epoch':int(os.environ['MAX_EPOCH']),
   'teacher':{'value_only':True,'qsearch_leaf':True,'teacher_depth':8,'max_nodes_per_search':100000},
   'training':{'optimizer':'adagrad','feature_learning_rate':3e-4,'output_learning_rate':3e-5,
               'batch_size':128,'search_mix':0.8,'value_weight':1.0,'rank_weight':0.0,'loss_power':2.5,
               'early_stop_patience':1000000,'swa':False}}
pathlib.Path(sys.argv[1]).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
PY
}

train_stage() {
  local target="$1" snapshot="$CHECKPOINTS/selected-through-$1.bin"
  [[ -s "$snapshot" ]] && return
  mkdir -p "$CHECKPOINTS"
  local args=(--init "$PARENT" --model-manifest "$PARENT_MANIFEST" --promotion-registry "$REGISTRY" --gate-report "$GATE" --parent-generation 6
    --output "$MODEL" --epochs "$target" --batch-size 128 --learning-rate .0003 --output-learning-rate .00003 --kappa-cp 1352.7871
    --search-mix .8 --search-mix-end .8 --value-weight 1 --rank-weight 0 --game-rank-weight 0 --game-vs-search-weight 0
    --loss-power 2.5 --gradient-clip-norm 1 --output-limit .20 --optimizer adagrad --threads "$WORKERS" --seed "$SEED"
    --early-stop-patience 1000000 --min-valid-improvement .000001 --swa-start-epoch 1000000
    --checkpoint-dir "$CHECKPOINTS" --log "$LOG" --train "$FINAL_TRAIN" --max-train-records "$MAX_TRAIN_RECORDS" --valid "$VALID" --test "$TEST")
  [[ -s "$MODEL.manifest.json" && -s "$MODEL" ]] && args+=(--resume)
  target/release/halfkp_search_train "${args[@]}"
  need "$MODEL"; need "$MODEL.manifest.json"; cp "$MODEL" "$snapshot"; cp "$MODEL.manifest.json" "$snapshot.manifest.json"
}

write_completion() {
  python3 - "$RUN_DIR/completion.json" "$MODEL" "$MODEL.manifest.json" "$CONFIG" "$FINAL_TRAIN" <<'PY'
import hashlib,json,pathlib,subprocess,sys,time
completion,model,manifest,config,train=map(pathlib.Path,sys.argv[1:])
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads(manifest.read_text())
x={'schema':'halfkp_v4_gen6_4m_completion_v1','completed_unix_secs':int(time.time()),
   'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
   'model_sha256':sha(model),'manifest_sha256':sha(manifest),'config_sha256':sha(config),'train_sha256':sha(train),
   'completed_epoch':int(m.get('completed_epoch',0)),'selected_epoch':int(m.get('selected_epoch',0)),
   'selected_validation_score':m.get('selected_validation_score'),'records_limit':4000000}
completion.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); print('completion:',completion)
PY
}

export NEW_RAW_TARGET MAX_TRAIN_RECORDS WORKERS SEED MAX_EPOCH CONFIG
validate_inputs; build_tools; build_raw; make_shards; make_teachers; deduplicate; write_config
train_stage 4; ((MAX_EPOCH >= 8)) && train_stage 8; ((MAX_EPOCH >= 12)) && train_stage 12; ((MAX_EPOCH >= 16)) && train_stage 16
write_completion; echo "gen6 unattended 4M pipeline complete: $RUN_DIR"

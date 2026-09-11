#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen6-2m-20260911}"
RAW_NEW="$RUN_DIR/raw-new"; RAW="$RUN_DIR/raw"; SHARDS="$RUN_DIR/shards-50k"
TEACHERS="$RUN_DIR/teachers-100k"; CHECKPOINTS="$RUN_DIR/checkpoints"
PARENT="$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen5-20260910/candidate/model-epoch24.bin"
PARENT_MANIFEST="$PARENT.manifest.json"
REGISTRY="$ROOT_DIR/training/halfkp64-scratch-v1/promotion_registry.json"
GATE="$ROOT_DIR/runs/halfkp64-scratch-v1/v4-value-gen5-20260910/match-confirm-20k-250pair/report.json"
BASE="$ROOT_DIR/runs/halfkp64-scratch-v1/phase3-2m-preserved-target-20260825/raw"
ARCHIVE="$ROOT_DIR/data/wdoor/archive/wdoor2015.7z"; EXTRACT="$ROOT_DIR/data/wdoor/extract/2015"
WORKERS="${WORKERS:-12}"; SEED="${SEED:-20260911}"; MAX_EPOCH="${MAX_EPOCH:-32}"
NEW_TARGET="${NEW_RAW_TARGET:-1800000}"; REPLAY_TARGET="${REPLAY_RAW_TARGET:-450000}"
MODEL="$RUN_DIR/model.bin"; LOG="$RUN_DIR/train.csv"; CONFIG="$RUN_DIR/config.json"
VALID_FILTERED="$RUN_DIR/valid.filtered.hkst"
mkdir -p "$RUN_DIR"; exec 9>"$RUN_DIR/.lock"; flock -n 9 || exit 0
exec > >(tee -a "$RUN_DIR/pipeline.log") 2>&1
need() { [[ -s "$1" ]] || { echo "missing: $1" >&2; exit 2; }; }
validate_parent() {
  need "$PARENT"; need "$PARENT_MANIFEST"; need "$REGISTRY"; need "$GATE"
  [[ "$(sha256sum "$PARENT" | awk '{print $1}')" == 162e17bacea3de80ecdbbdae765ca1157246c4febb6fa6a15f8d36b900d011dc ]] || exit 2
}
ensure_2015() {
  mkdir -p "$(dirname "$ARCHIVE")" "$EXTRACT"
  [[ -s "$ARCHIVE" ]] || curl -fL --retry 5 --retry-delay 2 -o "$ARCHIVE" https://wdoor.c.u-tokyo.ac.jp/shogi/archive/wdoor2015.7z
  [[ "$(sha256sum "$ARCHIVE" | awk '{print $1}')" == 24780b2b3a9ebbb094ff4199df9a0c47b2a1cc699ce205d91dcccb53665e81d2 ]] || exit 2
  local marker="$EXTRACT/.complete.json"
  if [[ ! -s "$marker" ]]; then
    while pgrep -f "7z x -y -o.*extract/2015 .*wdoor2015\\.7z" >/dev/null 2>&1; do
      echo "waiting for wdoor2015 extraction"; sleep 30
    done
    [[ -n "$(find "$EXTRACT" -type f -name '*.csa' -print -quit)" ]] || 7z x -y -o"$EXTRACT" "$ARCHIVE" >/dev/null
    local count; count="$(find "$EXTRACT" -type f -name '*.csa' | wc -l)"
    [[ "$count" -gt 0 ]] || exit 2
    python3 - "$marker" "$ARCHIVE" "$count" <<'PY'
import hashlib,json,sys
out,arc,count=sys.argv[1:]; h=hashlib.sha256()
with open(arc,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
json.dump({'schema':'wdoor_extract_complete_v1','archive_sha256':h.hexdigest(),'csa_files':int(count)},open(out,'w'),indent=2)
PY
  fi
}
build_tools() {
  local miss=0 b
  for b in dataset_build jsonl_shard halfkp_search_teacher halfkp_search_dedup halfkp_search_split_filter halfkp_search_train; do [[ -x "$ROOT_DIR/target/release/$b" ]] || miss=1; done
  if ((miss)); then cargo build --release --locked --features halfkp64,training-tools --bin dataset_build --bin jsonl_shard --bin halfkp_search_teacher --bin halfkp_search_dedup --bin halfkp_search_split_filter --bin halfkp_search_train; fi
}
build_raw() {
  mkdir -p "$RAW_NEW" "$RAW"
  if [[ ! -s "$RAW_NEW/manifest.json" ]]; then
    if compgen -G "$RAW_NEW/*.jsonl" >/dev/null; then echo "partial raw-new"; exit 2; fi
    target/release/dataset_build --input "$EXTRACT" --input "$ROOT_DIR/data/wdoor/extract/2016" --input "$ROOT_DIR/data/wdoor/extract/2017" \
      --output-dir "$RAW_NEW" --license-id user-authorized:wdoor-public-archive --source-uri https://wdoor.c.u-tokyo.ac.jp/shogi/ \
      --seed "$SEED" --shuffle-games --valid-percent 0 --test-percent 0 --max-records "$NEW_TARGET" --max-records-per-game 24 \
      --phase-records-per-game 8 8 8 --min-ply 8 --max-ply 200 --known-result-only \
      --base-dataset-manifest "$BASE/manifest.json" --exclude-base-train --freeze-base-eval
  fi
  python3 - "$RAW_NEW/manifest.json" "$NEW_TARGET" <<'PY'
import json,sys
n=int(json.load(open(sys.argv[1]))['stats']['train']['records'])
if n<int(sys.argv[2]): raise SystemExit(f'new raw records {n} < target')
PY
  if [[ ! -s "$RAW/manifest.json" ]]; then
    if compgen -G "$RAW/*" >/dev/null; then echo "partial raw composition"; exit 2; fi
    python3 - "$RAW_NEW" "$BASE" "$RAW" "$REPLAY_TARGET" <<'PY'
import pathlib,shutil,sys
new,old,out,replay=map(pathlib.Path,sys.argv[1:4]); replay=int(sys.argv[4]); out.mkdir(parents=True,exist_ok=True)
with (out/'train.jsonl').open('wb') as d:
  with (new/'train.jsonl').open('rb') as s: shutil.copyfileobj(s,d)
  with (old/'train.jsonl').open('rb') as s:
    for i,line in enumerate(s):
      if i>=replay: break
      d.write(line)
for k in ('valid','test'): shutil.copyfile(new/f'{k}.jsonl',out/f'{k}.jsonl')
PY
    python3 - "$RAW" "$RAW_NEW/manifest.json" "$BASE/manifest.json" "$NEW_TARGET" "$REPLAY_TARGET" <<'PY'
import hashlib,json,pathlib,subprocess,sys,time
root,newm,basem,newn,replay=sys.argv[1:]; root=pathlib.Path(root)
def inf(p):
 h=hashlib.sha256(); n=0
 with p.open('rb') as f:
  for l in f: h.update(l); n+=1
 return {'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest(),'records':n}
files={k:inf(root/f'{k}.jsonl') for k in ('train','valid','test')}
x={'schema_version':1,'schema':'halfkp_v4_raw_composed_v2','stage':'v4_t13_gen6_2m_raw','lineage_id':'halfkp64-scratch-v1',
 'created_unix_secs':int(time.time()),'git_rev':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 'license_id':'user-authorized:wdoor-public-archive','source_uri':'https://wdoor.c.u-tokyo.ac.jp/shogi/','freeze_base_eval':True,
 'exclude_base_train':True,'base_dataset_manifest':basem,'new_source_manifest':newm,
 'composition':{'new_records':int(newn),'replay_records':int(replay),'deduplication':'pending teacher dedup'},
 'outputs':files,'stats':{k:{'records':files[k]['records']} for k in files}}
(root/'manifest.json').write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
PY
  fi
  python3 - "$RAW/manifest.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); n=x['stats']['train']['records']
if n<2000000: raise SystemExit(f'composed raw records {n} < 2000000')
for k in ('train','valid','test'):
 p=sys.argv[1].replace('manifest.json',k+'.jsonl')
 if not open(p,'rb').read(1): raise SystemExit(f'empty {p}')
print('raw train records',n)
PY
}
make_shards() {
  mkdir -p "$SHARDS"
  for k in train valid test; do
    [[ -s "$SHARDS/$k-manifest.json" ]] && continue
    if compgen -G "$SHARDS/$k-*.jsonl" >/dev/null; then echo "partial shards: $k"; exit 2; fi
    target/release/jsonl_shard --input "$RAW/$k.jsonl" --output-prefix "$SHARDS/$k-" --lines 50000 --suffix-width 4 --parent-manifest "$RAW/manifest.json"
  done
}
teachers() {
  mkdir -p "$TEACHERS"; local i=0 input stem out
  for input in "$SHARDS"/train-*.jsonl "$SHARDS"/valid-*.jsonl "$SHARDS"/test-*.jsonl; do
    [[ -e "$input" ]] || continue; stem="${input##*/}"; stem="${stem%.jsonl}"; out="$TEACHERS/$stem.hkst"
    if [[ -s "$out.manifest.json" ]]; then ((i+=1)); continue; fi
    [[ ! -e "$out" && ! -e "$out.tmp" ]] || { echo "partial teacher: $out"; exit 2; }
    target/release/halfkp_search_teacher --weights "$PARENT" --input "$input" --parent-manifest "$PARENT_MANIFEST" --model-manifest "$PARENT_MANIFEST" \
      --promotion-registry "$REGISTRY" --gate-report "$GATE" --parent-generation 6 --output "$out" --value-only --qsearch-leaf --teacher-depth 8 \
      --min-completed-depth 1 --max-nodes-per-search 100000 --chunk-size 128 --jobs "$WORKERS" --seed "$((SEED+i))"
    ((i+=1))
  done
}
dedup_filter() {
  local dedup="$RUN_DIR/train.dedup.hkst" report="$RUN_DIR/train.dedup.json" p
  if [[ ! -s "$dedup.manifest.json" ]]; then
    local args=(--output "$dedup" --report "$report")
    for p in "$TEACHERS"/train-*.hkst; do [[ -e "$p" ]] && args+=(--input "$p"); done
    target/release/halfkp_search_dedup "${args[@]}"
  fi
  python3 - "$report" <<'PY'
import json,sys
n=int(json.load(open(sys.argv[1]))['retained_records'])
if n<2000000: raise SystemExit(f'dedup records {n} < 2000000')
print('dedup records',n)
PY
  if [[ ! -s "$VALID_FILTERED.manifest.json" ]]; then
    local args=(--output "$VALID_FILTERED" --report "$RUN_DIR/valid.filter.json")
    args+=(--train "$dedup")
    for p in "$TEACHERS"/valid-*.hkst; do [[ -e "$p" ]] && args+=(--valid "$p"); done
    target/release/halfkp_search_split_filter "${args[@]}"
  fi
  FINAL_TRAIN="$dedup"; FINAL_VALID="$VALID_FILTERED"; FINAL_TEST=""
  for p in "$TEACHERS"/test-*.hkst; do [[ -e "$p" ]] && FINAL_TEST="$p"; done
  [[ -n "$FINAL_TEST" ]] || exit 2
}
write_config() {
  [[ -s "$CONFIG" ]] && return
  python3 - "$CONFIG" <<'PY'
import json,os,pathlib
x={'schema':'halfkp_v4_gen6_2m_unattended_config_v1','lineage_id':'halfkp64-scratch-v1','parent_generation':6,
'parent_model_sha256':'162e17bacea3de80ecdbbdae765ca1157246c4febb6fa6a15f8d36b900d011dc',
'new_raw_target':int(os.environ['NEW_TARGET']),'replay_raw_target':int(os.environ['REPLAY_TARGET']),
'workers':int(os.environ['WORKERS']),'seed':int(os.environ['SEED']),'max_epoch':int(os.environ['MAX_EPOCH']),
'teacher':{'value_only':True,'qsearch_leaf':True,'teacher_depth':8,'max_nodes_per_search':100000},
'training':{'optimizer':'adagrad','feature_learning_rate':3e-4,'output_learning_rate':3e-5,'batch_size':128,'search_mix':.8,'value_weight':1.0,'rank_weight':0.0}}
pathlib.Path(os.environ['CONFIG']).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
PY
}
train_epoch() {
  local target="$1" snap="$CHECKPOINTS/model-epoch-$1.bin"
  [[ -s "$snap" ]] && return
  mkdir -p "$CHECKPOINTS"; local args=(--init "$PARENT" --model-manifest "$PARENT_MANIFEST" --promotion-registry "$REGISTRY" --gate-report "$GATE" --parent-generation 6
    --output "$MODEL" --epochs "$target" --batch-size 128 --learning-rate .0003 --output-learning-rate .00003 --kappa-cp 1352.7871
    --search-mix .8 --search-mix-end .8 --value-weight 1 --rank-weight 0 --game-rank-weight 0 --game-vs-search-weight 0 --loss-power 2.5
    --gradient-clip-norm 1 --output-limit .20 --optimizer adagrad --threads "$WORKERS" --seed "$SEED" --checkpoint-dir "$CHECKPOINTS" --save-last-epoch --log "$LOG"
    --train "$FINAL_TRAIN" --valid "$FINAL_VALID" --test "$FINAL_TEST")
  if [[ -s "$MODEL.manifest.json" ]]; then args+=(--resume); elif [[ -e "$MODEL" || -e "$MODEL.tmp" ]]; then exit 2; fi
  target/release/halfkp_search_train "${args[@]}"; need "$MODEL"
  local completed; completed="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["completed_epoch"])' "$MODEL.manifest.json")"
  cp "$MODEL" "$CHECKPOINTS/model-epoch-$completed.bin"; cp "$MODEL.manifest.json" "$CHECKPOINTS/model-epoch-$completed.bin.manifest.json"
}
validate_parent; ensure_2015; build_tools
export CONFIG NEW_TARGET REPLAY_TARGET WORKERS SEED MAX_EPOCH
write_config; build_raw; make_shards; teachers; dedup_filter
train_epoch 4; ((MAX_EPOCH>=8)) && train_epoch 8; ((MAX_EPOCH>=16)) && train_epoch 16; ((MAX_EPOCH>=32)) && train_epoch 32
echo "gen6 unattended 2M pipeline complete: $RUN_DIR"


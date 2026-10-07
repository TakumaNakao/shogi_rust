#!/usr/bin/env bash
set -Eeuo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"
run_dir="$root_dir/runs/halfkp64-scratch-v1/phase6-long-direct-v1-seed20260721"
checkpoint="$run_dir/checkpoint"
model="$run_dir/model.bin"
manifest="$run_dir/model.bin.manifest.json"
log="$run_dir/milestone-supervisor.log"
train_bin="$root_dir/target/release/halfkp_scratch_train"

train_args=(
  --train "$root_dir/runs/halfkp64-scratch-v1/exact-rank-v4-dev1k/train.hkcl"
  --train "$root_dir/runs/halfkp64-scratch-v1/exact-rank-v4-supp12k/train.hkcl"
  --train "$root_dir/runs/halfkp64-scratch-v1/exact-rank-v4-supp48k/train.hkcl"
  --train "$root_dir/runs/halfkp64-scratch-v1/legal-health-v7-anchor005/train.hkcl"
  --train "$root_dir/runs/halfkp64-scratch-v1/wdoor-highrate4000-1m-hkcl-v1/train.hkcl"
  --valid "$root_dir/runs/halfkp64-scratch-v1/exact-rank-v4-dev1k/valid.hkcl"
  --valid "$root_dir/runs/halfkp64-scratch-v1/wdoor-highrate4000-1m-hkcl-v1/valid.hkcl"
  --init "$root_dir/runs/halfkp64-scratch-v1/phase3-wdoor250k-v1-mix70-epoch2-seed20260721/model.bin"
  --init-manifest "$root_dir/runs/halfkp64-scratch-v1/phase3-wdoor250k-v1-mix70-epoch2-seed20260721/model.bin.manifest.json"
  --parent-gate-report "$root_dir/training/halfkp64-scratch-v1/promotion_gen2_report.json"
  --output "$model" --batch-size 256 --learning-rate 0.001 --kappa-cp 600
  --rank-temperature-cp 300 --symmetry-temperature-cp 30 --rank-weight 1
  --expert-preference-weight 1 --game-result-weight 1 --source-mix-exact 0.60
  --source-mix-expert 0.25 --source-mix-result 0.15 --source-mix-search 0
  --search-rank-weight 0 --stage calibrated --factorization direct --seed 20260721
  --l2 0 --weight-decay 0.01 --output-learning-rate-scale 0.4
  --score-anchor-weight 2 --score-anchor-cp 600 --activation-anchor-weight 0
  --checkpoint-dir "$checkpoint" --save-last-epoch --resume
)

wait_for_epoch() {
  local target="$1"
  while :; do
    if rg -q '"completed_epoch": '"$target"',' "$checkpoint/state.json" 2>/dev/null; then
      return
    fi
    sleep 60
  done
}

evaluate_epoch() {
  local epoch="$1"
  local out="$run_dir/epoch$epoch"
  mkdir -p "$run_dir/milestones/epoch$epoch"
  cp "$model" "$run_dir/milestones/epoch$epoch/model.bin"
  cp "$manifest" "$run_dir/milestones/epoch$epoch/model.bin.manifest.json"
  target/release/halfkp_scratch_eval --model "$model" --model-manifest "$manifest" --dataset "$root_dir/runs/halfkp64-scratch-v1/exact-rank-v4-dev1k/valid.hkcl" --json "$out-exact.json"
  target/release/halfkp_scratch_eval --model "$model" --model-manifest "$manifest" --dataset "$root_dir/runs/halfkp64-scratch-v1/wdoor-highrate4000-1m-hkcl-v1/valid.hkcl" --json "$out-highrate.json"
  target/release/halfkp_scratch_eval --model "$model" --model-manifest "$manifest" --dataset "$root_dir/runs/halfkp64-scratch-v1/wdoor-phase3-250k-hkcl-v1/valid.hkcl" --json "$out-broad.json"
  target/release/halfkp_scratch_eval --model "$model" --model-manifest "$manifest" --dataset "$root_dir/runs/halfkp64-scratch-v1/legal-health-v6-stratified/valid.hkcl" --json "$out-health.json"
  target/release/halfkp_static_gate --model "$model" --model-manifest "$manifest" --dataset "$root_dir/runs/halfkp64-scratch-v1/exact-rank-v4-dev1k/valid.hkcl" --json "$out-static.json"
  target/release/halfkp_search_gate --model "$model" --model-manifest "$manifest" --dataset "$root_dir/runs/halfkp64-scratch-v1/exact-rank-v4-dev1k/valid.hkcl" --nodes 200000 --json "$out-search200k.json"
  NEW_WEIGHTS="$model" BASELINE_WEIGHTS="$root_dir/policy_weights_halfkp64_kpp_distilled_v2.5.0.binary" GAMES=40 TIME_LIMIT_MS=100 DEPTH=32 MAX_PLIES=256 SEED=20260726 THREADS=1 JOBS=4 OUT_DIR="$out-screen" "$root_dir/scripts/bench_halfkp64_learned_vs_v250.sh"
}

{
  echo "supervisor started $(date -Is)"
  wait_for_epoch 32
  evaluate_epoch 32
  "$train_bin" "${train_args[@]}" --epochs 64 >>"$run_dir/train-epoch33-to64.log" 2>&1
  wait_for_epoch 64
  evaluate_epoch 64
  "$train_bin" "${train_args[@]}" --epochs 128 >>"$run_dir/train-epoch65-to128.log" 2>&1
  wait_for_epoch 128
  evaluate_epoch 128
  echo "supervisor completed $(date -Is)"
} >>"$log" 2>&1

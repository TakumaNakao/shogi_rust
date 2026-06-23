# MMTO tree 実践投入前手順（v2.1.0ベース）

目的は `mmto_tree_dump` / `mmto_tree_train` / `mmto_score_gate` / `mmto_rerank_gate` を本番導入前のゲートとして運用し、候補重みを作ることにあります。

方針:

- オフラインゲートは最優先（対局前に必ず通す）。
- 本文書は「本格学習前の採用前提手順」であり、1時間程度の試験（smoke）だけで止めない。
- `best_epoch=0` は baseline が最良のままなので**不採用**。
- 20局だけで採用判定しない。
- 直接採用候補を出さず、まず blend から対局へ進める。

## 1. 事前の検証コマンド

```bash
cd /home/nami_ride_trade/shogi_rust

env RUST_FONTCONFIG_DLOPEN=1 cargo fmt --check

# 既存実装
env RUST_FONTCONFIG_DLOPEN=1 cargo check \
  --bin mmto_tree_dump \
  --bin mmto_tree_train \
  --bin mmto_score_gate \
  --bin mmto_rerank_gate \
  --bin adjust_weights \
  --bin usi_benchmark

env RUST_FONTCONFIG_DLOPEN=1 cargo build --release \
  --bin mmto_tree_dump \
  --bin mmto_tree_train \
  --bin mmto_score_gate \
  --bin mmto_rerank_gate \
  --bin adjust_weights \
  --bin usi_benchmark

env RUST_FONTCONFIG_DLOPEN=1 cargo test --all-targets
```

## 2. Track

### Track A（基準）

- baseline は `policy_weights_v2.1.0.binary`。
- 学習・対局ゲートの比較基準は Track A。

### Track B（保守比較）

- Track A と同じパイプラインを別初期化で回し、再現性と安定性を比較する。

## 3. 本格学習前の標準フロー

1. 標準木データ作成（`mmto_tree_dump`）
2. 学習（`mmto_tree_train`）
3. offline gate（`mmto_score_gate`、`mmto_rerank_gate`）
4. hard-valid 対策（rerank候補の再収集）
5. DAgger再dump（hard局面再収集で再学習）
6. blend作成（`adjust_weights --blend-target --blend-ratio`）
7. 対局 gate（20局smoke + 100局以上）

offline gate を通過せずに対局へ進まない。

まとめて実行する場合は、以下のスクリプトを使う。

```bash
bash tools/run_mmto_rerank_pipeline.sh
```

24時間程度放置して長めに回す場合は、古いrunを消して空き容量を作ってから長時間用プリセットを使う。

```bash
bash tools/clean_mmto_runs.sh
bash tools/run_mmto_rerank_long.sh
```

長時間用プリセットの主なデフォルト:

- `POSITIONS=converted_records2016_10818.sfen`
- `MAX_POSITIONS=10000`
- `TEACHER_DEPTH=5`
- `STUDENT_DEPTH=4`
- `EPOCHS=8`
- `BLEND_RATIOS="0.02 0.05"`

主な環境変数:

```bash
MAX_POSITIONS=5000 EPOCHS=5 bash tools/run_mmto_rerank_pipeline.sh
TEACHER_DEPTH=5 STUDENT_DEPTH=4 RERANK_TEACHER_DEPTH=5 bash tools/run_mmto_rerank_pipeline.sh
MAX_POSITIONS=8000 EPOCHS=6 bash tools/run_mmto_rerank_long.sh
```

古いMMTO run生成物を消す場合:

```bash
bash tools/clean_mmto_runs.sh
```

確認なしで消す場合:

```bash
bash tools/clean_mmto_runs.sh --yes
```

## 4. `mmto_tree_dump`（本格前半）

```bash
RUN_DIR="data/mmto/runs/mmto_tree_full_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_tree_dump \
  --student-weights policy_weights_v2.1.0.binary \
  --teacher-weights policy_weights_v2.1.0.binary \
  --input taya36.sfen \
  --train-output "$RUN_DIR/train.rank.jsonl" \
  --valid-output "$RUN_DIR/valid.rank.jsonl" \
  --teacher-depth 4 \
  --student-depth 3 \
  --teacher-score-top 16 \
  --candidate-top 16 \
  --valid-percent 10 \
  --min-legal-moves 2 \
  --exclude-in-check \
  --max-positions 3000 \
  --seed 7101
```

## 5. `mmto_tree_train`（multi-threshold bad regret）

```bash
env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_tree_train \
  --weights policy_weights_v2.1.0.binary \
  --train "$RUN_DIR/train.rank.jsonl" \
  --valid "$RUN_DIR/valid.rank.jsonl" \
  --output "$RUN_DIR/candidate.binary" \
  --best-checkpoint-path "$RUN_DIR/best.binary" \
  --epochs 5 \
  --batch-size 128 \
  --learning-rate 0.0002 \
  --optimizer adagrad \
  --best-metric selected-regret \
  --bad-regret-cp 300 \
  --bad-regret-thresholds-cp 50,100,200,300 \
  --freeze-material \
  --anchor-l2 0.0002 \
  --max-weight-delta 0.05 \
  --log-path "$RUN_DIR/train.log"
```

必須チェック:

- `train.log` の `best_epoch=` が `0` なら **不採用**。
- `--bad-regret-thresholds-cp` は実行ログ上で `bad50` `bad100` `bad200` `bad300`（比率）が全部残ることを確認。
- 1 epoch目は baseline 行（`epoch=0`）として扱う。比較は epoch 0 より改善しているかで判定する。

## 6. offline gate（優先度最上位）

### 6.1 `mmto_score_gate`

```bash
env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_score_gate \
  --baseline-weights policy_weights_v2.1.0.binary \
  --candidate-weights "$RUN_DIR/best.binary" \
  --input taya36.sfen \
  --max-positions 2000 \
  --seed 7201 \
  --p95-limit-cp 50 \
  --max-limit-cp 200 \
  --mean-limit-cp 10 \
  --fail-on-material-drift-cp 5 \
  --json-output "$RUN_DIR/mmto_score_gate.json"
```

通過条件:

- `p95 <= 50`、`max <= 200`
- 必要なら `mean` と material drift も確認
- failなら再dump/データ条件見直し

### 6.2 `mmto_rerank_gate`

```bash
env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_rerank_gate \
  --baseline-weights policy_weights_v2.1.0.binary \
  --candidate-weights "$RUN_DIR/best.binary" \
  --teacher-weights policy_weights_v2.1.0.binary \
  --input taya36.sfen \
  --max-positions 2000 \
  --seed 7202 \
  --baseline-depth 3 \
  --candidate-depth 3 \
  --teacher-depth 5 \
  --bad-regret-thresholds-cp 50,100,200,300 \
  --allow-mean-regret-increase-cp 0 \
  --allow-p90-regret-increase-cp 0 \
  --allow-p95-regret-increase-cp 0 \
  --allow-bad-ratio-increase 0 \
  --hard-position-limit 1000 \
  --json-output "$RUN_DIR/mmto_rerank_gate.json"
```

`mmto_rerank_gate.json` には `hard_positions` が出力される。ここから hard-valid 用のSFENを抽出する。

## 7. blend候補作成

`adjust_weights --blend-target --blend-ratio` を使って、直接置換前の候補として blend を作る。

```bash
for R in 0.02 0.05 0.10 0.20; do
  env RUST_FONTCONFIG_DLOPEN=1 target/release/adjust_weights \
    --input policy_weights_v2.1.0.binary \
    --blend-target "$RUN_DIR/best.binary" \
    --blend-ratio "$R" \
    --output "$RUN_DIR/blend_${R}.binary"
done
```

## 8. hard-valid と DAgger再dump

hard局面の再収集（rerank結果）:

```bash
# rerank結果から hard_positions.sfen を作成
python3 - "$RUN_DIR/mmto_rerank_gate.json" > "$RUN_DIR/hard_positions.sfen" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
for pos in payload.get("hard_positions", []):
    print(pos["sfen"])
PY
```

hard局面を重点再収集して再ダンプ（DAgger再dump）:

```bash
env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_tree_dump \
  --student-weights "$RUN_DIR/best.binary" \
  --teacher-weights policy_weights_v2.1.0.binary \
  --input "$RUN_DIR/hard_positions.sfen" \
  --train-output "$RUN_DIR/hard_train.rank.jsonl" \
  --valid-output "$RUN_DIR/hard_valid.rank.jsonl" \
  --teacher-depth 6 \
  --student-depth 5 \
  --teacher-score-top 32 \
  --candidate-top 32 \
  --valid-percent 20 \
  --max-positions 2000 \
  --seed 7301

cat "$RUN_DIR/train.rank.jsonl" "$RUN_DIR/hard_train.rank.jsonl" > "$RUN_DIR/train_with_hard.rank.jsonl"
cat "$RUN_DIR/valid.rank.jsonl" "$RUN_DIR/hard_valid.rank.jsonl" > "$RUN_DIR/valid_with_hard.rank.jsonl"
```

hard-valid付き再学習:

```bash
env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_tree_train \
  --weights "$RUN_DIR/best.binary" \
  --train "$RUN_DIR/train_with_hard.rank.jsonl" \
  --valid "$RUN_DIR/valid_with_hard.rank.jsonl" \
  --extra-valid "hard=$RUN_DIR/hard_valid.rank.jsonl" \
  --output "$RUN_DIR/candidate_dagger.binary" \
  --best-checkpoint-path "$RUN_DIR/best_dagger.binary" \
  --epochs 3 \
  --batch-size 128 \
  --learning-rate 0.0002 \
  --bad-regret-cp 300 \
  --bad-regret-thresholds-cp 50,100,200,300 \
  --best-metric selected-regret \
  --freeze-material \
  --anchor-l2 0.0002 \
  --max-weight-delta 0.05 \
  --log-path "$RUN_DIR/train_dagger.log"
```

## 9. 対局 gate（採用は段階的）

- 20局は「破綻検知」。  
  **20局だけで採用しない**（これは条件ではない）。
- 20局を通過したら、seed変更で100局以上を比較。  
  - `usi_benchmark` で `seed` を複数変える  
  - 両者（baseline vs 候補）を同一局面で比較
- 100局で有意悪化がある場合は棄却。

## 10. 最終採用条件

- `mmto_tree_train` の `best_epoch > 0`
- `mmto_score_gate` 通過
- `mmto_rerank_gate` 通過
- blend候補の少なくとも1種類が、20局smokeと100局以上で一貫して悪化しない
- material 破綻・`best` 退化（`bad_*` 比率悪化）がないこと

## 11. 破棄条件

- `best_epoch=0`
- offline gateのいずれか失敗
- `train.log` / `mmto_score_gate.json` / `mmto_rerank_gate` の情報に異常がある
- 20局/100局どちらでも明確に悪化（勝率低下、引き分け異常増、時間超過・再現不能な挙動）

# Phase-balanced MMTO入力生成

- 作成日時: 2026-06-28 12:49:43 UTC
- 対象ブランチ: `training/strong-weight-learning-infra`
- 目的: 長時間学習を回す価値のある学習データを作るため、序盤・中盤・終盤の偏りを制御できる前処理を追加する。

## 背景

直近の重み学習実験では、静的なvalid指標やfeedback violationが改善しても、探索込みのrerank gateで悪化するケースが続いた。

GPT-5.5 xhigh分析では、単純にepochや時間を増やすよりも、以下を優先すべきと判断された。

- bench failureを通常lossへ雑に混ぜない。
- Bonanza/MMTO寄りに、探索結果と棋譜手が整合するデータを作る。
- 序盤・中盤・終盤・王手局面・合法手少数局面の偏りを管理する。
- 長時間学習前にカテゴリ別gateで悪化を検出する。

そのため、まず `dataset_build` が作る教師棋譜JSONLをphase別に抽出できるツールを追加した。

## 追加内容

新規バイナリ:

```text
mmto_balance_dataset
```

入力:

```text
dataset_build が出力する train/valid/test JSONL
```

出力:

```text
mmto_tree_dump に直接渡せる JSONL
```

出力レコードには以下を含める。

```text
sfen
teacher_move
sample_weight
phase
ply
legal_moves
in_check
```

機能:

- opening / middle / late ごとの件数上限を指定できる。
- phaseごとに `sample_weight` を指定できる。
- SFEN重複排除ができる。
- 王手中局面や合法手数のフィルタができる。
- reservoir sampling により、入力全体をメモリに保持しない。

## Pipeline連携

`tools/run_bonanza_root_pipeline.sh` に任意オプションを追加した。

```text
PHASE_BALANCE=1
PHASE_OPENING_LIMIT
PHASE_MIDDLE_LIMIT
PHASE_LATE_LIMIT
PHASE_OPENING_WEIGHT
PHASE_MIDDLE_WEIGHT
PHASE_LATE_WEIGHT
PHASE_DEDUPE_SFEN
```

デフォルトでは従来通り `dataset/all.jsonl` を `mmto_tree_dump` に渡す。`PHASE_BALANCE=1` のときだけ `dataset/balanced.jsonl` を作って、それをdump入力にする。

## 検証

Spark サブエージェントに検証を委任した。

確認済み:

```text
cargo build --release --bin mmto_balance_dataset --bin dataset_build --bin mmto_tree_dump --bin mmto_tree_train: 成功
cargo test --all-targets: 成功
```

手製JSONLでの確認:

```text
opening=2
middle=2
late=2
sample_weight:
  opening=1.2
  middle=1.0
  late=0.8
```

Pipeline smoke:

```text
PHASE_BALANCE=1
MAX_RECORDS=60
TREE_MAX_POSITIONS=30
TEACHER_DEPTH=2
STUDENT_DEPTH=1

dataset/all.jsonl: 60
dataset/balanced.jsonl: 10
dump: train records=9, valid records=1
mmto_tree_train: epoch 1 まで到達
巨大binary残存なし
```

短い2026入力ではopeningに偏った。これはsmoke入力が短手数に偏ったためであり、phase balance自体は手製JSONLで期待通り動作した。

## 次の使い方

長時間学習候補では、まずphase別に上限を切ってMMTO入力を作る。

例:

```bash
RUN_DIR=data/mmto/runs/bonanza_phase_balanced_$(date -u +%Y%m%d_%H%M%S) \
PHASE_BALANCE=1 \
PHASE_OPENING_LIMIT=20000 \
PHASE_MIDDLE_LIMIT=20000 \
PHASE_LATE_LIMIT=10000 \
PHASE_OPENING_WEIGHT=1.1 \
PHASE_MIDDLE_WEIGHT=1.0 \
PHASE_LATE_WEIGHT=0.8 \
MAX_RECORDS=120000 \
MAX_RECORDS_PER_GAME=12 \
MIN_PLY=16 \
MAX_PLY=160 \
TREE_MAX_POSITIONS=50000 \
TEACHER_DEPTH=4 \
STUDENT_DEPTH=3 \
bash tools/run_bonanza_root_pipeline.sh
```

採用候補にする条件:

- `mmto_tree_train` のカテゴリ別ログで opening/middle/late が大きく悪化しない。
- `in_check` と `low_legal` が悪化しない。
- score gate を通る。
- rerank gate で mean/p90/p95/bad50/bad100/match が非悪化。
- 最終的に対局ベンチ100局以上で確認する。

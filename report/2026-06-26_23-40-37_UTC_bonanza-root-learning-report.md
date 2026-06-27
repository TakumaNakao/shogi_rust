# Bonanza型root選択学習の導入レポート

- 作成日時: 2026-06-26 23:40:37 UTC
- ブランチ: `training/strong-weight-learning-infra`
- 目的: 既存の自己蒸留MMTO系が対局ベンチで安定しない問題を受け、外部棋譜手を探索候補に組み込むBonanza型root選択学習へ移行する。

## 背景

100K stream、hard replay、PV sibling weighted などのMMTO-lite実験では、オフライン指標は少し改善しても対局ベンチへ安定して移らなかった。

直近で最も良かったPV sibling 3K ultra-safe候補は、同一エンジン・重み比較100局で以下だった。

- seed 10901 / 20局: `11-7-2`
- seed 11001 / 20局: `10-8-2`
- seed 11101 / 60局: `30-25-5`
- 合算100局: `51-40-9`
- total score rate: `55.50%`

これは初めて100局で少し勝ち越したMMTO系重みだが、採用ラインには遠い。自己蒸留の拡大だけでは、現在の重み・探索が持つバイアスを超えにくいと判断した。

GPT-5.5 xhigh サブエージェントの分析でも、次のP0はPV sibling拡大ではなく、棋譜手を探索候補に入れたBonanza型root選択学習と判断された。

## 実装内容

`mmto_tree_dump` を拡張し、入力JSONLの `teacher_move` を読めるようにした。

- `{ "sfen": "...", "teacher_move": "7g7f" }` を受け付ける。
- `teacher_move` が合法なら探索候補へ必ず強制追加する。
- teacher search top-K、student search selected、棋譜手の和集合を候補にする。
- 出力JSONLへ `game_teacher_move` を追加する。
- 各候補へ `is_game_teacher_move` を追加する。
- 不正な `teacher_move` は局面自体を捨てず、teacher moveなしとして扱い、件数を表示する。

`mmto_tree_train` も拡張した。

- `selected_by_student` を読み、dump時の実探索selectedを直接bad候補に使う。
- `is_game_teacher_move` を読み、棋譜手をgood候補として追加margin lossに使えるようにした。
- 新規引数:
  - `--game-teacher-margin-weight`
  - `--game-teacher-max-regret-cp`
  - `--game-teacher-min-bad-regret-cp`
- 旧JSONLは引き続き読み込み可能。

再現用スクリプトを追加した。

- `tools/run_bonanza_root_pipeline.sh`
- CSAから `dataset_build` で棋譜手付きdatasetを作る。
- `mmto_tree_dump` で棋譜手をforce includeしたroot候補JSONLへ変換する。
- `mmto_tree_train` でgame-teacher margin付き学習を行う。
- score gate / rerank gateまで自動実行する。
- gate失敗または `best_epoch=0` の場合、大きな候補重みを自動削除する。

## スモーク検証

実行条件:

```bash
MAX_RECORDS=100 TREE_MAX_POSITIONS=100 RERANK_MAX_POSITIONS=40 \
JOBS=2 POSITION_CHUNK_SIZE=8 EPOCHS=1 BATCH_SIZE=32 \
TEACHER_DEPTH=3 STUDENT_DEPTH=2 TEACHER_SCORE_TOP=16 CANDIDATE_TOP=16 \
SCORE_ALL_LEGAL_FOR_VALID=0 \
RUN_DIR=data/mmto/runs/bonanza_root_smoke_20260626_233926 \
bash tools/run_bonanza_root_pipeline.sh
```

結果:

- dataset: 100局面
- dump: train 90 / valid 10
- root records: 100
- `game_teacher_move`: train 90/90
- `is_game_teacher_move`: train 90/90
- `selected_by_student`: train 90/90
- rank stats:
  - selected regret mean: `9.82`
  - p90: `28.84`
  - p95: `44.74`
  - bad50: `0.0500`
- train baseline:
  - selected regret mean: `34.07`
  - p95: `130.75`
- valid baseline:
  - selected regret mean: `31.12`
  - p95: `156.71`
- epoch 1:
  - valid p95 unchanged
  - `best_epoch=0`
  - candidate rejected and large weight files deleted

このスモークの目的は強い重みを作ることではなく、棋譜手force includeとtrainer側の目的関数が壊れずに一周することの確認である。結果として、必要なJSONL信号はすべて出力され、全ターゲットテストも通過した。

## 検証コマンド

```bash
cargo fmt --check
env RUST_FONTCONFIG_DLOPEN=1 cargo build --release --bin mmto_tree_dump --bin mmto_tree_train
env RUST_FONTCONFIG_DLOPEN=1 cargo test --all-targets
```

すべて成功。

## 次の実験計画

短期採用を狙うのではなく、本質的に強い重みへ進むため、次は以下の順で進める。

1. `MAX_RECORDS=2000`、depth `4/3` のBonanza-root実験を実行する。
2. `best_epoch > 0` かつ score/rerank gate 非悪化なら、20局ではなく複数seed合計100局で重み単体ベンチを行う。
3. 2Kで有望なら、Wdoor 2023-2026 高レートデータへ拡大し、20K root / 2K valid の24時間級実験へ進む。
4. game-teacher margin weight、max regret、listwise feature sourceを小さく振り、validだけでなくrerank gateと対局ベンチで判断する。

現時点ではリリース対象の重みはない。次の本命は、自己蒸留MMTO拡大ではなく、外部棋譜手を探索と整合させるBonanza型root選択学習である。

## Addendum: データ偏り対策

最初の小規模実験では、`dataset_build` がファイル名順に処理し、`--max-records` 到達時点で停止するため、100局面が2から3局程度の棋譜に偏る問題が見つかった。これは小規模実験ほど強く出るため、学習・valid指標の信頼性を落とす。

対策として `dataset_build` に以下を追加した。

- `--shuffle-games`: `--seed` に基づいてCSAファイル順をシャッフルする。
- `--max-records-per-game`: 1棋譜から採用する最大局面数を制限する。

`tools/run_bonanza_root_pipeline.sh` ではデフォルトで以下にした。

- `SHUFFLE_GAMES=1`
- `MAX_RECORDS_PER_GAME=8`

これにより、2026年Wdoor高レートデータの100局面スモークは以下になった。

- 変更前: 100局面が2から3局程度に偏る。
- 変更後: 100局面が13局から生成される。

## Addendum: 修正版pipeline smoke

実行条件:

```bash
MAX_RECORDS=100 MAX_RECORDS_PER_GAME=8 TREE_MAX_POSITIONS=100 \
RERANK_MAX_POSITIONS=40 JOBS=2 POSITION_CHUNK_SIZE=8 EPOCHS=1 BATCH_SIZE=32 \
TEACHER_DEPTH=3 STUDENT_DEPTH=2 TEACHER_SCORE_TOP=16 CANDIDATE_TOP=16 \
SCORE_ALL_LEGAL_FOR_VALID=0 \
RUN_DIR=data/mmto/runs/bonanza_root_pergame_smoke_20260627_000404 \
bash tools/run_bonanza_root_pipeline.sh
```

結果:

- dataset: 100局面、13局から生成
- dump: train 90 / valid 10
- rank stats:
  - selected regret mean: `8.19`
  - p90: `18.43`
  - p95: `24.05`
  - bad50: `0.0100`
- training:
  - baseline valid p95: `30.30`
  - epoch 1 valid p95: `29.55`
  - `best_epoch=1`
- score gate:
  - mean abs delta: `0.05cp`
  - p95: `0.12cp`
  - max: `0.20cp`
  - passed
- rerank gate:
  - baseline/candidateともに mean `6.91`, p90 `22.53`, p95 `28.19`
  - passed

この重みは100局面だけのスモークなので採用しない。`best.raw.binary` は削除済み。重要なのは、偏り対策後のBonanza-root pipelineがdataset生成、探索dump、game-teacher margin学習、score gate、rerank gateまで一通り完走した点である。

次は `MAX_RECORDS=500` 以上でも `MAX_RECORDS_PER_GAME=8` を維持し、valid件数を増やしたうえでオフライン指標を見る。2K以上の実験へ進む前に、game-teacher marginがvalid/rerankで安定して非悪化になる条件を探す。

## Addendum: 500局面実験

偏り対策後の500局面、depth `4/3` 実験を実行した。

実行条件:

```bash
MAX_RECORDS=500 MAX_RECORDS_PER_GAME=8 TREE_MAX_POSITIONS=500 \
RERANK_MAX_POSITIONS=200 JOBS=2 POSITION_CHUNK_SIZE=8 EPOCHS=3 BATCH_SIZE=64 \
TEACHER_DEPTH=4 STUDENT_DEPTH=3 TEACHER_SCORE_TOP=24 CANDIDATE_TOP=24 \
SCORE_ALL_LEGAL_FOR_VALID=0 \
RUN_DIR=data/mmto/runs/bonanza_root_pergame_500_d4d3_20260627_000542 \
bash tools/run_bonanza_root_pipeline.sh
```

データ:

- 500局面
- 63局から生成
- train 450 / valid 50
- skipped: 0

オフライン結果:

- rank stats selected regret:
  - mean `16.56`
  - p90 `57.98`
  - p95 `71.46`
  - bad50 `0.1180`
- baseline valid:
  - selected regret mean `10.43`
  - p90 `28.41`
  - p95 `54.51`
  - bad50 `0.0600`
- epoch 1:
  - selected regret mean `9.58`
  - p90 `28.41`
  - p95 `42.74`
  - bad50 `0.0400`
- score gate passed:
  - mean abs delta `0.16cp`
  - p95 `0.34cp`
  - max `0.41cp`
- rerank gate passed:
  - baseline/candidateともに mean `5.74`, p90 `11.55`, p95 `23.67`

20局ベンチ:

- seed `11201`: `10-10-0`
- paired starts: all 10 pairs split

この候補は重み差が小さすぎ、実戦への影響がほぼ見えなかったため不採用。

同じdumpから更新量を強めた追加実験も実施した。

### leaf gt0.10, max delta 0.005

Run:

`data/mmto/runs/bonanza_root_500_stronger_leaf_gt010_20260627_001350`

- score gate passed:
  - mean abs delta `0.62cp`
  - p95 `1.39cp`
  - max `1.73cp`
- rerank gate passed:
  - baseline mean `5.74`, match `36.00%`
  - candidate mean `5.72`, match `38.00%`
- 20局ベンチ:
  - seed `11301`: `10-9-1`
  - seed `11401`: `9-10-1`
  - combined 40 games: `19-19-2`

### move gt0.10, max delta 0.005

Run:

`data/mmto/runs/bonanza_root_500_stronger_move_gt010_20260627_001350`

- score gate passed:
  - mean abs delta `0.94cp`
  - p95 `1.73cp`
  - max `1.91cp`
- rerank gate passed:
  - baseline/candidateともに mean `5.74`, p95 `23.67`, match `36.00%`
- 対局ベンチには進めず。

結論:

Bonanza-root pipelineは正常に動作し、オフライン指標も改善する。しかし500局面規模では対局ベンチに移る強さはまだ出ていない。強め更新でも40局で50%に留まったため、500局面候補はすべて不採用とし、候補重みは削除した。

次はデータ規模を増やすだけでなく、以下を検討する。

- 2Kから5K局面に増やし、validを200局面以上にする。
- `teacher_depth=4/student_depth=3` を維持しつつ、`MAX_RECORDS_PER_GAME=8` で棋譜多様性を確保する。
- 対局に影響する更新量を出すため、`max_weight_delta=0.005` 前後を基本にする。
- `teacher-leaf` の方がrerank matchでわずかに良いため、次の本命は `listwise-feature-source=teacher-leaf` を優先する。

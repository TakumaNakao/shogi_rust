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

## Addendum: 2K局面実験

500局面では実戦影響が出なかったため、2K局面へ拡大した。

実行条件:

```bash
MAX_RECORDS=2000 MAX_RECORDS_PER_GAME=8 TREE_MAX_POSITIONS=2000 \
RERANK_MAX_POSITIONS=500 JOBS=2 POSITION_CHUNK_SIZE=8 EPOCHS=5 BATCH_SIZE=64 \
TEACHER_DEPTH=4 STUDENT_DEPTH=3 TEACHER_SCORE_TOP=24 CANDIDATE_TOP=24 \
SCORE_ALL_LEGAL_FOR_VALID=0 LEARNING_RATE=0.001 MAX_WEIGHT_DELTA=0.005 \
GAME_TEACHER_MARGIN_WEIGHT=0.10 LISTWISE_FEATURE_SOURCE=teacher-leaf \
LISTWISE_HARD_NEGATIVE_WEIGHT=0.02 \
RUN_DIR=data/mmto/runs/bonanza_root_pergame_2k_leaf_gt010_20260627_001929 \
bash tools/run_bonanza_root_pipeline.sh
```

データ:

- 2000局面
- 250局から生成
- train 1800 / valid 200
- skipped: 0
- unique score positions: 1596

オフライン学習:

- rank stats:
  - selected regret mean `15.51`
  - p90 `47.68`
  - p95 `70.68`
  - bad50 `0.0995`
- baseline valid:
  - selected regret mean `11.56`
  - p90 `27.20`
  - p95 `49.63`
  - bad50 `0.0500`
- epoch 5:
  - selected regret mean `10.35`
  - p90 `26.05`
  - p95 `33.34`
  - bad50 `0.0350`
  - `best_epoch=5`

Score gate:

- mean abs delta `1.08cp`
- p95 `2.41cp`
- max `3.26cp`
- passed

Rerank gate:

- baseline:
  - mean `5.73`
  - p90 `11.05`
  - p95 `27.09`
  - match `43.00%`
  - bad50 `0.0250`
- candidate:
  - mean `5.64`
  - p90 `11.05`
  - p95 `23.67`
  - match `42.50%`
  - bad50 `0.0250`
- failed reason: match rate `42.50% < 43.00%`

解釈:

2Kではvalid p95が大きく改善し、score gateも十分安全だった。一方で、rerank matchが200局面中1局面分だけ悪化してgate failedになった。mean/p95は改善しているため完全な失敗ではないが、過去の重み実験では小さなオフライン悪化が対局で大きく崩れることがあったため、この候補は対局に進めず不採用とした。候補重みは削除済み。

次の改善点:

- match rateを絶対条件にするか、mean/p95改善と悪化幅を合わせた複合gateにするか再検討する。
- `worst_delta[1]` は candidate が `B*4e -> 2b3a` に変わって regret +18.55cp。こうした単発悪化をhard replayへ戻す。
- 2K規模では学習信号は出ているため、次はhard replayを組み込んだ2K再学習か、rerank gateで生成されたhard positionsを追加validにする。

### hard replay追試

2K候補のrerank gateが出したhard positions 17件を `mmto_tree_dump` で再dumpし、extra-validとreplayに入れた。

Run:

`data/mmto/runs/bonanza_root_2k_leaf_gt010_hard17_20260627_004507`

設定:

- base train/valid: `bonanza_root_pergame_2k_leaf_gt010_20260627_001929`
- hard records: 17
- `--extra-valid hard=...`
- `--extra-valid-best-weight 0.5`
- `--replay-train hard.tree.jsonl`
- `--replay-weight 0.05`

結果:

- baseline hard:
  - selected regret mean `10.97`
  - p90 `26.18`
  - p95 `31.53`
- epoch 1 hardは一度悪化:
  - p95 `46.52`
- epoch 5 hardは回復:
  - selected regret mean `10.37`
  - p90 `26.11`
  - p95 `33.08`
- main valid epoch 5:
  - selected regret mean `10.36`
  - p95 `33.34`
- score gate:
  - mean abs delta `1.08cp`
  - p95 `2.40cp`
  - max `3.26cp`
  - passed
- rerank gate:
  - baseline match `43.00%`
  - candidate match `42.50%`
  - failed with the same match-rate regression

結論:

hard replayを追加しても、実探索rerank上の選択悪化は消えなかった。現在の追加lossは評価値分布とvalid p95を改善するが、探索が選ぶroot moveを十分に制御できていない。次は単純なhard replay増量ではなく、rerankで悪化した「candidate selected move」を明示的に下げる目的関数、またはgateで見ているdepthのsearched moveを直接学習対象にする必要がある。

### current-top margin loss追試

`selected_by_student` 固定のhard negativeでは、dump時のstudent手だけを罰してしまう。そこで、学習中モデルが候補集合内で現在topにしている手をbad側にする `current-top margin loss` を追加した。

実装:

- `mmto_tree_train`
  - `--current-top-margin-weight`
  - `--current-top-min-bad-regret-cp`
- pipeline scripts
  - `run_bonanza_root_pipeline.sh`
  - `run_mmto_from_dump.sh`
  - `run_mmto_rerank_pipeline.sh`

検証条件:

- base dump: `bonanza_root_pergame_2k_leaf_gt010_20260627_001929`
- train 1800 / valid 200
- `LOSS_MODE=listwise-leaf`
- `LISTWISE_FEATURE_SOURCE=teacher-leaf`
- `LISTWISE_HARD_NEGATIVE_WEIGHT=0`
- `GAME_TEACHER_MARGIN_WEIGHT=0.05`
- `CURRENT_TOP_MARGIN_WEIGHT=0.05`
- `MAX_WEIGHT_DELTA=0.005`

Run:

`data/mmto/runs/mmto_current_top_2k_20260627_033620`

結果:

- baseline valid:
  - selected regret mean `11.56`
  - p90 `27.20`
  - p95 `49.63`
- best epoch 2:
  - selected regret mean `10.60`
  - p90 `26.05`
  - p95 `34.43`
- score gate:
  - mean abs delta `0.70cp`
  - p95 `1.98cp`
  - max `2.89cp`
  - passed
- rerank gate:
  - baseline mean `5.73`, p90 `11.05`, p95 `27.09`, match `43.00%`
  - candidate mean `5.72`, p90 `11.20`, p95 `23.67`, match `44.00%`
  - failed: p90が `11.05 -> 11.20` に微悪化

`CURRENT_TOP_MARGIN_WEIGHT=0.10` も追試した。

Run:

`data/mmto/runs/mmto_current_top010_2k_20260627_033935`

結果:

- score gate passed
- rerank:
  - mean `5.73 -> 5.73`
  - p90 `11.05 -> 11.20`
  - p95 `27.09 -> 23.67`
  - match `43.00% -> 43.00%`
  - failed: p90微悪化

解釈:

current-top lossは、matchやp95を改善する兆候がある。一方で、最悪悪化手 `2b3a` は消えなかった。重みを0.10へ上げても改善せず、単純にこのlossを強くするだけでは実探索の選択悪化を十分に制御できない。

### DAgger replay stage追試

失敗候補重みをstudentにしてhard positionsを再dumpする `tools/run_mmto_dagger_from_run.sh` を追加した。目的は、baseline studentではなく、失敗候補が実際に選ぶ手を `selected_by_student` としてreplay学習へ戻すこと。

手順:

1. `KEEP_CANDIDATE_RAW=1` でcurrent-top候補を保持。
2. 失敗runの `hard_positions.sfen` を、失敗候補 `best.raw.binary` をstudentとして再dump。
3. 生成したDAgger dumpを `--replay-train` と `--extra-valid` に入れて再学習。
4. score gate / rerank gateを再確認。

保持run:

`data/mmto/runs/mmto_current_top_2k_keep_20260627_034440`

DAgger run:

`data/mmto/runs/mmto_dagger_current_top_2k_20260627_034724`

DAgger dump:

- hard positions: 16
- train records: 16
- selected regret mean `40.08`
- p90 `156.57`
- p95 `156.57`

再学習結果:

- best epoch 1
- score gate passed:
  - mean abs delta `0.95cp`
  - p95 `1.96cp`
  - max `2.46cp`
- rerank gate:
  - baseline mean `5.73`, p90 `11.05`, p95 `27.09`, match `43.00%`
  - candidate mean `6.51`, p90 `11.20`, p95 `27.09`, match `43.00%`
  - bad50 `0.0250 -> 0.0300`
  - bad100 `0.0150 -> 0.0200`
  - failed

新しい大悪化:

- `B*8d -> B*1d`
- delta `162.77`

解釈:

単純なDAgger replayは、候補由来の悪手を再収集する仕組みとしては動いた。しかし、hard 16局面だけをreplayに混ぜる現在の形では、対象hardを安定して改善できず、新しい大悪化を生んだ。したがって、長時間学習へ進む前に、以下のどちらかが必要。

- rerank gateで実際に出た `candidate_move` を明示したペア教師データを作り、候補集合内の探索選択手を直接下げる。
- replayを局面単位の小さな追加データではなく、候補重みで再探索した大きなrefresh dumpとして作り直し、通常train/validと同じ規模で評価する。

現在の結論:

「もっとエポックを回す」だけでは不十分。current-top lossでオフライン指標は改善するが、探索選択手の悪化を完全には防げない。次の実装対象は、rerank gateのhard outputから `(teacher_move, candidate_move)` を直接学習する明示ペア損失、またはcandidate refresh dumpを本体データとして扱う反復学習パイプラインである。

### explicit hard pair入力の実装

rerank gateのhard outputをそのまま明示ペア教師にできるようにした。

実装:

- `mmto_tree_dump`
  - JSONL入力で以下の別名を受け付ける。
    - teacher側: `teacher_move`, `teacher_best_move`
    - bad側: `student_move`, `selected_move`, `candidate_move`, `bad_move`
  - bad側の手が合法なら、通常のstudent探索結果ではなく、その手を `selected_by_student=true` として出力する。
  - teacher側の手は従来通り `is_game_teacher_move=true` として候補集合に強制追加する。
- `tools/extract_rerank_hard_pairs.py`
  - `mmto_rerank_gate` の `hard_positions` から、`sfen`, `teacher_move`, `student_move` のJSONLを生成する。
  - `--min-regret-delta-cp`
  - `--min-candidate-regret-cp`
  - `--limit`
- `tools/run_mmto_dagger_from_run.sh`
  - 既定で `SOURCE_RUN_DIR/rerank_gate.json` から明示ペアJSONLを作ってDAgger dumpに使う。
  - `USE_EXPLICIT_HARD_PAIRS=0` で従来の `hard_positions.sfen` 入力に戻せる。

検証:

```bash
tools/extract_rerank_hard_pairs.py \
  --input data/mmto/runs/mmto_current_top_2k_20260627_033620/rerank_gate.json \
  --output /tmp/mmto_explicit_pairs.jsonl \
  --limit 3

env RUST_FONTCONFIG_DLOPEN=1 target/release/mmto_tree_dump \
  --student-weights policy_weights_v2.1.0.binary \
  --teacher-weights policy_weights_v2.1.0.binary \
  --input /tmp/mmto_explicit_pairs.jsonl \
  --train-output /tmp/mmto_explicit_pairs.train.tree.jsonl \
  --valid-output /tmp/mmto_explicit_pairs.valid.tree.jsonl \
  --teacher-depth 4 \
  --student-depth 3 \
  --teacher-score-top 24 \
  --candidate-top 24 \
  --valid-percent 0 \
  --position-chunk-size 3 \
  --jobs 1 \
  --min-legal-moves 2 \
  --exclude-in-check \
  --max-positions 3
```

確認結果:

- `2b3c / 2b3a`
- `2a3c / 2b3a`
- `2b4d / 2b5e`

いずれも、teacher側が `is_game_teacher_move=true`、bad側が `selected_by_student=true` としてdumpされた。

次の検証:

明示ペア入力を使ったDAgger stageを2K条件で再実行する。これで `2b3a` 系の悪化手が消え、rerank p90/meanが非悪化になるかを確認する。通らなければ、hardだけの小replayではなく、候補重みによるrefresh dumpを数百から数千局面単位で作る。

### explicit hard pair DAgger追試

上記の明示ペア入力を使い、2K条件のbase train/validへhard 16件をreplayした。

Run:

`data/mmto/runs/mmto_explicit_hard_2k_20260627_035700`

設定:

- source: `data/mmto/runs/mmto_current_top_2k_20260627_033620`
- `CANDIDATE_WEIGHTS=policy_weights_v2.1.0.binary`
  - bad手は `student_move` で明示するため、dumpのstudent重みはbaselineで代用。
- hard pairs: 16
- `LISTWISE_HARD_NEGATIVE_WEIGHT=0.10`
- `CURRENT_TOP_MARGIN_WEIGHT=0.05`
- `GAME_TEACHER_MARGIN_WEIGHT=0.05`
- `REPLAY_WEIGHT=0.20`
- `EXTRA_VALID_BEST_WEIGHT=0.50`
- `MAX_WEIGHT_DELTA=0.005`

DAgger dump:

- selected regret mean `40.08`
- p90 `156.57`
- p95 `156.57`
- bad50 `0.2500`
- bad100 `0.1250`

再学習:

- best epoch 1
- main valid:
  - baseline p95 `49.63`
  - epoch 1 p95 `34.43`
- hard extra-valid:
  - baseline selected regret mean `14.90`, p90/p95 `31.65`
  - epoch 1 selected regret mean `18.40`, p90/p95 `46.52`
- score gate:
  - mean abs delta `1.16cp`
  - p95 `2.17cp`
  - max `2.55cp`
  - passed

Rerank gate:

- baseline:
  - mean `5.73`
  - p90 `11.05`
  - p95 `27.09`
  - match `43.00%`
  - bad50 `0.0250`
  - bad100 `0.0150`
- candidate:
  - mean `6.43`
  - p90 `11.05`
  - p95 `27.09`
  - match `42.50%`
  - bad50 `0.0300`
  - bad100 `0.0200`
- failed

観察:

- `2b3a` 系の最大悪化は弱まった。
  - 例: `2b3c` 局面は `candidate_move=2a1c`, delta `0.91`
- しかし別の大悪化が発生した。
  - `B*8d -> B*1d`
  - delta `162.77`

結論:

明示ペア入力は、狙った悪手を学習対象に入れる仕組みとしては正しく動く。ただし、hard 16件を小replayとして足すだけでは、重み更新の副作用を抑えられない。次は以下を優先する。

1. candidate refresh dumpをhard局面だけでなく、valid/trainから数百から数千局面規模で生成する。
2. refresh dumpをreplayではなく通常trainの一部として扱い、validもrefresh側から十分な数を取る。
3. 明示bad pairだけでなく、新しく出る大悪化を反復的にrefreshへ戻す。
4. 小replay重みを強くする方向は一旦避ける。今回の結果では局所対策が別局面の大悪化を生んだ。

### candidate refresh pipeline実装と追試

hard 16件だけのreplayでは副作用を抑えられなかったため、候補重みで通常train側を再探索し、数百局面規模のrefresh dumpを作るスクリプトを追加した。

実装:

- `tools/run_mmto_refresh_from_candidate.sh`
  - `CANDIDATE_WEIGHTS` をstudentとして `mmto_tree_dump` を実行。
  - `REFRESH_MAX_POSITIONS` / `REFRESH_VALID_PERCENT` でrefresh train/validを生成。
  - `TRAIN_MODE=mixed` ではbase train/validへrefresh train/validを連結して通常trainとして扱う。
  - `TRAIN_MODE=refresh-only` ではrefresh dumpだけで学習できる。
  - 長時間学習向けに `STREAM_TRAIN=1` を渡せる。
  - score gate / rerank gate / hard positions抽出 / 失敗時重み削除まで実行する。

Smoke:

`data/mmto/runs/mmto_refresh_smoke_20260627_040344`

設定:

- `CANDIDATE_WEIGHTS=policy_weights_v2.1.0.binary`
- `BASE_TRAIN_LINES=100`
- `BASE_VALID_LINES=20`
- `REFRESH_MAX_POSITIONS=40`
- `REFRESH_VALID_PERCENT=10`
- `TRAIN_MODE=mixed`
- `EPOCHS=1`

結果:

- refresh dump:
  - train 36 / valid 4
  - selected regret mean `15.42`
  - p90 `54.85`
  - p95 `68.51`
- mixed train/valid:
  - train 136 / valid 24
- best epoch `0`
- baselineが最良のため不採用
- 重み削除済み

このsmokeで、refresh dump作成、mixed train構築、学習、不採用時の重み削除まで動作確認した。

300局面refresh追試:

まずcurrent-top候補を再作成した。

`data/mmto/runs/mmto_current_top_for_refresh_20260627_040457`

結果は前回と同じ傾向:

- score gate passed
- rerank gate failed
- p95 `27.09 -> 23.67`
- match `43.00% -> 44.00%`
- p90 `11.05 -> 11.20` で微悪化
- worst: `B*4e -> 2b3a`, delta `18.55`

この候補をstudentとして300局面refreshを実行した。

`data/mmto/runs/mmto_refresh_candidate_20260627_040733`

設定:

- `CANDIDATE_WEIGHTS=data/mmto/runs/mmto_current_top_for_refresh_20260627_040457/best.raw.binary`
- `BASE_TRAIN_LINES=1800`
- `BASE_VALID_LINES=200`
- `REFRESH_MAX_POSITIONS=300`
- `REFRESH_VALID_PERCENT=10`
- `TRAIN_MODE=mixed`
- `EPOCHS=5`
- `MAX_WEIGHT_DELTA=0.005`

refresh dump:

- train 270 / valid 30
- selected regret mean `17.24`
- p90 `55.17`
- p95 `70.68`
- bad50 `0.1200`
- bad100 `0.0267`

mixed train:

- train 2070
- valid 230

再学習:

- best epoch 3
- mixed valid:
  - baseline p95 `42.74`
  - epoch 3 p95 `33.34`
- refresh extra-valid:
  - baseline selected regret mean `5.25`
  - epoch 3 selected regret mean `6.50`
  - p95 `27.28` 維持
- score gate:
  - mean abs delta `1.47cp`
  - p95 `2.57cp`
  - max `3.05cp`
  - passed

Rerank gate:

- baseline:
  - mean `5.73`
  - p90 `11.05`
  - p95 `27.09`
  - match `43.00%`
  - bad50 `0.0250`
  - bad100 `0.0150`
- candidate:
  - mean `6.38`
  - p90 `11.05`
  - p95 `27.09`
  - match `43.50%`
  - bad50 `0.0300`
  - bad100 `0.0200`
- failed

観察:

- `2b3a` 系の悪化はさらに弱まった。
  - `2b3c` 局面は `candidate_move=2a1c`, delta `0.91`
- しかし `B*8d -> B*1d` の大悪化が残った。
  - delta `162.77`
- matchは `43.00% -> 43.50%` と少し改善したが、meanとbad ratioが悪化した。

結論:

candidate refresh dumpはhard 16件replayより分布が広く、狙った悪化手を弱める効果はある。ただし、現状の目的関数では別の大悪化を抑えられず、長時間学習へ進むにはまだ危険。次は以下のどちらかが必要。

1. best metric / gateに最大悪化・bad100を強く入れ、`B*1d` のような少数大悪化を学習中に止める。
2. refreshを1回で終わらせず、rerank gateで出た新しい大悪化を次のrefreshへ戻す反復ループにする。

現時点では、単純な長時間学習はまだ回さない。長時間化の前に「大悪化を発生させない制約」を学習器に追加する。

### max-regret系best metricの追加

candidate refreshでp95やmatchが改善しても、少数の大悪化が残る問題が続いた。そのため、`mmto_tree_train` のbest checkpoint選択に大悪化を直接見るmetricを追加した。

追加metric:

- `p99-regret`
- `bad100-regret`
- `bad200-regret`
- `max-regret`

また、学習ログのsummaryに `max` を追加した。

目的:

- p95改善だけで `best_epoch` を選ばない。
- `B*1d` のような少数大悪化を含む候補を、対局前に落とす。
- 長時間学習時に、破壊的な重み更新を早期に検知する。

Probe:

`data/mmto/runs/mmto_max_metric_probe_20260627_041615`

入力:

- train/valid: `data/mmto/runs/mmto_refresh_candidate_20260627_040733`
- `--best-metric max-regret`
- 他条件はrefresh追試と同等

結果:

- baseline:
  - valid max `317.51`
  - refresh extra-valid max `32.06`
  - best metric score `325.53`
- epoch 1:
  - valid p95は改善 `42.74 -> 33.37`
  - しかしbest metric scoreは `325.86` に悪化
- epoch 5までbaselineを上回れず
- `best_epoch=0`

解釈:

`max-regret` は、今回のようにp95が改善しても最大悪化を改善できない候補を採用しない方向に働いた。これは長時間学習用の安全装置として有効。次の長めのrefresh実験では、まず `BEST_METRIC=max-regret` または `BEST_METRIC=bad100-regret` を使い、採用候補が出るかを確認する。

### refresh loop smoke

長時間学習では、1回のrefreshだけでなく「候補がgateを通った時だけ次のrefreshへ進む」反復制御が必要になる。そのため、`tools/run_mmto_refresh_loop.sh` を追加した。

実装:

- `INITIAL_CANDIDATE_WEIGHTS` を起点にする。
- 各iterationで `tools/run_mmto_refresh_from_candidate.sh` を実行する。
- 当初の既定 `BEST_METRIC=max-regret` は安全だが保守的すぎるため、best checkpoint guard追加後は `BEST_METRIC=p95-regret` とし、`BEST_GUARD_MAX_REGRET_INCREASE_CP=0` / `BEST_GUARD_BAD100_INCREASE=0` で尾部悪化を禁止する設定に変更した。
- offline gatesを通った場合だけ、`iter_N/best.raw.binary` を次iterationの候補にする。
- gate失敗、score失敗、best_epoch 0では停止する。
- `candidate.raw.binary` は各iterationで削除する。
- `KEEP_PASSED_WEIGHTS=0` の場合、過去iterationの採用済み重みも次候補以外は削除する。

Smoke:

`data/mmto/runs/mmto_refresh_loop_smoke_20260627_042019`

設定:

- `INITIAL_CANDIDATE_WEIGHTS=policy_weights_v2.1.0.binary`
- `ITERATIONS=1`
- `BEST_METRIC=max-regret`
- `BASE_TRAIN_LINES=100`
- `BASE_VALID_LINES=20`
- `REFRESH_MAX_POSITIONS=40`
- `EPOCHS=1`

結果:

- refresh dump:
  - train 36 / valid 4
  - selected regret mean `15.42`
  - p90 `54.85`
  - p95 `68.51`
  - max `165.62`
- mixed train/valid:
  - train 136 / valid 24
- baseline best metric score `35.23`
- epoch 1 best metric score `36.72`
- `best_epoch=0`
- loop stopped with `FINAL_CANDIDATE=policy_weights_v2.1.0.binary`
- `.binary` 生成物は削除済み

解釈:

反復制御は想定通り動作した。改善候補がない場合は次iterationへ進まず、長時間実行でも不採用重みを残さない。次の実験では、このloopにより `BEST_METRIC=max-regret` または `bad100-regret` で候補が出るまで、小さめのrefresh条件を広げていく。

### best checkpoint guard

日時: 2026-06-27 04:29 UTC

背景:

`BEST_METRIC=max-regret` は安全だが、p95やlossの改善を拾いにくい。一方で `BEST_METRIC=p95-regret` のままでは、p95は改善しても最大悪化やbad100率が悪化するepochをbest checkpointとして採用してしまう。そこで、主指標とは別に尾部リスクのガードを追加した。

実装:

- `mmto_tree_train` に以下を追加:
  - `--best-guard-max-regret-increase-cp`
  - `--best-guard-bad100-increase`
- 負値なら無効。
- 0を指定すると、baselineより `max-regret` または `bad100-regret` が少しでも悪化したepochをbest候補から除外する。
- extra-validを使う場合、主指標と同じく `extra-valid-best-weight` を掛けた合成スコアで判定する。
- 主要MMTOスクリプトから以下の環境変数で指定可能にした:
  - `BEST_GUARD_MAX_REGRET_INCREASE_CP`
  - `BEST_GUARD_BAD100_INCREASE`

Probe:

`data/mmto/runs/mmto_guard_probe_20260627_042736`

設定:

- train/valid: `data/mmto/runs/mmto_refresh_candidate_20260627_040733`
- `--best-metric p95-regret`
- `--best-guard-max-regret-increase-cp 0`
- `--best-guard-bad100-increase 0`
- `--extra-valid refresh=.../refresh.valid.tree.jsonl`
- `--extra-valid-best-weight 0.25`

結果:

- baseline:
  - best metric score `49.562458`
  - guard max-regret score `325.526733`
  - guard bad100 score `0.013043`
- epoch 1:
  - p95 best metric scoreは `40.642563` へ改善
  - しかしguard max-regret scoreが `325.855927` へ悪化
  - `best_guard_passed=false`
- epoch 1から5まで全てguard不合格
- `best_epoch=0`
- `.binary` 生成物は削除済み

解釈:

狙い通り、p95改善だけでは候補を採用せず、少数の大悪化を検出してbaselineに戻せた。次の長時間学習では、主指標は `p95-regret` や `capped-selected-regret` のままにしつつ、まず以下の厳しめのガードを使う。

```bash
BEST_GUARD_MAX_REGRET_INCREASE_CP=0
BEST_GUARD_BAD100_INCREASE=0
```

これで候補が全く出ない場合だけ、`BEST_GUARD_MAX_REGRET_INCREASE_CP=10` のように許容幅を小さく緩める。長時間学習は、このガードを通る候補が小規模で出ることを確認してから拡大する。

### refresh loop guard smoke

Probe:

`data/mmto/runs/mmto_refresh_loop_guard_smoke_20260627_043055`

設定:

- `BEST_METRIC=p95-regret`
- `BEST_GUARD_MAX_REGRET_INCREASE_CP=0`
- `BEST_GUARD_BAD100_INCREASE=0`
- `BASE_TRAIN_LINES=100`
- `BASE_VALID_LINES=20`
- `REFRESH_MAX_POSITIONS=40`
- `EPOCHS=1`

結果:

- baseline best metric score `34.101116`
- baseline guard max-regret score `35.231255`
- epoch 1 best metric score `35.257175`
- epoch 1 guard max-regret score `36.720306`
- `best_guard_passed=false`
- `best_epoch=0`
- loopは `FINAL_CANDIDATE=policy_weights_v2.1.0.binary` で停止
- `.binary` 生成物は削除済み

解釈:

長時間用の入口スクリプトでも、主指標だけで危険な候補へ進まず、baselineを維持して停止できた。これで「小さく試し、危険な候補を通さず、通った候補だけ次refreshへ進める」反復学習の安全性が上がった。

### low LR guarded candidate probe

Probe:

`data/mmto/runs/mmto_guard_lr_probe_20260627_043315`

目的:

高めの学習率ではp95が改善してもmax-regret guardに落ちたため、学習率と更新幅を絞れば、尾部を悪化させずに候補が出るか確認した。

設定:

- train/valid: `data/mmto/runs/mmto_refresh_candidate_20260627_040733`
- `--learning-rate 0.0002`
- `--max-weight-delta 0.001`
- `--anchor-l2 0.0005`
- `--best-metric p95-regret`
- `--best-guard-max-regret-increase-cp 0`
- `--best-guard-bad100-increase 0`
- `--extra-valid-best-weight 0.25`

trainer結果:

- baseline best metric score `49.562458`
- epoch 1 best metric score `38.875984`
- epoch 2 best metric score `38.503777`
- max-regret guard scoreはbaseline同値 `325.526733`
- bad100 guard scoreはbaseline同値 `0.013043`
- `best_epoch=2`

offline gates:

- score gate:
  - mean abs delta `0.27cp`
  - p95 `0.49cp`
  - max `0.58cp`
  - passed
- rerank gate:
  - baseline mean `6.01`, candidate mean `6.03`
  - baseline p95 `29.12`, candidate p95 `29.04`
  - match rate `16.00%` -> `16.00%`
  - failed: mean regret worsened by `0.02cp`

解釈:

小さい更新幅なら、trainer上ではp95を大きく改善しつつtail guardを通る候補を作れた。ただし実探索rerankでは平均regretがわずかに悪化し、採用には届かなかった。これは「長時間化すればよい」というより、rerankで出た悪化手を次の学習信号に戻す仕組みが必要であることを示している。候補の `.binary` は不採用として削除済み。

### weighted rerank-hard feedback

背景:

GPT-5.5 xhighの分析では、現在の主ボトルネックは「root候補集合上のproxy指標は改善できるが、実探索rerankで出る少数の悪化を安定して抑えられないこと」と判断した。単純にepoch数やデータ量を増やすのではなく、rerankで実際に悪化した `candidate_move` を次の学習で強く下げる必要がある。

既存問題:

- `tools/extract_rerank_hard_pairs.py` は `teacher_move` / `student_move` をJSONL化できる。
- しかし `regret_delta` や `candidate_regret` の大きさを `sample_weight` に反映していなかった。
- さらに `mmto_tree_dump` がJSONL入力の `sample_weight` を読んでいなかったため、仮に抽出側で重みを付けてもdump時に失われる状態だった。

実装:

- `mmto_tree_dump` のJSONL入力で `sample_weight` を受け取り、出力tree JSONLへ保持するようにした。
- `tools/extract_rerank_hard_pairs.py` に重み付けを追加:
  - `--weight-mode none|regret-delta|candidate-regret|combined`
  - `--weight-scale-cp`
  - `--max-sample-weight`
- `tools/run_mmto_dagger_from_run.sh` から以下の環境変数で制御可能にした:
  - `EXPLICIT_WEIGHT_MODE`
  - `EXPLICIT_WEIGHT_SCALE_CP`
  - `EXPLICIT_MAX_SAMPLE_WEIGHT`

Smoke:

`data/mmto/runs/mmto_hard_weight_smoke_20260627_043950`

入力:

- `data/mmto/runs/mmto_guard_lr_probe_20260627_043315/rerank_gate.json`
- `--weight-mode combined`
- `--weight-scale-cp 10`
- `--max-sample-weight 5`

結果:

- hard pair抽出: `written=9 skipped=0`
- 例:
  - `candidate_regret=5.711242`
  - `regret_delta=3.5024726`
  - `sample_weight=1.5711242`
- `mmto_tree_dump` 後のtree JSONLでもsample weightsが保持された:
  - `[1.5711242, 1.2766001, 1.2031225, 1.3831849, 1.2997372]`

解釈:

rerankで実際に悪化した手を、悪化度に応じて強く学習へ戻す経路ができた。次の実験では、低LR・小更新幅・tail guardに加えて、このweighted rerank-hardを通常refreshデータへ混ぜる。これにより、単純な長時間学習ではなく「実探索で失敗した手を反復的に回収する」方向へ進める。

追加smoke:

`data/mmto/runs/mmto_weighted_dagger_smoke_20260627_044127`

目的:

不採用候補の `.binary` を削除した後でも、rerank JSONからweighted hard pairを再利用できるか確認した。

変更:

- `tools/run_mmto_dagger_from_run.sh` は、`USE_EXPLICIT_HARD_PAIRS=1` かつ `CANDIDATE_WEIGHTS` が存在しない場合、dump scoring用に `WEIGHTS` へfallbackする。
- explicit hard pairでは `student_move` がJSONで強制されるため、候補重みを保持しなくてもhard pair dumpを再生成できる。

結果:

- 削除済み候補重みなしで起動できた。
- hard pair抽出: `written=9 skipped=0`
- dagger dump: train `5`, valid `0`
- trainerはtail guardで `best_epoch=0` となり安全停止。
- `.binary` 生成物は削除済み。

解釈:

これで、不採用候補の849MB重みを残さずに、rerank失敗情報だけを次の学習へ渡せる。ディスク制約下で反復的なhard feedbackを回すための前提が整った。

### refresh loop hard feedback automation

背景:

weighted rerank-hardの部品はできたが、手作業でDAggerを呼ぶだけでは長時間学習に向かない。refresh候補がrerank gateで落ちたとき、その `rerank_gate.json` を自動でweighted hard feedbackへ渡し、通った場合だけ次iterationへ進む制御が必要。

実装:

- `tools/run_mmto_refresh_loop.sh` を拡張。
- accepted candidateを次iterationの `WEIGHTS` として渡すようにした。
  - これにより、候補生成だけでなく重み更新も前回候補から継続する。
- `TEACHER_WEIGHTS` は既定で初期 `WEIGHTS` に固定。
- refresh stageがrerank失敗し、`rerank_gate.json` がある場合:
  - `tools/run_mmto_dagger_from_run.sh` を自動実行。
  - `BASE_TRAIN` / `BASE_VALID` は失敗refresh runのtree JSONLを使う。
  - `SCORE_POSITIONS` も失敗refresh runのものを使う。
  - hard feedbackがoffline gatesを通った場合だけ、その `best.raw.binary` を次candidateにする。
- `HARD_FEEDBACK_ON_FAILURE=0` で無効化可能。

Smoke:

`data/mmto/runs/mmto_refresh_loop_feedback_smoke_20260627_044540`

設定:

- `ITERATIONS=1`
- `REFRESH_MAX_POSITIONS=30`
- `EPOCHS=1`
- `LEARNING_RATE=0.0002`
- `MAX_WEIGHT_DELTA=0.001`
- `RERANK_REQUIRE_MEAN_REGRET_IMPROVEMENT_CP=999`
  - rerank失敗経路へ確実に入れるため、意図的に極端な要求にした。
- `DAGGER_MAX_POSITIONS=5`
- `EXPLICIT_WEIGHT_SCALE_CP=10`

結果:

- refresh stage:
  - trainer `best_epoch=1`
  - score gate passed
  - rerank gate failed
  - `candidate.raw.binary` / `best.raw.binary` は削除
- loop:
  - `iteration=1 starting weighted hard feedback...`
- hard feedback stage:
  - `written=1 skipped=0`
  - rerank-hard pair example:
    - `student_move=B*1d`
    - `candidate_regret=163.30379`
    - `sample_weight=5.0`
  - trainerは `best_epoch=0`
  - hard feedback offline gatesは通らず、安全停止
- final candidate:
  - `policy_weights_v2.1.0.binary`

解釈:

refresh失敗からweighted hard feedbackへ入る自動制御は動作した。今回は強制的なrerank失敗条件なので採用候補は出していないが、長時間学習では「refreshで候補生成 -> rerankで失敗 -> 失敗手をweighted hard pairとして回収 -> 通った候補だけ継続」というループが可能になった。smoke生成物は大容量化を避けるため削除する。

### refresh loop accept smoke

Probe:

`data/mmto/runs/mmto_refresh_loop_accept_smoke_20260627_044825`

目的:

強制失敗ではなく、通常のrerank gateでrefresh loopが候補を受理できるか確認した。小規模条件なので、重みの採用判断ではなく制御経路の確認である。

設定:

- `ITERATIONS=1`
- `REFRESH_MAX_POSITIONS=30`
- `EPOCHS=1`
- `LEARNING_RATE=0.0002`
- `MAX_WEIGHT_DELTA=0.001`
- `BEST_GUARD_MAX_REGRET_INCREASE_CP=1000`
- `BEST_GUARD_BAD100_INCREASE=1`
- `RERANK_MAX_POSITIONS=50`

結果:

- trainer:
  - baseline best metric score `34.000725`
  - epoch 1 best metric score `33.402618`
  - guard max-regret scoreはbaseline同値 `318.859650`
  - `best_epoch=1`
- score gate:
  - mean abs delta `0.23cp`
  - p95 `0.45cp`
  - max `0.54cp`
  - passed
- rerank gate:
  - baseline mean `5.10`, candidate mean `5.08`
  - baseline match `30.00%`, candidate match `32.00%`
  - p95は同値 `11.55`
  - passed
- loop:
  - `iteration=1 refresh accepted candidate=.../best.raw.binary`

解釈:

小規模ではあるが、低LR・小更新幅・tail guard付きrefreshで、offline gatesを通る候補を作り、loopが次candidateとして受け入れる経路を確認できた。これは「長時間学習の価値がある条件」に近づく重要な兆候。ただし検証局面が少ないため、この重みは採用せず削除する。次は同じ安全設定で `RERANK_MAX_POSITIONS` と `REFRESH_MAX_POSITIONS` を段階的に増やす。

### guarded100 refresh / hard-feedback trial

Probe:

`data/mmto/runs/mmto_refresh_loop_guarded100_20260627_045300`

目的:

smokeより一段大きい `REFRESH_MAX_POSITIONS=100` / `RERANK_MAX_POSITIONS=200` で、厳しめのtail guardを維持したまま候補が残るか確認した。

設定:

- `ITERATIONS=1`
- `REFRESH_MAX_POSITIONS=100`
- `EPOCHS=2`
- `LEARNING_RATE=0.0002`
- `MAX_WEIGHT_DELTA=0.001`
- `BEST_GUARD_MAX_REGRET_INCREASE_CP=0`
- `BEST_GUARD_BAD100_INCREASE=0`
- `RERANK_MAX_POSITIONS=200`
- hard feedback:
  - `DAGGER_MAX_POSITIONS=20`
  - `EXPLICIT_WEIGHT_SCALE_CP=50`
  - `EXPLICIT_MAX_SAMPLE_WEIGHT=5`

refresh stage:

- trainer:
  - baseline best metric score `39.474091`
  - epoch 1 best metric score `38.353870`
  - max-regret / bad100 guardはbaseline同値で通過
  - `best_epoch=1`
- score gate:
  - mean abs delta `0.25cp`
  - p95 `0.47cp`
  - max `0.55cp`
  - passed
- rerank gate:
  - baseline mean `6.26`
  - candidate mean `6.26`
  - baseline/candidate match `42.50%`
  - p95同値 `23.67`
  - failed because mean was marginally worse under exact comparison
- loop:
  - weighted hard feedbackへ自動遷移

hard feedback stage:

- hard pairs:
  - `written=7 skipped=0`
  - selected regret mean `94.60`
  - p95 `179.84`
  - bad50 `71.43%`
  - bad100 `42.86%`
- trainer:
  - baseline best metric score `45.060760`
  - epoch 1 best metric score `44.468342`
  - `best_epoch=1`
- score gate:
  - mean abs delta `0.13cp`
  - p95 `0.22cp`
  - max `0.26cp`
  - passed
- hard-feedback内rerank:
  - baseline/candidate mean `3.97`
  - baseline/candidate match `35.00%`
  - p95同値 `14.30`
  - passed
- loop:
  - `hard_feedback accepted candidate=.../best.raw.binary`

追加検証:

hard feedback候補を、より広い `base.valid.tree.jsonl` 200局面で再rerankした。

- baseline mean `6.01`
- candidate mean `6.02`
- p95 `29.12 -> 29.04`
- match `16.00% -> 16.00%`
- failed: meanが `0.01cp` 程度悪化

また、hard feedbackをfull valid 240件で再実行した。

- trainer上はp95改善とtail guard通過
- 200局面rerankでは:
  - baseline mean `6.39`
  - candidate mean `6.39`
  - p95同値 `27.09`
  - match `43.50% -> 43.00%`
  - failed: match低下

解釈:

100/200規模でもweighted hard feedbackは候補を作れるが、より広いrerankでは「regretはほぼ悪化しないがmatchが少し下がる」候補が残る。次のボトルネックは、trainerのbest選択がrerank match低下を事前に検出できないこと。

### teacher-match best guard

背景:

guarded100の追加検証では、regret指標がほぼ同等でもrerank matchが下がる候補が出た。これを長時間学習で増幅しないため、root候補上のteacher top一致率をtrainer側で計測し、best checkpoint選択のguardに加える。

実装:

- `mmto_tree_train` のmetricsに `teacher_match_count` を追加。
- ログ/CSVに `teacher_match` を出す。
- best metricとして `teacher-mismatch` を追加。
- best guardとして以下を追加:
  - `--best-guard-teacher-match-drop-pct`
- 主要スクリプトから以下の環境変数で渡せるようにした:
  - `BEST_GUARD_TEACHER_MATCH_DROP_PCT`
- `tools/run_mmto_refresh_loop.sh` では既定値を `0` とし、teacher-match低下を禁止する。

Probe:

`data/mmto/runs/mmto_teacher_match_guard_probe_20260627_050447`

入力:

- guarded100失敗runのtrain/valid
- full valid hard feedbackで使ったdagger trainをextra-valid/replayに指定
- `--best-guard-teacher-match-drop-pct 0`

結果:

- baseline:
  - valid teacher_match `17.92%`
  - extra-valid dagger teacher_match `28.57%`
  - guard teacher_match score `0.250595`
- epoch 1:
  - valid teacher_match `19.17%`
  - extra-valid dagger teacher_match `14.29%`
  - guard teacher_match score `0.227381`
  - `best_guard_passed=false`
- epoch 2:
  - valid teacher_match `20.00%`
  - extra-valid dagger teacher_match `14.29%`
  - guard teacher_match score `0.235714`
  - `best_guard_passed=false`
- `best_epoch=0`

解釈:

teacher-match guardは、今回の「広いrerankではmatchが落ちる候補」をtrainer段階で落とせた。これは長時間学習の安全装置として有効。ただしroot候補上のteacher matchは実探索rerank matchのproxyであり、完全な代替ではない。次はこのguardを有効にしたまま、100/200規模を再実行し、候補が残るか確認する。

### match-guarded100 rerun

Probe:

`data/mmto/runs/mmto_refresh_loop_matchguard100_20260627_050723`

目的:

teacher-match guardを有効にした状態で、guarded100と同じ規模を再実行し、以前のmatch低下候補をtrainer段階で落とせるか確認した。

設定:

- `REFRESH_MAX_POSITIONS=100`
- `RERANK_MAX_POSITIONS=200`
- `EPOCHS=2`
- `LEARNING_RATE=0.0002`
- `MAX_WEIGHT_DELTA=0.001`
- `BEST_GUARD_MAX_REGRET_INCREASE_CP=0`
- `BEST_GUARD_BAD100_INCREASE=0`
- `BEST_GUARD_TEACHER_MATCH_DROP_PCT=0`
- hard feedbackはfull train/validを使用。

refresh stage:

- baseline best metric score `39.474091`
- epoch 1 best metric score `38.353870`
- teacher-match guard score `0.229167 -> 0.241667`
- `best_epoch=1`
- score gate passed
- 200局面rerank:
  - baseline/candidate mean `6.26`
  - baseline/candidate match `42.50%`
  - failed by exact mean comparison
- loopはhard feedbackへ遷移。

hard feedback stage:

- hard pairs:
  - `written=7 skipped=0`
  - selected regret mean `94.60`
  - p95 `179.84`
  - bad100 `42.86%`
- baseline:
  - valid teacher_match `17.92%`
  - extra-valid dagger teacher_match `28.57%`
  - teacher-match guard score `0.250595`
- epoch 1:
  - valid teacher_match `19.17%`
  - extra-valid dagger teacher_match `14.29%`
  - teacher-match guard score `0.227381`
  - `best_guard_passed=false`
- epoch 2:
  - valid teacher_match `20.00%`
  - extra-valid dagger teacher_match `14.29%`
  - teacher-match guard score `0.235714`
  - `best_guard_passed=false`
- `best_epoch=0`
- loop final candidate: `policy_weights_v2.1.0.binary`

解釈:

teacher-match guardは、広いrerankでmatch低下していた種類の候補を、期待通りtrainer段階で拒否した。これにより長時間学習ループはさらに保守的になった。現時点では100/200規模で安全に採用できる候補はまだ出ていないが、「危険な候補を落とす」機構は機能している。次は候補を通すために、matchを落とさない目的関数側、具体的にはteacher topを直接維持するmargin項またはmatch-aware replay重みを検討する。

### match-top min-regret zero trial

Probe:

`data/mmto/runs/mmto_refresh_loop_matchtop100_20260627_051127`

目的:

teacher top一致率を落とす候補を避けるため、`CURRENT_TOP_MIN_BAD_REGRET_CP=0` にして、regretが小さいmodel top候補にもteacher top marginをかける設定を試した。

設定:

- match-guarded100と同じ
- 追加:
  - `CURRENT_TOP_MIN_BAD_REGRET_CP=0`

結果:

- baseline:
  - valid teacher_match `17.92%`
  - extra-valid refresh teacher_match `20.00%`
  - teacher-match guard score `0.229167`
- epoch 1:
  - valid teacher_match `17.08%`
  - extra-valid refresh teacher_match `10.00%`
  - teacher-match guard score `0.195833`
  - `best_guard_passed=false`
- epoch 2:
  - valid teacher_match `17.08%`
  - extra-valid refresh teacher_match `10.00%`
  - teacher-match guard score `0.195833`
  - `best_guard_passed=false`
- `best_epoch=0`

解釈:

`CURRENT_TOP_MIN_BAD_REGRET_CP=0` は、teacher top維持を強める意図とは逆に、extra-valid上のteacher matchを大きく落とした。現時点ではこの単純な設定変更は不採用。matchを守るには、current-top hard pairの閾値を下げるだけでは不十分で、teacher topを明示的に維持する別の目的関数またはhard局面の重み設計が必要。

### teacher-top CE objective

背景:

これまでのMMTO-lite/Bonanza-root系学習では、trainer上のlossや一部regret指標は下がる一方、rerank gateやteacher matchに効かない候補が多かった。単純に長時間回す前に、教師top手を直接維持する目的関数が必要と判断した。

実装:

- `mmto_tree_train` に `--teacher-top-ce-weight` を追加。
- listwise候補集合上で、`teacher_score` 最大の候補をteacher topとして扱う。
- 既存のteacher softmax CEに加え、`-w * ln(p_teacher_top)` を補助lossとして加算。
- gradientは `w * (p_i - 1[i == teacher_top]) / model_temperature_cp` を既存listwise gradientへ加算。
- weight `0` では既存挙動を維持する。
- 主要MMTOスクリプトから `TEACHER_TOP_CE_WEIGHT` で指定可能にした。
- `tools/run_mmto_from_dump.sh` は `BEST_METRIC` を環境変数で指定可能に修正した。

Probe:

`data/mmto/runs/mmto_teacher_top_ce_probe_20260627_051920`

入力:

- `data/mmto/runs/mmto_refresh_candidate_20260627_040733` のdumpからtrain 300 / valid 80を切り出し。
- `loss-mode=listwise-leaf`
- `learning-rate=0.0005`
- `max-weight-delta=0.003`
- `best-metric=teacher-mismatch`

weight感度:

- `TEACHER_TOP_CE_WEIGHT=0`
  - valid teacher_match: baseline `21.25%`
  - epoch 2: `21.25%`
  - `best_epoch=0`
- `0.02`
  - epoch 2: `21.25%`
  - `best_epoch=0`
- `0.05`
  - epoch 2: `21.25%`
  - `best_epoch=0`
- `0.2`
  - epoch 2 valid teacher_match: `23.75%`
  - `best_epoch=2`
- `0.5`
  - epoch 2 valid teacher_match: `26.25%`
  - bad50: `3.75% -> 2.50%`
  - `best_epoch=2`
- `1.0`
  - epoch 1 valid teacher_match: `26.25%`
  - `best_epoch=1`

Gate:

`data/mmto/runs/mmto_teacher_top_ce_gate_20260627_052255`

設定:

- `TEACHER_TOP_CE_WEIGHT=0.5`
- train 300 / valid 80
- `best-metric=teacher-mismatch`
- `best-guard-max-regret-increase-cp=0`
- `best-guard-bad100-increase=0`
- `best-guard-teacher-match-drop-pct=0`

結果:

- trainer:
  - baseline valid teacher_match `21.25%`
  - epoch 2 valid teacher_match `26.25%`
  - `best_epoch=2`
- score gate:
  - mean abs delta `0.47cp`
  - p95 `0.78cp`
  - max `0.88cp`
  - passed
- rerank gate 80 positions:
  - baseline mean regret `7.80`
  - candidate mean regret `7.42`
  - p90 `24.25 -> 23.67`
  - p95 `33.33 -> 32.47`
  - match `8.75% -> 10.00%`
  - passed

解釈:

teacher-top CEは、少なくとも小規模dumpではteacher matchとrerank regretを同時に改善した。これは、これまでの「lossは下がるがrerankに効かない」実験群と異なる良い兆候である。ただしtrain 300 / valid 80の小規模検証であり、この結果だけで長時間学習へ進むのは早い。

次の検証:

1. `TEACHER_TOP_CE_WEIGHT=0.5` を第一候補、`0.2` と `1.0` を比較候補として、train 1000から2000 / valid 200以上で再確認する。
2. `best-metric=teacher-mismatch` だけでなく、`p95-regret` とteacher-match guardを組み合わせた選択も比較する。
3. 200から500局面rerank gateでmean/p90/p95/matchが同時に悪化しないことを確認する。
4. ここまで通った場合のみ、10000局面級dumpで数時間学習へ進む。

### teacher-top CE 1k/200 validation

Probe:

`data/mmto/runs/mmto_teacher_top_ce_1k_20260627_052549`

目的:

train 300 / valid 80で見えたteacher-top CEの改善が、小規模ノイズではないかを確認するため、既存dumpからtrain 1000 / valid 200を切り出して再検証した。

設定:

- `SOURCE_RUN_DIR=data/mmto/runs/mmto_refresh_candidate_20260627_040733`
- `TRAIN_LINES=1000`
- `VALID_LINES=200`
- `LOSS_MODE=listwise-leaf`
- `TEACHER_TOP_CE_WEIGHT=0.5`
- `BEST_METRIC=teacher-mismatch`
- `BEST_GUARD_MAX_REGRET_INCREASE_CP=0`
- `BEST_GUARD_BAD100_INCREASE=0`
- `BEST_GUARD_TEACHER_MATCH_DROP_PCT=0`
- `EPOCHS=3`
- `LEARNING_RATE=0.0005`
- `MAX_WEIGHT_DELTA=0.003`
- `RERANK_MAX_POSITIONS=200`

trainer結果:

- baseline valid:
  - selected regret mean `11.56`
  - p90 `27.20`
  - p95 `49.63`
  - teacher_match `17.00%`
  - bad50 `5.00%`
- epoch 3:
  - selected regret mean `10.99`
  - p90 `26.69`
  - p95 `32.06`
  - teacher_match `20.50%`
  - bad50 `3.50%`
  - `best_epoch=3`

score gate:

- samples `1023`
- mean abs delta `1.06cp`
- p95 `1.64cp`
- max `1.83cp`
- passed

rerank gate 200 positions:

- baseline:
  - mean `6.01`
  - p90 `19.25`
  - p95 `29.12`
  - match `16.00%`
- candidate:
  - mean `5.70`
  - p90 `18.68`
  - p95 `28.34`
  - match `16.00%`
- passed

解釈:

teacher-top CEは、train 1000 / valid 200でもtrainer上のteacher match、tail regret、rerank regretを同時に改善した。match自体はrerankで横ばいだが、過去の候補で問題だったmatch低下は発生していない。これは長時間学習候補として明確に前進である。

注意:

`BLEND_RATIOS=""` を指定したが、`tools/run_mmto_from_dump.sh` は未指定扱いにして既定のblend重みを生成した。重み3本は削除済み。容量事故を避けるため、`BLEND_RATIOS` は未指定時のみ既定値を使い、空文字指定時はblend生成を止められるようスクリプトを修正した。

次の検証:

1. 同設定でtrain 2000 / valid 400から500、rerank 400から500を実施する。
2. その結果も通る場合、`TEACHER_TOP_CE_WEIGHT=0.2/0.5/1.0` の比較を2000規模で行う。
3. 2k/500で安定したら10k級の数時間学習に進む。

### teacher-top CE 2k comparison

Probe:

- `data/mmto/runs/mmto_teacher_top_ce_2k_20260627_052904` (`w=0.5`)
- `data/mmto/runs/mmto_teacher_top_ce_2k_w02_20260627_053205` (`w=0.2`)
- `data/mmto/runs/mmto_teacher_top_ce_2k_w10_20260627_053206` (`w=1.0`)

目的:

同じdump全体に近いtrain 2000 / valid 230で、teacher-top CE weightの感度を確認した。valid件数が230しかないため、rerankも230局面で実施した。

共通設定:

- `SOURCE_RUN_DIR=data/mmto/runs/mmto_refresh_candidate_20260627_040733`
- `TRAIN_LINES=2000`
- `VALID_LINES=230`
- `LOSS_MODE=listwise-leaf`
- `BEST_METRIC=teacher-mismatch`
- `BEST_GUARD_MAX_REGRET_INCREASE_CP=0`
- `BEST_GUARD_BAD100_INCREASE=0`
- `BEST_GUARD_TEACHER_MATCH_DROP_PCT=0`
- `EPOCHS=3`
- `LEARNING_RATE=0.0005`
- `MAX_WEIGHT_DELTA=0.003`
- `RERANK_MAX_POSITIONS=230`

baseline valid:

- selected regret mean `10.74`
- p90 `26.41`
- p95 `42.74`
- teacher_match `17.83%`
- bad50 `4.35%`

trainer結果:

| weight | best_epoch | valid selected regret | valid p95 | valid teacher_match | valid bad50 |
|---:|---:|---:|---:|---:|---:|
| `0.2` | 1 | `10.51` | `32.06` | `21.30%` | `3.04%` |
| `0.5` | 3 | `10.57` | `33.37` | `21.74%` | `3.04%` |
| `1.0` | 3 | `10.50` | `33.37` | `23.04%` | `3.04%` |

rerank 230結果:

| weight | mean | p90 | p95 | match |
|---:|---:|---:|---:|---:|
| baseline | `5.88` | `18.91` | `29.17` | `17.39%` |
| `0.2` | `5.63` | `18.48` | `28.62` | `17.39%` |
| `0.5` | `5.59` | `18.48` | `28.62` | `17.39%` |
| `1.0` | `5.55` | `18.48` | `28.62` | `17.39%` |

解釈:

2k/230では `0.2`, `0.5`, `1.0` がすべてoffline gateを通った。rerankのp90/p95/matchは同等だが、mean regretは `1.0` が最良で、trainer上のteacher_matchも最も高い。このデータ源では `TEACHER_TOP_CE_WEIGHT=1.0` を次の本命にする。

注意:

全runで `max_weight_delta=0.003` によるclampが多い。長時間学習では単にepochを伸ばすより、`learning-rate` と `max_weight_delta` の組み合わせを調整する必要がある。

次の検証:

1. より大きい `teacher_wdoor50k_d3s2_top16_20260626_164759` dumpでtrain 5000 / valid 500を実施する。
2. まず `TEACHER_TOP_CE_WEIGHT=1.0` を試し、通れば `0.5` を比較する。
3. rerank 500でmatch低下なし、mean/p90/p95改善を確認できた場合のみ、10k以上へ進む。

### teacher-top CE Wdoor 5k/500 validation

Probe:

- `data/mmto/runs/mmto_teacher_top_ce_wdoor5k_w10_20260627_053635` (`w=1.0`)
- `data/mmto/runs/mmto_teacher_top_ce_wdoor5k_w05_20260627_054227` (`w=0.5`)
- `data/mmto/runs/mmto_teacher_top_ce_wdoor5k_w02_20260627_054826` (`w=0.2`)
- `data/mmto/runs/mmto_teacher_top_ce_wdoor5k_w05_dedup_20260627_055717` (`w=0.5`, dedupe rerank)

目的:

2k/230で通ったteacher-top CEが、Wdoor由来の大きめdumpで長時間学習に進める水準か確認した。

共通設定:

- `SOURCE_RUN_DIR=data/mmto/runs/teacher_wdoor50k_d3s2_top16_20260626_164759`
- `TRAIN_LINES=5000`
- `VALID_LINES=500`
- `LOSS_MODE=listwise-leaf`
- `BEST_METRIC=teacher-mismatch`
- `EPOCHS=3`
- `LEARNING_RATE=0.0005`
- `MAX_WEIGHT_DELTA=0.003`
- `RERANK_MAX_POSITIONS=500`

trainer上の傾向:

- baseline valid teacher_match `20.60%`
- `w=1.0`:
  - best epoch 1
  - valid teacher_match `25.40%`
  - selected regret mean `36.63 -> 36.12`
- `w=0.5`:
  - best epoch 3
  - valid teacher_match `24.60%`
  - selected regret mean `36.63 -> 35.95`
- `w=0.2`:
  - best epoch 1
  - valid teacher_match `23.40%`
  - selected regret mean `36.63 -> 36.19`

通常rerank 500:

| weight | mean | p90 | p95 | match | result |
|---:|---:|---:|---:|---:|---|
| baseline | `208.00` | `26.62` | `42.42` | `42.80%` | - |
| `0.2` | `208.08` | `26.83` | `42.42` | `42.60%` | failed |
| `0.5` | `207.98` | `26.74` | `41.19` | `42.60%` | failed |
| `1.0` | `208.13` | `26.91` | `42.42` | `42.60%` | failed |

dedupe rerank:

同一SFENの重複がworst deltaに出ていたため、`mmto_rerank_gate` に `--dedupe-sfen` を追加し、`RERANK_DEDUPE_SFEN=1` でスクリプトから使えるようにした。Wdoor 5k/500では500件中21件が重複除外され、479ユニーク局面で評価された。

`w=0.5` dedupe rerank:

- baseline:
  - mean `216.78`
  - p90 `27.27`
  - p95 `42.50`
  - match `44.68%`
  - bad50 `3.55%`
- candidate:
  - mean `216.70`
  - p90 `27.27`
  - p95 `41.89`
  - match `44.47%`
  - bad50 `3.34%`
- failed:
  - matchが1局面分低下。

解釈:

teacher-top CEは、trainer上ではteacher matchを明確に改善し、rerank上でもmean/p95/bad50を改善する場合がある。しかし、Wdoor 5k/500では全weightでmatchがわずかに低下した。dedupeしてもmatch低下は残ったため、単純なteacher-top CEだけでは「長時間回す価値のある採用候補」にはまだ足りない。

重要な観察:

- 2k/230では通るが、5k/500では落ちる。
- rerank match低下は小さいが、長時間学習で増幅するリスクがある。
- worst deltaには序盤2手目付近の局面や、終盤の大きな外れ値が含まれる。
- `max_weight_delta=0.003` でclampが多く、epoch増加による改善は頭打ちになっている。

次の方針:

1. teacher-top CE単独で10k級へ進むのは停止。
2. rerankでmatchを落とした局面をhard feedbackとして再投入する。
3. 目的関数に「teacher top CE」だけでなく、「現在candidateがteacher一致を壊した局面への保持ペナルティ」を追加する。
4. Wdoor dumpではvalid/rerankをdedupeして使う。
5. 10k以上へ進む条件は、5k/500 dedupe rerankでmean/p90/p95/bad50改善かつmatch低下なし。

### teacher-top CE hard feedback trial

Probe:

`data/mmto/runs/mmto_teacher_top_ce_wdoor5k_w05_dagger_20260627_060452`

目的:

`w=0.5` のdedupe rerank失敗runからhard positions / hard pairsを抽出し、DAgger replayとして再投入すればmatch低下を消せるか確認した。

設定:

- source: `data/mmto/runs/mmto_teacher_top_ce_wdoor5k_w05_dedup_20260627_055717`
- explicit hard pairs:
  - written `45`
  - skipped `0`
- `DAGGER_MAX_POSITIONS=200`
- `TEACHER_TOP_CE_WEIGHT=0.5`
- `CURRENT_TOP_MARGIN_WEIGHT=0.05`
- `REPLAY_WEIGHT=0.10`
- `EXTRA_VALID_BEST_WEIGHT=0.25`
- `BEST_METRIC=teacher-mismatch`
- `RERANK_DEDUPE_SFEN=1`

注意:

このrunでは `run_mmto_dagger_from_run.sh` の既定によりrerankのbaseline/candidate depthが `3` になった。直前のfrom-dump検証はdepth `4` なので、絶対値は直接比較しない。

trainer結果:

- explicit hard pairsは45件生成。
- extra-valid[dagger] teacher_match:
  - baseline `20.00%`
  - epoch 3 `24.44%`
- valid teacher_match:
  - baseline `20.60%`
  - epoch 3 `24.60%`
- best_epoch `3`
- score gate passed:
  - mean abs delta `0.42cp`
  - p95 `1.28cp`
  - max `1.88cp`

dedupe rerank 479 positions, depth 3:

- baseline:
  - mean `433.16`
  - p90 `43.54`
  - p95 `86.46`
  - match `45.72%`
  - bad50 `9.39%`
- candidate:
  - mean `433.08`
  - p90 `43.64`
  - p95 `86.46`
  - match `45.30%`
  - bad50 `9.19%`
- failed:
  - p90 slight worse
  - match lower

解釈:

hard feedbackを足しても、trainer上のteacher_match改善はrerank match維持に変換されなかった。明示hard pairsの抽出は動作しているが、現状の学習目的では「teacher一致を壊した局面だけを確実に戻す」力が弱い。次は単なるreplayではなく、rerank JSONの `baseline_move != candidate_move` かつ `baseline_regret <= candidate_regret` の局面を直接teacher/baseline保持する専用lossが必要。

次の実装候補:

1. rerank feedback JSONLを `mmto_tree_train` が直接読む。
2. その中で `candidate_regret > baseline_regret` または `baseline exact match && candidate not match` の局面を保持対象にする。
3. 既存候補集合内にbaseline move / teacher moveがあれば、その手にone-hot CEまたはpairwise marginをかける。
4. 目的は「新候補が壊した局面を戻す」ことであり、全局面のteacher-top CEとは別扱いにする。

### explicit student margin loss

目的:

teacher-top CEとhard replayだけでは、rerank上でcandidateが壊した局面を十分に戻せなかった。そこで `mmto_tree_train` に `--explicit-student-margin-weight` を追加し、dump JSONL内の `game_teacher_move` / `teacher_best_move` と `selected_by_student` が異なる場合に、teacher側の手を上げ、student側の手を下げるsoftplus marginを直接追加した。

実装:

- `src/bin/mmto_tree_train.rs`
  - `LossOptions::explicit_student_margin_weight` を追加。
  - `explicit_student_hard_pair()` を追加。
  - pairwise / listwise の両モードで、明示teacher手 vs 明示student手のmargin lossを加算。
  - 既定値は `0.0` なので既存挙動は変えない。
- `tools/run_mmto_dagger_from_run.sh`
  - `EXPLICIT_STUDENT_MARGIN_WEIGHT` を追加し、`mmto_tree_train` に渡す。

検証:

- `cargo fmt --check`
- `FONTCONFIG_NO_PKG_CONFIG=1 RUST_FONTCONFIG_DLOPEN=1 cargo check --bin mmto_tree_train`
- `FONTCONFIG_NO_PKG_CONFIG=1 RUST_FONTCONFIG_DLOPEN=1 cargo test --all-targets`

#### explicit margin probe

基準run:

`data/mmto/runs/mmto_teacher_top_ce_wdoor5k_w05_dedup_20260627_055717`

共通設定:

- `TRAIN_LINES=3000`
- `VALID_LINES=500`
- `DAGGER_MAX_POSITIONS=100`
- `TEACHER_DEPTH=5`
- `STUDENT_DEPTH=4`
- `RERANK_TEACHER_DEPTH=5`
- `TEACHER_TOP_CE_WEIGHT=0.5`
- `REPLAY_WEIGHT=0.5`
- `LISTWISE_HARD_NEGATIVE_WEIGHT=0`
- `GAME_TEACHER_MARGIN_WEIGHT=0`
- `CURRENT_TOP_MARGIN_WEIGHT=0`
- `BEST_METRIC=teacher-mismatch`
- `RERANK_DEDUPE_SFEN=1`
- rerank positions: `479` unique SFEN

結果:

| explicit weight | run | score gate | rerank gate | mean | p90 | p95 | match | bad50 | bad100 | best_epoch | best_metric_score |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | - | - | - | `216.78363` | `27.265059` | `42.500510` | `44.6764%` | `3.5491%` | `0.8351%` | - | - |
| `0.5` | `data/mmto/runs/mmto_explicit_margin_w05_20260627_063557` | pass | pass | `216.74211` | `27.265059` | `42.500510` | `45.3027%` | `3.5491%` | `0.8351%` | `3` | `0.985909` |
| `1.0` | `data/mmto/runs/mmto_explicit_student_margin_probe_20260627_062100` | pass | pass | `216.72937` | `27.265059` | `42.500510` | `45.3027%` | `3.5491%` | `0.8351%` | `3` | `0.984227` |
| `2.0` | `data/mmto/runs/mmto_explicit_margin_w20_20260627_064850` | pass | pass | `216.72937` | `27.265059` | `42.500510` | `45.3027%` | `3.5491%` | `0.8351%` | `3` | `0.984227` |

解釈:

- explicit student marginは、少なくとも小規模probeではteacher-top CEのmatch低下問題を解消し、rerank gateを通過した。
- `0.5` より `1.0` / `2.0` が少し良いが、`1.0` と `2.0` は同一結果だった。
- 改善幅は非常に小さい。mean regretは約 `0.05cp` 改善、matchは3局面/479相当の改善に留まる。
- p90/p95/bad50/bad100は変化なし。長時間学習へ進む前に、より大きいdumpまたは別valid splitで再現性を確認する必要がある。

判断:

この変更は「長時間学習を回す価値のある目的関数」に一歩近づいたが、まだ本格長時間学習の根拠としては弱い。次は `EXPLICIT_STUDENT_MARGIN_WEIGHT=1.0` または `2.0` を候補に、5k/500とは別seedまたは10k級で再現性を確認する。採用条件は、dedupe rerankでmatch低下なし、mean悪化なし、できればp90/p95またはbad50改善を伴うこと。

運用メモ:

実験runに残った `best.raw.binary` は容量節約のため削除した。結果は `rerank_gate.json`、`score_gate.json`、`train_stdout.log` に残っている。

#### explicit margin 9k repro

目的:

5k/500 probeの通過が小さいvalid splitへの偶然ではないか確認するため、別dump `data/mmto/runs/mmto_100k_pilot_balanced9k_20260626_191235` を使って中規模再現を行った。

設定:

- source: `data/mmto/runs/mmto_100k_pilot_balanced9k_20260626_191235`
- run: `data/mmto/runs/mmto_explicit_margin_9k_repro_20260627_070444`
- `TRAIN_LINES=9000`
- `VALID_LINES=1000`
- `DAGGER_MAX_POSITIONS=200`
- `TEACHER_DEPTH=5`
- `STUDENT_DEPTH=4`
- `RERANK_TEACHER_DEPTH=5`
- `TEACHER_TOP_CE_WEIGHT=0.5`
- `EXPLICIT_STUDENT_MARGIN_WEIGHT=2.0`
- `REPLAY_WEIGHT=0.5`
- `BEST_METRIC=teacher-mismatch`
- `RERANK_DEDUPE_SFEN=1`

結果:

- score gate: pass
- rerank gate: failed
- best_epoch: `3`
- best_metric_score: `0.978502`
- baseline best_metric_score: `1.027150`

rerank:

| side | mean | p90 | p95 | match | bad50 | bad100 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | `243.65791` | `27.491108` | `42.636803` | `44.155845%` | `3.423849%` | `0.590319%` |
| candidate | `243.73343` | `28.151691` | `42.500510` | `43.919715%` | `3.305785%` | `0.590319%` |

失敗理由:

- mean regret worsened: `243.73 > 243.66`
- p90 regret worsened: `28.15 > 27.49`
- match rate failed: `43.92% < 44.16%`

解釈:

9k/1kではtrainer上のbest metricが改善しても、実際のrerankではmean/p90/matchが悪化した。5k/500で見えた改善は再現せず、単純に `EXPLICIT_STUDENT_MARGIN_WEIGHT` を強くして長時間回す根拠はない。特にteacher-mismatchをbest metricにしてもrerank matchが落ちるため、現在のoffline best selection自体が探索改善を十分に代理していない。

判断:

- explicit student marginは補助lossとして残せるが、これ単独で長時間学習に進むのは停止。
- 次は、rerank失敗局面をdump replayへ変換するだけでなく、rerank JSONのbaseline/candidate/teacherの比較結果を直接使うfeedback objectiveを実装する必要がある。
- 採用条件は、別dumpの9k/1kまたは同等規模で、mean/p90/matchを同時に悪化させないこと。best_metric改善だけでは不採用。

運用メモ:

このrunに残った `.binary` は削除済み。結果は `rerank_gate.json`、`score_gate.json`、`train_stdout.log` に残っている。

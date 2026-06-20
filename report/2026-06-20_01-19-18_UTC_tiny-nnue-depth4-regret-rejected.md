# TinyNNUE depth4 H=64 root regret良好だが対局棄却

- 作成日時: 2026-06-20 01:19:18 UTC
- ブランチ: `tooling/value-regret-engine-evaluator`
- 目的: 対局前にTinyNNUE候補のroot regretを測れるようにし、depth4教師値で学習したH=64候補を検証する。

## 実装内容

`value_regret_probe` を `EngineEvaluator` 対応に変更した。

- teacher/candidateのどちらにも既存KPP `.binary` とTinyNNUE `TNNUE001` を指定可能。
- KPP対KPP sanity checkでは8局面すべてregret 0、手一致100%。

## depth4データ

サブエージェントで生成:

```text
target/release/nnue_feature_dump \
  --input taya36.sfen \
  --output /tmp/nnue_taya_depth4_512.jsonl \
  --weights policy_weights_v2.1.0.binary \
  --depth 4 \
  --max-positions 512 \
  --jobs 4
```

結果:

```text
records: 512
skipped positions: 0
elapsed: 28.260s
file size: 206146 bytes
```

## H=64学習

```text
baseline valid rmse=9.72 mae=5.31 sign=85.94%
best epoch: 25 valid_rmse=6.68 valid_mae=4.30 valid_sign=84.38%
```

offline RMSEは既存KPP `static_eval` より改善した。

## root regret

条件:

```text
teacher: policy_weights_v2.1.0.binary
candidate: /tmp/tiny_nnue_depth4_h64_best.bin
teacher-depth: 4
candidate-depth: 4
positions: taya36.sfen
max-positions: 64
```

結果:

```text
mean_regret_cp: 5.51
p50_regret_cp: 0.58
p90_regret_cp: 5.58
p95_regret_cp: 6.81
max_regret_cp: 233.09
bad_regret_count_gt_300: 0 (0.00%)
teacher_move_match: 22 (34.38%)
```

root regretだけを見ると悪くない候補に見えた。

## 20局ベンチ

条件:

```text
new: /tmp/tiny_nnue_depth4_h64_best.bin
baseline: policy_weights_v2.1.0.binary
games: 20
depth: 5
time-limit-ms: 100
seed: 6103
```

結果:

```text
new wins: 2
baseline wins: 17
draws: 1
new decisive win rate: 10.53%
new total score rate: 12.50%
paired starts:
  new sweeps: 0
  baseline sweeps: 7
  splits: 2
  draw/mixed pairs: 1
average final score for new: -862.1
```

## 判断

`value_regret_probe` のTinyNNUE対応は採用する。候補評価関数を対局前に調べるために有用である。

ただし、depth4 H=64重みは棄却する。root regretが良くても対局では崩壊した。浅いroot局面だけでは、探索内部の静止探索・末端評価分布・終盤のスケールずれを検出できない。

次の方針:

1. TinyNNUEの次実験はroot valueだけでなく、子局面ranking/top-kまたは探索内部に近い局面集合で検証する。
2. `taya36.sfen` 512局面だけの重み更新は繰り返さない。
3. depth4/5教師値を使う場合も、局面供給を通常局面・敗局tail・探索内部ノードに広げる。
4. この候補重みは削除し、リリース対象にしない。

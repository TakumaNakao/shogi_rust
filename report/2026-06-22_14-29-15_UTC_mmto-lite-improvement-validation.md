# MMTO-lite改善実装と検証結果

- 作成日時: 2026-06-22 14:29:15 UTC
- 対象ブランチ: `feature/mmto-lite-pairwise-gates`
- 目的: MMTO-liteを実戦投入できる水準へ近づけるため、学習信号、validation、pairwise補助を改善して検証する。

## 結論

実装改善は有効だったが、実戦投入レベルにはまだ届いていない。

full-legal validationでは、従来のtop16 trainよりもtop128 trainの方が明確にoffline regretを改善した。最良候補は `valid selected regret mean` を 76.99 から 64.78 まで下げた。

しかし対局では以下に留まった。

```text
blend_0.02 / 20局: 11-9, score rate 55.00%
blend_0.02 / 40局: 20-19-1, score rate 51.25%
blend_0.05 / 20局: 7-10-3, score rate 42.50%
```

40局gateの55%以上には届かないため、リリース候補・実戦投入候補としては採用しない。

## GPT-5.5 xhighレビューの要点

重要判断:

1. full-legal validationを標準にする。
2. train専用gap/spanフィルタを追加し、validを簡単にしない。
3. listwiseに低重みのpairwise margin補助を追加する。
4. top8/top16 validでは揺らぎが大きく、topN外の危険手選択を見落とす。
5. 完全MMTOへはまだ進まず、MMTO-liteの信号品質を改善する。

## 実装した改善

対象:

```text
src/bin/mmto_train.rs
```

追加CLI:

```text
--loss listwise|listwise-pairwise
--train-min-teacher-gap
--train-max-teacher-gap
--train-min-score-span
--valid-filter none|same
--min-candidates
--pairwise-weight
--pairwise-gap
--pairwise-margin
--pairwise-max-pairs-per-sample
```

改善点:

- train専用teacher gapフィルタ。
- train専用score spanフィルタ。
- validは既定でフィルタなしにし、評価を簡単にしない。
- pairwise hinge補助損失。
- pairwise勾配を有効ペア数で正規化。
- teacher gap mean/p50をログ出力。
- pairwise accuracyをログ出力。
- `--pairwise-gap-cp` / `--pairwise-margin-cp` はaliasとして残した。

## 検証1: 小規模既存top8データ

既存データ:

```text
data/mmto/runs/d3_top8_200_20260622_131843
```

`--loss listwise-pairwise --train-min-teacher-gap 1 --pairwise-weight 0.1` では、filtered valid 20局面で以下のように順位精度は改善した。

```text
valid top1: 25.00% -> 55.00% / 60.00%
valid pairwise: 55.64% -> 62%台
```

ただしselected regretは安定改善せず、p95が悪化したため採用しない。

## 検証2: top16 / full-legal valid

データ:

```text
data/mmto/runs/d3_top16_1000_fullvalid_20260622_140525
```

条件:

- depth 3
- max positions 1000
- train top16
- valid full-legal
- valid records 100

結果:

```text
baseline valid selected regret mean: 60.09
best listwise valid selected regret mean: 59.98
```

改善が小さすぎる。top16 trainではfull-legal valid上の危険手を十分に抑えられないと判断した。

## 検証3: top128 / full-legal valid

データ:

```text
data/mmto/runs/d3_top128_1000_fullvalid_20260622_140845
```

条件:

- depth 3
- max positions 1000
- train top128
- valid full-legal
- train records 898
- valid records 100

top128は多くの通常局面で実質full-legal trainになる。JSONLサイズは約4MBで現実的だった。

### 通常delta

`max-weight-delta 0.02/0.05` では改善が小さい。

```text
baseline valid selected regret mean: 76.99
best around: 76.67
```

### delta 0.2

`max-weight-delta 0.2` まで広げるとoffline改善が出た。

最良:

```text
learning-rate: 50
model-temperature: 30
teacher-temperature: 100
loss: listwise
max-weight-delta: 0.2
anchor-l2: 0.0001
```

valid推移:

```text
epoch 0: selected regret 76.99, p90 260.09, p95 265.19
epoch 1: selected regret 74.24, p90 235.57, p95 265.19
epoch 2: selected regret 69.23, p90 230.18, p95 264.14
epoch 3: selected regret 69.22, p90 230.18, p95 264.14
epoch 4: selected regret 69.09, p90 230.18, p95 264.14
epoch 5: selected regret 64.78, p90 209.76, p95 264.14
```

offline gateは通過と判断し、blend対局へ進めた。

## 対局検証

候補:

```text
data/mmto/runs/candidate_top128_lr50_mt30_d02_20260622_141859
```

### blend 0.02 / 20局

```text
record-dir: bench20_blend002_seed7101
new wins: 11
baseline wins: 9
draws: 0
score rate: 55.00%
paired starts:
  new sweeps: 3
  baseline sweeps: 2
  splits: 5
```

20局smokeは通過。

### blend 0.02 / 40局

```text
record-dir: bench40_blend002_seed7201
new wins: 20
baseline wins: 19
draws: 1
score rate: 51.25%
paired starts:
  new sweeps: 2
  baseline sweeps: 2
  splits: 15
  draw/mixed pairs: 1
```

40局gate不通過。保留ではなく不採用寄り。

### blend 0.05 / 20局

```text
record-dir: bench20_blend005_seed7301
new wins: 7
baseline wins: 10
draws: 3
score rate: 42.50%
paired starts:
  new sweeps: 0
  baseline sweeps: 2
  splits: 6
  draw/mixed pairs: 2
```

blendを強めると悪化した。

## 観察

1. full-legal validationを導入したことで、topN外の危険手選択が見えるようになった。
2. top16 trainではfull-legal validをほとんど改善できない。
3. top128 train + delta 0.2ではoffline selected regretが約16%改善した。
4. offline改善は対局改善へ十分には転化しなかった。
5. pairwise補助はpairwise accuracyを上げるが、selected regretや対局力には直結しなかった。
6. blend 0.05で悪化したため、学習重みの方向には有害成分も含まれている。

## 判断

今回の改善でMMTO-liteは検証基盤として前進したが、実戦投入候補は作れていない。

リリース、タグ付け、重み採用は行わない。

## 次の改善案

優先順位:

1. full-legal validの高regret局面を抽出し、なぜ静的KPPが悪手を選ぶか分類する。
2. selected regretを直接下げるhard-negative lossを追加する。
3. model argmaxがteacher top外へ飛ぶ局面を重点学習する。
4. pairwiseは全ペアではなく「modelが選びそうな悪手 vs teacher上位」に限定する。
5. KPPだけで限界が見える場合、小型NNUE residualへ戻る。

次の実装候補:

```text
mmto_train:
  --loss listwise-hard-negative
  --hard-negative-weight
  --hard-negative-margin
  --hard-negative-top-model

mmto_probe:
  full-legal validでmodel argmax rank、selected regret、悪手USIを出す
```

完全MMTOへ進むのはまだ早い。まずroot full-legal上の悪手選択を潰す。
